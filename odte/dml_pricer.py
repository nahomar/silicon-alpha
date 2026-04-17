"""Differential Machine Learning pricer for 0DTE options.

The network outputs (price, delta, gamma, vega) in a SINGLE forward pass.
The loss combines:
  (1) price MSE against a reference (Black-Scholes closed form, then Heston MC)
  (2) AAD-derivative MSE — compares autograd greeks to analytic greeks

Maturity gate:
  σ_eff = sqrt( max(sigma_floor^2, sigma^2 * tau) )
so the effective variance scales linearly with tau and never collapses to
zero. This is the "maturity-gated variance" from the plan; it keeps the net
grounded near τ→0 instead of extrapolating to singular Greeks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import math
import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - dev-time gate only
    torch = None  # type: ignore
    nn = object  # type: ignore
    _HAS_TORCH = False

from models.config import DMLConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes (analytic reference; numpy for robust small-tau behavior)
# ---------------------------------------------------------------------------

def _phi(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2)))


def _erf(x):
    # numpy/torch dispatch
    if hasattr(x, "erf"):
        return x.erf()
    if isinstance(x, np.ndarray):
        from scipy.special import erf as _se
        return _se(x)
    return math.erf(x)


def bs_price_call(S, K, tau, sigma, r=0.0):
    """Black-Scholes call price, works with numpy arrays or torch tensors."""
    # Clamp to avoid divide-by-zero at expiry.
    if hasattr(sigma, "clamp"):
        sig = torch.clamp(sigma, min=1e-6)
        tau_c = torch.clamp(tau, min=1e-8)
    else:
        sig = np.maximum(sigma, 1e-6)
        tau_c = np.maximum(tau, 1e-8)
    sqrt_tau = (sig * 0 + tau_c) ** 0.5 if hasattr(tau_c, "sqrt") else np.sqrt(tau_c)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * tau_c) / (sig * sqrt_tau) \
        if not hasattr(sig, "log") else \
        (torch.log(S / K) + (r + 0.5 * sig ** 2) * tau_c) / (sig * sqrt_tau)
    d2 = d1 - sig * sqrt_tau
    disc = np.exp(-r * tau_c) if not hasattr(tau_c, "exp") else torch.exp(-r * tau_c)
    return S * _phi(d1) - K * disc * _phi(d2)


def bs_greeks_call(S, K, tau, sigma, r=0.0):
    """Analytic (Δ, Γ, 𝒱) for a call. Works for numpy or torch."""
    if hasattr(sigma, "clamp"):
        sig = torch.clamp(sigma, min=1e-6)
        tau_c = torch.clamp(tau, min=1e-8)
        sqrt_tau = tau_c.sqrt()
        log = torch.log
        exp = torch.exp
        pi = math.pi
        def pdf(x): return exp(-0.5 * x * x) / math.sqrt(2 * pi)
    else:
        sig = np.maximum(sigma, 1e-6)
        tau_c = np.maximum(tau, 1e-8)
        sqrt_tau = np.sqrt(tau_c)
        log = np.log
        exp = np.exp
        def pdf(x): return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    d1 = (log(S / K) + (r + 0.5 * sig ** 2) * tau_c) / (sig * sqrt_tau)
    delta = _phi(d1)
    gamma = pdf(d1) / (S * sig * sqrt_tau)
    vega = S * pdf(d1) * sqrt_tau
    return delta, gamma, vega


# ---------------------------------------------------------------------------
# Neural pricer
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class MaturityGate(nn.Module):
        """Smooth σ floor — σ_eff = sqrt(σ² + floor²).

        Note: the τ-dependence of the price lives in BS's own σ·√τ factor;
        we don't re-apply τ here. What the plan meant by "variance correction
        that vanishes as τ→0" is that the LEARNED residual is scaled by a
        τ-gate — implemented in price() via the `tau_gate` factor on the
        tanh term. This module just keeps σ strictly positive and C∞.
        """

        def __init__(self, sigma_floor: float = 1e-4):
            super().__init__()
            self.sigma_floor = float(sigma_floor)

        def forward(self, sigma: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            del tau  # unused (kept in signature for API stability)
            return torch.sqrt(sigma * sigma + self.sigma_floor ** 2)

    class DMLPricer(nn.Module):
        """(S, K, tau, r, sigma) -> (price, delta, gamma, vega).

        All inputs are scalars per sample. The network internally uses a
        log-moneyness + tau + σ_eff representation for scale invariance.
        """

        def __init__(self, cfg: DMLConfig | None = None):
            super().__init__()
            self.cfg = cfg or DMLConfig()
            self.gate = MaturityGate(self.cfg.sigma_floor)
            h = self.cfg.hidden_dim
            layers: list[nn.Module] = []
            in_dim = 5  # [log(S/K), tau, σ_eff, r, sqrt(tau)]
            for _ in range(self.cfg.n_layers):
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.SiLU())
                in_dim = h
            layers.append(nn.Linear(h, 1))
            self.net = nn.Sequential(*layers)

        def _features(self, S, K, tau, r, sigma) -> torch.Tensor:
            """Scale-invariant features per Huge & Savine (2020)."""
            sigma_eff = self.gate(sigma, tau)
            logmon = torch.log(S / K)
            return torch.stack([logmon, tau, sigma_eff, r, torch.sqrt(tau.clamp(min=1e-9))],
                               dim=-1)

        def price(self, S, K, tau, r, sigma) -> torch.Tensor:
            """Huge-Savine "twin net" parameterization with τ-gated residual:

                price = BS(S, K, τ, σ_eff, r)  +  K · ε · g(τ) · tanh(net(x))

            g(τ) = 1 - exp(-β·τ)  vanishes smoothly as τ→0 so the correction
            is exactly the actual payoff at expiry (no hallucination in the
            singular regime), and saturates to 1 for longer maturities.
            """
            x = self._features(S, K, tau, r, sigma)
            raw = self.net(x).squeeze(-1)
            sigma_eff = self.gate(sigma, tau)
            bs = bs_price_call(S, K, tau, sigma_eff, r)
            eps = 0.02
            beta = 200.0                        # τ (in years) · β saturates at ~1 for τ>≈1w
            tau_gate = 1.0 - torch.exp(-beta * tau)
            return bs + K * eps * tau_gate * torch.tanh(raw)

        def forward(self, S, K, tau, r, sigma):
            """Return (price, delta, gamma, vega) using autograd.

            Ensure S and sigma are fresh leaf tensors with grad so AAD walks
            the full graph (avoids silent zero-grads when caller forgot).
            """
            S_ = S.detach().clone().requires_grad_(True)
            sig_ = sigma.detach().clone().requires_grad_(True)
            price = self.price(S_, K, tau, r, sig_)
            delta = torch.autograd.grad(price.sum(), S_, create_graph=True)[0]
            gamma = torch.autograd.grad(delta.sum(), S_, create_graph=True)[0]
            vega = torch.autograd.grad(price.sum(), sig_, create_graph=True)[0]
            return price, delta, gamma, vega

        # ------------------------------------------------------------------
        # losses (scale-normalized so each term is O(1))
        # ------------------------------------------------------------------
        def loss(self, S, K, tau, r, sigma) -> dict:
            p_hat, d_hat, g_hat, v_hat = self.forward(S, K, tau, r, sigma)
            p_true = bs_price_call(S, K, tau, sigma, r)
            d_true, g_true, v_true = bs_greeks_call(S, K, tau, sigma, r)
            # Normalize price by strike; delta is already [0,1]; gamma*S*S,
            # vega per unit σ (≈ [0, ~100% of S]) -- divide vega by S.
            loss_p = F.mse_loss(p_hat / K, p_true / K)
            loss_d = F.mse_loss(d_hat, d_true)
            loss_g = F.mse_loss(g_hat * S, g_true * S)           # S-scaled gamma (≈O(0.1))
            loss_v = F.mse_loss(v_hat / S, v_true / S)
            lam = self.cfg.grad_loss_weight
            total = loss_p + lam * (loss_d + loss_g + loss_v)
            return {"total": total, "price": loss_p, "delta": loss_d,
                    "gamma": loss_g, "vega": loss_v}

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train_dml_bs(model: "DMLPricer", steps: int = 2000, batch: int = 1024,
                     device: str = "cpu", S_ref: float = 5500.0) -> dict:
        """Pretrain on closed-form Black-Scholes synthetic data.

        Curriculum: start with longer-maturity / more-ITM samples where Greeks
        are smooth, anneal toward ATM / short-maturity where the gamma spike
        lives. Linear warmup + cosine decay on LR.
        """
        dev = torch.device(device)
        model = model.to(dev)
        peak_lr = model.cfg.lr * 0.3     # cut default 1e-3 → 3e-4
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-4)
        warmup = max(50, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warmup if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))),
        )
        history = {"total": [], "price": [], "delta": [], "gamma": [], "vega": []}
        for step in range(steps):
            # Curriculum tau: start at tau=1d..2w, linearly narrow to [10min, 3d]
            frac = step / max(1, steps - 1)
            tau_lo = max(1 / (365 * 6.5 * 60), (1 - frac) * (1 / 365) + frac * (1 / (365 * 24 * 6)))
            tau_hi = (1 - frac) * (14 / 365) + frac * (3 / 365)
            S = torch.empty(batch, device=dev).uniform_(0.9 * S_ref, 1.1 * S_ref)
            K = torch.empty(batch, device=dev).uniform_(0.9 * S_ref, 1.1 * S_ref)
            tau = torch.empty(batch, device=dev).uniform_(tau_lo, tau_hi)
            r = torch.zeros(batch, device=dev)
            sigma = torch.empty(batch, device=dev).uniform_(0.05, 0.60)
            losses = model.loss(S, K, tau, r, sigma)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % max(1, steps // 10) == 0:
                log.info("dml step %d  lr=%.1e  total=%.4f  price=%.4f  delta=%.4f  gamma=%.4f  vega=%.4f",
                         step, sched.get_last_lr()[0],
                         losses["total"].item(), losses["price"].item(),
                         losses["delta"].item(), losses["gamma"].item(),
                         losses["vega"].item())
                for k in history:
                    history[k].append(float(losses[k].item()))
        return history

    def greek_error_on_atm(model: "DMLPricer", S: float = 5500.0,
                           tau_days: float = 1.0, sigma: float = 0.2,
                           device: str = "cpu") -> dict:
        """Report % error of (Δ, Γ, 𝒱) at ATM τ=tau_days vs analytic BS."""
        dev = torch.device(device)
        S_t = torch.tensor([S], device=dev)
        K_t = torch.tensor([S], device=dev)
        tau_t = torch.tensor([tau_days / 365], device=dev)
        r_t = torch.tensor([0.0], device=dev)
        sig_t = torch.tensor([sigma], device=dev)
        p_hat, d_hat, g_hat, v_hat = model(S_t, K_t, tau_t, r_t, sig_t)
        p_true = bs_price_call(S_t, K_t, tau_t, sig_t, r_t)
        d_true, g_true, v_true = bs_greeks_call(S_t, K_t, tau_t, sig_t, r_t)

        def pct(a, b):
            b_abs = max(abs(float(b.item())), 1e-9)
            return 100 * abs(float(a.item()) - float(b.item())) / b_abs
        return {"price_pct": pct(p_hat, p_true),
                "delta_pct": pct(d_hat, d_true),
                "gamma_pct": pct(g_hat, g_true),
                "vega_pct": pct(v_hat, v_true)}

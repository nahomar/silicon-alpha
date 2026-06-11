"""Rigorous validation harness for the Phase-0 Differential-ML option pricer.

This is the *honest* accuracy story for the DML pricer, written to survive
scrutiny rather than to flatter. It answers three questions, with committed
numbers and plots:

  1. How accurately does the network reproduce analytic Black-Scholes Greeks
     across the whole 0DTE grid (moneyness x maturity x vol) -- not just the
     one ATM point that is easy?

     Caveat we state up front: the architecture hard-codes BS as its base and
     only learns a small (eps=2%) tau-gated tanh residual, so high BS-grid
     accuracy is *partly architectural*. We therefore report the full error
     DISTRIBUTION (median / p95 / max) over the grid, including the short-tau
     ATM corner where gamma spikes, which is the genuinely hard region.

  2. After the Heston fine-tune, does the model price the Heston surface
     BETTER than raw Black-Scholes does? This is the real contribution of
     Phase 0: BS has no closed form under stochastic vol, so the residual has
     to learn the Heston correction. We compute, against a common Heston
     Monte-Carlo reference (with reported MC standard error):

         err_BS    = | BS_price       - Heston_MC |
         err_DML   = | DML_price      - Heston_MC |

     and check that err_DML is materially below err_BS. If it is not, the
     fine-tune bought nothing and we say so.

  3. How far do the Greeks drift during the Heston fine-tune? The fine-tune
     fits PRICE to Heston MC while regularizing Greeks toward analytic BS
     (there is no cheap closed-form Heston Greek on a laptop), so we quantify
     how much delta/gamma/vega move relative to the BS-only model.

Reference: Huge & Savine, "Differential Machine Learning" (2020),
arXiv:2005.02347 -- twin-network parameterization + differential
(derivative-matching) loss.

Usage:
    PYTHONPATH=. python -m odte.eval.validate_dml --scale smoke
    PYTHONPATH=. python -m odte.eval.validate_dml --scale full
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from models.config import DMLConfig
from odte.dml_pricer import (
    DMLPricer,
    bs_price_call,
    bs_greeks_call,
    train_dml_bs,
)
from odte.synth_options import HestonParams
from odte.train.train_dml import finetune_heston, heston_mc_call_price

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "phase0"


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    """0DTE-relevant evaluation grid. SPX-scaled (S_ref ~ 5500)."""
    s_ref: float = 5500.0
    moneyness: tuple = (0.97, 1.03)      # K / S band (0DTE chains are tight)
    n_moneyness: int = 25
    tau_hours: tuple = (0.5, 72.0)       # 30 min .. 3 trading days
    n_tau: int = 20
    sigmas: tuple = (0.10, 0.20, 0.40)   # representative implied-vol levels

    def mesh(self):
        """Return flat arrays (S, K, tau_years, sigma) over the full grid."""
        kov = np.linspace(*self.moneyness, self.n_moneyness)        # K/S
        tau_y = np.linspace(self.tau_hours[0] / (365 * 24),
                            self.tau_hours[1] / (365 * 24), self.n_tau)
        S, K, T, SIG = [], [], [], []
        for sig in self.sigmas:
            for kk in kov:
                for tt in tau_y:
                    S.append(self.s_ref)
                    K.append(self.s_ref * kk)
                    T.append(tt)
                    SIG.append(sig)
        return (np.asarray(S), np.asarray(K), np.asarray(T), np.asarray(SIG))


# ---------------------------------------------------------------------------
# Distribution helper
# ---------------------------------------------------------------------------

def _dist(abs_err: np.ndarray) -> dict:
    a = np.asarray(abs_err, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"median": float("nan"), "p95": float("nan"),
                "max": float("nan"), "mean": float("nan")}
    return {
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


# ---------------------------------------------------------------------------
# (1) Grid evaluation vs analytic Black-Scholes
# ---------------------------------------------------------------------------

def grid_eval_vs_bs(model: DMLPricer, grid: Grid, device: str = "cpu") -> dict:
    """Per-point relative error of model price/Greeks vs analytic BS.

    Reported as percentage-error distributions over the whole grid. Price and
    vega are normalized by strike/spot so the percentages are well-scaled even
    deep OTM where absolute magnitudes are tiny.
    """
    S, K, T, SIG = grid.mesh()
    dev = torch.device(device)
    S_t = torch.tensor(S, dtype=torch.float32, device=dev)
    K_t = torch.tensor(K, dtype=torch.float32, device=dev)
    T_t = torch.tensor(T, dtype=torch.float32, device=dev)
    R_t = torch.zeros_like(S_t)
    SIG_t = torch.tensor(SIG, dtype=torch.float32, device=dev)

    p_hat, d_hat, g_hat, v_hat = model(S_t, K_t, T_t, R_t, SIG_t)
    p_hat = p_hat.detach().cpu().numpy()
    d_hat = d_hat.detach().cpu().numpy()
    g_hat = g_hat.detach().cpu().numpy()
    v_hat = v_hat.detach().cpu().numpy()

    p_true = bs_price_call(S, K, T, SIG, 0.0)
    d_true, g_true, v_true = bs_greeks_call(S, K, T, SIG, 0.0)

    # Price/vega: normalize by spot so OTM near-zero prices do not blow up %.
    price_pct = 100.0 * np.abs(p_hat - p_true) / np.maximum(np.abs(p_true), 0.01 * S)
    # Delta lives in [0, 1]; the honest metric is ABSOLUTE error in delta-points,
    # not relative % (deep OTM delta ~ 0 makes any relative metric meaningless).
    delta_abs = np.abs(d_hat - d_true)
    # Gamma normalized by its own grid-max (it is a spike near ATM short-tau).
    gamma_abs = np.abs(g_hat - g_true)
    gamma_pct = 100.0 * gamma_abs / np.maximum(np.abs(g_true), np.max(np.abs(g_true)) * 0.01)
    vega_pct = 100.0 * np.abs(v_hat - v_true) / np.maximum(np.abs(v_true), 0.01 * S)

    return {
        "n_points": int(S.size),
        "price_pct": _dist(price_pct),
        "delta_abs_points": _dist(delta_abs),
        "gamma_pct": _dist(gamma_pct),
        "vega_pct": _dist(vega_pct),
        # raw arrays kept for plotting (not serialized to JSON)
        "_arrays": {
            "S": S, "K": K, "T": T, "SIG": SIG,
            "gamma_pct": gamma_pct, "delta_abs": delta_abs,
        },
    }


# ---------------------------------------------------------------------------
# (2) Heston reference: does fine-tuned DML beat BS at pricing Heston?
# ---------------------------------------------------------------------------

def heston_reference_grid(grid: Grid, heston: HestonParams,
                          n_points: int, n_paths: int, n_steps: int,
                          tau_band_hours: tuple = (0.5, 72.0),
                          moneyness_band: tuple | None = None,
                          seed: int = 0) -> dict:
    """Sample n_points option specs in the given maturity band and compute a
    Heston MC price + MC stderr for each. Returns the specs and references.

    tau_band_hours lets us probe two regimes: the 0DTE band (where the Heston
    correction to BS is tiny because SV barely accumulates) and a multi-day
    "swing" band (where the correction is materially larger than MC noise).
    """
    rng = np.random.default_rng(seed)
    mny = moneyness_band or grid.moneyness
    S = np.full(n_points, grid.s_ref)
    K = grid.s_ref * rng.uniform(mny[0], mny[1], n_points)
    T = rng.uniform(tau_band_hours[0] / (365 * 24),
                    tau_band_hours[1] / (365 * 24), n_points)
    SIG = rng.uniform(min(grid.sigmas), max(grid.sigmas), n_points)
    R = np.zeros(n_points)

    price = heston_mc_call_price(S, K, T, SIG, R, heston,
                                 n_paths=n_paths, n_steps=n_steps, seed=seed)
    # MC standard error: rerun cheaply with a second seed to estimate spread.
    # (Cheap proxy: a half-size independent batch; reported as guidance.)
    price_b = heston_mc_call_price(S, K, T, SIG, R, heston,
                                   n_paths=max(256, n_paths // 2),
                                   n_steps=n_steps, seed=seed + 7919)
    mc_stderr = np.abs(price - price_b) / np.sqrt(2.0)
    return {"S": S, "K": K, "T": T, "SIG": SIG, "R": R,
            "price": price, "mc_stderr": mc_stderr}


def eval_against_heston(model: DMLPricer, ref: dict, device: str = "cpu") -> dict:
    """Compare DML and raw-BS prices to the Heston MC reference."""
    dev = torch.device(device)
    S_t = torch.tensor(ref["S"], dtype=torch.float32, device=dev)
    K_t = torch.tensor(ref["K"], dtype=torch.float32, device=dev)
    T_t = torch.tensor(ref["T"], dtype=torch.float32, device=dev)
    R_t = torch.zeros_like(S_t)
    SIG_t = torch.tensor(ref["SIG"], dtype=torch.float32, device=dev)

    p_dml = model(S_t, K_t, T_t, R_t, SIG_t)[0].detach().cpu().numpy()
    p_bs = bs_price_call(ref["S"], ref["K"], ref["T"], ref["SIG"], 0.0)
    heston = ref["price"]

    # Normalize price errors by spot (consistent, scale-stable %).
    spot = ref["S"]
    err_bs = np.abs(p_bs - heston) / spot * 100.0
    err_dml = np.abs(p_dml - heston) / spot * 100.0
    mc_floor = ref["mc_stderr"] / spot * 100.0

    improved = float(np.mean(err_dml < err_bs))  # fraction of points DML wins

    # Paired significance: is the per-point reduction (err_BS - err_DML) real,
    # or within the noise? Positive mean + large t => DML genuinely closer to
    # Heston than BS. We also report the MC noise floor so the reader can judge
    # whether any apparent edge clears it.
    paired = err_bs - err_dml
    n = paired.size
    mean_gain = float(np.mean(paired))
    t_stat = float(mean_gain / (np.std(paired, ddof=1) / np.sqrt(n))) if n > 1 else float("nan")

    return {
        "n_points": int(spot.size),
        "err_BS_vs_heston_pct_of_spot": _dist(err_bs),
        "err_DML_vs_heston_pct_of_spot": _dist(err_dml),
        "mc_stderr_pct_of_spot": _dist(mc_floor),
        "fraction_points_DML_beats_BS": improved,
        "paired_mean_gain_pct_of_spot": mean_gain,
        "paired_t_stat": t_stat,
        "_arrays": {"err_bs": err_bs, "err_dml": err_dml,
                    "K_over_S": ref["K"] / ref["S"], "T": ref["T"]},
    }


# ---------------------------------------------------------------------------
# (3) Greek drift from BS-only -> Heston-fine-tuned
# ---------------------------------------------------------------------------

def greek_drift(model_bs: DMLPricer, model_ft: DMLPricer, grid: Grid,
                device: str = "cpu") -> dict:
    S, K, T, SIG = grid.mesh()
    dev = torch.device(device)
    args = [torch.tensor(a, dtype=torch.float32, device=dev) for a in (S, K, T)]
    R_t = torch.zeros_like(args[0])
    SIG_t = torch.tensor(SIG, dtype=torch.float32, device=dev)

    _, d0, g0, v0 = model_bs(args[0], args[1], args[2], R_t, SIG_t)
    _, d1, g1, v1 = model_ft(args[0], args[1], args[2], R_t, SIG_t)

    def rel(a, b):
        a = a.detach().cpu().numpy(); b = b.detach().cpu().numpy()
        return _dist(100.0 * np.abs(a - b) / np.maximum(np.abs(a), 1e-6))

    return {
        "delta_drift_pct": rel(d0, d1),
        "gamma_drift_pct": rel(g0, g1),
        "vega_drift_pct": rel(v0, v1),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(loss_bs: dict, loss_ft: list, bs_eval: dict,
                heston_eval: dict, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # (a) training loss curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(loss_bs["total"], label="total"); ax[0].plot(loss_bs["gamma"], label="gamma")
    ax[0].set_title("BS pretrain loss"); ax[0].set_xlabel("logged step"); ax[0].set_yscale("log")
    ax[0].legend()
    ft_total = [h["total"] for h in loss_ft]
    ax[1].plot(ft_total, color="darkorange"); ax[1].set_title("Heston fine-tune loss")
    ax[1].set_xlabel("step"); ax[1].set_yscale("log")
    fig.tight_layout(); p = out_dir / "loss_curves.png"; fig.savefig(p, dpi=110); plt.close(fig)
    written.append(p.name)

    # (b) gamma % error heatmap over (moneyness x tau) at the mid sigma slice
    arr = bs_eval["_arrays"]
    sigs = np.unique(arr["SIG"]); mid_sig = sigs[len(sigs) // 2]
    msk = arr["SIG"] == mid_sig
    kos = arr["K"][msk] / arr["S"][msk]
    tau_h = arr["T"][msk] * 365 * 24
    gpct = arr["gamma_pct"][msk]
    nK = len(np.unique(kos)); nT = len(np.unique(tau_h))
    try:
        order = np.lexsort((tau_h, kos))
        H = gpct[order].reshape(nK, nT)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(H, aspect="auto", origin="lower",
                       extent=[tau_h.min(), tau_h.max(), kos.min(), kos.max()],
                       cmap="viridis")
        ax.set_xlabel("maturity (hours)"); ax.set_ylabel("K / S")
        ax.set_title(f"Gamma %% error vs BS  (sigma={mid_sig:.0%})")
        fig.colorbar(im, ax=ax, label="% error")
        fig.tight_layout(); p = out_dir / "gamma_error_heatmap.png"
        fig.savefig(p, dpi=110); plt.close(fig); written.append(p.name)
    except Exception as e:  # pragma: no cover - plotting robustness
        log.warning("gamma heatmap skipped: %s", e)

    # (c) the money plot: BS-vs-Heston vs DML-vs-Heston error
    he = heston_eval["_arrays"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(he["err_bs"], he["err_dml"], s=10, alpha=0.5)
    lim = max(float(np.nanmax(he["err_bs"])), float(np.nanmax(he["err_dml"]))) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (no improvement)")
    ax.set_xlabel("|BS - Heston|  (% of spot)")
    ax.set_ylabel("|DML - Heston|  (% of spot)")
    ax.set_title("Heston pricing error: DML vs raw Black-Scholes\n(points below the line = DML closer to Heston)")
    ax.legend(); fig.tight_layout()
    p = out_dir / "heston_improvement.png"; fig.savefig(p, dpi=110); plt.close(fig)
    written.append(p.name)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SCALES = {
    # (bs_steps, ft_steps, ft_batch, mc_paths_ft, heston_eval_pts, mc_paths_eval)
    "smoke": (600, 60, 64, 800, 80, 2000),
    "full": (4000, 600, 96, 3000, 400, 12000),
}


def run(scale: str = "smoke", device: str = "cpu", seed: int = 0,
        out_dir: Path = OUT_DIR) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bs_steps, ft_steps, ft_batch, mc_ft, he_pts, mc_eval = SCALES[scale]
    grid = Grid()
    heston = HestonParams()

    t_start = time.time()
    cfg = DMLConfig()
    model = DMLPricer(cfg)

    log.info("Stage 1/2: Black-Scholes pretrain (%d steps)", bs_steps)
    loss_bs = train_dml_bs(model, steps=bs_steps, batch=512, device=device,
                           S_ref=grid.s_ref)

    # snapshot BS-only Greeks before fine-tune (for drift measurement)
    import copy
    model_bs_only = copy.deepcopy(model)

    bs_eval = grid_eval_vs_bs(model, grid, device=device)

    # Production model: fine-tune on the 0DTE maturity band (the actual product).
    log.info("Stage 2/2: Heston MC fine-tune, 0DTE band (%d steps)", ft_steps)
    ft = finetune_heston(model, steps=ft_steps, batch=ft_batch, device=device,
                         S_ref=grid.s_ref, heston_params=heston,
                         n_mc_paths=mc_ft, n_mc_steps=32,
                         tau_lo=0.5 / (365 * 24), tau_hi=72.0 / (365 * 24))
    loss_ft = ft["history"]

    # Method-validation control: a SECOND model fine-tuned across a wide
    # maturity band (0.5h .. 30d) so the swing regime is in-distribution. This
    # isolates the question "does the residual learn the Heston correction when
    # it is material?" from the product question "is it material at 0DTE?".
    log.info("Control: wide-band fine-tune (0.5h..30d) for method validation")
    model_wide = copy.deepcopy(model_bs_only)
    finetune_heston(model_wide, steps=ft_steps, batch=ft_batch, device=device,
                    S_ref=grid.s_ref, heston_params=heston,
                    n_mc_paths=mc_ft, n_mc_steps=48,
                    tau_lo=0.5 / (365 * 24), tau_hi=30 * 24.0 / (365 * 24))

    # Two maturity regimes against a common Heston MC reference:
    #   - 0dte:  0.5h .. 72h  (correction tiny; should be ~MC noise) -> product model
    #   - swing: 5d   .. 30d  (correction material; should clear MC noise) -> wide model
    log.info("Building Heston MC references (0DTE + swing), %d pts x %d paths",
             he_pts, mc_eval)
    ref_0dte = heston_reference_grid(grid, heston, n_points=he_pts,
                                     n_paths=mc_eval, n_steps=48,
                                     tau_band_hours=(0.5, 72.0), seed=seed)
    ref_swing = heston_reference_grid(grid, heston, n_points=he_pts,
                                      n_paths=mc_eval, n_steps=64,
                                      tau_band_hours=(5 * 24.0, 30 * 24.0),
                                      seed=seed + 101)
    heston_eval = eval_against_heston(model, ref_0dte, device=device)
    heston_eval_swing = eval_against_heston(model_wide, ref_swing, device=device)
    drift = greek_drift(model_bs_only, model, grid, device=device)

    plots = _make_plots(loss_bs, loss_ft, bs_eval, heston_eval, out_dir)

    metrics = {
        "scale": scale,
        "seed": seed,
        "wall_clock_sec": round(time.time() - t_start, 1),
        "config": asdict(cfg),
        "heston_params": asdict(heston),
        "grid": asdict(grid),
        "bs_grid_eval": {k: v for k, v in bs_eval.items() if not k.startswith("_")},
        "heston_eval_0dte": {k: v for k, v in heston_eval.items() if not k.startswith("_")},
        "heston_eval_swing": {k: v for k, v in heston_eval_swing.items() if not k.startswith("_")},
        "greek_drift_bs_to_finetune": drift,
        "plots": plots,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics -> %s", out_dir / "metrics.json")
    return metrics


def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=list(SCALES), default="smoke")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = run(scale=a.scale, device=a.device, seed=a.seed)

    print("\n=== Phase 0 DML validation summary ===")
    print(f"scale={m['scale']}  wall={m['wall_clock_sec']}s  "
          f"grid points={m['bs_grid_eval']['n_points']}")
    print("BS-grid accuracy (median / p95 / max):")
    da = m["bs_grid_eval"]["delta_abs_points"]
    print(f"  delta abs-pts  {da['median']:.5f} / {da['p95']:.5f} / {da['max']:.5f}")
    for g in ("gamma_pct", "vega_pct"):
        d = m["bs_grid_eval"][g]
        print(f"  {g:<11} %  {d['median']:.3f} / {d['p95']:.3f} / {d['max']:.3f}")
    for label, key in (("0DTE  (0.5-72h)  [product model]", "heston_eval_0dte"),
                       ("swing (5-30d)   [wide-band control]", "heston_eval_swing")):
        he = m[key]
        print(f"Heston pricing error, {label}  (% of spot):")
        print(f"  raw BS  vs Heston (median): {he['err_BS_vs_heston_pct_of_spot']['median']:.4f}")
        print(f"  DML     vs Heston (median): {he['err_DML_vs_heston_pct_of_spot']['median']:.4f}")
        print(f"  MC stderr floor   (median): {he['mc_stderr_pct_of_spot']['median']:.4f}")
        print(f"  paired gain={he['paired_mean_gain_pct_of_spot']:+.5f}  "
              f"t={he['paired_t_stat']:+.1f}  "
              f"DML wins {he['fraction_points_DML_beats_BS']*100:.0f}% of pts")


if __name__ == "__main__":
    _cli()

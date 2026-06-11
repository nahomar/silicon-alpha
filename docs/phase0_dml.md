# Phase 0 — Differential-ML Option Pricer: Validation

**TL;DR.** A 50k-parameter twin-network learns SPX option prices and Greeks in
a single forward pass (price + Δ/Γ/𝒱 via automatic adjoint differentiation).
Against analytic Black-Scholes it reproduces Greeks to **≤ 2e-5 delta-points,
0.09% max gamma error, 0.01% max vega error** across a 0DTE grid. Fine-tuned on
a Heston Monte-Carlo reference, the headline finding is a *decision*, not a
brag:

- **At 0DTE the Heston correction is not worth learning.** Raw Black-Scholes at
  the instantaneous vol is already optimal; the MC-fine-tuned residual is a
  statistically-significant **−0.00002% of spot worse** (paired *t* = −3.9).
  This is financially correct — stochastic vol barely accumulates over hours —
  and it says the 0DTE product should price off the BS-stage model.
- **At multi-day maturities the residual learns a real correction**: median
  Heston pricing error drops **~14%** (0.069% → 0.059% of spot, *t* = +2.0)
  when the swing band is in-distribution. The machinery works where there is
  something to learn.

Every number here is produced by `odte/eval/validate_dml.py` and committed to
[`../reports/phase0/metrics.json`](../reports/phase0/metrics.json); the doc is
written to be falsifiable, not flattering. Run: `full` scale, seed 0, ~440s CPU.

---

## 1. What this is

A *Differential Machine Learning* pricer (Huge & Savine, 2020,
[arXiv:2005.02347](https://arxiv.org/abs/2005.02347)). The network maps
`(S, K, τ, r, σ) → price`; the Greeks are **automatic adjoint derivatives** of
that price through the network:

```
Δ = ∂price/∂S      Γ = ∂²price/∂S²      𝒱 = ∂price/∂σ
```

via `torch.autograd.grad(..., create_graph=True)` — one forward pass yields
price and all three Greeks consistently, no finite differences. The loss
matches price **and** the autograd Greeks to a reference (the "differential"
loss that gives the method its sample efficiency).

### Architecture choice — and its honest consequence

The network does **not** predict price from scratch. It is a *residual on top
of Black-Scholes*:

```
price = BS(S, K, τ, σ_eff, r)  +  K · ε · g(τ) · tanh(net(x)),    ε = 0.02
g(τ) = 1 − exp(−β·τ)   →  0  as τ → 0
```

`g(τ)` is a maturity gate: the learned correction vanishes smoothly at expiry,
so near τ→0 the price collapses to the exact intrinsic payoff instead of a
hallucinated singular value.

**The honest consequence:** because BS is hard-coded as the base and the
residual is capped at ε = 2% of strike, high accuracy *against BS* is partly
architectural. So we do not report one flattering ATM number — we report the
full error **distribution** over the grid (§2), and we separately test whether
the residual can learn what BS does **not** know: the Heston correction (§3).

---

## 2. Accuracy vs analytic Black-Scholes (whole grid)

Grid: S = 5500, K/S ∈ [0.97, 1.03], τ ∈ [30 min, 3 trading days], σ ∈ {10%,
20%, 40%} — **1,500 points**, including the short-τ ATM corner where gamma
spikes and the method is genuinely hard.

| Greek | metric | median | p95 | max |
|-------|--------|-------:|----:|----:|
| Δ | absolute (delta-points) | 0.00000 | 0.00000 | **0.00002** |
| Γ | % error | 0.000 | 0.004 | **0.086** |
| 𝒱 | % error | 0.002 | 0.008 | **0.013** |

Delta is reported in **absolute delta-points** (it lives in [0,1]; a relative %
is meaningless deep OTM where true delta ≈ 0). Gamma is normalized by its
grid-maximum (a near-ATM short-τ spike). Plots:
[`gamma_error_heatmap.png`](../reports/phase0/gamma_error_heatmap.png),
[`loss_curves.png`](../reports/phase0/loss_curves.png).

These are excellent — and, per §1, partly a property of the architecture. The
contribution that is *not* architectural is §3.

---

## 3. Does the residual learn the Heston correction?

Black-Scholes has no closed form under stochastic volatility, so this is the
real test. We fine-tune the residual against a **Heston Monte-Carlo** price
reference and ask whether the model prices the Heston surface *better than raw
BS does*. Per point, against a common MC reference (with reported MC standard
error):

```
err_BS  = |BS_price  − Heston_MC| / spot
err_DML = |DML_price − Heston_MC| / spot
```

Heston: κ = 4.0, θ = 0.04 (20% long-run vol), ξ = 0.6 (vol-of-vol),
ρ = −0.7 (leverage). Euler full-truncation, 12,000 paths/point.

| regime | model | BS err (med, % spot) | DML err (med) | MC floor | paired gain | *t* | DML wins |
|--------|-------|---------------------:|--------------:|---------:|------------:|----:|---------:|
| **0DTE** (0.5–72h) | product (0DTE fine-tune) | 0.0070 | 0.0070 | 0.0041 | −0.00002 | **−3.9** | 43% |
| **swing** (5–30d) | wide-band control | 0.0687 | 0.0588 | 0.0177 | +0.00765 | **+2.0** | 51% |

**Reading this honestly:**

- **0DTE: the correction is not worth learning — and we can prove it.** The
  product model is a *statistically-significant* −0.00002% of spot *worse* than
  raw BS (*t* = −3.9 across 400 points). The magnitude is financially
  negligible, but the sign is the point: at 0.5–72h, stochastic vol has almost
  no time to diffuse, so Heston ≈ BS at the instantaneous vol, and fine-tuning
  on noisy MC targets injects a hair of noise rather than signal. **Design
  implication: the 0DTE pricer should use the BS-stage model; the Heston
  fine-tune does not earn its keep at this tenor.** (The earlier smoke run's
  *t* = +2.9 "improvement" was sampling noise — caught precisely because the
  full run drives the MC floor down to 0.0041%.)

- **Swing: the residual learns a real correction.** With the 5–30d band
  in-distribution, median Heston pricing error falls ~14% (0.0687 → 0.0588% of
  spot) at *t* = +2.0 — modest but real, and the effect size (paired gain
  +0.00765) is ~380× the 0DTE effect. This is the method validation: the
  twin-net captures the SV correction when there *is* one, rather than merely
  memorizing BS. See [`heston_improvement.png`](../reports/phase0/heston_improvement.png).

The product pricer is the 0DTE model; the swing result is a *methodological
control* trained on a wide maturity band purely to show the residual has real
expressive power.

### Greek drift during fine-tune

The fine-tune fits **price** to Heston MC while regularizing Greeks toward
analytic BS (no cheap closed-form Heston Greek at batch scale on a laptop).
Median Greek movement from the BS-only model to the fine-tuned model:

| Greek | median drift |
|-------|-------------:|
| Δ | 0.001% |
| Γ | 0.000% |
| 𝒱 | 0.07% |

Small — the price-only fine-tune did not destabilize the Greek surface.
(Upper-percentile drift is dominated by points where the true Greek ≈ 0 and any
relative metric explodes; the median is the meaningful figure.)

---

## 4. Limitations (read before trusting any of this)

- **Synthetic references only.** Targets are closed-form BS and a Heston MC
  simulator (`odte/synth_options.py`), not market quotes. This validates the
  *numerical machinery*, not tradeable edge. Real CBOE DataShop / OPRA data is
  required before any claim about the live market.
- **Greeks under Heston are regularized to BS**, not fit to Heston Greeks. The
  model learns the Heston *price* surface; a full treatment needs pathwise /
  Malliavin or bumped-MC Greek targets.
- **BS-grid accuracy is partly architectural** (§1). The honest contribution is
  the §3 Heston correction, not the §2 BS reproduction.
- **Heston MC has discretization + sampling error.** The reported MC standard
  error (0.004% of spot at 0DTE) is the floor any "improvement" must clear —
  which is exactly why the 0DTE effect is judged negligible and the swing
  effect only modestly significant.

---

## Reproducing

```bash
# fast sanity (~20s CPU)
PYTHONPATH=. python -m odte.eval.validate_dml --scale smoke

# committed numbers (~440s CPU; what this doc reports)
PYTHONPATH=. python -m odte.eval.validate_dml --scale full

# reproducible gate test (~16s CPU)
PYTHONPATH=. pytest tests/odte/test_dml_pricer.py -xvs
```

Outputs: [`../reports/phase0/metrics.json`](../reports/phase0/metrics.json) +
PNG plots. Seed = 0; wall-clock 440s on CPU (Apple Silicon, no GPU).

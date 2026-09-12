"""Decisive probe: does Eskom loadshedding transmit to PGM miners with a lag?

This is the gate for the entire AFRIMIN track (`docs/afrimin_track.md`). It is
the direct analogue of `infra/modal/dir_baseline.py` — a $0 diagnostic that
decides whether a large build is worth starting.

## Hypothesis

South Africa is ~70%+ of mined global platinum, and deep-level PGM mining is
power-intensive. If African supply events reach global price discovery with a
lag, an *escalation* in national loadshedding stage should predict subsequent
underperformance in SA PGM miners.

## What would make this fake, and what is done about it

**1. Look-ahead on the event.** Stages are announced and revised intraday, so
same-day stage is not cleanly knowable before the close. The feature is lagged a
full session (`--lag`, minimum 1, enforced in `eskom.align_to_sessions`).

**2. Beta dressed up as alpha.** Predicting raw miner returns from loadshedding
mostly rediscovers "miners move with platinum." So miners are first
**factor-neutralized** against the metal and the broad market, and the test asks
whether loadshedding predicts the *residual*. Raw-return results are printed
alongside purely as the contrast.

**3. Leakage through the factor model.** Factor betas are fit on the TRAIN split
only and applied to EVAL. Fitting them on the full sample would let eval-period
information into the residual definition. The out-of-sample column is the one
that counts.

**4. Autocorrelation inflating significance.** Loadshedding is highly persistent
(stage runs last weeks), so plain OLS standard errors would be badly
understated. t-statistics use **Newey-West HAC** errors.

**5. A macro artifact that isn't about power.** A placebo runs the identical
test on global gold miners (GDX), which have minimal SA grid exposure. If the
placebo shows the same effect, the mechanism is macro, not curtailment.

## Reading the result

Criteria were declared in the design doc *before* the first run:

- |t| < 2 on the out-of-sample residual  → **no transmission lag. Stop.**
- t ≥ 2, correct sign                    → plausible; run ONE more independent
                                            case before building anything.
- t ≥ 2, wrong sign                      → specification error, not contrarian alpha.
- directional accuracy > 60%             → assume leakage and hunt for it.

Usage:
    PYTHONPATH=. python -m afrimin.probe.eskom_pgm --stages data/eskom_stages.csv
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from afrimin.data import eskom, prices

log = logging.getLogger(__name__)

SIGNIFICANCE_T = 2.0
LEAKAGE_ACC = 0.60


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    resid: np.ndarray
    n: int
    names: list[str]

    def t_of(self, name: str) -> float:
        return float(self.tstat[self.names.index(name)])

    def beta_of(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])


def _nw_lags(n: int) -> int:
    """Newey-West bandwidth, standard rule of thumb: 4(n/100)^(2/9)."""
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def ols(y: np.ndarray, X: np.ndarray, names: list[str],
        hac: bool = True) -> OLSResult:
    """OLS with optional Newey-West HAC standard errors.

    HAC matters here: loadshedding stage is strongly autocorrelated, and
    classical OLS errors on a persistent regressor understate uncertainty badly
    enough to turn noise into a 'significant' result.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n}, k={k}")

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    if not hac:
        sigma2 = float(resid @ resid) / (n - k)
        V = sigma2 * XtX_inv
    else:
        L = _nw_lags(n)
        u = X * resid[:, None]              # (n, k) score contributions
        Omega = u.T @ u                     # lag 0
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1.0)       # Bartlett kernel
            G = u[lag:].T @ u[:-lag]
            Omega += w * (G + G.T)
        V = XtX_inv @ Omega @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return OLSResult(beta=beta, se=se, tstat=t, resid=resid, n=n, names=names)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    ticker: str
    t_oos: float
    beta_oos: float
    t_is: float
    acc_oos: float
    n_train: int
    n_eval: int
    is_placebo: bool = False

    @property
    def significant(self) -> bool:
        return abs(self.t_oos) >= SIGNIFICANCE_T

    @property
    def correct_sign(self) -> bool:
        # Hypothesis: escalating loadshedding -> miner UNDERperformance.
        return self.beta_oos < 0


class InsufficientPower(SystemExit):
    """The test cannot distinguish 'no effect' from 'no experiment'."""


def _require_power(x: np.ndarray, split: int, min_events: int = 20) -> None:
    """Refuse to render a verdict when a split has no regressor variance.

    South Africa ran ~10 months with no loadshedding at all from mid-2024. A
    time-split that puts the eval period inside that window gives a regressor
    that is constant zero, so every coefficient is exactly 0.0 and every
    t-statistic is exactly 0.0 — and the probe would print a confident
    "NO TRANSMISSION LAG DETECTED" having tested precisely nothing.

    A null is only evidence when the test could have detected an effect. This
    is the same class of error as the corrupted directional target in
    `docs/data_integrity_finding.md`: a run that completes successfully while
    measuring nothing is far more dangerous than one that crashes.
    """
    for name, seg in (("train", x[:split]), ("eval", x[split:])):
        events = int(np.count_nonzero(seg))
        if events < min_events:
            raise InsufficientPower(
                f"\nINCONCLUSIVE — the {name} split contains only {events} "
                f"non-zero stage changes in {len(seg)} sessions "
                f"(minimum {min_events}).\n\n"
                f"  A regressor with no variance yields coefficients of exactly "
                f"zero, so this run\n"
                f"  would report a confident null while testing nothing. That is "
                f"not evidence of\n"
                f"  no effect; it is the absence of an experiment.\n\n"
                f"  Cause: South Africa had ~10 months with zero loadshedding "
                f"from mid-2024, so a\n"
                f"  plain time-split can land an entire split inside a dead "
                f"zone.\n\n"
                f"  Fix: restrict the window to the period that actually "
                f"contains variance, e.g.\n"
                f"    --start 2022-07-01 --end 2024-05-01\n"
            )


def _factor_neutralize(miner: pd.Series, factors: pd.DataFrame,
                       split: int) -> pd.Series:
    """Residualize miner returns against factors, fitting betas on train only.

    Returns the full residual series; only the eval slice is out-of-sample.
    """
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    y = miner.to_numpy()
    fit = ols(y[:split], X[:split], ["const", *factors.columns], hac=False)
    resid = y - X @ fit.beta
    return pd.Series(resid, index=miner.index, name=f"{miner.name}_resid")


def run(stages_path: str, start: str, end: str | None, lag: int,
        train_frac: float) -> list[ProbeResult]:
    hist = eskom.load_stages(stages_path)
    print(f"\nstage history: {hist.describe()}\n")

    tickers = (
        list(prices.PGM_MINERS) + list(prices.PGM_METAL)
        + list(prices.FACTORS) + list(prices.CONTROLS)
    )
    panel = prices.fetch(tickers, start=start, end=end)
    rets = panel.returns

    stage = eskom.align_to_sessions(hist, rets.index, lag_days=lag)
    # The regressor is the CHANGE in stage: a market that knows the country is
    # at Stage 4 should have priced Stage 4. What is potentially unpriced is the
    # escalation. Levels are reported separately in the doc's follow-up work.
    d_stage = stage.diff().rename("d_stage")

    df = rets.join(d_stage, how="inner").dropna()
    if len(df) < 100:
        raise SystemExit(
            f"only {len(df)} overlapping observations between stage history and "
            f"price data — too few to test. Widen the stage CSV or --start."
        )

    split = int(len(df) * train_frac)
    print(f"aligned: {len(df)} sessions, train={split}, eval={len(df) - split}, "
          f"stage lagged {lag} session(s)\n")
    _require_power(df["d_stage"].to_numpy(), split)

    metal = "PPLT" if "PPLT" in df else next(
        (t for t in prices.PGM_METAL if t in df), None)
    if metal is None:
        raise SystemExit("no PGM metal series available; cannot factor-neutralize")
    factor_cols = [c for c in (metal, "SPY") if c in df]

    results: list[ProbeResult] = []
    targets = [(t, False) for t in prices.PGM_MINERS if t in df]
    targets += [(t, True) for t in prices.CONTROLS if t in df]

    for ticker, is_placebo in targets:
        resid = _factor_neutralize(df[ticker], df[factor_cols], split)
        x = df["d_stage"].to_numpy()
        X = np.column_stack([np.ones(len(x)), x])
        names = ["const", "d_stage"]

        fit_is = ols(resid.to_numpy()[:split], X[:split], names)
        fit_oos = ols(resid.to_numpy()[split:], X[split:], names)

        r_eval = resid.to_numpy()[split:]
        x_eval = x[split:]
        moved = x_eval != 0
        acc = (
            float(np.mean(np.sign(r_eval[moved]) == -np.sign(x_eval[moved])))
            if moved.sum() > 0 else float("nan")
        )

        results.append(ProbeResult(
            ticker=ticker,
            t_oos=fit_oos.t_of("d_stage"),
            beta_oos=fit_oos.beta_of("d_stage"),
            t_is=fit_is.t_of("d_stage"),
            acc_oos=acc,
            n_train=split,
            n_eval=len(df) - split,
            is_placebo=is_placebo,
        ))
    return results


def report(results: list[ProbeResult]) -> None:
    print(f"{'ticker':<8} {'t (oos)':>9} {'beta (oos)':>12} {'t (in-samp)':>12} "
          f"{'dir acc':>9}  note")
    print("-" * 68)
    for r in results:
        note = "PLACEBO" if r.is_placebo else ""
        acc = "n/a" if np.isnan(r.acc_oos) else f"{r.acc_oos:.1%}"
        print(f"{r.ticker:<8} {r.t_oos:>9.2f} {r.beta_oos:>12.2e} "
              f"{r.t_is:>12.2f} {acc:>9}  {note}")

    real = [r for r in results if not r.is_placebo]
    placebo = [r for r in results if r.is_placebo]
    hits = [r for r in real if r.significant and r.correct_sign]
    wrong = [r for r in real if r.significant and not r.correct_sign]
    leaky = [r for r in real if not np.isnan(r.acc_oos) and r.acc_oos > LEAKAGE_ACC]
    placebo_hits = [r for r in placebo if r.significant]

    print("\n" + "=" * 68)
    if leaky:
        print("VERDICT: SUSPECT LEAKAGE — investigate before believing anything.")
        print(f"  {[r.ticker for r in leaky]} exceed {LEAKAGE_ACC:.0%} daily "
              f"directional accuracy. Daily miner residuals should not be that "
              f"predictable; find the look-ahead.")
    elif placebo_hits and hits:
        print("VERDICT: MACRO ARTIFACT, not a power-curtailment mechanism.")
        print(f"  the placebo {[r.ticker for r in placebo_hits]} shows the same "
              f"effect despite minimal SA grid exposure.")
    elif wrong and not hits:
        print("VERDICT: SIGNIFICANT BUT WRONG SIGN — treat as specification error.")
        print("  loadshedding escalation predicting OUTperformance is not a "
              "contrarian signal; it means the model is misspecified.")
    elif hits:
        print(f"VERDICT: TRANSMISSION LAG PLAUSIBLE ({len(hits)}/{len(real)} miners).")
        print("  Per docs/afrimin_track.md this licenses exactly ONE more "
              "independent case (DRC cobalt export policy), NOT an ingestion "
              "build and NOT vendor data.")
    else:
        print("VERDICT: NO TRANSMISSION LAG DETECTED.")
        print("  This is the honest null and the expected outcome. The best-")
        print("  instrumented case of the thesis — dominant supply share, causal")
        print("  mechanism, daily data, liquid instruments — shows nothing out of")
        print("  sample. Thesis (c) is unsupported at the daily horizon.")
        print("  Do not build the ingestion layer. Do not buy vendor data.")
        print("  Claim (a), the value-capture research view, is unaffected.")
    print("=" * 68)
    print("\nSignal presence is not tradeable alpha: a daily effect must still")
    print("survive spread, borrow and ADR tracking error to mean anything.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Eskom loadshedding -> PGM miner transmission probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=eskom._HOWTO,
    )
    ap.add_argument("--stages", required=True,
                    help="CSV of daily loadshedding stages (see epilog)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lag", type=int, default=1,
                    help="sessions to lag the stage feature (min 1)")
    ap.add_argument("--train-frac", type=float, default=0.70)
    args = ap.parse_args()

    try:
        results = run(args.stages, args.start, args.end, args.lag,
                      args.train_frac)
    except (eskom.MissingStageData, prices.PriceDataUnavailable) as exc:
        raise SystemExit(f"\n{exc}\n")
    report(results)


if __name__ == "__main__":  # pragma: no cover
    main()

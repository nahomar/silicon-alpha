"""Decisive probe: does Eskom loadshedding transmit to PGM miners with a lag?

This is the gate for the entire CHOKEPOINT track (`docs/chokepoint_track.md`). It is
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
    PYTHONPATH=. python -m chokepoint.probe.eskom_pgm --stages data/eskom_stages.csv
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chokepoint.data import eskom, prices

log = logging.getLogger(__name__)

from chokepoint.stats import (  # shared so a probe cannot omit a guard
    LEAKAGE_ACC, SIGNIFICANCE_T, InsufficientPower, OLSResult,
    factor_neutralize as _factor_neutralize, ols,
    require_power as _require_power,
)


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
        print("  Per docs/chokepoint_track.md this licenses exactly ONE more "
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

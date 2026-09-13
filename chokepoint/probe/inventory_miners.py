"""Part 2 probe: do exchange copper inventories predict copper-miner returns?

## Hypothesis

Falling exchange inventories mean physical copper is being drawn down faster
than it is replaced — supply tightening. If that is not fully in the price,
copper miners should outperform after inventories fall. So the coefficient on
inventory change should be NEGATIVE.

## Read the prior before reading the result

This is deliberately a **weak-prior test**, and saying so in advance is the
point.

LME, COMEX and SHFE publish warehouse stocks *daily*, and copper inventory is
among the most-watched numbers in the entire metals complex. Every desk with a
copper book sees it the same morning we would. There is no informational
advantage here whatsoever — this is the opposite of the thesis that motivated
the track, which was about information reaching price discovery slowly.

So the honest expectation is **no effect**, and a strong effect would be more
likely to indicate a mistake in this file than an inefficiency in the copper
market. That is written down now so it cannot be rationalised later.

Why run it at all: it is free, the data is already parsed, and it exercises the
Part 2 machinery on a case where we know roughly what the answer should be. A
pipeline that finds large alpha in daily-published LME stocks is a pipeline with
a bug, and better to learn that here than on a signal we actually believe.

## Design

Miner returns are residualized against copper (CPER) and the market (SPY), with
factor betas fit on TRAIN only — "miners move with copper" is not alpha. The
inventory signal is lagged one month: Cochilco publishes monthly in arrears, so
month M's figure is not in a bulletin until M+1, even though the underlying
exchange prints were public daily.

A gold-miner placebo (GDX) runs the identical test. Copper inventories should
say nothing about gold miners; if they do, the effect is a macro factor rather
than anything about copper supply.

Usage:
    PYTHONPATH=. python -m chokepoint.probe.inventory_miners
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chokepoint.data import cochilco, prices
from chokepoint.stats import (
    SIGNIFICANCE_T, InsufficientPower, factor_neutralize, ols, require_power,
)

log = logging.getLogger(__name__)

COPPER_MINERS = {
    "FCX": "Freeport-McMoRan",
    "SCCO": "Southern Copper",
    "TECK": "Teck Resources",
}
COPPER_METAL = "CPER"       # US Copper Index Fund
MARKET = "SPY"
PLACEBO = "GDX"             # gold miners — no copper-supply exposure


@dataclass
class Result:
    ticker: str
    t_oos: float
    beta_oos: float
    t_is: float
    n_train: int
    n_eval: int
    is_placebo: bool = False

    @property
    def significant(self) -> bool:
        return abs(self.t_oos) >= SIGNIFICANCE_T

    @property
    def correct_sign(self) -> bool:
        # Falling inventory (negative change) -> miners outperform.
        return self.beta_oos < 0


def run(start: str, train_frac: float) -> tuple[list[Result], int]:
    inv = cochilco.inventory_history()
    inv = inv[inv.index >= pd.Timestamp(start)]
    if len(inv) < 36:
        raise SystemExit(f"only {len(inv)} months of inventory from {start}")

    tickers = [*COPPER_MINERS, COPPER_METAL, MARKET, PLACEBO]
    panel = prices.fetch(tickers, start=start)

    # Month-end closes -> monthly log returns, aligned to the inventory index.
    monthly_close = panel.close.resample("ME").last()
    rets = np.log(monthly_close).diff().dropna(how="any")

    # Percent change in total exchange stock, lagged one month: Cochilco
    # publishes in arrears, so month M is not readable until M+1.
    signal = (inv["total"].pct_change().shift(1).rename("inv_chg"))

    df = rets.join(signal, how="inner").dropna()
    if len(df) < 36:
        raise SystemExit(
            f"only {len(df)} aligned months — inventory and price histories "
            f"barely overlap. Widen --start or check ticker availability."
        )

    split = int(len(df) * train_frac)
    print(f"\naligned: {len(df)} months {df.index[0].date()} → "
          f"{df.index[-1].date()}  (train {split} / eval {len(df) - split})\n")
    require_power(df["inv_chg"].to_numpy(), split)

    factor_cols = [c for c in (COPPER_METAL, MARKET) if c in df]
    if not factor_cols:
        raise SystemExit("neither copper nor market factor available")

    x = df["inv_chg"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    names = ["const", "inv_chg"]

    results: list[Result] = []
    targets = [(t, False) for t in COPPER_MINERS if t in df]
    if PLACEBO in df:
        targets.append((PLACEBO, True))

    for ticker, is_placebo in targets:
        resid = factor_neutralize(df[ticker], df[factor_cols], split)
        fit_is = ols(resid.to_numpy()[:split], X[:split], names)
        fit_oos = ols(resid.to_numpy()[split:], X[split:], names)
        results.append(Result(
            ticker=ticker,
            t_oos=fit_oos.t_of("inv_chg"),
            beta_oos=fit_oos.beta_of("inv_chg"),
            t_is=fit_is.t_of("inv_chg"),
            n_train=split, n_eval=len(df) - split,
            is_placebo=is_placebo,
        ))
    return results, len(df)


def report(results: list[Result], n: int) -> None:
    print(f"{'ticker':<8} {'t (oos)':>9} {'beta (oos)':>12} "
          f"{'t (in-samp)':>12}  note")
    print("-" * 56)
    for r in results:
        note = "PLACEBO" if r.is_placebo else ""
        print(f"{r.ticker:<8} {r.t_oos:>9.2f} {r.beta_oos:>12.3f} "
              f"{r.t_is:>12.2f}  {note}")

    real = [r for r in results if not r.is_placebo]
    hits = [r for r in real if r.significant and r.correct_sign]
    placebo_hits = [r for r in results if r.is_placebo and r.significant]

    print("\n" + "=" * 56)
    if placebo_hits:
        # Checked FIRST and unconditionally. An earlier version gated this on
        # `hits` as well, so a firing placebo with no real hits fell through to
        # "no predictive content" — the verdict was blind in precisely the case
        # the placebo exists to catch.
        worst = max(placebo_hits, key=lambda r: abs(r.t_oos))
        print(f"VERDICT: TEST IS UNSOUND — the placebo fired "
              f"(|t| = {abs(worst.t_oos):.2f}).")
        print(f"  {worst.ticker} has no copper-supply exposure, so copper "
              f"inventories cannot")
        print("  legitimately predict its factor residual. Something is "
              "leaking: the")
        print("  residualization is leaving a macro factor that inventory "
              "co-moves with,")
        print("  or this is the best of several tests and should be judged "
              "against the")
        print("  number of tries, not against t=2.")
        print("  NOTHING about the real targets can be read until this is "
              "explained —")
        print("  a null from an unsound test is not evidence of absence.")
    elif hits:
        print(f"VERDICT: SIGNIFICANT ({len(hits)}/{len(real)}) — and that is a "
              f"reason for suspicion, not celebration.")
        print("  Exchange stocks are published DAILY and watched by every "
              "copper desk alive.")
        print("  A real edge here would mean the most-watched number in the "
              "complex is")
        print("  mispriced for a month. Audit the alignment and the lag before "
              "believing it.")
    else:
        print("VERDICT: NO PREDICTIVE CONTENT — the expected result.")
        print("  Copper inventories are public daily and already in the price. "
              "This is")
        print("  what an efficient, well-covered market is supposed to look "
              "like, and it")
        print("  confirms the pipeline does not manufacture signal from a "
              "public series.")
    print("=" * 56)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--train-frac", type=float, default=0.70)
    args = ap.parse_args()
    try:
        results, n = run(args.start, args.train_frac)
    except InsufficientPower as exc:
        raise SystemExit(str(exc))
    report(results, n)


if __name__ == "__main__":  # pragma: no cover
    main()

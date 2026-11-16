# Partners for B0: candidates tested one at a time (owner 2026-11-15: "keep testing candidates")

A partner must do two things: make money on its own, and not lose on the same days as B0.
Test M (close momentum, docs/momentum_plan.md) failed the first. Every candidate below is fixed in
writing and committed before its numbers are computed. Development data only (through 2026-11-06).

## G: sell same-day options on gold (GLD) and Treasury bonds (TLT) (fixed 2026-11-15, before computing)

### Why
- Gold and long bonds move for reasons other than stocks (real rates, the dollar, inflation, flight to
  safety), so their bad days need not line up with B0's.
- Option sellers earn a premium in most markets because buyers pay for protection. Whether that premium
  survives the wider spreads of ETF options is the question.
- Data already bought: same-day GLD and TLT options, 1-minute quotes, 2024-11-11 .. 2026-11-06
  (data/odte_etf/{GLD,TLT}/; same-day expiries mostly Mon/Wed/Fri).

### The rule (no fitted parameters)
- On every session with a same-day GLD or TLT expiry, sell the standard iron fly (zoo FLY: iron_fly,
  wings 1.5 sigma, sigma = ATM straddle / 0.8, the same fly B0 trades) at **10:00, 12:00 and 14:00 ET**.
  Each trade sized at $1k worst-case risk (P&L in $ per $1k, as in the stress test).
- Costs: **touch** (full mid-to-touch distance on every leg) plus $1.20 per contract.
- These options settle into shares, so (as in odte/research/e4_spy_itm.py) every leg that is in the money
  at 15:59 is closed then at the touch (+$1.20 per contract); the rest expire worthless (worth their
  intrinsic value at the close if they crossed in the last minute).
- Day P&L = the sum of that day's trades over both underlyings.

### Pass rules
- **G1 (stands on its own):** pooled GLD+TLT daily P&L at the touch:
  - mean > 0 in both halves (2024-11-11 .. 2025-10-31 and 2025-11-01 .. 2026-11-06), AND
  - pooled t-stat of the daily P&L >= 2.0.
- **G2 (helps B0):** on B0's walk-forward days in the same period (reports/stress/stress_spx_daily.parquet;
  G is 0 on days with no GLD/TLT expiry):
  - correlation of the two daily P&Ls <= 0.10, AND
  - the combination, each side scaled to equal daily volatility measured on the first half only, has a
    higher Sharpe than B0 alone AND a lower probability of a losing year (stationary bootstrap: mean
    block 20 days, 10,000 one-year paths of 252 days, same seed for both).
- **Verdict:**
  - PASS (G1 and G2) -> paper contender next to B0 (needs GLD/TLT option quote lines: owner decides).
  - LEAD (both halves > 0, t >= 1.0, and G2) -> shadow-only in paper for forward evidence, no claim of an edge.
  - FAIL otherwise. Reported as it is, no tuning.
- Power, stated up front: two years is short. A real edge with Sharpe 1 reaches t >= 2 only about a third
  of the time, which is why the LEAD tier exists.

### Reported only (not used for the verdict)
- GLD and TLT separately; each decision time; half-gap costs.
- Held to the close ignoring share delivery (upper bound).
- Same rule on USO, EEM and XLF (fewer expiry days).
- Rich-only filter: trade only when the 10:00 straddle exceeds the average |open-to-close move| of the
  previous 20 sessions (both known at 10:00).

Code: odte/research/partner_g.py -> reports/partners/.

### Result (2026-11-15): FAIL, decisively (reports/partners/g.json)
- **G1 FAIL.** 1,313 GLD+TLT decisions on 326 days. Pooled daily P&L at the touch −$1,067 per day
  (3-6 trades a day, $1k risk each; t −14.6); both halves negative (−$1,248, −$915).
  - Per trade at the touch: GLD −$257, TLT −$283 per $1k. At half-gap: GLD −$152, TLT −$224.
  - Even held to the close ignoring share delivery, at half-gap: GLD −$42, TLT −$144 per $1k.
  - Why: whatever premium exists before costs is small (GLD held to the close at half-gap is only −$42,
    of which fees alone are ~$33), and the costs are large relative to risk. A TLT
    fly risks ~$35 per contract, so $1.20 per contract on 4 legs alone is ~14% of the risk; GLD's
    relative spreads are several times SPX's. Closing in-the-money legs at 15:59 adds more.
- **G2 FAIL.** B0 over the same days: Sharpe 1.22. Correlation with B0 +0.04 (low, as hoped), but the combination's Sharpe falls to −5.5.
- Reported only: every symbol, every time and the rich-only filter are strongly negative
  (USO −$903/day, EEM −$1,226, XLF −$580; 10:00 −$439, 12:00 −$305, 14:00 −$369; rich-only −$840).
- **Verdict:** not added. Same-day ETF options outside SPY/QQQ/IWM are not a place to sell premium.

## O: hold the S&P overnight (fixed 2026-11-15, before computing)

### Why
- B0 is an intraday bet that ends at 16:00. The night (16:00 -> 09:30) is a different window, so its
  results should not line up with B0's.
- Published evidence that most of the stock market's long-run gain arrives overnight while the trading
  day earns little: Cooper, Cliff & Gulen (2008); Lou, Polk & Skouras (2019). Honest label: this is
  mostly a risk premium (being long stocks while markets are closed), not a pricing mistake.

### The rule (no fitted parameters)
- Every session D: buy 1 MES at the close of the 15:59 ET bar, sell at the open of the 09:30 bar of the
  next session (weekends included). ES minute data (data/futures/es_1m.parquet), same contract at both
  ends (nights spanning a contract roll are skipped).
- Cost 0.75 index points per round trip (as test M). P&L in points; $5 per point per MES.

### Pass rules
- **O1 (stands on its own):** mean net P&L per night > 0 in both 2016-2021 and 2022-2026 (by the date the
  night ends), AND pooled t-stat >= 2.0.
- **O2 (helps B0):** B0's walk-forward days 2019-06-26 .. 2026-11-06. Each night's P&L is booked on the
  first B0 day on or after the morning it ends (pre-2022 Tue/Thu nights roll forward; nothing is
  dropped). Correlation <= 0.10, AND the combination (equal daily volatility, weights from 2019-06-26 ..
  2021-12-31 only) has a higher Sharpe AND a lower P(losing year) than B0 alone (same bootstrap and
  seed as before).
- **Verdict:** PASS (O1 and O2) -> paper contender (needs MES quotes: CME data, owner decides). LEAD
  (both eras > 0, t >= 1.0, O2) -> shadow only. FAIL otherwise. No tuning.

### Reported only
- The day leg (long 09:30 open -> 15:59 close) for comparison; by year; costs 0 and 1.5 points;
  weekend vs weekday nights; worst night and max drawdown per MES.

Code: odte/research/partner_o.py -> reports/partners/o.json.

### Result (2026-11-15): LEAD, not PASS (reports/partners/o.json)
- **O1 FAIL (t too low).** 2,624 nights. After 0.75-point costs: +0.60 points a night (+$759 a year per
  MES), t 1.12; positive in both eras (2016-21 +0.77, 2022-26 +0.40) but well short of t 2.
  - Before costs +1.35 points, t 2.52: the overnight premium is there; MES costs eat over half of it.
  - At 1.5-point costs it loses (−0.15). By year: losing in 2016, 2022 (−2.73 a night) and 2023.
  - The day leg (09:30 -> 16:00) earns nothing (−0.10 points, t −0.14), as the papers say.
  - Worst night −218 points (−$1,090 per MES); max drawdown −$5.3k per MES.
- **O2 PASS.** Correlation with B0 0.04. Combined Sharpe 0.59 -> 0.76; P(losing year) 25.2% -> 19.95%.
  It earned in 2020 and 2021, the two years B0 lost (scaled: +41 and +65 a day vs B0 −35 and −72), but
  lost in 2022 (−51) when B0 was flat.
  - Size check: equal volatility means ~3.4 MES per B0 unit (~$115k of S&P overnight per $1k of B0 risk
    per trade). That is a large overnight position for a small account.
- **Verdict: LEAD.** Both eras positive, t >= 1, O2 passes. Per the rule: forward evidence only, no claim
  of an edge. Forward nights can be scored from public daily open/close data, so no bot change is needed.
- Not a pass, and not changed after the fact: a cheaper way to hold it (SPY bought in the closing auction,
  sold in the opening auction) would cost far less than MES, but that is a new test to fix in writing
  before it is run, not a reason to re-score this one.

## T: the turn of the month (fixed 2026-11-15, before computing)

### Why
- Ariel (1987), Lakonishok & Smidt (1988), McConnell & Xu (2008, RFS): a large share of the stock
  market's monthly gain has arrived from the last trading day of a month through the first three
  trading days of the next (salary and pension money coming in, month-end rebalancing).
- B0 is an intraday bet every day; T is a few-days-long position once a month, so their results
  should not line up.
- Evidence against, recorded up front: later studies find the effect weaker after it was published;
  and T is long the S&P over the same nights as partner O, so the two overlap.

### The rule (no fitted parameters)
- Trading days: the NYSE calendar (weekdays minus NYSE holidays and the 2018-12-05 and 2025-01-09
  closures). Checked against the ES data before writing this: every full or half ES session is an
  NYSE day, and the only NYSE days missing from the ES file are the quarterly expiry Fridays
  (mid-month, never inside a window).
- Day −1 = the last trading day of a month; days +1, +2, +3 = the first three of the next month.
- Buy 1 MES at the close of day −2 (the last regular-hours bar before 16:00 ET: 15:59, or 13:14 on half
  days) and sell at the close of day +3. Held over 4 trading days and the 4 nights before them. Same
  contract at both ends (ES rolls mid-month, so none should be skipped); a missing close skips the window.
- Cost 0.75 index points per round trip (as M and O). $5 per point per MES.
- Windows that end 2016-01-01 .. 2026-11-06 (about 130).

### Pass rules
- **T1 (stands on its own):** mean net P&L per window > 0 in both 2016-2021 and 2022-2026 (by the date
  the window ends), AND t-stat >= 2.0.
- **T2 (helps B0):** each held day is marked close to close (cost booked on the exit day) and booked on
  the first B0 day on or after it, as for O. B0's walk-forward days 2019-06-26 .. 2026-11-06.
  Correlation <= 0.10, AND the combination (equal daily volatility, weights from 2019-06-26 ..
  2021-12-31 only) has a higher Sharpe AND a lower P(losing year) than B0 alone (same bootstrap and seed).
- **Verdict:** PASS (T1 and T2) -> paper contender (MES quotes needed: owner decides). LEAD (both eras > 0,
  t >= 1.0, T2) -> shadow only. FAIL otherwise. No tuning.
- Power, stated up front: if these days earned no more than ordinary days, being long 4 days a month
  gives t of about 0.9 over these 10.8 years; t >= 2 needs them to earn about 2.5 times an ordinary day.
  If the whole month's gain still arrived in these 4 days, t would be about 5.

### Reported only
- **Is it the calendar or just being long?** Mean daily close-to-close return on window days vs all other
  trading days, Welch t-stat. Without a real difference T is only part-time buy-and-hold.
- Each day (−1, +1, +2, +3) separately; the nights vs the 09:30-16:00 days inside the window.
- Costs 0 and 1.5 points; by year; worst window; max drawdown per MES.
- Correlation with O's daily series; B0 + O + T together (same equal-volatility method).

Code: odte/research/partner_t.py -> reports/partners/t.json.

### Result (2026-11-15): LEAD by the rule, but there is no turn-of-month effect (reports/partners/t.json)
- **T1 FAIL (t too low).** 131 windows, none skipped. After costs +8.8 points a window (+$529 a year per
  MES), t 1.30; positive in both eras (2016-21 +8.4, t 1.21; 2022-26 +9.3, t 0.75). Costs barely matter
  (one round trip a month; +9.6 before costs). Worst window −255 points; max drawdown −$2.7k per MES.
- **T2 PASS.** Correlation with B0 −0.001. Combined Sharpe 0.59 -> 0.72; P(losing year) 25.2% -> 19.5%.
  It earned in 2020 (scaled +81 a day vs B0 −35) but little in 2021 (+8 vs −72). Weight ~5.1 MES per B0 unit.
- **The calendar check (reported only) says the effect is not there.** Window days averaged +0.060% a
  day, other days +0.054% (Welch t 0.13). Day −1 lost (−1.3 points a day). T is part-time buy-and-hold:
  it made money because stocks rose over 2016-26, and it helps B0 because B0 does not move with the market.
- Inside the windows the nights lost (−0.6 points, t −0.5) and the 09:30-16:00 sessions gained (+3.0,
  t 2.0), the reverse of O's pattern; several splits were reported, so this is treated as noise.
- Correlation with O 0.28 (shared nights). B0 + O + T: Sharpe 0.81, P(losing year) 17.2%.
- **Verdict: LEAD by the rule, recorded as it is, with the label "equity premium, not a calendar
  effect".** What O and T share is being long stocks, so the fair benchmark is plain buy-and-hold (B).

## B: plain buy-and-hold next to B0, the benchmark O and T must beat (fixed 2026-11-15, before computing)

### Why
- O and T both earn mostly the stock market's normal gain, and both help B0 because B0 does not move with
  the market. If simply owning the index all the time helps B0 as much, O and T only add complexity.
- It is also the simplest possible partner: an index fund, no bot change.

### The rule (no fitted parameters)
- Long 1 MES at all times, 2016-01-01 .. 2026-11-06, marked close to close on each trading day (last
  regular-hours close, NYSE calendar, as T).
- When the contract changes between two closes (the quarterly roll), that day's mark is 0 (the file holds
  one contract per minute, so the roll spread cannot be priced) and 0.75 points of cost is charged.
  Recorded up front: this leaves out about 4 days of market moves a year.

### Pass rules
- **B1:** mean daily net mark > 0 in both 2016-2021 and 2022-2026, AND t-stat >= 2.0. Power: at the
  market's usual Sharpe (~0.7 a year), t over 10.8 years is about 2.3, so this can fail by chance.
- **B2:** as O2 (booked on the first B0 day on or after each day; equal volatility, weights from
  2019-06-26 .. 2021-12-31; correlation <= 0.10; higher Sharpe AND lower P(losing year); same bootstrap
  and seed).
- **Verdict:** PASS / LEAD / FAIL as before.
- **Head-to-head, fixed now:** O (or T) is worth more than an index fund only if B0+O (or B0+T) beats
  B0+B on BOTH Sharpe and P(losing year). Otherwise the honest partner is an index fund and O/T are
  dropped as separate ideas.

### Reported only
- Sharpe and P(losing year) of B0+B, B0+O, B0+T, B0+O+T and B0+B+O (each partner at equal volatility).
- Correlation of O and T with B.

Code: odte/research/partner_b.py -> reports/partners/b.json.

# Go / no-go rules for real money

Fixed 2026-11-10, before the first live paper day. Results must never be used to loosen
these rules. A change needs a written, dated reason here, made before the data it would
affect.

## 0. Hard rule
- No real money until the account owner explicitly says so. The code refuses live mode
  unless `allow_live=True` (`odte/live/config.py`).

## 1. The bot itself must be reliable first
- At least 60 paper trading days.
- In the last 20 of those days:
  - no unexplained crash;
  - no unresolved reconcile mismatch (end-of-day IBKR records vs. the journal);
  - no missed day that went unnoticed (the evening check catches these).

## 2. A strategy must prove itself on live data
Scored at the touch on XSP's own quotes. Paper fills are simulated by IBKR and do not
count as cost evidence.
- **Event-day seller** (binding, from the research plan):
  - passes the one-shot SPX exam (locked data from 2026-09-08);
  - AND at least 60 live event days with mean P&L > 0 and a 95% day-bootstrap CI
    lower bound > 0, per index.
- **Model sellers and the side seller (edge #1):**
  - at least 60 live days and at least 30 trades;
  - mean daily P&L > 0 with a 95% CI lower bound > 0;
  - AND beats always-sell on the same days by the weekly loop's promotion rule
    (paired SE, Bonferroni over the candidates).
- **Anything else:** pre-registered first, then the same bar.

## 3. How real money would start (incubation)
- One XSP lot per trade, at most $300 risk per trade and $600 per day, for at least one
  year of live data.
- No increase in size during that year.
- The first purpose is to measure real fills: the share of the bid/ask gap actually paid.

## 4. Stop rules (decided now)
- **Costs worse than assumed:** if, over at least 20 real fills, the average share of
  the gap paid is above 0.75 (the backtests assumed 0.5), stop real money and rerun the
  backtests at the measured cost.
- **Drawdowns:** two drawdown pauses ($1,500, the existing rule) within 6 months stops
  real money and returns to paper.
- **Project:** if after 120 live trading days no strategy meets section 2, stop adding
  options strategies and re-evaluate. No "one more tweak".
- **Never:**
  - increase size to win back losses;
  - trade through an unresolved reconcile mismatch;
  - change a rule after seeing the results it would change.

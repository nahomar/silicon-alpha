"""Automated daily 0DTE trader (paper first).

Design rule: live decisions are computed by the SAME code as the backtests.
The bot records the day's option quotes in the historical file format and
runs them through odte.eval.expiry_edge.session_rows, so a live feature row
equals the row the research would have built from that day's data.

Modules
  config    settings and risk limits
  model     train / load the ridge model the backtests used
  features  quotes so far + calendar + context -> one feature row
  fly       iron-fly construction, sizing, the limit-price ladder
  broker    IBKR (ib_async) and a replay broker over historical days
  runner    the trading day: record, decide, order, settle, journal
"""

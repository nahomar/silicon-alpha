"""Strategy zoo: test many disclosed 0DTE option strategies on SPX minute quotes at once.

Each strategy is a rule in a small vocabulary (docs/strategy_zoo_catalog.json): entry
time, structure, strikes in sigma units (sigma = ATM straddle / 0.8, one implied
standard move to the close), wings, direction, filters, exits. Pre-registration and
the batch gate: docs/strategy_zoo_plan.md.

Pricing:
  - each leg at mid; trading cost = `share` x the leg's half-spread, scaled to today's
    level by the day's factor k = (today's median ATM straddle relative spread) /
    (that day's at 10:00), clipped to [0.1, 2]. share 0.5 = "half-gap" (the convention
    of every earlier test), 1.0 = "touch" (pay the full mid-to-touch distance);
  - fees $1.20 per contract per side;
  - held to cash settlement on the official close unless an exit fires (checked
    every minute on mids, exit priced with that minute's spreads);
  - sized to $1,000 of worst-case loss per trade (worst case of the expiry payoff).

  python3 -m odte.research.zoo run docs/strategy_zoo_catalog.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "data" / "odte_0dte"
EXAM_FROM = pd.Timestamp("2026-09-08")
ERA_SPLIT = pd.Timestamp("2022-01-01")
COST_WINDOW = (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-04"))
FEE, BUDGET = 1.20, 1000.0
STRUCTURES = ("iron_fly", "iron_condor", "put_credit_spread", "call_credit_spread", "long_straddle",
              "long_strangle", "long_call_butterfly", "long_put_butterfly", "broken_wing_put_butterfly",
              "broken_wing_call_butterfly", "put_debit_spread", "call_debit_spread")


# ---------------------------------------------------------------------------
# one day of quotes
# ---------------------------------------------------------------------------

class Day:
    def __init__(self, day: pd.Timestamp, chain: pd.DataFrame, settle: float, prev_settle: float, info: dict):
        from odte.eval.expiry_edge import implied_forward, minute_grid
        g = minute_grid(chain)
        self.day, self.settle, self.prev, self.info = day, settle, prev_settle, info
        self.bid, self.ask = g["bid"], g["ask"]
        ok = (self.bid > 0) & (self.ask >= self.bid)
        self.mid = ((self.bid + self.ask) / 2).where(ok)
        self.half = ((self.ask - self.bid) / 2).where(ok)
        self.F = implied_forward(g)
        self.minutes = self.mid.index

    def at(self, hhmm: str) -> pd.Timestamp:
        return self.day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))

    def forward(self, t) -> float:
        f = self.F[self.F.index <= t].dropna()
        return float(f.iloc[-1]) if len(f) else float("nan")

    def strikes(self, t, right) -> np.ndarray:
        row = self.mid.loc[t]
        return np.array(sorted(k for (r, k), v in row.items() if r == right and np.isfinite(v)), float)

    def straddle(self, t, F) -> tuple:
        c, p = self.strikes(t, "C"), self.strikes(t, "P")
        both = np.intersect1d(c, p)
        if not len(both) or not np.isfinite(F):
            return np.nan, np.nan
        K = float(both[np.argmin(np.abs(both - F))])
        m = self.mid.loc[t]
        return K, float(m[("C", K)] + m[("P", K)])

    def straddle_rel_spread(self, t) -> float:
        K, st = self.straddle(t, self.forward(t))
        if not np.isfinite(st) or st <= 0:
            return np.nan
        h = self.half.loc[t]
        return float((h[("C", K)] + h[("P", K)]) / st)


def nearest(ks: np.ndarray, x: float):
    if not len(ks) or not np.isfinite(x):
        return None
    return float(ks[np.argmin(np.abs(ks - x))])


# ---------------------------------------------------------------------------
# structures -> legs [(right, strike, qty)]  (qty > 0 long, < 0 short)
# ---------------------------------------------------------------------------

def legs_for(rule: dict, d: Day, t, F: float, sigma: float, direction: str | None) -> list | None:
    s = rule["structure"]
    kc, kp = d.strikes(t, "C"), d.strikes(t, "P")
    k = float(rule.get("short_k", 0.0))
    if "wing_points" in rule:
        W = float(rule["wing_points"])
    else:
        W = float(rule.get("wing", 1.0)) * sigma
    K0 = nearest(np.intersect1d(kc, kp), F)
    if K0 is None or not W > 0:
        return None

    def put_spread(short_side: bool):
        Ks = nearest(kp, F - k * sigma)
        Kw = nearest(kp, (Ks - W) if Ks is not None else np.nan)
        if Ks is None or Kw is None or Kw >= Ks:
            return None
        return [("P", Ks, -1), ("P", Kw, 1)] if short_side else [("P", Ks, 1), ("P", Kw, -1)]

    def call_spread(short_side: bool):
        Ks = nearest(kc, F + k * sigma)
        Kw = nearest(kc, (Ks + W) if Ks is not None else np.nan)
        if Ks is None or Kw is None or Kw <= Ks:
            return None
        return [("C", Ks, -1), ("C", Kw, 1)] if short_side else [("C", Ks, 1), ("C", Kw, -1)]

    if s == "iron_fly":
        Wc, Wp = nearest(kc, K0 + W), nearest(kp, K0 - W)
        if Wc is None or Wp is None or Wc <= K0 or Wp >= K0:
            return None
        return [("C", K0, -1), ("P", K0, -1), ("C", Wc, 1), ("P", Wp, 1)]
    if s == "iron_condor":
        a, b = put_spread(True), call_spread(True)
        return (a + b) if a and b and a[0][1] < b[0][1] else None
    if s in ("put_credit_spread", "call_credit_spread", "put_debit_spread", "call_debit_spread") or (
            direction is not None and s in ("credit_spread", "debit_spread")):
        side = {"put_credit_spread": ("P", True), "call_credit_spread": ("C", True),
                "put_debit_spread": ("P", False), "call_debit_spread": ("C", False)}.get(s, (None, True))
        if direction is not None:          # direction "put" = bullish view (sell puts / buy calls)
            credit = side[1]
            right = ("P" if direction == "put" else "C") if credit else ("C" if direction == "put" else "P")
            side = (right, credit)
        return put_spread(side[1]) if side[0] == "P" else call_spread(side[1])
    if s == "long_straddle":
        return [("C", K0, 1), ("P", K0, 1)]
    if s == "long_strangle":
        Kc, Kp = nearest(kc, F + k * sigma), nearest(kp, F - k * sigma)
        return [("C", Kc, 1), ("P", Kp, 1)] if Kc and Kp and Kc > Kp else None
    if s in ("long_call_butterfly", "long_put_butterfly", "broken_wing_put_butterfly", "broken_wing_call_butterfly"):
        right = "C" if "call" in s else "P"
        ks = kc if right == "C" else kp
        Kc_ = nearest(ks, F + float(rule.get("center_k", 0.0)) * sigma)
        if Kc_ is None:
            return None
        if s.startswith("broken_wing"):
            near, far = (Kc_ + W, Kc_ - 2 * W) if right == "P" else (Kc_ - W, Kc_ + 2 * W)
        else:
            near, far = Kc_ - W, Kc_ + W
        a, b = nearest(ks, near), nearest(ks, far)
        if a is None or b is None or len({a, b, Kc_}) < 3:
            return None
        return [(right, a, 1), (right, Kc_, -2), (right, b, 1)]
    return None


def payoff(legs: list, S: float) -> float:
    return sum(q * (max(S - k, 0.0) if r == "C" else max(k - S, 0.0)) for r, k, q in legs)


def worst_payoff(legs: list, F: float) -> tuple:
    """Lowest expiry payoff (per 1x) and whether it is bounded (no negative tail slope)."""
    pts = sorted({k for _, k, _ in legs} | {0.0, F * 3})
    lo = min(payoff(legs, s) for s in pts)
    slope_up = sum(q for r, _, q in legs if r == "C")
    return lo, slope_up >= 0


# ---------------------------------------------------------------------------
# one trade
# ---------------------------------------------------------------------------

def trade(rule: dict, d: Day, entry_hhmm: str, k_day: float, share: float) -> dict | None:
    t = d.at(entry_hhmm)
    if t not in d.minutes:
        return None
    F = d.forward(t)
    K0, st = d.straddle(t, F)
    if not (np.isfinite(st) and st > 0):
        return None
    sigma = st / 0.8
    direction = None
    dr = rule.get("direction")
    if dr:
        F0 = d.forward(d.at("09:31"))
        move = F - F0
        if dr == "fixed_put":
            direction = "put"
        elif dr == "fixed_call":
            direction = "call"
        elif dr in ("with_morning_move", "against_morning_move"):
            if not np.isfinite(move) or move == 0:
                return None
            up = move > 0
            sell_puts = up if dr == "with_morning_move" else not up
            direction = "put" if sell_puts else "call"
    legs = legs_for(rule, d, t, F, sigma, direction)
    if not legs:
        return None
    m, h = d.mid.loc[t], d.half.loc[t]
    try:
        V = sum(q * float(m[(r, k)]) for r, k, q in legs)
        cost_in = sum(abs(q) * float(h[(r, k)]) for r, k, q in legs) * k_day * share
    except KeyError:
        return None
    if not np.isfinite(V) or not np.isfinite(cost_in):
        return None
    n_contracts = sum(abs(q) for _, _, q in legs)
    cash_in = -V - cost_in
    lo, bounded = worst_payoff(legs, F)
    if not bounded:
        return None
    risk = -(cash_in * 100 - n_contracts * FEE + lo * 100)
    if not risk > 0:
        return None
    ex = rule.get("exit", {}) or {}
    cash_out, exited = None, None
    tp, sl, tx = ex.get("take_profit"), ex.get("stop_loss"), ex.get("time_exit")
    if tp is not None or sl is not None or tx is not None:
        t_end = d.at(tx) if tx else d.minutes.max() - pd.Timedelta(minutes=1)
        win = d.minutes[(d.minutes > t) & (d.minutes <= t_end)]
        cols = [(r, k) for r, k, _ in legs]
        qv = np.array([q for _, _, q in legs], float)
        vm = d.mid.loc[win, cols].to_numpy(float) @ qv
        hm = np.abs(qv) @ d.half.loc[win, cols].to_numpy(float).T
        credit = V < 0
        base = abs(V)
        hit = np.zeros(len(win), bool)
        if tp is not None:
            hit |= (-vm <= (1 - tp) * base) if credit else (vm >= (1 + tp) * base)
        if sl is not None:
            hit |= (-vm >= (1 + sl) * base) if credit else (vm <= (1 - min(sl, 1.0)) * base)
        hit &= np.isfinite(vm) & np.isfinite(hm)
        if hit.any():
            i = int(np.argmax(hit))
            cash_out, exited = vm[i] - hm[i] * k_day * share, "tp/sl"
        elif tx is not None:
            good = np.isfinite(vm) & np.isfinite(hm)
            if good.any():
                i = int(np.where(good)[0][-1])
                cash_out, exited = vm[i] - hm[i] * k_day * share, "time"
    if cash_out is None:
        cash_out = payoff(legs, d.settle)
        fees = n_contracts * FEE
    else:
        fees = 2 * n_contracts * FEE
    pnl = (cash_in + cash_out) * 100 - fees
    return {"pnl": pnl, "risk": risk, "ret": pnl / risk * BUDGET, "exited": exited}


def passes_filters(rule: dict, d: Day, entry_hhmm: str) -> bool:
    f = rule.get("filters", {}) or {}
    info = d.info
    vix = info.get("vix_prev", np.nan)
    if "vix_min" in f and not (np.isfinite(vix) and vix >= f["vix_min"]):
        return False
    if "vix_max" in f and not (np.isfinite(vix) and vix <= f["vix_max"]):
        return False
    ev = bool(info.get("event_day", False))
    if f.get("skip_event_days") and ev:
        return False
    if f.get("only_event_days") and not ev:
        return False
    if "days_of_week" in f and d.day.dayofweek not in f["days_of_week"]:
        return False
    if "max_gap_pct" in f:
        F0 = d.forward(d.at("09:31"))
        if not (np.isfinite(F0) and np.isfinite(d.prev) and abs(F0 / d.prev - 1) * 100 <= f["max_gap_pct"]):
            return False
    if "min_morning_move_sigma" in f or "max_morning_move_sigma" in f:
        t = d.at(entry_hhmm)
        F0, Ft = d.forward(d.at("09:31")), d.forward(t)
        _, st = d.straddle(t, Ft)
        if not (np.isfinite(st) and st > 0 and np.isfinite(F0) and np.isfinite(Ft)):
            return False
        mv = abs(Ft - F0) / (st / 0.8)
        if mv < f.get("min_morning_move_sigma", -1) or mv > f.get("max_morning_move_sigma", 1e9):
            return False
    return True


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def day_info() -> pd.DataFrame:
    from odte.research import factory as fx
    d = fx.load()
    g = d.groupby("day").first()
    ev = (g[["ev_cpi", "ev_nfp", "ev_fomc", "ev_fomc_sep"]].fillna(0).sum(axis=1) > 0)
    return pd.DataFrame({"vix_prev": g["vix_prev"], "event_day": ev})


def entry_times(rule: dict) -> list:
    e = rule["entry_time"]
    return list(e) if isinstance(e, list) else [e]


def run_batch(rules: list, days: list | None = None, shares=(0.5, 1.0)) -> pd.DataFrame:
    """Daily P&L per rule and share: index day, columns (rule_id, share)."""
    info = day_info()
    st = pd.read_csv(CHAINS / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    files = sorted(p for p in CHAINS.glob("20??-??-??.parquet") if pd.Timestamp(p.stem) < EXAM_FROM)
    if days is not None:
        files = [p for p in files if pd.Timestamp(p.stem) in set(days)]
    # today's typical straddle relative spread (for the cost scale)
    rel_today = []
    for p in files:
        if COST_WINDOW[0] <= pd.Timestamp(p.stem) <= COST_WINDOW[1]:
            dd = pd.Timestamp(p.stem)
            if dd in st.index and dd in info.index:
                x = Day(dd, pd.read_parquet(p), float(st[dd]), np.nan, {}).straddle_rel_spread(dd + pd.Timedelta(hours=10))
                if np.isfinite(x):
                    rel_today.append(x)
    R_today = float(np.median(rel_today)) if rel_today else 0.01
    out = {}
    for p in files:
        dd = pd.Timestamp(p.stem)
        if dd not in st.index or dd not in info.index:
            continue
        prev = st[st.index < dd]
        d = Day(dd, pd.read_parquet(p), float(st[dd]), float(prev.iloc[-1]) if len(prev) else np.nan,
                info.loc[dd].to_dict())
        r_day = d.straddle_rel_spread(d.at("10:00"))
        k_day = float(np.clip(R_today / r_day, 0.1, 2.0)) if np.isfinite(r_day) and r_day > 0 else 1.0
        row = {}
        for rule in rules:
            for share in shares:
                tot = 0.0
                for hh in entry_times(rule):
                    if not passes_filters(rule, d, hh):
                        continue
                    tr = trade(rule, d, hh, k_day, share)
                    if tr is not None:
                        tot += tr["ret"]
                row[(rule["id"], share)] = tot
        out[dd] = row
    res = pd.DataFrame.from_dict(out, orient="index").sort_index()
    res.attrs["R_today"] = R_today
    return res


def summarize(daily: pd.DataFrame, rules: list, q: float = 0.05) -> pd.DataFrame:
    from odte.research.factory import bh
    rows = []
    for rule in rules:
        x = daily[(rule["id"], 0.5)]
        n = len(x)
        mu, sd = x.mean(), x.std(ddof=1)
        t = mu / (sd / math.sqrt(n)) if sd > 0 else 0.0
        p = math.erfc(abs(t) / math.sqrt(2))
        e1, e2 = x[x.index < ERA_SPLIT].sum(), x[x.index >= ERA_SPLIT].sum()
        touch = daily[(rule["id"], 1.0)]
        rows.append({"id": rule["id"], "family": rule.get("family", ""), "fidelity": rule.get("fidelity", ""),
                     "mean_day": mu, "t": t, "p": p, "total": x.sum(), "2016-21": e1, "2022-26": e2,
                     "touch_total": touch.sum(), "trade_days": int((x != 0).sum())})
    tab = pd.DataFrame(rows)
    tab["fdr_pass"] = bh(tab["p"].to_numpy(), q) & (tab["mean_day"] > 0)
    tab["survives"] = tab["fdr_pass"] & (tab["2016-21"] > 0) & (tab["2022-26"] > 0) & (tab["touch_total"] > 0)
    return tab.sort_values("p")


def load_rules(path) -> list:
    cat = json.loads(Path(path).read_text())
    rules = []
    for e in cat:
        if e.get("fidelity") == "untestable":
            continue
        r = dict(e.get("rule", {}))
        r["id"], r["family"], r["fidelity"] = e["id"], e.get("family", ""), e.get("fidelity", "")
        if r.get("structure") in STRUCTURES or r.get("direction"):
            rules.append(r)
    return rules


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        rules = load_rules(sys.argv[2])
        daily = run_batch(rules)
        out = ROOT / "reports" / "factory"
        daily.to_parquet(out / "zoo_daily.parquet")
        tab = summarize(daily, rules)
        tab.to_csv(out / "zoo_results.csv", index=False)
        print(f"{len(rules)} rules, {len(daily)} days, R_today {daily.attrs.get('R_today'):.4f}")
        print(tab.round(3).to_string(index=False))
        print(f"\nSURVIVORS: {int(tab['survives'].sum())}")

"""Signal factory: many candidate signals x several targets, judged with FDR control.

Protocol (pre-registered 2026-11-09, docs/expiry_edge_plan.md):
  * data: seen SPX fixed-time decisions 2016-01 .. 2026-09-04 (exam excluded)
  * ~50 signals from six families, all known at decision time t
  * targets (z = (settle - F) / straddle mid):
      T_fly   half-gap return on risk of selling the iron fly (today's cost)
      T_dir   z          (which side the close lands on)
      T_size  |z|        (move vs priced)
      T_tail  |z| > 1.5  (danger)
  * statistic: Spearman IC over rows, t-stat clustered by day
  * survive: Benjamini-Hochberg q <= 0.05 over ALL tests AND same IC sign in
    2016-2021 and 2022-2026
  * survivors -> walk-forward ridge composite per target -> live challenger

  python3 -m odte.research.factory
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / "reports" / "expiry_edge" / "decisions-sealed-c1.parquet",
         ROOT / "reports" / "expiry_edge" / "decisions-dev-gex-es.parquet"]
EXAM_FROM = pd.Timestamp("2026-09-08")
TODAY_COST, FEES4 = 0.0169, 4 * 1.20
ERA_SPLIT = pd.Timestamp("2022-01-01")
FDR_Q = 0.05
EVENT_FLAGS = ["ev_cpi", "ev_nfp", "ev_ppi", "ev_jolts", "ev_claims", "ev_ism_mfg", "ev_ism_svc", "ev_fomc",
               "ev_fomc_sep", "ev_fomc_minutes", "ev_auction_long", "ev_opex", "ev_quad_witching",
               "ev_vix_expiry", "ev_month_end", "ev_quarter_end", "ev_pre_holiday", "ev_post_holiday",
               "megacap_earnings"]


# ---------------------------------------------------------------------------
# data, targets, signals
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
    d = d[d["on6"]].copy()
    d["day"] = pd.to_datetime(d["day"])
    d = d[d["day"] < EXAM_FROM].drop_duplicates(["day", "hhmm"]).sort_values(["day", "hhmm"])
    d = d[np.isfinite(d["fly_credit_mid"]) & (d["fly_credit_mid"] > 0) & np.isfinite(d["W"])
          & np.isfinite(d["settle"]) & (d["straddle_mid"] > 0)].reset_index(drop=True)
    cm, W = d["fly_credit_mid"].to_numpy(), d["W"].to_numpy()
    intrinsic = cm - d["pnl_short_fly_mid"].to_numpy() / 100
    c = cm * (1 - 0.5 * TODAY_COST)
    risk = (W - c) * 100 + FEES4
    d["T_fly"] = np.where(risk > 0, ((c - intrinsic) * 100 - FEES4) / risk, np.nan)
    z = (d["settle"] - d["F"]) / d["straddle_mid"]
    d["T_dir"], d["T_size"], d["T_tail"] = z, z.abs(), (z.abs() > 1.5).astype(float)
    return d


def _hist_z(d: pd.DataFrame, col: str, n: int = 20) -> pd.Series:
    """Value vs its own previous n days at the same time of day (prior days only)."""
    g = d.groupby("hhmm")[col]
    mu = g.transform(lambda s: s.shift(1).rolling(n, min_periods=10).mean())
    sd = g.transform(lambda s: s.shift(1).rolling(n, min_periods=10).std())
    return (d[col] - mu) / sd.replace(0, np.nan)


def signals(d: pd.DataFrame) -> pd.DataFrame:
    s = {}
    ann = np.sqrt(390 * 252)
    im = d["implied_move"].replace(0, np.nan)
    # option pricing
    s["iv_over_rv"] = d["iv_over_rv"]
    s["iv_level_vs_hist"] = _hist_z(d, "iv_per_rootmin")
    s["iv_vs_vix"] = d["iv_per_rootmin"] * ann / (d["vix_prev"] / 100)
    s["skew"] = d["skew"]
    s["skew_vs_hist"] = _hist_z(d, "skew")
    s["prev_ratio"] = d["prev_ratio"]
    s["vix9d_over_vix"] = d["vix9d_over_vix"]
    s["vix_over_vix3m"] = d["vix_over_vix3m"]
    s["vvix_over_vix"] = d["vvix_prev"] / d["vix_prev"]
    s["vix_chg"] = d["vix_chg"]
    s["quoted_spread"] = d["straddle_spread"] / d["straddle_mid"]
    s["vix_level"] = d["vix_prev"]
    # price action (scaled by today's implied move)
    s["gap_vs_implied"] = d["gap"] / im
    s["move_vs_implied"] = d["move_so_far"] / im
    s["absmove_vs_implied"] = d["move_so_far"].abs() / im
    s["range_vs_implied"] = d["range_so_far"] / im
    s["rv30_over_rvday"] = d["rv_last30"] / d["rv_so_far"]
    s["spx_ret_prev"] = d["spx_ret_prev"]
    s["spx_rv5_over_rv20"] = d["spx_rv5"] / d["spx_rv20"]
    # calendar
    for f in EVENT_FLAGS:
        s[f] = d[f]
    s["events_ahead"] = d["events_ahead"]
    s["fomc_ahead"] = d["fomc_ahead"]
    s["dow"] = d["dow"]
    s["minutes_to_close"] = d["minutes_to_close"]
    # dealer positioning
    s["gex_net"] = d["gex_net"]
    s["gex_0dte"] = d["gex_0dte"]
    s["gex_flip_dist"] = d["gex_flip_dist"]
    s["maxoi_dist"] = d["maxoi_dist"]
    s["maxoi_absdist"] = d["maxoi_dist"].abs()
    s["oi_pcr_0dte"] = d["oi_pcr_0dte"]
    # cross-market
    for c in ("tnx_chg", "dxy_ret", "crude_ret", "gold_ret", "hyg_ret", "btc_ret", "es_on_ret",
              "es_on_vol", "es_last15"):
        s[c] = d[c]
    s["es_on_rv_vs_implied"] = d["es_on_rv"] / 1e4 / im
    s["es_on_range_vs_implied"] = d["es_on_range"] / 1e4 / im
    return pd.DataFrame(s, index=d.index).replace([np.inf, -np.inf], np.nan)


def regime_encodings(d: pd.DataFrame, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Idea B: k-means market state per day, refit each year on earlier days only.
    Each row gets, per target, the mean of that target over EARLIER days in the
    same state (walk-forward target encoding)."""
    from odte.data import context as ctx
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    wide = ctx.load()
    spx = wide["spx"].dropna()
    day_rows = d.groupby("day").first()
    feats = pd.DataFrame(index=day_rows.index)
    feats["vix"] = day_rows["vix_prev"]
    feats["term"] = day_rows["vix_over_vix3m"]
    feats["rv20"] = day_rows["spx_rv20"]
    feats["vvix"] = day_rows["vvix_prev"] / day_rows["vix_prev"]
    feats["ret20"] = [np.log(spx[spx.index < dd].iloc[-1] / spx[spx.index < dd].iloc[-21])
                      if (spx.index < dd).sum() > 21 else np.nan for dd in feats.index]
    feats = feats.dropna()
    daily_t = d.groupby("day")[["T_fly", "T_dir", "T_size", "T_tail"]].mean()
    out = {t: pd.Series(np.nan, index=feats.index) for t in daily_t.columns}
    years = sorted({dd.year for dd in feats.index})
    for y in years:
        prior = feats[feats.index < pd.Timestamp(f"{y}-01-01")]
        cur = feats[(feats.index >= pd.Timestamp(f"{y}-01-01")) & (feats.index < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(prior) < 200 or cur.empty:
            continue
        sc = StandardScaler().fit(prior)
        km = KMeans(k, n_init=10, random_state=seed).fit(sc.transform(prior))
        lab = pd.Series(km.predict(sc.transform(pd.concat([prior, cur]))), index=pd.concat([prior, cur]).index)
        for dd in cur.index:
            earlier = lab[(lab.index < dd) & (lab == lab[dd])].index
            for t in daily_t.columns:
                v = daily_t.loc[daily_t.index.intersection(earlier), t]
                out[t][dd] = v.mean() if len(v) >= 20 else np.nan
    enc = pd.DataFrame({f"regime_enc_{t}": s for t, s in out.items()})
    return enc.reindex(d["day"]).set_index(d.index)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def ic_test(sig: pd.Series, tgt: pd.Series, day: pd.Series) -> dict:
    ok = sig.notna() & tgt.notna()
    if ok.sum() < 200 or sig[ok].nunique() < 2:
        return {"n": int(ok.sum()), "ic": np.nan, "t": np.nan, "p": np.nan}
    rs, rt = sig[ok].rank(), tgt[ok].rank()
    zs, zt = (rs - rs.mean()) / rs.std(), (rt - rt.mean()) / rt.std()
    prod = zs * zt
    ic = float(prod.mean())
    per_day = (prod - ic).groupby(day[ok]).sum()
    se = float(np.sqrt((per_day ** 2).sum()) / ok.sum())
    t = ic / se if se > 0 else np.nan
    p = float(math.erfc(abs(t) / math.sqrt(2))) if np.isfinite(t) else np.nan
    return {"n": int(ok.sum()), "days": int(day[ok].nunique()), "ic": ic, "t": t, "p": p}


def bh(pvals: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = ok.sum()
    thresh = q * np.arange(1, m + 1) / m
    passed = p[idx] <= thresh
    if passed.any():
        out[idx[: np.max(np.where(passed)[0]) + 1]] = True
    return out


# ---------------------------------------------------------------------------
# composites
# ---------------------------------------------------------------------------

def walk_forward_ridge(d: pd.DataFrame, X: pd.DataFrame, y: pd.Series, min_days=250, refit=21) -> np.ndarray:
    from odte.eval.expiry_edge import _make_model
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    out = np.full(len(d), np.nan)
    Xa, ya = X.to_numpy(float), y.to_numpy(float)
    for k in range(min_days, len(days), refit):
        tr = (dd < days[k]) & np.isfinite(ya)
        te = dd >= days[k]
        if k + refit < len(days):
            te &= dd < days[k + refit]
        m = _make_model("ml_ridge").fit(Xa[tr], ya[tr])
        out[te] = m.predict(Xa[te])
    return out


def day_series(d, pnl):
    return pd.Series(pnl).groupby(d["day"].to_numpy()).sum()


def summarize(x: pd.Series) -> dict:
    from odte.eval.panel_stats import day_block_bootstrap
    x = x.dropna()
    if len(x) < 20:
        return {"days": int(len(x))}
    b = day_block_bootstrap(x.to_numpy())
    era = {"2016-21": float(x[x.index < ERA_SPLIT].sum()), "2022-26": float(x[x.index >= ERA_SPLIT].sum())}
    return {"days": int(len(x)), "total": float(x.sum()), "sharpe": b["sharpe"],
            "sharpe_ci": [b["ci_low"], b["ci_high"]], **era}


def main() -> int:
    d = load()
    S = signals(d)
    S = pd.concat([S, regime_encodings(d)], axis=1)
    targets = ["T_fly", "T_dir", "T_size", "T_tail"]
    era1 = d["day"] < ERA_SPLIT
    rows = []
    for t in targets:
        for name in S.columns:
            if name.startswith("regime_enc_") and name != f"regime_enc_{t}":
                continue                                  # each encoding only vs its own target
            r = ic_test(S[name], d[t], d["day"])
            r1 = ic_test(S[name][era1], d[t][era1], d["day"][era1])
            r2 = ic_test(S[name][~era1], d[t][~era1], d["day"][~era1])
            rows.append({"target": t, "signal": name, **r, "ic_2016_21": r1["ic"], "ic_2022_26": r2["ic"]})
    tab = pd.DataFrame(rows)
    tab["fdr_pass"] = bh(tab["p"].to_numpy(), FDR_Q)
    tab["same_sign"] = np.sign(tab["ic_2016_21"]) == np.sign(tab["ic_2022_26"])
    tab["survives"] = tab["fdr_pass"] & tab["same_sign"] & (np.sign(tab["ic"]) == np.sign(tab["ic_2016_21"]))
    n_tests = int(tab["p"].notna().sum())
    print(f"{d['day'].nunique()} days, {len(d):,} rows; {S.shape[1]} signals; {n_tests} tests; "
          f"FDR passes {int(tab['fdr_pass'].sum())}; survivors {int(tab['survives'].sum())}\n", flush=True)

    comps = {}
    for t in targets:
        surv = tab[(tab["target"] == t) & tab["survives"]].sort_values("p")
        print(f"== {t}: {len(surv)} survivors")
        if len(surv):
            print(surv[["signal", "ic", "t", "ic_2016_21", "ic_2022_26"]].round(4).to_string(index=False))
        if not len(surv):
            continue
        pred = walk_forward_ridge(d, S[surv["signal"].tolist()], d[t])
        comps[t] = {"signals": surv["signal"].tolist(),
                    "oos_ic": ic_test(pd.Series(pred), d[t], d["day"])}
        if t == "T_fly":
            sell = pred > 0
        elif t in ("T_size", "T_tail"):
            thr = np.nanquantile(pred, 0.8)
            sell = np.isfinite(pred) & (pred < thr)          # skip the predicted-riskiest 20%
        else:
            sell = None
        if sell is not None:
            tested = np.isfinite(pred)
            comps[t]["seller"] = summarize(day_series(d[tested], np.where(sell[tested], d["T_fly"][tested] * 1000, 0.0)))
            comps[t]["always_sell_same_days"] = summarize(day_series(d[tested], d["T_fly"][tested] * 1000))
        else:
            tested = np.isfinite(pred)
            hit = (np.sign(pred[tested]) == np.sign(d["T_dir"][tested])).mean()
            comps[t]["direction_hit_rate"] = float(hit)
        print(f"   composite out-of-sample IC {comps[t]['oos_ic']['ic']:+.4f} (t {comps[t]['oos_ic']['t']:+.2f})"
              + (f"; seller {comps[t]['seller']}" if "seller" in comps[t] else "")
              + (f"; always-sell {comps[t]['always_sell_same_days']}" if "always_sell_same_days" in comps[t] else "")
              + (f"; direction hit {comps[t]['direction_hit_rate']:.3f}" if "direction_hit_rate" in comps[t] else ""))
        print()
    out = ROOT / "reports" / "factory"
    out.mkdir(parents=True, exist_ok=True)
    tab.sort_values("p").to_csv(out / "all_tests.csv", index=False)
    (out / "composites.json").write_text(json.dumps(comps, indent=2, default=float))
    top = tab.sort_values("p").head(15)[["target", "signal", "ic", "t", "p", "fdr_pass", "same_sign", "survives"]]
    print("Top 15 by p-value (for context):")
    print(top.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

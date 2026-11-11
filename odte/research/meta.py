"""Phase 2-3 (docs/master_plan.md): one meta-model over every signal, choosing among five trades.

For each decision row it predicts the half-gap return of each candidate trade (short fly,
short condor, put spread, call spread, long straddle) and takes the best one if its
prediction is > 0, else no trade. Walk-forward on development data only.

Inputs come from an explicit whitelist (FEATURE_GROUPS): nothing computed from the
settlement or from quotes after the decision time can enter.

  python3 -m odte.research.meta zoo          # every model type, walk-forward, report
  python3 -m odte.research.meta ablate MODEL  # group ablation for one model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import exam_guard as eg
from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "data" / "research"
OUT = ROOT / "reports" / "meta"
TRADES = ("fly", "condor", "put_spread", "call_spread", "long_straddle")
EVAL_FROM = pd.Timestamp("2022-01-01")
MIN_DAYS, REFIT, BUDGET = 500, 63, 1000.0
INDEXES = ("SPX", "NDX", "RUT")


def feature_groups(cols) -> dict:
    from odte.data.calendar import CAL_FEATURES
    from odte.data.context import CTX_FEATURES
    from odte.data.gamma import GEX_FEATURES
    from odte.eval.expiry_edge import FEATURES
    from odte.research.term_structure import SIGNALS as TS
    g = {"session": list(FEATURES), "calendar": list(CAL_FEATURES), "context": list(CTX_FEATURES) + ["iv_vs_vix1d"],
         "gex": list(GEX_FEATURES), "term": list(TS), "flow": ["fly_chg5", "fly_chg15"],
         "strategy": ["sig_event_day", "sig_morning_dir", "sig_calm5"],
         "cross": [c for c in cols if c.startswith("x_")], "factory": [c for c in cols if c.startswith("f_")],
         "identity": ["is_SPX", "is_NDX", "is_RUT", "tod"]}
    return {k: [c for c in dict.fromkeys(v) if c in cols] for k, v in g.items()}


MATRIX = LIB / "meta_matrix.parquet"


def load() -> pd.DataFrame:
    """Combined development table. Uses the cached matrix if present (e.g. on Sol, where the raw
    option files are not copied); `python3 -m odte.research.meta matrix` rebuilds it."""
    if MATRIX.exists():
        d = pd.read_parquet(MATRIX)
        eg.check(d["day"])
        return d
    return build_matrix()


def build_matrix() -> pd.DataFrame:
    from odte.research import term_structure as ts
    parts = []
    for sym in INDEXES:
        p = LIB / f"library_{sym}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[d["day"] < eg.EXAM_FROM].reset_index(drop=True)
        eg.check(d["day"])
        for c in ("es_on_range", "es_on_rv", "es_on_ret", "es_on_vol", "es_last15", "gex_0dte", "gex_flip_dist",
                  "gex_net", "maxoi_dist", "oi_pcr_0dte"):
            if c not in d.columns:
                d[c] = np.nan                                             # not available for this index
        d = d.sort_values(["day", "hhmm"]).reset_index(drop=True)
        S = fx.signals(d).add_prefix("f_")                                # factory signals, per index
        d = pd.concat([d, S], axis=1)
        if sym == "SPX":
            d = d.join(ts.build(d)[ts.SIGNALS], how="left")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    for s in INDEXES:
        d[f"is_{s}"] = (d["index"] == s).astype(float)
    d["tod"] = d["hhmm"].str[:2].astype(int) * 60 + d["hhmm"].str[3:].astype(int)
    # cross-index richness: each index vs the others at the same (day, time)
    piv = d.pivot_table(index=["day", "hhmm"], columns="index", values=["iv_over_rv", "implied_move"], aggfunc="first")
    for col in ("iv_over_rv", "implied_move"):
        if col not in piv:
            continue
        m = piv[col]
        others_mean = {s: m.drop(columns=[s], errors="ignore").mean(axis=1) for s in m.columns}
        key = pd.MultiIndex.from_frame(d[["day", "hhmm"]])
        rel = np.full(len(d), np.nan)
        for s in m.columns:
            idx = d["index"].to_numpy() == s
            rel[idx] = (m[s] / others_mean[s]).reindex(key[idx]).to_numpy()
        d[f"x_{col}_vs_others"] = rel
    d["post"] = (d["min_since_event"] <= 60).fillna(False)
    return d


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

SEARCH_GROUPS = ("session", "calendar", "context", "gex", "term", "flow", "strategy", "cross", "factory")
SEARCH_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def make(name: str):
    from odte.eval.expiry_edge import _make_model
    if name.startswith("ridge_a"):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(name[7:])))
    if name in ("ml_ridge", "ml_lgbm"):
        return _make_model(name)
    if name == "enet":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5))
    if name == "mlp":
        return TorchMLP()
    raise ValueError(name)


class TorchMLP:
    """Small multi-layer perceptron (PyTorch, Apple GPU when available). Inputs median-imputed and
    standardized on the training window only."""

    def __init__(self, hidden=64, epochs=60, lr=1e-3, wd=1e-3, seed=0):
        self.hidden, self.epochs, self.lr, self.wd, self.seed = hidden, epochs, lr, wd, seed

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = Xi.mean(0), Xi.std(0) + 1e-9
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dev = dev
        Xt = torch.tensor((Xi - self.mu) / self.sd, dtype=torch.float32, device=dev)
        yt = torch.tensor(np.clip(y, -1.5, 1.5), dtype=torch.float32, device=dev).view(-1, 1)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden), torch.nn.ReLU(),
                                       torch.nn.Dropout(0.1), torch.nn.Linear(self.hidden, 1)).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(Xt)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        Xi = np.where(np.isfinite(X), X, self.med)
        with torch.no_grad():
            Xt = torch.tensor((Xi - self.mu) / self.sd, dtype=torch.float32, device=self.dev)
            return self.net.eval()(Xt).cpu().numpy().ravel()


def walk_forward(d: pd.DataFrame, X: np.ndarray, Y: dict, model: str) -> dict:
    """Per trade, expanding-window predictions (refit every REFIT days, first after MIN_DAYS days)."""
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    train_ok = ~d["post"].to_numpy()
    preds = {k: np.full(len(d), np.nan) for k in Y}
    for k0 in range(MIN_DAYS, len(days), REFIT):
        tr_days = dd < days[k0]
        te = (dd >= days[k0]) & ((dd < days[k0 + REFIT]) if k0 + REFIT < len(days) else True)
        for name, y in Y.items():
            tr = tr_days & train_ok & np.isfinite(y)
            if tr.sum() < 1000:
                continue
            m = make(model).fit(X[tr], y[tr])
            preds[name][te] = m.predict(X[te])
    return preds


def decide(d: pd.DataFrame, preds: dict, tag: str = "half") -> pd.DataFrame:
    """Best predicted trade if > 0 (never inside the post-release hour); realized return of that trade."""
    P = np.column_stack([preds[k] for k in TRADES])
    Yh = np.column_stack([d[f"y_{k}_{tag}"].to_numpy(float) for k in TRADES])
    has = np.isfinite(P).any(axis=1)
    Pf = np.where(np.isfinite(P), P, -np.inf)
    best = Pf.argmax(axis=1)
    take = has & (Pf.max(axis=1) > 0) & ~d["post"].to_numpy()
    ret = np.where(take, Yh[np.arange(len(d)), best], 0.0)
    ret = np.where(np.isfinite(ret), ret, 0.0)
    return pd.DataFrame({"day": d["day"], "index": d["index"], "tested": has, "take": take,
                         "trade": np.where(take, np.array(TRADES)[best], "none"), "ret": ret})


def score(dec: pd.DataFrame) -> dict:
    """Daily P&L (sum over indexes and decisions), per era."""
    t = dec[dec["tested"]]
    daily = t.groupby("day")["ret"].sum()
    out = {}
    for label, x in (("2022_26", daily[daily.index >= EVAL_FROM]), ("2016_21", daily[daily.index < EVAL_FROM])):
        if len(x) < 20:
            continue
        b = fx.summarize(x)
        out[label] = {"days": int(len(x)), "mean_day": float(x.mean()), "total": float(x.sum()),
                      "sharpe": b.get("sharpe"), "sharpe_ci": b.get("sharpe_ci")}
    out["trade_mix"] = t.loc[t["take"], "trade"].value_counts(normalize=True).round(3).to_dict()
    out["take_rate"] = float(t["take"].mean())
    return out


def baseline_always(d: pd.DataFrame, trade: str = "fly", tag: str = "half") -> dict:
    x = d.assign(ret=np.where(d["post"], 0.0, d[f"y_{trade}_{tag}"].fillna(0.0)))
    daily = x.groupby("day")["ret"].sum()
    e = daily[daily.index >= EVAL_FROM]
    b = fx.summarize(e)
    return {"mean_day": float(e.mean()), "sharpe": b.get("sharpe"), "sharpe_ci": b.get("sharpe_ci")}


def run_zoo(models=("ml_ridge", "enet", "ml_lgbm", "mlp")) -> dict:
    d = load()
    groups = feature_groups(d.columns)
    feats = [c for g in groups.values() for c in g]
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Y = {k: d[f"y_{k}_half"].to_numpy(float) for k in TRADES}
    res = {"rows": int(len(d)), "features": len(feats), "groups": {k: len(v) for k, v in groups.items()},
           "always_fly": baseline_always(d, "fly"), "always_put_spread": baseline_always(d, "put_spread")}
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        preds = walk_forward(d, X, Y, m)
        res[m] = {"half": score(decide(d, preds, "half")), "touch": score(decide(d, preds, "touch"))}
        pd.DataFrame({k: v for k, v in preds.items()}).assign(day=d["day"], index=d["index"], hhmm=d["hhmm"]) \
            .to_parquet(OUT / f"preds_{m}.parquet", index=False)
        print(m, json.dumps(res[m]["half"], default=float), flush=True)
    (OUT / "zoo.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def ablate(model: str, drop: str | None) -> dict:
    """Walk-forward with one feature group removed (drop=None: all groups). Phase 3."""
    d = load()
    groups = feature_groups(d.columns)
    feats = [c for g, cols in groups.items() if g != drop for c in cols]
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Y = {k: d[f"y_{k}_half"].to_numpy(float) for k in TRADES}
    preds = walk_forward(d, X, Y, model)
    res = {"model": model, "dropped": drop or "none", "features": len(feats),
           "half": score(decide(d, preds, "half")), "touch": score(decide(d, preds, "touch"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ablate_{model}_{drop or 'none'}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def search(task: int) -> dict:
    """One point of the group-subset x ridge-strength grid (identity always kept). task in
    [0, 2**len(SEARCH_GROUPS) * len(SEARCH_ALPHAS)). Development data only; scored on both eras."""
    n = 2 ** len(SEARCH_GROUPS)
    mask, alpha = task % n, SEARCH_ALPHAS[task // n]
    d = load()
    groups = feature_groups(d.columns)
    keep = [g for i, g in enumerate(SEARCH_GROUPS) if mask >> i & 1] + ["identity"]
    feats = [c for g in keep for c in groups.get(g, [])]
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Y = {k: d[f"y_{k}_half"].to_numpy(float) for k in TRADES}
    preds = walk_forward(d, X, Y, f"ridge_a{alpha:g}")
    res = {"task": task, "mask": mask, "groups": keep, "alpha": alpha, "features": len(feats),
           "half": score(decide(d, preds, "half")), "touch": score(decide(d, preds, "touch"))}
    out = OUT / "search"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"s{task:05d}.json").write_text(json.dumps(res, default=float))
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["search"]:
        tasks = [int(x) for x in sys.argv[2:]]
        for t in tasks:
            if not (OUT / "search" / f"s{t:05d}.json").exists():
                search(t)
    elif sys.argv[1:2] == ["matrix"]:
        m = build_matrix()
        m.to_parquet(MATRIX, index=False)
        print(f"{len(m):,} rows x {m.shape[1]} columns -> {MATRIX}")
    elif sys.argv[1:2] == ["ablate"]:
        r = ablate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "none" else None)
        print(json.dumps(r, default=float))
    elif sys.argv[1:2] == ["zoo"]:
        r = run_zoo()
        print(json.dumps({k: v for k, v in r.items() if k not in ("ml_ridge", "enet", "ml_lgbm", "mlp")}, indent=2,
                         default=float))

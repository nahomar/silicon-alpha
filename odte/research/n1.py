"""N1 evaluation: leader vs leader + path group (pre-registered keep rule in docs/master_plan.md)."""
import json

import numpy as np
import pandas as pd

from odte.research import meta

LEADER = ["session", "calendar", "context", "flow", "strategy", "factory"]


def evaluate(groups, alpha=1.0):
    d = meta.load()
    g = meta.feature_groups(d.columns)
    feats = [c for k in groups + ["identity"] for c in g.get(k, [])]
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Y = {k: d[f"y_{k}_half"].to_numpy(float) for k in meta.TRADES}
    preds = meta.walk_forward(d, X, Y, f"ridge_a{alpha:g}")
    return {"features": len(feats), "half": meta.score(meta.decide(d, preds, "half")),
            "touch": meta.score(meta.decide(d, preds, "touch"))}, preds, d


if __name__ == "__main__":
    out = {}
    for name, groups in (("leader", LEADER), ("leader_path", LEADER + ["path"]), ("path_only", ["path"])):
        r, preds, d = evaluate(groups)
        out[name] = r
        if name == "leader_path":
            pd.DataFrame(preds).assign(day=d["day"], index=d["index"], hhmm=d["hhmm"]).to_parquet(
                meta.OUT / "preds_leader_path.parquet", index=False)
        h = r["half"]
        print(f"{name:12s} ({r['features']} inputs) 2022-26 ${h['2022_26']['mean_day']:+.1f}/day Sharpe {h['2022_26']['sharpe']:+.2f}"
              f" | 2018-21 ${h['2016_21']['mean_day']:+.1f}/day Sharpe {h['2016_21']['sharpe']:+.2f}"
              f" | touch 2022-26 ${r['touch']['2022_26']['mean_day']:+.1f}", flush=True)
    (meta.OUT / "n1.json").write_text(json.dumps(out, indent=2, default=float))

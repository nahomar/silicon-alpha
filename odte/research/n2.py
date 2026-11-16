"""N2 evaluation: GPU sequence model (3 seeds averaged) vs the leader, and a 50/50 blend."""
import glob
import json

import numpy as np
import pandas as pd

from odte.research import meta

KEYS = ["day", "index", "hhmm"]


def aligned(d, frame):
    m = d[KEYS].merge(frame, on=KEYS, how="left", validate="one_to_one")
    return {k: m[k].to_numpy(float) for k in meta.TRADES}


def line(name, r):
    h = r["half"]
    return (f"{name:14s} 2022-26 ${h['2022_26']['mean_day']:+.1f}/day Sharpe {h['2022_26']['sharpe']:+.2f}"
            f" | 2018-21 ${h['2016_21']['mean_day']:+.1f}/day Sharpe {h['2016_21']['sharpe']:+.2f}"
            f" | worse era {min(h['2022_26']['sharpe'], h['2016_21']['sharpe']):+.2f}"
            f" | touch 2022-26 ${r['touch']['2022_26']['mean_day']:+.1f} | take {h['take_rate']:.0%}")


if __name__ == "__main__":
    d = meta.load()
    seeds = [pd.read_parquet(f) for f in sorted(glob.glob(str(meta.OUT / "seq" / "preds_seed*.parquet")))]
    per_seed = [aligned(d, s) for s in seeds]
    seq = {k: np.mean([p[k] for p in per_seed], axis=0) for k in meta.TRADES}
    lead = aligned(d, pd.read_parquet(meta.OUT / "preds_leader.parquet"))
    blend = {k: 0.5 * seq[k] + 0.5 * lead[k] for k in meta.TRADES}
    out = {}
    for name, p in [("leader", lead), ("seq_avg", seq), ("blend_50_50", blend)] + [(f"seq_seed{i}", s) for i, s in enumerate(per_seed)]:
        out[name] = {"half": meta.score(meta.decide(d, p, "half")), "touch": meta.score(meta.decide(d, p, "touch"))}
        print(line(name, out[name]), flush=True)
    both = np.isfinite(seq["fly"]) & np.isfinite(lead["fly"])
    out["pred_corr"] = {k: float(np.corrcoef(seq[k][both], lead[k][both])[0, 1]) for k in meta.TRADES}
    print("prediction correlation seq vs leader:", {k: round(v, 2) for k, v in out["pred_corr"].items()})
    (meta.OUT / "n2.json").write_text(json.dumps(out, indent=2, default=float))

"""Plan B for the option-buying idea: a sequence transformer on minute paths.

Step 1 (odte/eval/direction.py) failed: logistic regression and LightGBM on
87 summary inputs could not call the S&P's 30-60 minute direction better than
"always up". This tests whether a model that reads the session minute by
minute does better, on the same rows, targets, development days and gate.

Input per decision (t = 10:00 .. 15:00, every 15 min):
  * the last 120 minutes up to t, one token per minute: 1-minute move of
    the parity forward, its size, move since the open, implied vol with the
    decay to the close taken out, straddle spread, time of day (left-padded
    early in the day);
  * the same 87 static inputs as step 1 (scaled with training-data stats).
Output: P(up) for H = 30 and H = 60 (two heads, binary cross-entropy).
Walk-forward: train on all earlier days (first 250 training only), refit
every 63 days, early stopping on the last 15% of training days. 2 seeds.
~40k parameters.

Gate: identical to step 1 (vs the best simple rule, accuracy AND captured
move, 99% day-resampled CIs above 0). Decided before the run. If it fails,
TradeFM (524M, tick data) is not retrained for this purpose: a bigger
version of a model with no signal at 30 s (tick gate) and none here has
nothing to scale.

Usage:
  python3 -m odte.eval.direction_seq \\
      reports/expiry_edge/decisions-lockbox-odte_0dte-2021H1.parquet \\
      reports/expiry_edge/decisions-dev-gex-es.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from odte.eval import direction as dr

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "data" / "odte_0dte"
L, N_TOK = 120, 6
REFIT_DAYS, N_SEEDS = 63, 2


def _path(day: str):
    from odte.eval.seq_model import minute_paths
    return day, minute_paths(pd.read_parquet(CHAINS / f"{day}.parquet"))


def load_paths(days) -> dict:
    with ProcessPoolExecutor(6) as ex:
        return {pd.Timestamp(d): p for d, p in
                ex.map(_path, [f"{pd.Timestamp(x):%Y-%m-%d}" for x in days], chunksize=16)}


def tokens(path: pd.DataFrame, t: pd.Timestamp) -> tuple:
    """(L x N_TOK, mask) for minutes <= t, the last L of them, left-padded."""
    p = path.loc[:t]
    day = t.normalize()
    open_t, close_t = day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=16)
    F = p["F"].ffill().to_numpy(float)
    ok = np.isfinite(F)
    X = np.zeros((L, N_TOK), np.float32)
    m = np.zeros(L, bool)
    if not ok.any():
        return X, m
    f0 = F[ok][0]
    lf = np.log(np.where(ok, F, f0))
    r = np.diff(lf, prepend=lf[0]) * 1e4
    mins = ((p.index - open_t).total_seconds() / 60).to_numpy()
    left = ((close_t - p.index).total_seconds() / 60).to_numpy()
    ivol = p["straddle"].to_numpy(float) / np.where(ok, F, f0) / np.sqrt(np.maximum(left, 1) / 390) * 100
    spr = p["spread"].to_numpy(float) / p["straddle"].to_numpy(float)
    tok = np.stack([r / 10, np.abs(r) / 10, (lf - np.log(f0)) * 100, np.nan_to_num(ivol),
                    np.nan_to_num(np.clip(spr, 0, 1)) * 10, mins / 390], axis=1).astype(np.float32)
    tok = np.nan_to_num(tok[-L:])
    X[-len(tok):], m[-len(tok):] = tok, True
    return X, m


def _net(n_static: int, d: int = 32):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(N_TOK, d)
            self.pos = nn.Embedding(L, d)
            layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=64, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.stat = nn.Sequential(nn.Linear(n_static, 32), nn.GELU())
            self.head = nn.Sequential(nn.Linear(2 * d + 32, 32), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(32, 2))

        def forward(self, x, m, s):
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(x.shape[0], L)
            h = self.enc(self.inp(x) + self.pos(pos), src_key_padding_mask=~m)
            mf = m.unsqueeze(-1).float()
            mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
            return self.head(torch.cat([mean, h[:, -1, :], self.stat(s)], dim=1))
    return Net()


def _fit_predict(X, M, S, Y, W, tr, te, seed, epochs=20, patience=3, bs=256):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    idx = np.flatnonzero(tr)
    n_val = max(int(len(idx) * 0.15), 256)
    fit, val = idx[:-n_val], idx[-n_val:]                      # chronological split
    net = _net(S.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def loss_on(rows):
        out = net(T(X[rows]), T(M[rows], torch.bool), T(S[rows]))
        return (bce(out, T(Y[rows])) * T(W[rows])).sum() / T(W[rows]).sum().clamp(min=1.0)

    def predict(rows):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(rows), 1024):
                b = rows[i:i + 1024]
                outs.append(torch.sigmoid(net(T(X[b]), T(M[b], torch.bool), T(S[b]))).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 2), np.float32)

    best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(fit), max(1, len(fit) // bs)):
            opt.zero_grad()
            loss_on(b).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(float(loss_on(val[i:i + 1024])) * len(val[i:i + 1024])
                     for i in range(0, len(val), 1024)) / len(val)
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    out = predict(np.flatnonzero(te))
    if dev == "mps":
        torch.mps.empty_cache()
    return out


def walk_forward(d: pd.DataFrame, X, M, cols) -> dict:
    S_raw = d[cols].to_numpy(float)
    ys = [d[f"fwd{h}"].to_numpy() for h in dr.HORIZONS]
    Y = np.stack([(y > 0).astype(np.float32) for y in ys], 1)
    W = np.stack([(np.isfinite(y) & (y != 0)).astype(np.float32) for y in ys], 1)
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    pred = np.full((len(d), 2), np.nan)
    for k in range(dr.MIN_TRAIN_DAYS, len(days), REFIT_DAYS):
        tr = (dd < days[k]) & (W.sum(1) > 0)
        te = dd >= days[k]
        if k + REFIT_DAYS < len(days):
            te &= dd < days[k + REFIT_DAYS]
        mu, sd = np.nanmean(S_raw[tr], 0), np.nanstd(S_raw[tr], 0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        S = np.clip(np.nan_to_num((S_raw - mu) / sd), -10, 10).astype(np.float32)
        pred[te] = np.mean([_fit_predict(X, M, S, Y, W, tr, te, seed=s) for s in range(N_SEEDS)], axis=0)
        print(f"  refit {(k - dr.MIN_TRAIN_DAYS) // REFIT_DAYS + 1}: {int(tr.sum()):,} train rows, "
              f"predicted {pd.Timestamp(days[k]):%Y-%m-%d} +{REFIT_DAYS}d", flush=True)
    return {("transformer", h): pred[:, i] for i, h in enumerate(dr.HORIZONS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "reports" / "direction_seq.json"))
    a = ap.parse_args()
    d = dr.add_targets(dr.load(a.decisions))
    cols = dr.feature_cols(d)
    paths = load_paths(sorted(d["day"].unique()))
    X = np.zeros((len(d), L, N_TOK), np.float32)
    M = np.zeros((len(d), L), bool)
    for i, (day, t) in enumerate(zip(d["day"], d["t"])):
        X[i], M[i] = tokens(paths[pd.Timestamp(day)], pd.Timestamp(t))
    print(f"{d['day'].nunique()} days, {len(d):,} sequences of {L} minutes, {len(cols)} static inputs",
          flush=True)
    preds = walk_forward(d, X, M, cols)
    res = dr.evaluate(d, preds)
    Path(a.out).write_text(json.dumps(res, indent=2))
    for h in dr.HORIZONS:
        r, v = res[f"H{h}"], res[f"H{h}"]["transformer"]
        print(f"\nH={h}: always-up {r['rules']['always_up']['accuracy']:.2%}, transformer "
              f"{r['rules']['transformer']['accuracy']:.2%} (capture {r['rules']['transformer']['capture_bp']:+.2f} bp); "
              f"vs best rule {v['accuracy_minus_best_rule']:+.2%} [{v['ci'][0]:+.2%}, {v['ci'][1]:+.2%}], "
              f"capture {v['capture_minus_best_rule_bp']:+.2f} bp [{v['capture_ci'][0]:+.2f}, {v['capture_ci'][1]:+.2f}]; "
              f"top 20%: {v['top20pct_accuracy']:.1%}, {v['top20pct_capture_bp']:+.2f} bp vs break-even "
              f"{v['top20pct_hurdle_bp']:.1f} bp -> {'PASS' if v['passes'] else 'fail'}")
    print(f"\nVERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

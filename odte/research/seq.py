"""N2: GPU sequence model for the meta decision (development data only).

Pre-registered in docs/master_plan.md (2026-11-11). Earlier sequence models in this repo
(odte/eval/seq_model.py, direction_seq.py) tied ridge or failed, so the prior is low.

Each decision row (index, day, time t) gets:
  * a window of the last W = 180 minutes of the day up to and including t (left-padded),
    5 channels, all normalised with information available at t only:
      0  (log F_s - log F_t) / implied_move_t       where price was, in today's implied moves
      1  1-minute log change / (implied_move_t / sqrt(minutes left at t))   (clipped +-10)
      2  log of the time-decay-adjusted straddle at s relative to its value at t
      3  ATM straddle relative half-spread (clipped 0..1)
      4  mask (1 = a real minute <= t)
  * the leader's tabular inputs (session, calendar, context, flow, strategy, factory, identity),
    standardised with training statistics only.

Network: 3 dilated 1-D convolutions -> masked mean + last step, joined with a small MLP of the
tabular inputs -> 5 outputs (one per trade). Huber loss on per-trade standardised returns,
missing labels masked. Walk-forward exactly as meta.walk_forward (first fit after MIN_DAYS days,
refit every REFIT days, post-release rows never trained on); early stopping on the last 15%
of each training window's days.

  python3 -m odte.research.seq run SEED        # writes reports/meta/seq/preds_seedSEED.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import meta

ROOT = Path(__file__).resolve().parents[2]
W = 180
N_CH = 5
LEADER = ["session", "calendar", "context", "flow", "strategy", "factory", "identity"]
OUT = meta.OUT / "seq"


def decision_minute(hhmm: str) -> int:
    h, m = int(hhmm[:2]), int(hhmm[3:])
    return h * 60 + m - (9 * 60 + 31)


def window(path: np.ndarray, m: int, implied_move: float) -> np.ndarray:
    """(W, N_CH) window ending at minute index m; uses path[: m + 1] only."""
    out = np.zeros((W, N_CH), np.float32)
    lo = max(0, m - W + 1)
    seg = path[lo: m + 1].astype(float)
    n = len(seg)
    if n == 0 or not np.isfinite(implied_move) or implied_move <= 0:
        return out
    lf = pd.Series(seg[:, 0]).ffill().to_numpy()
    lf_now = lf[-1]
    left = np.arange(390 - lo, 390 - m - 1, -1, dtype=float)[:n]           # minutes left at each s
    st = pd.Series(seg[:, 1]).ffill().to_numpy() / np.sqrt(np.maximum(left, 1) / 390)
    valid = np.isfinite(seg[:, 0])
    blk = np.zeros((n, N_CH))
    if np.isfinite(lf_now):
        blk[:, 0] = (lf - lf_now) / implied_move
    r = np.diff(lf, prepend=lf[0])
    blk[:, 1] = np.clip(r / (implied_move / np.sqrt(max(390 - m, 1))), -10, 10)
    if np.isfinite(st[-1]) and st[-1] > 0:
        blk[:, 2] = np.log(np.clip(st / st[-1], 1e-3, 1e3))
    blk[:, 3] = np.clip(seg[:, 2], 0, 1)
    blk[:, 4] = valid
    blk[~valid, :4] = 0.0
    out[W - n:] = np.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_inputs(d: pd.DataFrame):
    paths = {}
    for sym in meta.INDEXES:
        z = np.load(meta.LIB / f"paths_{sym}.npz")
        paths[sym] = (z["paths"], {s: i for i, s in enumerate(z["days"])})
    Xs = np.zeros((len(d), W, N_CH), np.float32)
    days = d["day"].dt.strftime("%Y-%m-%d").to_numpy()
    for i, (sym, day, hhmm, im) in enumerate(zip(d["index"], days, d["hhmm"], d["implied_move"])):
        arr, pos = paths[sym]
        j = pos.get(day)
        if j is not None:
            Xs[i] = window(arr[j], decision_minute(hhmm), float(im))
    g = meta.feature_groups(d.columns)
    feats = [c for k in LEADER for c in g.get(k, [])]
    Xt = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    Y = np.column_stack([d[f"y_{k}_half"].to_numpy(np.float32) for k in meta.TRADES])
    return Xs, Xt, Y


def _net(n_tab: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(N_CH, 32, 5, padding=2), nn.GELU(),
                nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GELU(),
                nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.GELU())
            self.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(64 + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, len(meta.TRADES)))

        def forward(self, xs, xt):
            mask = xs[:, :, 4:5]                                  # (B, W, 1)
            h = self.conv(xs.transpose(1, 2)).transpose(1, 2)     # (B, W, 32)
            mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.head(torch.cat([mean, h[:, -1], self.tab(xt)], dim=1))

    return Net()


def fit_predict(Xs, Xt, Y, tr, te, seed: int, device: str):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    med = np.nanmedian(Xt[tr], axis=0)
    sd = np.nanstd(Xt[tr], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    T = np.clip(np.nan_to_num((Xt - med) / sd, nan=0.0), -5, 5).astype(np.float32)
    ys = np.nanstd(Y[tr], axis=0)
    ys = np.where(np.isfinite(ys) & (ys > 0), ys, 1.0)
    Yn = (Y / ys).astype(np.float32)
    idx = np.flatnonzero(tr)
    cut = idx[int(len(idx) * 0.85)]                               # rows are in day order
    fit_i, val_i = idx[idx < cut], idx[idx >= cut]
    to = lambda a: torch.from_numpy(a).to(device)
    Xs_t, T_t, Y_t = to(Xs), to(T), to(Yn)
    net = _net(T.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_f = torch.nn.HuberLoss(reduction="none")

    def batch_loss(ii):
        y = Y_t[ii]
        ok = torch.isfinite(y)
        out = net(Xs_t[ii], T_t[ii])
        l = loss_f(out, torch.where(ok, y, torch.zeros_like(y)))
        return (l * ok).sum() / ok.sum().clamp(min=1)

    best, best_state, bad = np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(40):
        net.train()
        perm = torch.from_numpy(fit_i)[torch.randperm(len(fit_i), generator=g)].to(device)
        for b in range(0, len(perm), 512):
            opt.zero_grad()
            batch_loss(perm[b: b + 512]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean([batch_loss(to(val_i[b: b + 4096])).item() for b in range(0, len(val_i), 4096)]))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    te_i = np.flatnonzero(te)
    with torch.no_grad():
        P = np.concatenate([net(Xs_t[to(te_i[b: b + 4096])], T_t[to(te_i[b: b + 4096])]).cpu().numpy()
                            for b in range(0, len(te_i), 4096)])
    return te_i, P * ys


def run(seed: int) -> Path:
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    d = meta.load().sort_values(["day", "index", "hhmm"]).reset_index(drop=True)
    Xs, Xt, Y = build_inputs(d)
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    train_ok = ~d["post"].to_numpy()
    P = np.full(Y.shape, np.nan, np.float32)
    for k0 in range(meta.MIN_DAYS, len(days), meta.REFIT):
        tr = (dd < days[k0]) & train_ok
        te = (dd >= days[k0]) & ((dd < days[k0 + meta.REFIT]) if k0 + meta.REFIT < len(days) else True)
        if tr.sum() < 1000 or not te.any():
            continue
        te_i, p = fit_predict(Xs, Xt, Y, tr, te, seed, device)
        P[te_i] = p
        print(f"seed {seed} refit {days[k0]:%Y-%m-%d}: train {tr.sum():,} rows, test {te.sum():,}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"preds_seed{seed}.parquet"
    pd.DataFrame(P, columns=list(meta.TRADES)).assign(day=d["day"], index=d["index"], hhmm=d["hhmm"]).to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        print(run(int(sys.argv[2])))

"""A small sequence transformer for the hold-to-expiry 0DTE study.

The ridge and LightGBM models see a summary of the session up to t (realized
vol so far, move so far, ...). This model sees the session itself: one token
per minute from 09:31 to t, carrying

  r        1-minute log change of the parity forward (bp / 10)
  |r|      its size
  cum      move since the open (%)
  ivol     ATM straddle / F, divided by sqrt(minutes left / 390): the
           market's implied vol path, with the deterministic decay to the
           close taken out
  spread   ATM straddle spread / mid
  tod      minute of day / 390

plus the same static feature vector the tabular models get. It predicts the
same target (realized / implied, clipped at 4) and is scored by the same
rule, so the comparison with ridge and LightGBM is like for like.

Size is deliberate: ~30k parameters for ~1,500 labelled decisions. A big
model would memorize them, and the walk-forward would show exactly that.

Walk-forward: refit every `refit_every` sessions on all earlier sessions,
with early stopping on the chronologically last 15% of the training days.
It refits less often than ridge / LightGBM (which refit daily); that is a
small handicap, noted in the report. Averages `n_seeds` seeds.

Leakage: a decision at t uses only minutes <= t (quotes visible at
ceil(stamp)); static features are the same causal ones as the other models.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .expiry_edge import _mids, implied_forward, minute_grid

L_MAX = 331          # 09:30 .. 15:00 inclusive, the latest decision time
N_TOK = 6


def minute_paths(day_chain: pd.DataFrame) -> pd.DataFrame:
    """Per-minute forward, ATM straddle and its spread for one session."""
    g = minute_grid(day_chain)
    F = implied_forward(g)
    mid = _mids(g)
    if not {"C", "P"} <= set(mid.columns.get_level_values(0)):
        return pd.DataFrame(index=g["bid"].index)
    ks = mid["C"].columns.intersection(mid["P"].columns)
    cm, pm = mid["C"][ks].to_numpy(), mid["P"][ks].to_numpy()
    cb, ca = g["bid"]["C"][ks].to_numpy(), g["ask"]["C"][ks].to_numpy()
    pb, pa = g["bid"]["P"][ks].to_numpy(), g["ask"]["P"][ks].to_numpy()
    k = ks.to_numpy(dtype=float)
    ok = np.isfinite(cm) & np.isfinite(pm)
    dist = np.where(ok, np.abs(k[None, :] - F.to_numpy()[:, None]), np.inf)
    j = np.argmin(dist, axis=1)
    r = np.arange(len(j))
    good = np.isfinite(dist[r, j])
    straddle = np.where(good, cm[r, j] + pm[r, j], np.nan)
    spread = np.where(good, (ca[r, j] + pa[r, j]) - (cb[r, j] + pb[r, j]), np.nan)
    return pd.DataFrame({"F": F.to_numpy(), "straddle": straddle, "spread": spread},
                        index=g["bid"].index)


def _tokens(path: pd.DataFrame, t: pd.Timestamp) -> tuple:
    """(L_MAX x N_TOK tokens, mask, positions) for minutes <= t, left-padded."""
    p = path.loc[:t]
    day = t.normalize()
    open_t = day + pd.Timedelta(hours=9, minutes=30)
    close_t = day + pd.Timedelta(hours=16)
    F = p["F"].ffill().to_numpy()
    r = np.diff(np.log(F), prepend=np.log(F[0])) * 1e4
    r = np.nan_to_num(r)
    f0 = F[np.isfinite(F)][0] if np.isfinite(F).any() else np.nan
    cum = np.nan_to_num(np.log(F / f0) * 100)
    mins_left = np.array([(close_t - x).total_seconds() / 60 for x in p.index])
    ivol = p["straddle"].to_numpy() / F / np.sqrt(np.maximum(mins_left, 1) / 390) * 100
    spread = p["spread"].to_numpy() / p["straddle"].to_numpy()
    tod = np.array([(x - open_t).total_seconds() / 60 for x in p.index]) / 390
    tok = np.stack([r / 10, np.abs(r) / 10, cum, np.nan_to_num(ivol),
                    np.nan_to_num(np.clip(spread, 0, 1)) * 10, tod], axis=1).astype(np.float32)
    tok = tok[-L_MAX:]
    n = len(tok)
    X = np.zeros((L_MAX, N_TOK), np.float32)
    X[-n:] = tok
    mask = np.zeros(L_MAX, bool)
    mask[-n:] = True
    pos = np.zeros(L_MAX, np.int64)
    pos[-n:] = np.clip(np.round(tod[-n:] * 390).astype(int), 0, L_MAX - 1)
    return X, mask, pos


def build_sequences(ds: pd.DataFrame, paths: dict) -> tuple:
    X = np.zeros((len(ds), L_MAX, N_TOK), np.float32)
    M = np.zeros((len(ds), L_MAX), bool)
    P = np.zeros((len(ds), L_MAX), np.int64)
    for i, (d, t) in enumerate(zip(ds["day"], ds["t"])):
        X[i], M[i], P[i] = _tokens(paths[pd.Timestamp(d)], pd.Timestamp(t))
    return X, M, P


def _net(n_static: int, d: int = 32):
    import torch
    from torch import nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(N_TOK, d)
            self.pos = nn.Embedding(L_MAX, d)
            layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=64, dropout=0.1,
                                               batch_first=True)
            # enable_nested_tensor=False: the nested-tensor fast path for
            # padded batches is not implemented on Apple's MPS backend.
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.stat = nn.Sequential(nn.Linear(n_static, 32), nn.GELU())
            self.head = nn.Sequential(nn.Linear(2 * d + 32, 32), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(32, 1))

        def forward(self, x, m, p, s):
            h = self.enc(self.inp(x) + self.pos(p), src_key_padding_mask=~m)
            mf = m.unsqueeze(-1).float()
            mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
            last = h[:, -1, :]                       # left-padded: last = minute t
            return self.head(torch.cat([mean, last, self.stat(s)], dim=1)).squeeze(-1)
    return SeqNet()


def _fit_predict(Xtr, Mtr, Ptr, Str, ytr, Xte, Mte, Pte, Ste, seed: int,
                 epochs: int = 60, patience: int = 8, val_frac: float = 0.15):
    import torch
    dev = ("cuda" if torch.cuda.is_available()           # ASU Sol / any NVIDIA GPU
           else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(ytr)
    n_val = max(int(n * val_frac), 12)
    tr, va = np.arange(n - n_val), np.arange(n - n_val, n)   # chronological split
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    net = _net(Str.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for b in np.array_split(np.random.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            out = net(T(Xtr[b]), T(Mtr[b], torch.bool), T(Ptr[b], torch.long), T(Str[b]))
            loss = torch.mean((out - T(ytr[b])) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = net(T(Xtr[va]), T(Mtr[va], torch.bool), T(Ptr[va], torch.long), T(Str[va]))
            vl = float(torch.mean((v - T(ytr[va])) ** 2))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(T(Xte), T(Mte, torch.bool), T(Pte, torch.long), T(Ste)).cpu().numpy()


def walk_forward_seq(ds: pd.DataFrame, paths: dict, static_cols: list,
                     min_train_days: int = 40, refit_every: int = 21,
                     n_seeds: int = 2, verbose: bool = True) -> np.ndarray:
    """Predictions for every row on a tested day (NaN before min_train_days)."""
    X, M, P = build_sequences(ds, paths)
    S_raw = ds[static_cols].to_numpy(dtype=float)
    y = ds["ratio"].clip(upper=4.0).to_numpy()
    # Both as datetime64: np.isin silently matches nothing when one side is
    # numpy datetime64 and the other pandas Timestamps.
    days = np.array(sorted(ds["day"].unique()), dtype="datetime64[ns]")
    dvals = ds["day"].to_numpy(dtype="datetime64[ns]")
    pred = np.full(len(ds), np.nan)
    for start in range(min_train_days, len(days), refit_every):
        block = days[start:start + refit_every]
        tr = (dvals < block[0]) & np.isfinite(y)
        te = np.isin(dvals, block)
        mu = np.nanmean(S_raw[tr], axis=0)
        sd = np.nanstd(S_raw[tr], axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        S = np.nan_to_num((S_raw - mu) / sd).astype(np.float32)   # train-only scaling
        outs = [_fit_predict(X[tr], M[tr], P[tr], S[tr], y[tr].astype(np.float32),
                             X[te], M[te], P[te], S[te], seed=s) for s in range(n_seeds)]
        pred[te] = np.mean(outs, axis=0)
        if verbose:
            print(f"  seq transformer: refit {start // refit_every + 1}, trained on "
                  f"{int(tr.sum())} rows, predicted {pd.Timestamp(block[0]):%Y-%m-%d}..{pd.Timestamp(block[-1]):%Y-%m-%d}",
                  flush=True)
    return pred

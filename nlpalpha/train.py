"""Training routines: MLM pretraining, tweet embedding, day-model fitting."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_model import (
    DayClassifier, MLMHead, TweetEncoder, Vocab, mask_tokens, pad_batch,
    set_seed,
)

MAX_LEN = 40
MAX_TWEETS_PER_DAY = 32


# ---------------------------------------------------------------------------
# Stage 1: masked language modelling
# ---------------------------------------------------------------------------

def pretrain_mlm(token_lists, vocab: Vocab, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4, epochs: int = 3,
                 batch_size: int = 256, lr: float = 3e-4, seed: int = 0,
                 max_len: int = MAX_LEN, log_every: int = 200,
                 verbose: bool = True) -> tuple[TweetEncoder, dict]:
    """Train the encoder to fill in masked tokens.

    `token_lists` must contain training-window tweets only. The objective
    never sees a return, so nothing about the label can be memorized here.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    encoder = TweetEncoder(len(vocab), d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_len=max_len,
                           pad_id=vocab.pad_id)
    head = MLMHead(d_model, len(vocab))
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    encoded = [vocab.encode(t, max_len) for t in token_lists]
    n = len(encoded)
    history = []
    t0 = time.time()

    for epoch in range(epochs):
        encoder.train(); head.train()
        order = torch.randperm(n, generator=gen).numpy()
        tot_loss, tot_tok, tot_correct = 0.0, 0, 0

        for start in range(0, n, batch_size):
            batch = [encoded[i] for i in order[start:start + batch_size]]
            ids, pad_mask = pad_batch(batch, vocab.pad_id, max_len)
            corrupted, labels = mask_tokens(ids, vocab, generator=gen)

            hidden = encoder(corrupted, pad_mask)
            logits = head(hidden)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            scored = labels.ne(-100)
            k = int(scored.sum())
            if k:
                tot_loss += float(loss.detach()) * k
                tot_tok += k
                tot_correct += int((logits.argmax(-1)[scored] == labels[scored]).sum())

        mean_loss = tot_loss / max(tot_tok, 1)
        acc = tot_correct / max(tot_tok, 1)
        history.append({"epoch": epoch, "mlm_loss": mean_loss,
                        "mlm_token_acc": acc,
                        "perplexity": float(np.exp(min(mean_loss, 20)))})
        if verbose:
            print(f"  [mlm] epoch {epoch}  loss {mean_loss:.4f}  "
                  f"ppl {np.exp(min(mean_loss,20)):.1f}  "
                  f"tok-acc {acc:.4f}  ({time.time()-t0:.0f}s)")

    return encoder, {"history": history, "n_tweets": n,
                     "wall_sec": round(time.time() - t0, 1)}


@torch.no_grad()
def embed_tweets(encoder: TweetEncoder, vocab: Vocab, token_lists,
                 batch_size: int = 512, max_len: int = MAX_LEN) -> np.ndarray:
    """One frozen vector per tweet, in input order. (n_tweets, d_model)"""
    encoder.eval()
    encoded = [vocab.encode(t, max_len) for t in token_lists]
    out = np.zeros((len(encoded), encoder.d_model), dtype=np.float32)
    for start in range(0, len(encoded), batch_size):
        ids, pad_mask = pad_batch(encoded[start:start + batch_size],
                                  vocab.pad_id, max_len)
        out[start:start + batch_size] = encoder.embed(ids, pad_mask).numpy()
    return out


# ---------------------------------------------------------------------------
# Day-level tensors
# ---------------------------------------------------------------------------

def build_day_tensors(panel, tweet_emb: np.ndarray,
                      max_tweets: int = MAX_TWEETS_PER_DAY):
    """Stack each ticker-day's tweet vectors into (N, T, d) plus a mask.

    Days with more than `max_tweets` keep the most recent ones: they are
    closest to the decision point, and truncating the oldest is the choice
    that cannot import information from further back than the rest.
    """
    n = len(panel)
    d = tweet_emb.shape[1]
    x = np.zeros((n, max_tweets, d), dtype=np.float32)
    mask = np.zeros((n, max_tweets), dtype=bool)
    for i, idx in enumerate(panel["tweet_idx"].to_numpy()):
        sel = idx[-max_tweets:] if len(idx) > max_tweets else idx
        if not len(sel):
            continue
        x[i, :len(sel)] = tweet_emb[np.asarray(sel)]
        mask[i, :len(sel)] = True
    return x, mask


# ---------------------------------------------------------------------------
# Stage 2: day-level classifier
# ---------------------------------------------------------------------------

def train_day_model(train_data, val_data, d_model: int, n_features: int,
                    use_text: bool = True, epochs: int = 30,
                    batch_size: int = 256, lr: float = 1e-3, seed: int = 0,
                    patience: int = 6, weight_decay: float = 1e-4,
                    verbose: bool = False):
    """Fit the day model, selecting the epoch by validation AUC.

    Early stopping and every other choice look only at validation. The test
    split is untouched here by construction -- it is not even passed in.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    xt, mt, ft, yt = train_data
    xv, mv, fv, yv = val_data
    model = DayClassifier(d_model, n_features=n_features, use_text=use_text)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def to_t(x, m, f, y):
        return (torch.from_numpy(x) if x is not None else None,
                torch.from_numpy(m) if m is not None else None,
                torch.from_numpy(f) if f is not None else None,
                torch.from_numpy(y.astype(np.float32)))

    Xt, Mt, Ft, Yt = to_t(xt, mt, ft, yt)
    Xv, Mv, Fv, Yv = to_t(xv, mv, fv, yv)

    n = len(Yt)
    best = {"auc": -np.inf, "epoch": -1, "state": None}
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            sel = order[start:start + batch_size]
            logit, _ = model(
                tweets=Xt[sel] if Xt is not None else None,
                mask=Mt[sel] if Mt is not None else None,
                feats=Ft[sel] if Ft is not None else None)
            loss = F.binary_cross_entropy_with_logits(logit, Yt[sel])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vlogit, _ = model(tweets=Xv, mask=Mv, feats=Fv)
        auc = roc_auc(yv, vlogit.numpy())
        if verbose:
            print(f"    epoch {epoch:2d}  val_auc {auc:.4f}")
        if auc > best["auc"] + 1e-5:
            best = {"auc": auc, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_val_auc": float(best["auc"]),
                   "best_epoch": int(best["epoch"])}


@torch.no_grad()
def predict(model: DayClassifier, x, m, f) -> np.ndarray:
    """Sigmoid probabilities for a fitted day model."""
    model.eval()
    logit, _ = model(
        tweets=torch.from_numpy(x) if x is not None else None,
        mask=torch.from_numpy(m) if m is not None else None,
        feats=torch.from_numpy(f) if f is not None else None)
    return torch.sigmoid(logit).numpy()


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC via the rank identity; ties get average ranks."""
    y = np.asarray(y_true).astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    s = np.sort(score)
    i = 0
    while i < len(s):                      # average ranks within tie groups
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

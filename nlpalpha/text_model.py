"""A small transformer language model over financial tweets, in PyTorch.

Architecture is hierarchical, because the prediction unit is a *ticker-day*
holding an unordered bag of tweets, not a single document:

    tokens --> TweetEncoder (transformer, [CLS] pooled) --> tweet vector
    tweet vectors of one ticker-day --> attention pooling --> day vector
    day vector (+ optional tabular features) --> MLP --> direction logit

Training runs in two stages:

  1. Masked language modelling on the tweet corpus. The encoder learns the
     domain -- cashtags, "beats", "downgrade", "PT raised" -- with no access
     to returns, so it cannot memorize labels.
  2. The encoder is frozen and used to embed every tweet once; the day-level
     attention pooling and the head are then trained on the return task.

Freezing after stage 1 is a deliberate trade. It costs some accuracy versus
end-to-end fine-tuning, and buys the ability to run many seeds cheaply on CPU
-- which matters more here, because a single-seed deep learning result on
20k noisy samples is not evidence of anything. Attention pooling is where the
day-level model can still learn which tweets matter.

Scale honesty: this is roughly a 2-4M parameter model trained on ~10^5 short
texts. It is a language model in the architectural sense, not a large one. It
is trained from scratch rather than fine-tuned from an off-the-shelf financial
LM for a reason beyond availability: models like FinBERT were pretrained on
corpora overlapping 2014-2016, so fine-tuning one here would import
look-ahead contamination that no backtest could detect. The vocabulary and the
MLM objective see *training-window text only*.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, MASK, CLS = "<pad>", "<unk>", "<mask>", "<cls>"
SPECIALS = [PAD, UNK, MASK, CLS]


def set_seed(seed: int) -> None:
    """Seed every generator the study touches."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@dataclass
class Vocab:
    itos: list[str] = field(default_factory=list)
    stoi: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, token_lists, min_freq: int = 5, max_size: int = 20000) -> "Vocab":
        """Build from token lists -- pass *training-window tweets only*.

        A vocabulary fitted on the full corpus leaks: which words exist in the
        test period is itself future information, and rare-word cutoffs shift
        with it.
        """
        counts = Counter()
        for toks in token_lists:
            counts.update(t.lower() for t in toks)
        itos = list(SPECIALS)
        for tok, n in counts.most_common():
            if n < min_freq or len(itos) >= max_size:
                break
            if tok not in SPECIALS:
                itos.append(tok)
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)})

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def mask_id(self) -> int:
        return self.stoi[MASK]

    @property
    def cls_id(self) -> int:
        return self.stoi[CLS]

    def encode(self, tokens, max_len: int) -> list[int]:
        """[CLS] + tokens, truncated to max_len, unknowns mapped to <unk>."""
        unk = self.stoi[UNK]
        ids = [self.cls_id]
        for t in tokens[: max_len - 1]:
            ids.append(self.stoi.get(t.lower(), unk))
        return ids


def pad_batch(seqs: list[list[int]], pad_id: int, max_len: int):
    """Right-pad to a rectangle; return (ids, key_padding_mask)."""
    n = len(seqs)
    out = np.full((n, max_len), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        k = min(len(s), max_len)
        out[i, :k] = s[:k]
    ids = torch.from_numpy(out)
    return ids, ids.eq(pad_id)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TweetEncoder(nn.Module):
    """Transformer encoder producing one vector per tweet from its [CLS]."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, max_len: int = 40,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) ids -> (B, L, d_model) contextual states."""
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.tok(ids) * math.sqrt(self.d_model) + self.pos(positions)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.norm(h)

    def embed(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Pooled tweet vector: the [CLS] position. (B, d_model)"""
        return self.forward(ids, pad_mask)[:, 0, :]


class MLMHead(nn.Module):
    """Tied-free projection back to the vocabulary for masked prediction."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(F.gelu(self.fc(h)))


def mask_tokens(ids: torch.Tensor, vocab: Vocab, prob: float = 0.15,
                generator: torch.Generator | None = None):
    """Standard BERT corruption: 80% <mask>, 10% random, 10% unchanged.

    Specials are never masked, and unmasked positions are set to -100 so the
    loss ignores them.
    """
    labels = ids.clone()
    special = torch.zeros_like(ids, dtype=torch.bool)
    for sid in (vocab.pad_id, vocab.cls_id):
        special |= ids.eq(sid)

    prob_matrix = torch.full(ids.shape, prob)
    prob_matrix.masked_fill_(special, 0.0)
    chosen = torch.bernoulli(prob_matrix, generator=generator).bool()
    labels[~chosen] = -100

    replace = torch.bernoulli(torch.full(ids.shape, 0.8),
                              generator=generator).bool() & chosen
    ids = ids.clone()
    ids[replace] = vocab.mask_id

    randomize = (torch.bernoulli(torch.full(ids.shape, 0.5),
                                 generator=generator).bool()
                 & chosen & ~replace)
    random_ids = torch.randint(len(SPECIALS), len(vocab), ids.shape,
                               generator=generator)
    ids[randomize] = random_ids[randomize]
    return ids, labels


# ---------------------------------------------------------------------------
# Day-level model
# ---------------------------------------------------------------------------

class DayAttentionPool(nn.Module):
    """Additive attention over the tweets of one ticker-day.

    Mean pooling would let a hundred boilerplate retweets drown the one tweet
    that carries news. A learned query lets the model weight them.
    """

    def __init__(self, d_model: int, d_attn: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, d_attn)
        self.query = nn.Linear(d_attn, 1, bias=False)

    def forward(self, tweets: torch.Tensor, mask: torch.Tensor):
        """(B, T, d) tweet vectors + (B, T) valid mask -> (B, d), (B, T)."""
        scores = self.query(torch.tanh(self.proj(tweets))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        # A day with no valid tweets would softmax to NaN; guard it.
        empty = ~mask.any(dim=1, keepdim=True)
        scores = torch.where(empty.expand_as(scores),
                             torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(empty.expand_as(weights),
                              torch.zeros_like(weights), weights)
        return torch.einsum("bt,btd->bd", weights, tweets), weights


class DayClassifier(nn.Module):
    """Attention-pooled text, optionally concatenated with tabular features."""

    def __init__(self, d_model: int, n_features: int = 0, d_hidden: int = 64,
                 dropout: float = 0.2, use_text: bool = True):
        super().__init__()
        self.use_text = use_text
        self.pool = DayAttentionPool(d_model) if use_text else None
        d_in = (d_model if use_text else 0) + n_features
        if d_in == 0:
            raise ValueError("DayClassifier needs text, features, or both")
        self.head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, tweets=None, mask=None, feats=None):
        parts = []
        weights = None
        if self.use_text:
            pooled, weights = self.pool(tweets, mask)
            parts.append(pooled)
        if feats is not None:
            parts.append(feats)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1), weights


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

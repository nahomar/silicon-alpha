"""TradeFM — decoder-only transformer for market microstructure tokens.

Architecture (HRT-style, runs at any scale):
  - Token embedding + rotary position encoding
  - N × { RMSNorm → MultiHeadAttention (causal) → RMSNorm → SwiGLU FFN }
  - LM head tied to embedding matrix

Phase-0 preset `MiniTradeFM` is ~10M params and trains on MPS/CPU.
Phase-2 preset `TradeFMConfig.tradefm_524m` is the 524M target for 8×H100.

Gating:
  - FlashAttention-3 path is enabled when cfg.use_flash_attn AND flash_attn
    is importable; otherwise eager causal attention is used.
  - FP8 path via transformer-engine is enabled only when cfg.fp8 AND TE is
    importable AND CUDA is active. Mac runs in bf16/fp32.

The tensor layout is identical across sizes so the same persistent CUDA
inference kernel (Phase 3) can consume any checkpoint.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import TradeFMConfig
from . import HAS_FLASH_ATTN, HAS_TE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------

def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary(xq: torch.Tensor, xk: torch.Tensor,
                  freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms) * self.weight


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalAttention(nn.Module):
    """Eager multi-head causal attention (FlashAttention-3 shim swaps in).

    Shape contract (identical to flash_attn):  input (B, T, D),  out (B, T, D).
    """

    def __init__(self, cfg: TradeFMConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = cfg.d_model // cfg.n_heads
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _eager(self, xq, xk, xv) -> torch.Tensor:
        # xq/xk/xv: (B, T, H, Dh) → attention with causal mask.
        # Materializes an (B,H,T,T) score tensor — OOMs on long ctx. Kept as
        # the last-resort path; prefer _sdpa or _flash.
        B, T, H, Dh = xq.shape
        xq = xq.transpose(1, 2)                   # (B, H, T, Dh)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(Dh)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=scores.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ xv                           # (B, H, T, Dh)
        return out.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def _sdpa(self, xq, xk, xv) -> torch.Tensor:
        """PyTorch 2.x built-in scaled_dot_product_attention.

        Picks mem-efficient or flash-1 backend automatically on CUDA, cuts
        peak memory ~3-6× vs _eager at long ctx, no extra pip deps. This is
        the right default for A100 / H100 when flash-attn-3 isn't installed.
        """
        B, T, H, Dh = xq.shape
        q = xq.transpose(1, 2)                    # (B, H, T, Dh)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def _flash(self, xq, xk, xv) -> torch.Tensor:
        from flash_attn import flash_attn_func  # type: ignore
        # flash_attn expects (B, T, H, Dh) with causal=True
        out = flash_attn_func(xq, xk, xv, causal=True)
        B, T, H, Dh = out.shape
        return out.contiguous().view(B, T, H * Dh)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh); k = k.view(B, T, H, Dh); v = v.view(B, T, H, Dh)
        if self.cfg.rotary:
            q, k = _apply_rotary(q, k, freqs_cis)
        use_flash = self.cfg.use_flash_attn and HAS_FLASH_ATTN \
            and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)
        if use_flash:
            out = self._flash(q, k, v)
        elif x.is_cuda:
            # On any CUDA device, SDPA is always better than eager.
            out = self._sdpa(q, k, v)
        else:
            out = self._eager(q, k, v)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or (4 * dim * 2 // 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Block + stack
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TradeFMConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), freqs_cis))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class TradeFM(nn.Module):
    def __init__(self, cfg: TradeFMConfig | None = None):
        super().__init__()
        self.cfg = cfg or TradeFMConfig.mini()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab, c.d_model)
        # Cross-asset fusion scaffold: if modality_vocab > 0, each token carries
        # an extra channel-ID embedding that the attention layers can learn to
        # route lead/lag signals through (ES -> OPRA, ETF-arb -> OPRA, etc).
        # Single-modality runs (modality_vocab == 0) are unaffected.
        self.modality_emb = (nn.Embedding(c.modality_vocab, c.d_model)
                             if getattr(c, "modality_vocab", 0) > 0 else None)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        # LM head tied to the embedding (saves params).
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        head_dim = c.d_model // c.n_heads
        self.register_buffer("freqs_cis",
                             _precompute_freqs_cis(head_dim, c.ctx_len * 2),
                             persistent=False)

    def forward(self, tokens: torch.Tensor,
                modality_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = tokens.shape
        x = self.tok_emb(tokens)
        if self.modality_emb is not None and modality_ids is not None:
            # modality_ids: (B, T) int64, aligned with tokens. Added (not
            # concatenated) to keep d_model and downstream kernel shapes fixed.
            x = x + self.modality_emb(modality_ids)
        freqs = self.freqs_cis[:T]
        ckpt = getattr(self.cfg, "grad_checkpointing", False) and self.training
        for blk in self.blocks:
            if ckpt:
                # Recompute activations on backward → ~60% less act memory
                # at the cost of a second forward through each block.
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, freqs, use_reentrant=False)
            else:
                x = blk(x, freqs)
        x = self.norm(x)
        return self.head(x)

    def loss(self, tokens: torch.Tensor,
             modality_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard causal LM loss on shifted tokens.

        modality_ids (optional, same shape as tokens): per-token source ID for
        cross-asset fusion. Sliced the same way as tokens for the forward pass.
        """
        mids = modality_ids[:, :-1] if modality_ids is not None else None
        logits = self.forward(tokens[:, :-1], modality_ids=mids)
        target = tokens[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab),
                               target.reshape(-1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def MiniTradeFM() -> TradeFM:
    return TradeFM(TradeFMConfig.mini())


def wrap_fp8_autocast():
    """Context manager that is fp8_autocast on H100 + TE, else no-op."""
    if HAS_TE and torch.cuda.is_available():
        try:
            from transformer_engine.pytorch import fp8_autocast  # type: ignore
            return fp8_autocast(enabled=True)
        except Exception:
            pass
    from contextlib import nullcontext
    return nullcontext()

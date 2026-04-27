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
        # LM head — independent from tok_emb.
        # Originally tied via `self.head.weight = self.tok_emb.weight` to
        # save ~vocab*d_model params, but under FSDP the tie produces a
        # flat 1-D root FlatParameter that breaks `nn.Embedding`'s 2-D
        # weight expectation at forward. Caught by the 8-GPU NCCL smoke.
        # Cost of untying: +vocab*d_model params (e.g. 524M -> ~532M, 1.5%);
        # benefit: clean FSDP sharding, no weight-dim edge cases, and
        # slightly more expressive independent in/out embeddings.
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        # Auxiliary directional head — 3-layer MLP from last hidden state to a
        # binary "majority direction" logit per position. Disabled by default;
        # opt-in via cfg.dir_head_enabled. Shape contract: (B, T, d_model)
        # → (B, T) binary logit.
        if getattr(c, "dir_head_enabled", False):
            d_h = c.d_model
            self.dir_head = nn.Sequential(
                nn.Linear(d_h, d_h // 2, bias=False),
                nn.GELU(),
                nn.Linear(d_h // 2, d_h // 4, bias=False),
                nn.GELU(),
                nn.Linear(d_h // 4, 1, bias=True),
            )
        else:
            self.dir_head = None
        head_dim = c.d_model // c.n_heads
        self.register_buffer("freqs_cis",
                             _precompute_freqs_cis(head_dim, c.ctx_len * 2),
                             persistent=False)

    def forward(self, tokens: torch.Tensor,
                modality_ids: Optional[torch.Tensor] = None,
                return_aux: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        lm_logits = self.head(x)
        if return_aux and self.dir_head is not None:
            dir_logits = self.dir_head(x).squeeze(-1)  # (B, T)
            return lm_logits, dir_logits
        return lm_logits

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

    def _build_dir_targets(self, tokens: torch.Tensor,
                           feature_offset: torch.Tensor | int = 0
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (dir_target, mask) for tokens of shape (B, T).

        At each position p in the *input* (B, T-1) window, the target is:
            1 if the count of return tokens in [p+1, p+1+H*N) that are above
            the per-batch return-token median is > H/2, else 0.
        where N = cfg.dir_n_features (one full feature cycle per H step) and
        H = cfg.dir_horizon (rows-ahead).

        `mask` is True where the lookahead window fits inside `tokens` AND
        the position is itself a return-feature position (feature index 0).
        Targets at masked-False positions are arbitrary (zero) and excluded
        from the BCE reduction.

        Shapes: dir_target (B, T-1) float32, mask (B, T-1) bool.
        """
        c = self.cfg
        N = int(getattr(c, "dir_n_features", 7))
        H = int(getattr(c, "dir_horizon", 10))
        B, T = tokens.shape
        device = tokens.device

        # Per-batch threshold computed from return-token positions only.
        # offsets shape (B,). For sample b, pos p is a return position iff
        # (p + off_b) % N == 0.
        if isinstance(feature_offset, int):
            offsets = torch.full((B,), feature_offset, dtype=torch.long, device=device)
        else:
            offsets = feature_offset.to(device=device, dtype=torch.long)

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        feat_idx = (positions + offsets.unsqueeze(1)) % N
        ret_pos_mask = (feat_idx == 0)

        # Threshold: median over the union of all return-token values in the batch.
        ret_vals = tokens[ret_pos_mask].float()
        if ret_vals.numel() == 0:
            empty = torch.zeros((B, T - 1), device=device, dtype=torch.float32)
            mask = torch.zeros((B, T - 1), device=device, dtype=torch.bool)
            return empty, mask
        thresh = ret_vals.median()

        above = (tokens.float() > thresh).to(torch.float32)  # (B, T)

        # For each input position p (0..T-2), look at next H * N tokens
        # starting at p+1 and count how many of them are above-threshold
        # AND at a return-feature position. Then compare to H/2 (i.e.,
        # majority among the H return positions that fall inside the window).
        T_in = T - 1
        target = torch.zeros((B, T_in), device=device, dtype=torch.float32)
        mask = torch.zeros((B, T_in), device=device, dtype=torch.bool)

        # Slow but correct: iterate over offsets within the lookahead window.
        # Vectorised over (B, T_in). H*N is small (≤100 at default cfg) so
        # this is cheap relative to the transformer forward.
        max_lookahead = H * N
        if T < 2 + max_lookahead:
            return target, mask  # window too small to score any position

        # Boolean: at position q = p + 1 + k, is q a return position AND q < T?
        # For k in [0, H*N): pos_q = positions + 1 + k (broadcast over k).
        ks = torch.arange(max_lookahead, device=device)              # (H*N,)
        q_idx = (positions[:, :T_in].unsqueeze(-1) + 1 + ks)         # (B, T_in, H*N)
        q_in_range = q_idx < T                                       # (B, T_in, H*N)
        q_feat_idx = (q_idx + offsets.view(B, 1, 1)) % N
        q_is_ret = (q_feat_idx == 0) & q_in_range                    # (B, T_in, H*N)

        # `above_q` at q_idx — gather from (B, T) above tensor.
        # Clamp out-of-range indices to 0; mask them out via q_is_ret.
        q_clamped = torch.where(q_in_range, q_idx, torch.zeros_like(q_idx))
        above_q = torch.gather(above, 1, q_clamped.view(B, -1)).view(B, T_in, max_lookahead)
        ups = (above_q * q_is_ret.float()).sum(dim=-1)               # (B, T_in)
        n_rets = q_is_ret.float().sum(dim=-1)                        # (B, T_in)

        # Position p is "labeled" iff it's a return position AND the full
        # lookahead window contains H return positions.
        target = (ups * 2 > n_rets).float()
        mask = ret_pos_mask[:, :T_in] & (n_rets >= H)
        return target, mask

    def joint_loss(self, tokens: torch.Tensor,
                   feature_offset: torch.Tensor | int = 0,
                   modality_ids: Optional[torch.Tensor] = None,
                   ) -> dict:
        """LM + auxiliary directional loss. cfg.dir_head_enabled must be True.

        Returns a dict with:
          - total: weighted sum (alpha * L_lm + beta * L_dir)
          - L_lm:  cross-entropy on next-token prediction
          - L_dir: BCE-with-logits on majority-direction at H rows ahead
          - n_dir: number of valid directional positions in the batch

        Targets are constructed from the input batch, mirroring the
        h=`dir_horizon` majority-direction labeling used by the dir_baseline
        diagnostics.
        """
        assert self.dir_head is not None, "joint_loss requires dir_head_enabled"
        c = self.cfg
        mids = modality_ids[:, :-1] if modality_ids is not None else None
        lm_logits, dir_logits = self.forward(
            tokens[:, :-1], modality_ids=mids, return_aux=True)
        target_lm = tokens[:, 1:]
        L_lm = F.cross_entropy(
            lm_logits.reshape(-1, c.vocab), target_lm.reshape(-1))

        dir_target, dir_mask = self._build_dir_targets(tokens, feature_offset)
        n_valid = int(dir_mask.sum().item())
        if n_valid == 0:
            L_dir = torch.zeros((), device=tokens.device, dtype=lm_logits.dtype)
        else:
            logits_flat = dir_logits[dir_mask]
            target_flat = dir_target[dir_mask]
            L_dir = F.binary_cross_entropy_with_logits(
                logits_flat, target_flat, reduction="mean")

        total = float(c.dir_alpha) * L_lm + float(c.dir_beta) * L_dir
        return {"total": total, "L_lm": L_lm, "L_dir": L_dir, "n_dir": n_valid}

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

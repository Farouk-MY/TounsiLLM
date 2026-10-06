"""
Core Transformer components for TounsiLM.

Implements from scratch:
  - RMSNorm (Root Mean Square Layer Normalization)
  - RotaryEmbedding (Rotary Position Embeddings / RoPE)
  - CausalSelfAttention (Multi-Head Attention with causal mask and RoPE)
  - SwiGLUMLP (SwiGLU Feed-Forward Network)
  - TransformerBlock (Pre-norm Transformer decoder block)

All components use standard PyTorch primitives (nn.Module, nn.Linear, etc.)
and are initialized from random weights — no pretrained weights are loaded.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tounsilm.config import ModelConfig


# ──────────────────────────────────────────────
# RMSNorm
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes input by the RMS of its elements, without centering.
    More efficient than LayerNorm since it skips mean computation.

    Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(input_dtype)


# ──────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ──────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Encodes absolute position as rotations applied to query/key vectors.
    This allows the attention score to depend on relative position.

    Reference: Su et al. (2021), "RoFormer: Enhanced Transformer with
    Rotary Position Embedding"
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        # Create [cos, sin] pairs for rotation
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for positions [0, seq_len).

        Returns:
            Tuple of (cos, sin) each of shape [seq_len, dim]
        """
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        cos: Cosine part of shape [seq_len, head_dim]
        sin: Sine part of shape [seq_len, head_dim]

    Returns:
        Rotated (q, k) tensors.
    """
    # Reshape for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ──────────────────────────────────────────────
# Causal Self-Attention
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE.

    Implements scaled dot-product attention with a causal mask,
    preventing each position from attending to future positions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Q, K, V projections (combined for efficiency)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_sequence_length,
            base=config.rope_base,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional [batch_size, 1, seq_len, seq_len] additive mask

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )

        # Output projection
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        return output


# ──────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ──────────────────────────────────────────────

class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    Uses the SwiGLU activation: SwiGLU(x) = (xW₁ ⊙ Swish(xV)) W₂

    This is more expressive than standard ReLU/GELU FFNs and is used
    in modern architectures like LLaMA, Mistral, etc.

    Reference: Shazeer (2020), "GLU Variants Improve Transformer"
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Gate and up projections (SwiGLU has two "up" projections)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(xW_gate) ⊙ xW_up) W_down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        return self.dropout(output)


# ──────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block.

    Structure:
        x -> RMSNorm -> CausalSelfAttention -> + residual
        x -> RMSNorm -> SwiGLUMLP -> + residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

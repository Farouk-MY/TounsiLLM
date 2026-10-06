"""
Tests for TounsiLM Model Architecture and Components.

Verifies:
  - Model initialization from random weights
  - Parameter calculation breakdown
  - Embedding weight tying
  - RMSNorm, RoPE, SwiGLU, and TransformerBlock mechanics
  - Checkpoint saving and loading
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tounsilm.config import ModelConfig
from tounsilm.model.components import (
    RMSNorm,
    RotaryEmbedding,
    CausalSelfAttention,
    SwiGLUMLP,
    TransformerBlock,
)
from tounsilm.model.transformer import TounsiLMModel


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_sequence_length=64,
        tie_word_embeddings=True,
    )


def test_rmsnorm():
    norm = RMSNorm(hidden_size=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape
    # Check that variance/RMS is approximately 1
    rms = torch.sqrt(out.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rotary_embedding():
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    cos, sin = rope(16)
    assert cos.shape == (16, dim)
    assert sin.shape == (16, dim)


def test_swiglu_mlp(small_config):
    mlp = SwiGLUMLP(small_config)
    x = torch.randn(2, 8, small_config.hidden_size)
    out = mlp(x)
    assert out.shape == x.shape


def test_transformer_block(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(2, 8, small_config.hidden_size)
    out = block(x)
    assert out.shape == x.shape


def test_model_initialization_and_parameter_counts(small_config):
    model = TounsiLMModel(small_config)
    counts = model.count_parameters()

    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]
    assert counts["embedding"] == small_config.vocab_size * small_config.hidden_size
    assert counts["lm_head"] == 0  # Tied embeddings
    assert model.lm_head.weight is model.token_embedding.weight


def test_model_save_and_load(small_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model = TounsiLMModel(small_config)
        model.save_pretrained(tmp_path)

        assert (tmp_path / "model.pt").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "parameter_count.json").exists()

        # Load back
        loaded_model = TounsiLMModel.load_checkpoint(tmp_path)
        assert loaded_model.config.vocab_size == small_config.vocab_size
        assert loaded_model.config.hidden_size == small_config.hidden_size

        # Verify weights match exactly
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded_model.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2)

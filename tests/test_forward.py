"""
Tests for TounsiLM Forward Pass, Loss, and Autograd.

Verifies:
  - Forward output shapes [batch_size, seq_len, vocab_size]
  - Next-token cross-entropy loss computation
  - Causal masking integrity
  - Backward pass gradient propagation
  - Gradient checkpointing backward pass
"""

import pytest
import torch

from tounsilm.config import ModelConfig
from tounsilm.model.transformer import TounsiLMModel


@pytest.fixture
def small_model():
    cfg = ModelConfig(
        vocab_size=500,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_sequence_length=32,
    )
    return TounsiLMModel(cfg)


def test_forward_shape(small_model):
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, 500, (batch_size, seq_len))
    out = small_model(x)
    assert "logits" in out
    assert out["logits"].shape == (batch_size, seq_len, 500)


def test_forward_with_loss(small_model):
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, 500, (batch_size, seq_len))
    labels = torch.randint(0, 500, (batch_size, seq_len))
    out = small_model(x, labels=labels)

    assert "loss" in out
    loss = out["loss"]
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0


def test_causal_masking_property(small_model):
    """Verify that earlier token logits are invariant to later token changes."""
    small_model.eval()
    seq_len = 8

    # Sequence A
    tokens_a = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    # Sequence B (identical prefix up to pos 4, different suffix from pos 4)
    tokens_b = torch.tensor([[10, 20, 30, 40, 99, 99, 99, 99]])

    with torch.no_grad():
        logits_a = small_model(tokens_a)["logits"]
        logits_b = small_model(tokens_b)["logits"]

    # Positions 0, 1, 2, 3 must have identical logits
    for pos in range(4):
        assert torch.allclose(logits_a[0, pos], logits_b[0, pos], atol=1e-5), (
            f"Causal violation at position {pos}!"
        )


def test_gradient_flow_backward(small_model):
    small_model.train()
    x = torch.randint(0, 500, (2, 8))
    labels = torch.randint(0, 500, (2, 8))

    out = small_model(x, labels=labels)
    out["loss"].backward()

    for name, param in small_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"


def test_gradient_checkpointing_forward_backward(small_model):
    small_model.enable_gradient_checkpointing()
    small_model.train()
    x = torch.randint(0, 500, (2, 8))
    labels = torch.randint(0, 500, (2, 8))

    out = small_model(x, labels=labels)
    out["loss"].backward()
    small_model.disable_gradient_checkpointing()

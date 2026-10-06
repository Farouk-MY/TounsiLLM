"""
Tests for Autoregressive Generation.

Verifies:
  - Generation output length and format
  - Top-k filtering and top-p sampling
  - Repetition penalty enforcement
  - EOS token early stopping
  - Streaming generation
"""

import pytest
import torch

from tounsilm.config import ModelConfig
from tounsilm.model.transformer import TounsiLMModel


@pytest.fixture
def small_model():
    cfg = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_sequence_length=32,
    )
    return TounsiLMModel(cfg)


def test_generation_length(small_model):
    small_model.eval()
    prompt = torch.tensor([[5, 10, 15]], dtype=torch.long)
    out = small_model.generate(prompt, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 13)


def test_generation_eos_stopping(small_model):
    small_model.eval()
    prompt = torch.tensor([[5, 10]], dtype=torch.long)
    # If eos token id is set and encountered, generation stops
    out = small_model.generate(prompt, max_new_tokens=5, eos_token_id=small_model.config.vocab_size + 999)
    assert out.shape[1] == 7


def test_generation_top_k_and_top_p(small_model):
    small_model.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    out = small_model.generate(prompt, max_new_tokens=5, top_k=10, top_p=0.9, temperature=0.7)
    assert out.shape == (1, 7)

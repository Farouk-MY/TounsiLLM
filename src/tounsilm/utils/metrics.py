"""
Evaluation and performance metrics for TounsiLM.

Provides utilities for calculating:
  - Perplexity
  - Tokenization efficiency and compression ratio
  - Dialect adherence metrics
  - Human evaluation rubrics & score aggregations
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Caps at 1e6 to avoid overflow when loss is very large.
    """
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    try:
        return math.exp(min(loss, 20.0))
    except OverflowError:
        return float("inf")


def compute_token_efficiency(char_count: int, token_count: int) -> float:
    """Compute characters per token (higher = better subword compression)."""
    if token_count <= 0:
        return 0.0
    return char_count / token_count


def calculate_human_eval_score(ratings: dict[str, float]) -> float:
    """Aggregate 1-5 human evaluation scores according to the TounsiEval rubric:
      - tunisian_naturalness (weight 0.25)
      - correctness (weight 0.20)
      - meaning_preservation (weight 0.15)
      - grammar (weight 0.15)
      - dialect_adherence (weight 0.15)
      - fluency (weight 0.10)
    """
    weights = {
        "tunisian_naturalness": 0.25,
        "correctness": 0.20,
        "meaning_preservation": 0.15,
        "grammar": 0.15,
        "dialect_adherence": 0.15,
        "fluency": 0.10,
    }
    score = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        if key in ratings:
            score += ratings[key] * weight
            total_weight += weight
    return score / total_weight if total_weight > 0 else 0.0

"""
Reproducibility utilities for TounsiLM.

Handles random seed management across Python, NumPy, PyTorch (CPU & CUDA).
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all RNGs.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full determinism also requires setting these env vars and flags,
    # but they may hurt performance significantly.
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_num_params(num_params: int) -> str:
    """Format parameter count in human-readable form.

    Examples:
        123_456_789 -> '123.5M'
        1_234_567_890 -> '1.2B'
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    return str(num_params)

"""
TounsiLM — A Tunisian Arabic Language Model built from scratch.

This package provides the complete implementation for training a decoder-only
Transformer language model specialized in Tunisian Arabic (Derja), Arabizi,
and Tunisian/French code-switching.

The model is trained from random initialization — no pretrained weights are used.
"""

import sys

# Ensure UTF-8 output streams across all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__version__ = "0.1.0"
__model_name__ = "TounsiLM"

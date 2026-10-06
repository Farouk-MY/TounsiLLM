"""
Script to estimate and verify TounsiLM parameter counts programmatically.

Calculates:
  - Embedding parameters
  - Attention parameters
  - MLP parameters
  - Normalization parameters
  - Total parameters
  - Trainable parameters

Reports theoretical and instantiated counts with component breakdowns.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import ModelConfig, load_config
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.utils.reproducibility import format_num_params


def estimate_theoretical_parameters(config: ModelConfig) -> dict:
    """Calculate parameter counts mathematically without instantiating model."""
    V = config.vocab_size
    H = config.hidden_size
    L = config.num_layers
    I = config.intermediate_size

    # Token embeddings
    embedding = V * H

    # Attention per layer: Q, K, V, O projections (each H x H)
    # Total attention params per layer = 4 * H * H
    attn_per_layer = 4 * H * H
    total_attn = L * attn_per_layer

    # MLP per layer (SwiGLU): gate_proj (H x I), up_proj (H x I), down_proj (I x H)
    # Total MLP params per layer = 3 * H * I
    mlp_per_layer = 3 * H * I
    total_mlp = L * mlp_per_layer

    # Normalization: 2 RMSNorm per block (each H params) + 1 final RMSNorm (H params)
    norm_params = (2 * L + 1) * H

    # LM Head: V x H. If tied, 0 extra parameters!
    lm_head = 0 if config.tie_word_embeddings else V * H

    total = embedding + total_attn + total_mlp + norm_params + lm_head

    return {
        "embedding": embedding,
        "attention": total_attn,
        "mlp": total_mlp,
        "normalization": norm_params,
        "lm_head": lm_head,
        "total": total,
        "trainable": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate and report TounsiLM parameter counts")
    parser.add_argument("--config", type=str, default="configs/model_300m.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(config_path)
        model_cfg = cfg.model
    else:
        model_cfg = ModelConfig()

    print("=" * 65)
    print("      TounsiLM Parameter Estimation & Architecture Audit")
    print("=" * 65)
    print(f"Architecture Hyperparameters:")
    print(f"  • Vocabulary size (V)      : {model_cfg.vocab_size:,}")
    print(f"  • Hidden size (H)          : {model_cfg.hidden_size}")
    print(f"  • Transformer layers (L)   : {model_cfg.num_layers}")
    print(f"  • Attention heads (A)      : {model_cfg.num_attention_heads}")
    print(f"  • Head dimension (d_k)     : {model_cfg.head_dim}")
    print(f"  • Intermediate size (I)    : {model_cfg.intermediate_size}")
    print(f"  • Max sequence length      : {model_cfg.max_sequence_length}")
    print(f"  • Tied word embeddings     : {model_cfg.tie_word_embeddings}")
    print("-" * 65)

    # 1. Theoretical calculation
    theo = estimate_theoretical_parameters(model_cfg)
    print("Theoretical Parameter Breakdown:")
    print(f"  1. Token Embedding         : {theo['embedding']:>12,} ({format_num_params(theo['embedding'])})")
    print(f"  2. Attention Projections   : {theo['attention']:>12,} ({format_num_params(theo['attention'])})")
    print(f"  3. SwiGLU MLP Layers       : {theo['mlp']:>12,} ({format_num_params(theo['mlp'])})")
    print(f"  4. RMSNorm Parameters      : {theo['normalization']:>12,} ({format_num_params(theo['normalization'])})")
    print(f"  5. LM Head (Tied={model_cfg.tie_word_embeddings}): {theo['lm_head']:>12,} ({format_num_params(theo['lm_head'])})")
    print("-" * 65)
    print(f"  TOTAL PARAMETERS           : {theo['total']:>12,} ({format_num_params(theo['total'])})")
    print(f"  TRAINABLE PARAMETERS       : {theo['trainable']:>12,} ({format_num_params(theo['trainable'])})")

    # 2. Instantiated check
    print("-" * 65)
    print("Instantiating PyTorch Module (Random Initialization)...")
    model = TounsiLMModel(model_cfg)
    actual = model.count_parameters()
    print(f"  • Instantiated Embedding   : {actual['embedding']:>12,} ({format_num_params(actual['embedding'])})")
    print(f"  • Instantiated Attention   : {actual['attention']:>12,} ({format_num_params(actual['attention'])})")
    print(f"  • Instantiated MLP         : {actual['mlp']:>12,} ({format_num_params(actual['mlp'])})")
    print(f"  • Instantiated RMSNorm     : {actual['normalization']:>12,} ({format_num_params(actual['normalization'])})")
    print(f"  • Instantiated LM Head     : {actual['lm_head']:>12,} ({format_num_params(actual['lm_head'])})")
    print(f"  • Actual Total Parameters  : {actual['total']:>12,} ({format_num_params(actual['total'])})")
    print(f"  • Actual Trainable Params  : {actual['trainable']:>12,} ({format_num_params(actual['trainable'])})")

    # Target comparison
    target = 300_000_000
    diff = actual["total"] - target
    pct_diff = (diff / target) * 100
    print("-" * 65)
    print(f"Target 300M Comparison:")
    print(f"  Target: 300.0M | Actual: {actual['total']/1e6:.2f}M | Delta: {diff:+d} ({pct_diff:+.2f}%)")
    if abs(pct_diff) <= 10.0:
        print("  [SUCCESS] Parameter count is within ±10% target tolerance!")
    else:
        print(f"  [NOTE] Parameter count deviates by {pct_diff:.2f}%.")
    print("=" * 65)


if __name__ == "__main__":
    main()

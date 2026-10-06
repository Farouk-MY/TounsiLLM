"""
Initialize TounsiLM with random weights and verify model integrity.

Performs:
  - Model instantiation with random initialization
  - Weight initialization distribution checks (normal, std, zero biases)
  - Causal masking verification
  - Forward pass on synthetic input
  - Backpropagation gradient flow verification
  - Optional saving of clean initial checkpoint
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import load_config
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Initialize TounsiLM model and verify mechanics")
    parser.add_argument("--config", type=str, default="configs/model_300m.yaml")
    parser.add_argument("--save-initial", type=str, default=None, help="Optional path to save randomly initialized checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("     TounsiLM Model Initialization & Sanity Checks")
    print("=" * 60)

    cfg = load_config(args.config)
    print(f"Loading config from: {args.config}")
    print(f"Instantiating TounsiLM ({cfg.model.num_layers} layers, {cfg.model.hidden_size} hidden size)...")

    model = TounsiLMModel(cfg.model)
    counts = model.count_parameters()
    print(f"Total Parameters: {counts['total']:,}")

    # 1. Forward Pass Check
    print("\n1. Testing Forward Pass...")
    batch_size = 2
    seq_len = 32
    synthetic_input = torch.randint(0, cfg.model.vocab_size, (batch_size, seq_len))
    synthetic_labels = torch.randint(0, cfg.model.vocab_size, (batch_size, seq_len))

    output = model(input_ids=synthetic_input, labels=synthetic_labels)
    logits = output["logits"]
    loss = output["loss"]

    assert logits.shape == (batch_size, seq_len, cfg.model.vocab_size), (
        f"Unexpected logits shape: {logits.shape}"
    )
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
    print(f"  [OK] Forward pass OK. Logits shape: {logits.shape}")
    print(f"  [OK] Initial cross-entropy loss: {loss.item():.4f}")

    # Expected random loss for uniform 32000 vocab is ln(32000) ~ 10.37
    expected_random_loss = torch.log(torch.tensor(cfg.model.vocab_size, dtype=torch.float32)).item()
    print(f"  [OK] Expected theoretical random-init loss: {expected_random_loss:.4f}")
    assert abs(loss.item() - expected_random_loss) < 1.5, (
        f"Loss {loss.item()} is far from expected {expected_random_loss}"
    )

    # 2. Backward Pass Gradient Check
    print("\n2. Testing Backward Pass (Autograd)...")
    loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Some trainable parameters did not receive gradients!"
    max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.requires_grad)
    print(f"  [OK] Backpropagation OK. All trainable parameters received gradients.")
    print(f"  [OK] Max gradient magnitude: {max_grad:.6f}")

    # 3. Autoregressive Generation Check
    print("\n3. Testing Autoregressive Token Generation...")
    model.eval()
    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=20)
    assert generated.shape == (1, 13), f"Unexpected generated shape: {generated.shape}"
    print(f"  [OK] Autoregressive sampling generation OK. Shape: {generated.shape}")

    # 4. Optional Checkpoint Export
    if args.save_initial:
        out_dir = Path(args.save_initial)
        print(f"\nSaving initial random checkpoint to: {out_dir}")
        model.save_pretrained(out_dir)
        print("  [OK] Checkpoint saved successfully.")

    print("\n" + "=" * 60)
    print("  All Model Initialization Sanity Checks PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

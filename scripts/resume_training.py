"""
Resume training from an existing checkpoint.

Recovers complete training state:
  - Model weights
  - AdamW optimizer state & momentum buffers
  - Cosine learning rate scheduler position
  - Mixed precision GradScaler state
  - Step count & tokens processed counter
  - Configuration & random seeds
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import load_config
from tounsilm.data.dataset import TounsiLMDataset, create_packed_dataset
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.tokenizer.tokenizer import TounsiTokenizer
from tounsilm.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Resume TounsiLM training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest", help="Path to checkpoint directory or 'checkpoints/latest'")
    parser.add_argument("--config", type=str, default="configs/pretraining_local.yaml", help="Path to base config")
    parser.add_argument("--additional-steps", type=int, default=1000, help="Steps to train past resumed step")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.name == "latest" and ckpt_path.is_file():
        # Read target from pointer file
        with open(ckpt_path, "r") as f:
            target_name = f.read().strip()
        ckpt_dir = ckpt_path.parent / target_name
    else:
        ckpt_dir = ckpt_path

    if not ckpt_dir.exists():
        print(f"[Error] Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    print("=" * 60)
    print("      TounsiLM Training Resume")
    print("=" * 60)
    print(f"Resuming from: {ckpt_dir}")

    cfg = load_config(args.config)
    tok_path = Path("data/tokenizer/tounsilm_tokenizer.model")
    tokenizer = TounsiTokenizer(tok_path) if tok_path.exists() else None

    # Load datasets
    train_bin = Path("data/processed/train.bin")
    val_bin = Path("data/processed/val.bin")

    if not train_bin.exists():
        train_txt = Path("data/processed/train.txt")
        val_txt = Path("data/processed/val.txt")
        with open(train_txt, "r", encoding="utf-8") as f:
            train_lines = [l.strip() for l in f if l.strip()]
        with open(val_txt, "r", encoding="utf-8") as f:
            val_lines = [l.strip() for l in f if l.strip()]
        create_packed_dataset(train_lines, tokenizer, train_bin)
        create_packed_dataset(val_lines, tokenizer, val_bin)

    model = TounsiLMModel(cfg.model)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        tokenizer=tokenizer,
    )

    trainer.load_checkpoint(ckpt_dir)
    trainer.train_config.max_steps = trainer.global_step + args.additional_steps
    print(f"Training for an additional {args.additional_steps} steps (up to step {trainer.train_config.max_steps})...")

    stats = trainer.train()
    print("\nResumed training completed successfully:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

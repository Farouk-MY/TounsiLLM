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

    # Automatically load model architecture from checkpoint if saved with it
    ckpt_config_file = ckpt_dir / "config.json"
    ckpt_meta_file = ckpt_dir / "metadata.json"
    if ckpt_config_file.exists():
        import json
        with open(ckpt_config_file, "r", encoding="utf-8") as f:
            model_dict = json.load(f)
        for k, v in model_dict.items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
        print(f"Loaded checkpoint architecture: {cfg.model.num_layers}L, {cfg.model.hidden_size}H, {cfg.model.vocab_size}V, seq_len={cfg.model.max_sequence_length}")

    if ckpt_meta_file.exists():
        import json
        with open(ckpt_meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "config" in meta and "training" in meta["config"]:
            for k, v in meta["config"]["training"].items():
                if hasattr(cfg.training, k) and k not in ["max_steps", "checkpoint_dir"]:
                    setattr(cfg.training, k, v)

    tok_path = Path("data/tokenizer/tounsilm_tokenizer.model")
    tokenizer = TounsiTokenizer(tok_path) if tok_path.exists() else None

    # Load datasets
    train_bin = Path("data/processed/train.bin")
    val_bin = Path("data/processed/val.bin")

    if not train_bin.exists():
        train_txt = Path("data/processed/train.txt")
        val_txt = Path("data/processed/val.txt")
        if train_txt.exists():
            with open(train_txt, "r", encoding="utf-8") as f:
                train_lines = [l.strip() for l in f if l.strip()]
            create_packed_dataset(train_lines, tokenizer, train_bin)
        if val_txt.exists():
            with open(val_txt, "r", encoding="utf-8") as f:
                val_lines = [l.strip() for l in f if l.strip()]
            create_packed_dataset(val_lines, tokenizer, val_bin)

    # Adapt sequence length to model and data bounds
    if hasattr(cfg.model, "max_sequence_length") and cfg.training.sequence_length > cfg.model.max_sequence_length:
        cfg.training.sequence_length = cfg.model.max_sequence_length

    train_tokens = train_bin.stat().st_size // 2 if train_bin.exists() else 0
    if train_tokens > 0 and train_tokens <= cfg.training.sequence_length:
        cfg.training.sequence_length = max(16, train_tokens - 1)

    train_ds = TounsiLMDataset(train_bin, seq_len=cfg.training.sequence_length)
    val_ds = None
    if val_bin.exists():
        try:
            val_ds = TounsiLMDataset(val_bin, seq_len=cfg.training.sequence_length)
        except ValueError as e:
            print(f"[Warning] Validation dataset skipped: {e}")
            val_ds = None

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

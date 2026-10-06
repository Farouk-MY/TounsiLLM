"""
Export and package TounsiLM model for deployment and sharing.

Exports:
  - PyTorch state_dict (model.pt)
  - Model architecture config (config.json)
  - Tokenizer model & vocabulary (tounsilm_tokenizer.model, vocab)
  - Comprehensive model card metadata (model_card.json)
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.model.transformer import TounsiLMModel
from tounsilm.tokenizer.tokenizer import TounsiTokenizer


def main():
    parser = argparse.ArgumentParser(description="Export TounsiLM model checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint-final")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/tounsilm_tokenizer.model")
    parser.add_argument("--output-dir", type=str, default="exported_model/TounsiLM-300M")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    tok_path = Path(args.tokenizer_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.exists():
        print(f"[Error] Checkpoint not found: {ckpt_dir}")
        sys.exit(1)

    print("=" * 60)
    print("       TounsiLM Model Export & Distribution Packaging")
    print("=" * 60)
    print(f"Source Checkpoint : {ckpt_dir}")
    print(f"Target Export Dir : {out_dir}")

    # Load model to verify integrity
    model = TounsiLMModel.load_checkpoint(ckpt_dir)
    counts = model.count_parameters()

    # Copy model files
    shutil.copy2(ckpt_dir / "model.pt", out_dir / "model.pt")
    shutil.copy2(ckpt_dir / "config.json", out_dir / "config.json")

    # Copy tokenizer
    if tok_path.exists():
        shutil.copy2(tok_path, out_dir / "tounsilm_tokenizer.model")

    # Generate Model Card Metadata
    model_card = {
        "model_name": "TounsiLM-300M",
        "description": "Tunisian Arabic Decoder-Only Transformer built from scratch",
        "parameters": counts,
        "languages": ["Tunisian Arabic (Derja)", "Arabizi", "Tunisian French Code-Switching"],
        "license": "Apache-2.0",
        "export_date": datetime.now().isoformat(),
        "architecture": {
            "num_layers": model.config.num_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "max_sequence_length": model.config.max_sequence_length,
            "tie_word_embeddings": model.config.tie_word_embeddings,
        },
    }

    with open(out_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    print("\n[OK] Model exported successfully!")
    print(f"  • Total parameters : {counts['total']:,}")
    print(f"  • Artifacts at     : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Train SentencePiece BPE tokenizer from scratch for TounsiLM.

Uses Tunisian corpus text to learn a 32,000 subword vocabulary covering:
  - Arabic-script Tunisian (Derja)
  - Arabizi (Latin script with numbers 3, 5, 7, 8, 9)
  - Tunisian/French code-switching
  - Technical vocabulary
  - Instruction tuning special tokens (<|im_start|>, <|im_end|>, etc.)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import TokenizerConfig, load_config
from tounsilm.tokenizer.tokenizer import TounsiTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train TounsiLM Tokenizer from scratch")
    parser.add_argument("--config", type=str, default="configs/tokenizer.yaml", help="Path to tokenizer config")
    parser.add_argument("--data-file", type=str, default="data/tokenizer/corpus.txt", help="Training corpus text file")
    parser.add_argument("--output-dir", type=str, default="data/tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = load_config(cfg_path).tokenizer
    else:
        cfg = TokenizerConfig()

    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"[Error] Training corpus file not found: {data_file}")
        print("Please run `python scripts/prepare_data.py` first to generate the tokenizer corpus.")
        sys.exit(1)

    print("=" * 60)
    print("      TounsiLM Tokenizer Training (From Scratch)")
    print("=" * 60)
    print(f"Corpus file   : {data_file}")
    print(f"Target vocab  : {cfg.vocab_size:,}")
    print(f"Model type    : {cfg.model_type.upper()}")
    print(f"Output dir    : {cfg.output_dir}")
    print("-" * 60)

    tokenizer = TounsiTokenizer.train([data_file], cfg)

    print("\n" + "=" * 60)
    print(f"Tokenizer trained successfully! Final vocabulary size: {tokenizer.vocab_size:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()

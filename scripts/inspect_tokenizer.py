"""
Inspect and evaluate TounsiLM tokenizer quality.

Displays:
  - Original text
  - Token IDs
  - Decoded tokens (pieces)
  - Number of tokens
  - Characters/token ratio (compression efficiency)
  - Round-trip fidelity check
  - Examples from:
      1. Arabic Tunisian
      2. Arabizi
      3. Mixed Tunisian/French
      4. English technical terms
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.tokenizer.tokenizer import TounsiTokenizer


DEFAULT_TEST_SAMPLES = [
    # 1. Arabic Tunisian
    ("Arabic Tunisian", "شنوة تعمل توّا؟"),
    ("Arabic Tunisian", "علاش ما جيتش؟"),
    ("Arabic Tunisian", "برشا ناس يحبو القهوة."),
    ("Arabic Tunisian", "ما عادش نجم نصبر أكثر من هكّا."),
    # 2. Arabizi
    ("Arabizi", "chnowa ta3mel?"),
    ("Arabizi", "3lech ma jitich?"),
    ("Arabizi", "barsha ness y7ebbou l9ahwa."),
    ("Arabizi", "winek ya 5ouya labes 3lik?"),
    # 3. Mixed Tunisian/French
    ("Mixed Tunisian/French", "لازم نعمل update للapplication."),
    ("Mixed Tunisian/French", "توّا عندي réunion."),
    ("Mixed Tunisian/French", "ما نجمش نبعث الـdocument اليوم."),
    ("Mixed Tunisian/French", "باش نعمل deploy بعد شوية."),
    # 4. English Technical Terms
    ("English Technical Terms", "API"),
    ("English Technical Terms", "database"),
    ("English Technical Terms", "mobile application"),
    ("English Technical Terms", "machine learning"),
]


def inspect_text(tokenizer: TounsiTokenizer, category: str, text: str):
    info = tokenizer.inspect(text)
    print(f"\n[{category}]")
    print(f"  Input text       : \"{info['original']}\"")
    print(f"  Token pieces     : {info['pieces']}")
    print(f"  Token IDs        : {info['token_ids']}")
    print(f"  Number of tokens : {info['num_tokens']}")
    print(f"  Chars/token      : {info['chars_per_token']:.2f}")
    print(f"  Round-trip match : {'[YES]' if info['roundtrip_match'] else '[NO]'}")


def main():
    parser = argparse.ArgumentParser(description="Inspect TounsiLM tokenizer tokenization quality")
    parser.add_argument("--model-path", type=str, default="data/tokenizer/tounsilm_tokenizer.model")
    parser.add_argument("--custom-text", type=str, default=None, help="Inspect a specific custom sentence")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[Error] Tokenizer model not found at: {model_path}")
        print("Please train the tokenizer first with `python scripts/train_tokenizer.py`.")
        sys.exit(1)

    tokenizer = TounsiTokenizer(model_path)

    print("=" * 70)
    print("         TounsiLM Tokenizer Inspection & Fragmentation Audit")
    print("=" * 70)
    print(f"Model path   : {model_path}")
    print(f"Vocab size   : {tokenizer.vocab_size:,}")
    print(f"Special IDs  : PAD={tokenizer.pad_id}, UNK={tokenizer.unk_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
    print("=" * 70)

    if args.custom_text:
        inspect_text(tokenizer, "Custom Input", args.custom_text)
    else:
        for category, sample in DEFAULT_TEST_SAMPLES:
            inspect_text(tokenizer, category, sample)

    print("\n" + "=" * 70)
    print("  Tokenizer Inspection Completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

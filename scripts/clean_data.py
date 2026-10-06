"""
Clean, normalize, and filter the raw Tunisian corpus.

Pipeline stages:
  1. Tunisian-aware normalizer (Unicode NFC, alef normalization, repeated chars)
  2. PII filtering (phone, email, IP addresses)
  3. Quality filtering (length, symbol ratio, url ratio, repeated characters)
  4. Dialect scoring & filtering (Tunisian lexical indicators)
  5. Contamination check against TounsiEval benchmark items
  6. Dataset statistics report generation
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import DataConfig, load_config
from tounsilm.data.normalizer import TunisianNormalizer
from tounsilm.data.filters import QualityFilter, DialectFilter, PIIFilter
from tounsilm.evaluation.tounsi_eval import load_benchmark_file


def main():
    print("=" * 60)
    print("       TounsiLM Data Pipeline: Cleaning & Normalization")
    print("=" * 60)

    cfg = DataConfig()
    normalizer = TunisianNormalizer()
    quality_filter = QualityFilter(cfg)
    dialect_filter = DialectFilter(min_score=cfg.min_tunisian_score)
    pii_filter = PIIFilter(enabled=cfg.pii_filter)

    raw_path = Path("data/raw/tounsi_raw_corpus.jsonl")
    if not raw_path.exists():
        print(f"[Error] Raw corpus not found at {raw_path}. Run prepare_data.py first.")
        sys.exit(1)

    inter_dir = Path("data/intermediate")
    inter_dir.mkdir(parents=True, exist_ok=True)
    clean_path = inter_dir / "tounsi_cleaned.jsonl"

    # Load TounsiEval samples to guard against data contamination
    eval_benchmark = load_benchmark_file("data/eval/tounsi_eval.jsonl")
    eval_inputs = {s.input.strip().lower() for s in eval_benchmark}

    total_in = 0
    passed_quality = 0
    passed_dialect = 0
    pii_count = 0
    contaminated_count = 0

    clean_docs = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_in += 1
            item = json.loads(line)
            raw_text = item["text"]

            # 1. Normalization
            norm_text = normalizer.normalize(raw_text)

            # 2. PII Filter
            norm_text, pii_detected = pii_filter.filter(norm_text)
            if pii_detected:
                pii_count += 1

            # 3. Quality Filter
            q_res = quality_filter.filter(norm_text)
            if not q_res.passed:
                continue
            passed_quality += 1

            # 4. Dialect Filter
            d_res = dialect_filter.score(norm_text)
            if d_res["score"] < cfg.min_tunisian_score:
                continue
            passed_dialect += 1

            # 5. Contamination check
            if norm_text.strip().lower() in eval_inputs:
                contaminated_count += 1
                continue

            clean_docs.append({
                "id": item.get("id", f"clean-{len(clean_docs)+1}"),
                "text": norm_text,
                "script": item.get("script", "unknown"),
                "dialect_score": round(d_res["score"], 3),
            })

    # Save cleaned documents
    with open(clean_path, "w", encoding="utf-8") as f:
        for doc in clean_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Save corpus for tokenizer training as well
    tok_corpus_dir = Path("data/tokenizer")
    tok_corpus_dir.mkdir(parents=True, exist_ok=True)
    with open(tok_corpus_dir / "corpus.txt", "w", encoding="utf-8") as f:
        for doc in clean_docs:
            f.write(doc["text"] + "\n")

    print(f"Total raw docs processed : {total_in:,}")
    print(f"Passed quality filters   : {passed_quality:,}")
    print(f"Passed dialect filters   : {passed_dialect:,}")
    print(f"PII occurrences sanitized: {pii_count:,}")
    print(f"Contaminated docs dropped: {contaminated_count}")
    print(f"Final cleaned documents  : {len(clean_docs):,}")
    print(f"Saved cleaned corpus to   : {clean_path}")
    print(f"Saved tokenizer text to  : {tok_corpus_dir / 'corpus.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

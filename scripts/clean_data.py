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

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import DataConfig, load_config
from tounsilm.data.normalizer import TunisianNormalizer
from tounsilm.data.filters import QualityFilter, DialectFilter, PIIFilter
from tounsilm.evaluation.tounsi_eval import load_benchmark_file


def main():
    parser = argparse.ArgumentParser(description="Clean, normalize, and filter raw datasets for TounsiLM")
    parser.add_argument("--input", type=str, default="data/raw", help="Path to raw file (.jsonl or .txt) or directory")
    parser.add_argument("--output", type=str, default="data/intermediate/tounsi_cleaned.jsonl", help="Output cleaned .jsonl path")
    parser.add_argument("--skip-dialect-filter", action="store_true", help="Skip dialect keyword scoring for custom domain datasets")
    args = parser.parse_args()

    print("=" * 60)
    print("       TounsiLM Data Pipeline: Cleaning & Normalization")
    print("=" * 60)

    cfg = DataConfig()
    normalizer = TunisianNormalizer()
    quality_filter = QualityFilter(cfg)
    dialect_filter = DialectFilter(min_score=cfg.min_tunisian_score)
    pii_filter = PIIFilter(enabled=cfg.pii_filter)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] Input path not found: {input_path}")
        print("Please place your dataset file (.jsonl or .txt) in `data/raw/` or pass `--input <path>`.")
        sys.exit(1)

    # Collect files to process (.parquet, .jsonl, .txt)
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = sorted(
            list(input_path.glob("*.parquet")) +
            list(input_path.glob("*.jsonl")) +
            list(input_path.glob("*.txt"))
        )

    if not files_to_process:
        print(f"[Error] No .parquet, .jsonl, or .txt files found in: {input_path}")
        sys.exit(1)

    print(f"Discovered {len(files_to_process)} raw data file(s):")
    for f in files_to_process:
        print(f"  • {f}")

    inter_dir = Path("data/intermediate")
    inter_dir.mkdir(parents=True, exist_ok=True)
    clean_path = Path(args.output)
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    # Load TounsiEval samples to guard against data contamination
    eval_benchmark = load_benchmark_file("data/eval/tounsi_eval.jsonl")
    eval_inputs = {s.input.strip().lower() for s in eval_benchmark}

    total_in = 0
    passed_quality = 0
    passed_dialect = 0
    pii_count = 0
    contaminated_count = 0

    clean_docs = []

    for fpath in files_to_process:
        raw_items = []
        if fpath.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(fpath)
            # Detect text column
            text_col = None
            for candidate in ["text", "Tweet", "tweet", "content", "document", "sentence", "comment"]:
                if candidate in df.columns:
                    text_col = candidate
                    break
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == object or str(df[col].dtype) == "string":
                        text_col = col
                        break
            if text_col is None:
                print(f"[Warning] Could not find text column in {fpath}. Skipping.")
                continue
            print(f"  Processing Parquet file '{fpath.name}' using column '{text_col}' ({len(df):,} rows)...")
            for idx, val in enumerate(df[text_col]):
                if pd.notna(val) and str(val).strip():
                    raw_items.append({"text": str(val).strip(), "id": f"{fpath.stem}-{idx+1}"})
        else:
            is_jsonl = fpath.suffix == ".jsonl"
            with open(fpath, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if is_jsonl:
                        try:
                            item = json.loads(line_str)
                            raw_text = item.get("text", line_str)
                            raw_items.append({"text": raw_text, "id": item.get("id", f"{fpath.stem}-{line_idx+1}")})
                        except Exception:
                            raw_items.append({"text": line_str, "id": f"{fpath.stem}-{line_idx+1}"})
                    else:
                        raw_items.append({"text": line_str, "id": f"{fpath.stem}-{line_idx+1}"})

        for item in raw_items:
            raw_text = item["text"]
            total_in += 1
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
            if not args.skip_dialect_filter:
                d_res = dialect_filter.score(norm_text)
                if d_res["score"] < cfg.min_tunisian_score:
                    continue
                passed_dialect += 1
                dialect_score = round(d_res["score"], 3)
            else:
                passed_dialect += 1
                dialect_score = 1.0

            # 5. Contamination check
            if norm_text.strip().lower() in eval_inputs:
                contaminated_count += 1
                continue

            clean_docs.append({
                "id": item.get("id", f"clean-{len(clean_docs)+1}"),
                "text": norm_text,
                "script": item.get("script", "unknown"),
                "dialect_score": dialect_score,
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

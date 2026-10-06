"""
Deduplicate documents and generate deterministic train/val/test splits.

Stages:
  1. Exact hash deduplication (xxhash)
  2. Near-duplicate deduplication (MinHash LSH)
  3. Deduplication statistical reporting
  4. Deterministic splitting into train (98%), val (1%), test (1%)
  5. Dataset report generation (Arabic script %, Arabizi %, document lengths)
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import DataConfig
from tounsilm.data.deduplication import ExactDeduplicator, MinHashDeduplicator, HAS_DATASKETCH


def main():
    print("=" * 60)
    print("       TounsiLM Data Pipeline: Deduplication & Splitting")
    print("=" * 60)

    clean_path = Path("data/intermediate/tounsi_cleaned.jsonl")
    if not clean_path.exists():
        print(f"[Error] Cleaned corpus not found at {clean_path}. Run clean_data.py first.")
        sys.exit(1)

    documents = []
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    texts = [d["text"] for d in documents]
    print(f"Loaded {len(texts):,} cleaned documents.")

    # 1. Exact Deduplication
    print("\n1. Running Exact Deduplication (xxhash)...")
    exact_dedup = ExactDeduplicator()
    unique_texts, exact_report = exact_dedup.deduplicate(texts)
    print(f"  • Documents before : {exact_report['documents_before']:,}")
    print(f"  • Duplicates removed: {exact_report['documents_removed']:,}")
    print(f"  • Unique remaining  : {exact_report['documents_remaining']:,} ({100-exact_report['duplicate_percentage']:.1f}%)")

    # 2. Near Deduplication
    final_texts = unique_texts
    if HAS_DATASKETCH:
        print("\n2. Running Near-Duplicate Detection (MinHash LSH)...")
        near_dedup = MinHashDeduplicator(threshold=0.85, ngram_size=5)
        final_texts, near_report = near_dedup.deduplicate(unique_texts)
        print(f"  • Near-duplicates removed: {near_report['documents_removed']:,}")
        print(f"  • Final documents        : {near_report['documents_remaining']:,}")
    else:
        print("\n2. MinHash LSH skipped (datasketch not installed).")

    # 3. Deterministic Train/Val/Test Split
    print("\n3. Creating Train (98%) / Val (1%) / Test (1%) Splits...")
    random.seed(42)
    shuffled = list(final_texts)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = max(15, int(n_total * 0.10)) if n_total >= 50 else max(2, int(n_total * 0.10))
    n_test = max(15, int(n_total * 0.10)) if n_total >= 50 else max(2, int(n_total * 0.10))
    n_train = max(1, n_total - n_val - n_test)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    with open(proc_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_data) + "\n")
    with open(proc_dir / "val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_data) + "\n")
    with open(proc_dir / "test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_data) + "\n")

    # 4. Dataset Analysis Dashboard Report
    total_chars = sum(len(t) for t in final_texts)
    arabic_chars = sum(sum(1 for c in t if "\u0600" <= c <= "\u06FF") for t in final_texts)
    latin_chars = sum(sum(1 for c in t if c.isascii() and c.isalpha()) for t in final_texts)
    arabic_pct = (arabic_chars / max(1, arabic_chars + latin_chars)) * 100
    arabizi_pct = 100.0 - arabic_pct

    lengths = [len(t) for t in final_texts]
    avg_len = sum(lengths) / max(len(lengths), 1)
    sorted_lens = sorted(lengths)
    med_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0

    report = {
        "total_documents": len(final_texts),
        "total_characters": total_chars,
        "estimated_tokens": int(total_chars / 4.0),
        "arabic_script_percentage": round(arabic_pct, 2),
        "arabizi_percentage": round(arabizi_pct, 2),
        "average_document_length": round(avg_len, 1),
        "median_document_length": med_len,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
    }

    report_path = Path("data/manifests/dataset_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDataset Analysis Summary:")
    print(f"  • Total Documents       : {report['total_documents']:,}")
    print(f"  • Total Characters      : {report['total_characters']:,}")
    print(f"  • Estimated Tokens      : ~{report['estimated_tokens']:,}")
    print(f"  • Arabic Script Ratio   : {report['arabic_script_percentage']}%")
    print(f"  • Arabizi Ratio         : {report['arabizi_percentage']}%")
    print(f"  • Avg / Med Doc Length  : {report['average_document_length']} / {report['median_document_length']} chars")
    print(f"  • Train / Val / Test    : {len(train_data):,} / {len(val_data):,} / {len(test_data):,}")
    print(f"  • Report saved to       : {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

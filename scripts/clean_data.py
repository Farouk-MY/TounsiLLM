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


def _detect_text_column(df) -> str | None:
    """Auto-detect text column in tabular data."""
    candidates = [
        "Tweet", "tweet", "text", "Text", "content", "Content",
        "document", "sentence", "comment", "Comment", "body", "Body",
        "message", "Message", "transcript", "arabic", "derja"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            return c
    return None


def extract_raw_items_from_file(fpath: Path) -> list[dict]:
    """Universal extractor for any file format: Parquet, CSV, TSV, JSON, JSONL, Subtitles, HTML, Text, Markdown."""
    ext = fpath.suffix.lower()
    items = []

    # 1. Parquet Table
    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(fpath)
        col = _detect_text_column(df)
        if col:
            print(f"  • Reading Parquet '{fpath.name}' column '{col}' ({len(df):,} rows)")
            for i, val in enumerate(df[col]):
                if pd.notna(val) and str(val).strip():
                    items.append({"text": str(val).strip(), "id": f"{fpath.stem}-{i+1}"})

    # 2. CSV / TSV Table
    elif ext in [".csv", ".tsv"]:
        import pandas as pd
        sep = "\t" if ext == ".tsv" else ","
        try:
            df = pd.read_csv(fpath, sep=sep, on_bad_lines="skip", low_memory=False)
            col = _detect_text_column(df)
            if col:
                print(f"  • Reading {ext.upper()} '{fpath.name}' column '{col}' ({len(df):,} rows)")
                for i, val in enumerate(df[col]):
                    if pd.notna(val) and str(val).strip():
                        items.append({"text": str(val).strip(), "id": f"{fpath.stem}-{i+1}"})
        except Exception as e:
            print(f"  [Warning] Failed to parse {fpath.name}: {e}")

    # 3. JSON & JSONL
    elif ext in [".json", ".jsonl"]:
        if ext == ".jsonl":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        txt = obj.get("text", obj.get("Tweet", obj.get("content", s))) if isinstance(obj, dict) else str(obj)
                    except Exception:
                        txt = s
                    if str(txt).strip():
                        items.append({"text": str(txt).strip(), "id": f"{fpath.stem}-{i+1}"})
        else:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for i, obj in enumerate(data):
                        if isinstance(obj, dict):
                            keys = ["text", "Tweet", "content", "message", "sentence"]
                            txt = next((obj[k] for k in keys if k in obj), str(obj))
                        else:
                            txt = str(obj)
                        if str(txt).strip():
                            items.append({"text": str(txt).strip(), "id": f"{fpath.stem}-{i+1}"})
                elif isinstance(data, dict):
                    for k, val in data.items():
                        if isinstance(val, str) and val.strip():
                            items.append({"text": val.strip(), "id": f"{fpath.stem}-{k}"})
            except Exception as e:
                print(f"  [Warning] Failed to parse JSON {fpath.name}: {e}")

    # 4. Subtitles (.srt, .vtt) — e.g. Choufli Hal / Tunisian TV dialogues
    elif ext in [".srt", ".vtt"]:
        import re
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content = re.sub(r"\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3}", "", content)
        content = re.sub(r"<[^>]+>", "", content)
        lines = [line.strip() for line in content.splitlines()]
        clean_lines = [l for l in lines if l and not l.isdigit()]
        print(f"  • Extracted {len(clean_lines):,} spoken dialogue lines from subtitle '{fpath.name}'")
        for i, l in enumerate(clean_lines):
            items.append({"text": l, "id": f"{fpath.stem}-{i+1}"})

    # 5. Web Pages (.html, .htm, .xml)
    elif ext in [".html", ".htm", ".xml"]:
        import re
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content = re.sub(r"<script[^>]*>.*?</script>", " ", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", " ", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;", " ", content)
        paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 15]
        print(f"  • Extracted {len(paragraphs):,} text blocks from HTML '{fpath.name}'")
        for i, p in enumerate(paragraphs):
            items.append({"text": p, "id": f"{fpath.stem}-{i+1}"})

    # 6. Plain Text & Markdown (.txt, .md, .markdown)
    elif ext in [".txt", ".md", ".markdown"]:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if s and not s.startswith("```"):
                    items.append({"text": s, "id": f"{fpath.stem}-{i+1}"})

    return items


def main():
    parser = argparse.ArgumentParser(description="Universal Data Cleaning, Normalization & Filtering for TounsiLM")
    parser.add_argument("--input", type=str, default="data/raw", help="Path to raw file or directory (supports Parquet, CSV, TSV, JSON, JSONL, Subtitles, HTML, Text, Markdown)")
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
        print("Please place your files in `data/raw/` or pass `--input <path>`.")
        sys.exit(1)

    # Universal file discovery
    patterns = ["*.parquet", "*.csv", "*.tsv", "*.jsonl", "*.json", "*.txt", "*.md", "*.markdown", "*.srt", "*.vtt", "*.html", "*.htm", "*.xml"]
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = []
        for pat in patterns:
            files_to_process.extend(input_path.glob(pat))
        files_to_process = sorted(list(set(files_to_process)))

    if not files_to_process:
        print(f"[Error] No supported data files found in: {input_path}")
        sys.exit(1)

    print(f"Discovered {len(files_to_process)} raw data file(s):")
    for f in files_to_process:
        print(f"  • {f.name} ({f.suffix})")

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
        raw_items = extract_raw_items_from_file(fpath)

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

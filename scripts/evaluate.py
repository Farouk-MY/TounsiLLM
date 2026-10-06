"""
Run TounsiEval benchmark suite on a trained TounsiLM checkpoint.

Measures:
  - 14 linguistic categories (Derja, Arabizi, Code-switching, Grammar, etc.)
  - Output token efficiency & fragmentation
  - Formats results into structured JSON and human-readable Markdown reports
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import EvalConfig
from tounsilm.evaluation.evaluator import ModelEvaluator
from tounsilm.inference.generator import TextGenerator
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.tokenizer.tokenizer import TounsiTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate TounsiLM on TounsiEval benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint-final", help="Path to checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/tounsilm_tokenizer.model")
    parser.add_argument("--benchmark-file", type=str, default="data/eval/tounsi_eval.jsonl")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    tok_path = Path(args.tokenizer_path)

    if not ckpt_dir.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_dir}")
        sys.exit(1)
    if not tok_path.exists():
        print(f"[Error] Tokenizer not found at: {tok_path}")
        sys.exit(1)

    print("=" * 60)
    print("      TounsiEval Benchmark Evaluation Suite")
    print("=" * 60)
    print(f"Model Checkpoint : {ckpt_dir}")
    print(f"Tokenizer Path   : {tok_path}")
    print(f"Benchmark File   : {args.benchmark_file}")
    print("-" * 60)

    generator = TextGenerator.from_checkpoint(ckpt_dir, tok_path)
    eval_cfg = EvalConfig(
        eval_dataset_path=args.benchmark_file,
        output_dir=args.output_dir,
    )
    evaluator = ModelEvaluator(generator=generator, config=eval_cfg)

    print("Running evaluation over 14 benchmark categories...")
    report = evaluator.evaluate_benchmark()

    md_report = evaluator.generate_markdown_report(report)
    md_file = Path(args.output_dir) / "latest_eval_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\n[OK] Evaluation completed! Evaluated {report['total_benchmark_samples']} items.")
    print(f"[OK] Markdown report saved to: {md_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

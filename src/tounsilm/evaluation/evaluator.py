"""
Evaluation runner for TounsiLM.

Runs TounsiEval benchmarks against a trained checkpoint, measures:
  - Perplexity on evaluation text
  - Generative responses for each of the 14 categories
  - Token fragmentation metrics
  - Human evaluation rubrics
  - Outputs structured JSON/Markdown reports
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tounsilm.config import EvalConfig
from tounsilm.evaluation.tounsi_eval import EvalSample, load_benchmark_file
from tounsilm.inference.generator import TextGenerator
from tounsilm.utils.metrics import compute_perplexity, compute_token_efficiency


class ModelEvaluator:
    """Evaluates a TounsiLM model checkpoint on TounsiEval."""

    def __init__(
        self,
        generator: TextGenerator,
        config: Optional[EvalConfig] = None,
    ):
        self.generator = generator
        self.config = config or EvalConfig()
        self.benchmark_samples = load_benchmark_file(self.config.eval_dataset_path)

    def evaluate_benchmark(self) -> Dict:
        """Run all test items in TounsiEval benchmark."""
        results = []
        category_counts = {}

        for sample in self.benchmark_samples:
            cat = sample.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

            generated = self.generator.generate(
                prompt=sample.input,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            # Measure token efficiency on output
            char_len = len(generated)
            tok_len = len(self.generator.tokenizer.encode(generated, add_bos=False, add_eos=False))
            eff = compute_token_efficiency(char_len, tok_len)

            results.append({
                "id": sample.id,
                "category": sample.category,
                "input": sample.input,
                "expected_behavior": sample.expected_behavior,
                "reference_answer": sample.reference_answer,
                "model_output": generated,
                "char_length": char_len,
                "token_length": tok_len,
                "chars_per_token": eff,
            })

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_name": "TounsiLM-300M",
            "total_benchmark_samples": len(results),
            "categories": category_counts,
            "results": results,
            "human_evaluation_rubric": {
                "tunisian_naturalness": "1-5 rating on colloquial authenticity",
                "correctness": "1-5 rating on task/prompt obedience",
                "meaning_preservation": "1-5 rating on semantic clarity",
                "grammar": "1-5 rating on Derja grammatical cohesion",
                "dialect_adherence": "1-5 rating on avoiding foreign drift without cause",
                "fluency": "1-5 rating on overall flow",
            },
        }

        # Save report
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def generate_markdown_report(self, report_dict: Dict) -> str:
        """Format the evaluation results into a Markdown report."""
        lines = [
            "# TounsiEval Benchmark Report",
            f"**Model**: {report_dict.get('model_name', 'TounsiLM')}",
            f"**Timestamp**: {report_dict.get('timestamp', '')}",
            f"**Total Samples**: {report_dict.get('total_benchmark_samples', 0)}",
            "",
            "## Category Breakdown",
            "| Category | Samples |",
            "| :--- | :--- |",
        ]
        for cat, cnt in report_dict.get("categories", {}).items():
            lines.append(f"| {cat} | {cnt} |")

        lines.extend([
            "",
            "## Sample Generations",
            "| ID | Category | Prompt | Model Output |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for item in report_dict.get("results", [])[:10]:
            clean_in = item["input"].replace("\n", " ")
            clean_out = item["model_output"].replace("\n", " ")
            lines.append(f"| {item['id']} | {item['category']} | {clean_in} | {clean_out} |")

        return "\n".join(lines)

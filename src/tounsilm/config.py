"""
Central configuration system for TounsiLM.

All model, training, data, and tokenizer configurations are defined as
dataclasses. Configurations can be loaded from YAML files and overridden
via command-line arguments.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml
import torch


# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for the TounsiLM Transformer architecture."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 2304
    max_sequence_length: int = 512
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    rope_base: float = 10000.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.max_sequence_length > 0, "max_sequence_length must be positive"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        assert 0.0 <= self.attention_dropout <= 1.0, "attention_dropout must be in [0, 1]"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuration for pretraining."""

    # Precision and memory
    precision: str = "bf16"  # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 32
    micro_batch_size: int = 1

    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 500
    max_steps: int = 100000

    # Checkpointing
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 10
    max_checkpoints: int = 5

    # Data
    sequence_length: int = 512
    num_workers: int = 2

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "tounsilm-300m"

    # Seed
    seed: int = 42

    # Distributed
    distributed: bool = False

    def validate(self) -> None:
        """Validate training configuration."""
        assert self.precision in ("fp32", "fp16", "bf16"), (
            f"precision must be one of fp32, fp16, bf16; got {self.precision}"
        )
        assert self.micro_batch_size >= 1, "micro_batch_size must be >= 1"
        assert self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be >= 1"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_steps > 0, "max_steps must be positive"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def resolve_precision(self) -> torch.dtype:
        """Resolve precision string to torch dtype, with hardware fallback."""
        if self.precision == "bf16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                print(
                    "[WARNING] BF16 not supported on this hardware. "
                    "Falling back to FP16."
                )
                self.precision = "fp16"
                return torch.float16
        elif self.precision == "fp16":
            return torch.float16
        else:
            return torch.float32


# ──────────────────────────────────────────────
# Tokenizer Configuration
# ──────────────────────────────────────────────

@dataclass
class TokenizerConfig:
    """Configuration for tokenizer training."""

    vocab_size: int = 32000
    model_type: str = "bpe"  # "bpe" or "unigram"
    character_coverage: float = 0.9999
    max_sentence_length: int = 16384
    num_threads: int = 4
    input_sentence_size: int = 5_000_000
    shuffle_input_sentence: bool = True
    seed_sentencepiece_size: int = 1_000_000

    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"

    # User-defined symbols
    user_defined_symbols: list[str] = field(default_factory=lambda: [
        "<|im_start|>",
        "<|im_end|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ])

    # Paths
    output_dir: str = "data/tokenizer"
    model_prefix: str = "tounsilm_tokenizer"

    def validate(self) -> None:
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.model_type in ("bpe", "unigram"), (
            f"model_type must be 'bpe' or 'unigram'; got {self.model_type}"
        )


# ──────────────────────────────────────────────
# Data Configuration
# ──────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration for data pipeline."""

    raw_dir: str = "data/raw"
    intermediate_dir: str = "data/intermediate"
    processed_dir: str = "data/processed"
    manifests_dir: str = "data/manifests"

    # Splitting
    train_ratio: float = 0.98
    val_ratio: float = 0.01
    test_ratio: float = 0.01

    # Filtering
    min_doc_length: int = 10
    max_doc_length: int = 100000
    max_symbol_ratio: float = 0.3
    max_url_ratio: float = 0.3
    max_repeated_char_ratio: float = 0.4
    min_tunisian_score: float = 0.1

    # Deduplication
    exact_dedup: bool = True
    near_dedup: bool = True
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.8
    minhash_ngram_size: int = 5

    # PII filtering
    pii_filter: bool = True

    # Contamination
    eval_exclusion_file: Optional[str] = None

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, (
            f"Split ratios must sum to 1.0; got {total}"
        )


# ──────────────────────────────────────────────
# Evaluation Configuration
# ──────────────────────────────────────────────

@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    eval_dataset_path: str = "data/eval/tounsi_eval.jsonl"
    batch_size: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    output_dir: str = "evaluation_results"

    def validate(self) -> None:
        assert self.batch_size >= 1


# ──────────────────────────────────────────────
# Full Project Configuration
# ──────────────────────────────────────────────

@dataclass
class TounsiLMConfig:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        """Validate all sub-configurations."""
        self.model.validate()
        self.training.validate()
        self.tokenizer.validate()
        self.data.validate()
        self.evaluation.validate()

        # Cross-config consistency
        assert self.model.vocab_size == self.tokenizer.vocab_size, (
            f"Model vocab_size ({self.model.vocab_size}) must match "
            f"tokenizer vocab_size ({self.tokenizer.vocab_size})"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────
# Configuration I/O
# ──────────────────────────────────────────────

def load_config(path: str | Path) -> TounsiLMConfig:
    """Load a TounsiLMConfig from a YAML file.

    The YAML file may contain any subset of the configuration fields.
    Missing fields use their default values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = TounsiLMConfig()

    if "model" in raw:
        config.model = ModelConfig(**{
            k: v for k, v in raw["model"].items()
            if k in ModelConfig.__dataclass_fields__
        })

    if "training" in raw:
        config.training = TrainingConfig(**{
            k: v for k, v in raw["training"].items()
            if k in TrainingConfig.__dataclass_fields__
        })

    if "tokenizer" in raw:
        config.tokenizer = TokenizerConfig(**{
            k: v for k, v in raw["tokenizer"].items()
            if k in TokenizerConfig.__dataclass_fields__
        })

    if "data" in raw:
        config.data = DataConfig(**{
            k: v for k, v in raw["data"].items()
            if k in DataConfig.__dataclass_fields__
        })

    if "evaluation" in raw:
        config.evaluation = EvalConfig(**{
            k: v for k, v in raw["evaluation"].items()
            if k in EvalConfig.__dataclass_fields__
        })

    config.validate()
    return config


def save_config(config: TounsiLMConfig, path: str | Path) -> None:
    """Save a TounsiLMConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_system_info() -> dict[str, Any]:
    """Collect system information for reproducibility."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        info["bf16_supported"] = torch.cuda.is_bf16_supported()

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        info["git_commit"] = "N/A"

    return info

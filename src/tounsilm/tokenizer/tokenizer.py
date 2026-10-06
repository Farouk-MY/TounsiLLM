"""
TounsiLM Tokenizer — trained from scratch on Tunisian Arabic corpus.

Wraps SentencePiece BPE tokenizer with Tunisian-aware handling for:
  - Arabic-script Tunisian Derja
  - Arabizi (Latin-script Tunisian)
  - Tunisian/French code-switching
  - English technical terms
  - Special tokens for instruction tuning

The tokenizer is trained on the project's own corpus — no pretrained
tokenizer from another model is used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm

from tounsilm.config import TokenizerConfig


class TounsiTokenizer:
    """SentencePiece BPE tokenizer for Tunisian Arabic.

    This tokenizer is trained from scratch using the project's corpus.
    It handles Arabic script, Arabizi, French code-switching, and
    English technical vocabulary.
    """

    # Special token IDs (SentencePiece defaults)
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, model_path: Optional[str | Path] = None):
        """Initialize tokenizer, optionally loading a trained model.

        Args:
            model_path: Path to a trained .model file. If None, the
                       tokenizer must be trained before use.
        """
        self.sp: Optional[spm.SentencePieceProcessor] = None
        self.model_path: Optional[Path] = None

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str | Path) -> None:
        """Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        self.model_path = model_path

    @classmethod
    def train(
        cls,
        input_files: list[str | Path],
        config: TokenizerConfig,
    ) -> "TounsiTokenizer":
        """Train a new SentencePiece BPE tokenizer.

        Args:
            input_files: List of text files to train on.
            config: Tokenizer configuration.

        Returns:
            Trained TounsiTokenizer instance.
        """
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_prefix = str(output_dir / config.model_prefix)
        input_str = ",".join(str(f) for f in input_files)

        # Build user_defined_symbols string
        user_symbols = ",".join(config.user_defined_symbols) if config.user_defined_symbols else ""

        print(f"[Tokenizer] Training SentencePiece {config.model_type.upper()} tokenizer...")
        print(f"[Tokenizer] Vocab size: {config.vocab_size}")
        print(f"[Tokenizer] Input files: {input_str}")

        train_kwargs = {
            "input": [str(f) for f in input_files],
            "model_prefix": model_prefix,
            "vocab_size": config.vocab_size,
            "model_type": config.model_type,
            "character_coverage": config.character_coverage,
            "max_sentence_length": config.max_sentence_length,
            "num_threads": config.num_threads,
            "input_sentence_size": config.input_sentence_size,
            "shuffle_input_sentence": config.shuffle_input_sentence,
            "seed_sentencepiece_size": config.seed_sentencepiece_size,
            "pad_id": cls.PAD_ID,
            "unk_id": cls.UNK_ID,
            "bos_id": cls.BOS_ID,
            "eos_id": cls.EOS_ID,
            "pad_piece": config.pad_token,
            "unk_piece": config.unk_token,
            "bos_piece": config.bos_token,
            "eos_piece": config.eos_token,
            "byte_fallback": True,
            "split_digits": True,
            "allow_whitespace_only_pieces": True,
            "normalization_rule_name": "identity",
        }
        if config.user_defined_symbols:
            train_kwargs["user_defined_symbols"] = config.user_defined_symbols

        spm.SentencePieceTrainer.train(**train_kwargs)

        print(f"[Tokenizer] Model saved to: {model_prefix}.model")

        tokenizer = cls()
        tokenizer.load(f"{model_prefix}.model")
        return tokenizer

    def _ensure_loaded(self) -> None:
        """Ensure the tokenizer model is loaded."""
        if self.sp is None:
            raise RuntimeError(
                "Tokenizer is not loaded. Either train a new tokenizer with "
                "TounsiTokenizer.train() or load an existing one with "
                "TounsiTokenizer(model_path='path/to/model.model')"
            )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.

        Returns:
            List of token IDs.
        """
        self._ensure_loaded()
        ids = self.sp.Encode(text)
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        self._ensure_loaded()
        return self.sp.Decode(ids)

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """Encode a batch of texts.

        Args:
            texts: List of text strings.
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.

        Returns:
            List of token ID lists.
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(self, ids_list: list[list[int]]) -> list[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(ids) for ids in ids_list]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into subword pieces (strings).

        Args:
            text: Input text.

        Returns:
            List of subword strings.
        """
        self._ensure_loaded()
        return self.sp.EncodeAsPieces(text)

    def id_to_piece(self, token_id: int) -> str:
        """Convert a single token ID to its string piece."""
        self._ensure_loaded()
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Convert a string piece to its token ID."""
        self._ensure_loaded()
        return self.sp.PieceToId(piece)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        self._ensure_loaded()
        return self.sp.GetPieceSize()

    @property
    def bos_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def unk_id(self) -> int:
        return self.UNK_ID

    def inspect(self, text: str) -> dict:
        """Inspect tokenization of a text string.

        Returns a dictionary with detailed tokenization analysis.
        """
        self._ensure_loaded()
        token_ids = self.encode(text, add_bos=False, add_eos=False)
        pieces = self.tokenize(text)
        decoded = self.decode(token_ids)

        return {
            "original": text,
            "token_ids": token_ids,
            "pieces": pieces,
            "decoded": decoded,
            "num_tokens": len(token_ids),
            "num_chars": len(text),
            "chars_per_token": len(text) / max(len(token_ids), 1),
            "roundtrip_match": decoded.strip() == text.strip(),
        }

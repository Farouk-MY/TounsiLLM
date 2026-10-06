"""
PyTorch Dataset for TounsiLM pretraining.

Implements:
  - TounsiLMDataset: Memory-mapped dataset of packed token sequences
  - create_packed_dataset: Tokenize and pack documents into fixed-length chunks
  - Collation and padding utilities

Documents are packed into continuous token streams, with EOS tokens
separating documents. This maximizes GPU utilization by avoiding padding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TounsiLMDataset(Dataset):
    """Memory-mapped dataset of packed token sequences for causal LM training.

    Each item is a fixed-length sequence of token IDs. Documents are
    concatenated with EOS separation and split into chunks of seq_len.

    The dataset uses numpy memmap for memory-efficient access to large
    token arrays without loading everything into RAM.
    """

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        dtype: np.dtype = np.uint16,
    ):
        """
        Args:
            data_path: Path to the .bin file containing packed token IDs.
            seq_len: Sequence length for each training example.
            dtype: NumPy dtype of the stored tokens (uint16 for vocab < 65536).
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        self.seq_len = seq_len
        self.data = np.memmap(str(data_path), dtype=dtype, mode="r")
        self.num_tokens = len(self.data)

        # Number of complete sequences we can form
        # Each example needs seq_len + 1 tokens (input + shifted label)
        self.num_examples = max(0, (self.num_tokens - 1) // seq_len)

        if self.num_examples == 0:
            raise ValueError(
                f"Dataset has {self.num_tokens} tokens but needs at least "
                f"{seq_len + 1} for seq_len={seq_len}. "
                f"Add more data or reduce sequence length."
            )

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training example.

        Returns:
            Dictionary with:
              - input_ids: Token IDs [seq_len]
              - labels: Target IDs [seq_len] (shifted by one position)
        """
        if idx < 0 or idx >= self.num_examples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_examples})")

        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for the label shift

        # Ensure we don't go past the end
        end = min(end, self.num_tokens)
        chunk = self.data[start:end].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())

        # Pad if we hit the end of data
        if len(input_ids) < self.seq_len:
            pad_len = self.seq_len - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.zeros(pad_len, dtype=torch.long),
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])

        return {"input_ids": input_ids, "labels": labels}

    @property
    def total_tokens(self) -> int:
        return self.num_tokens

    def close(self) -> None:
        """Close memmap to release underlying file handle."""
        if hasattr(self, "data") and self.data is not None:
            if hasattr(self.data, "_mmap") and self.data._mmap is not None:
                try:
                    self.data._mmap.close()
                except Exception:
                    pass
            self.data = None

    def __del__(self) -> None:
        self.close()


def create_packed_dataset(
    documents: list[str],
    tokenizer,
    output_path: str | Path,
    dtype: np.dtype = np.uint16,
) -> dict:
    """Tokenize documents and pack them into a binary file.

    Documents are tokenized, concatenated with EOS separation, and
    written as a flat numpy array for memory-mapped access.

    Args:
        documents: List of text documents.
        tokenizer: TounsiTokenizer instance.
        output_path: Path to write the .bin file.
        dtype: NumPy dtype for token storage.

    Returns:
        Dictionary with dataset statistics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    total_docs = 0

    for doc in documents:
        if not doc.strip():
            continue

        # Encode with BOS and EOS
        tokens = tokenizer.encode(doc, add_bos=True, add_eos=True)
        all_tokens.extend(tokens)
        total_docs += 1

    if not all_tokens:
        raise ValueError("No tokens produced. Check your documents and tokenizer.")

    # Convert to numpy array
    token_array = np.array(all_tokens, dtype=dtype)

    # Check that all token IDs fit in the dtype
    max_id = token_array.max()
    if dtype == np.uint16 and max_id > 65535:
        raise ValueError(
            f"Token ID {max_id} exceeds uint16 range. "
            f"Use uint32 or reduce vocab size."
        )

    # Write to disk
    token_array.tofile(str(output_path))

    stats = {
        "num_documents": total_docs,
        "num_tokens": len(all_tokens),
        "file_size_bytes": output_path.stat().st_size,
        "dtype": str(dtype),
        "output_path": str(output_path),
    }

    # Write metadata alongside
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats

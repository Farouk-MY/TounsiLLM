"""
Deduplication for TounsiLM data pipeline.

Implements:
  - ExactDeduplicator: Hash-based exact duplicate removal
  - MinHashDeduplicator: Locality-sensitive hashing for near-duplicate detection

Uses xxhash for fast hashing and datasketch for MinHash/LSH.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import xxhash

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

from tounsilm.config import DataConfig


class ExactDeduplicator:
    """Hash-based exact duplicate removal.

    Uses xxhash for fast document-level hashing. Documents that
    produce the same hash (after normalization) are considered duplicates.
    """

    def __init__(self):
        self.seen_hashes: set[str] = set()
        self.total_seen = 0
        self.duplicates_found = 0

    def _hash_text(self, text: str) -> str:
        """Compute xxhash of normalized text."""
        # Normalize whitespace for hashing
        normalized = " ".join(text.split())
        return xxhash.xxh64(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if a document is a duplicate.

        Args:
            text: Document text.

        Returns:
            True if this exact text has been seen before.
        """
        self.total_seen += 1
        h = self._hash_text(text)
        if h in self.seen_hashes:
            self.duplicates_found += 1
            return True
        self.seen_hashes.add(h)
        return False

    def deduplicate(self, documents: list[str]) -> tuple[list[str], dict]:
        """Remove exact duplicates from a list of documents.

        Args:
            documents: List of document strings.

        Returns:
            Tuple of (deduplicated_documents, report_dict).
        """
        self.seen_hashes.clear()
        self.total_seen = 0
        self.duplicates_found = 0

        unique = []
        for doc in documents:
            if not self.is_duplicate(doc):
                unique.append(doc)

        report = {
            "documents_before": len(documents),
            "documents_removed": self.duplicates_found,
            "documents_remaining": len(unique),
            "duplicate_percentage": (
                self.duplicates_found / max(len(documents), 1) * 100
            ),
        }

        return unique, report


class MinHashDeduplicator:
    """Near-duplicate detection using MinHash + LSH.

    Uses character n-gram shingling to create MinHash signatures,
    then uses Locality-Sensitive Hashing to find similar document pairs.

    Does not remove all similar Tunisian sentences just because they
    share vocabulary — the similarity threshold is configurable.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5,
    ):
        """
        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate).
            threshold: Jaccard similarity threshold for near-duplicate detection.
            ngram_size: Character n-gram size for shingling.
        """
        if not HAS_DATASKETCH:
            raise ImportError(
                "datasketch is required for MinHash deduplication. "
                "Install it with: pip install datasketch"
            )

        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size

    def _create_minhash(self, text: str) -> MinHash:
        """Create a MinHash signature for a document."""
        m = MinHash(num_perm=self.num_perm)
        # Character n-gram shingling
        normalized = " ".join(text.split())
        for i in range(max(0, len(normalized) - self.ngram_size + 1)):
            shingle = normalized[i:i + self.ngram_size]
            m.update(shingle.encode("utf-8"))
        return m

    def deduplicate(self, documents: list[str]) -> tuple[list[str], dict]:
        """Remove near-duplicates from a list of documents.

        Args:
            documents: List of document strings.

        Returns:
            Tuple of (deduplicated_documents, report_dict).
        """
        if len(documents) <= 1:
            return documents, {
                "documents_before": len(documents),
                "documents_removed": 0,
                "documents_remaining": len(documents),
                "duplicate_percentage": 0.0,
            }

        # Create LSH index
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        minhashes = []
        duplicate_indices = set()

        for i, doc in enumerate(documents):
            mh = self._create_minhash(doc)
            minhashes.append(mh)

            # Check for near-duplicates
            try:
                result = lsh.query(mh)
                if result:
                    duplicate_indices.add(i)
                else:
                    lsh.insert(str(i), mh)
            except ValueError:
                # Key already exists (shouldn't happen with unique str(i))
                duplicate_indices.add(i)

        unique = [
            doc for i, doc in enumerate(documents)
            if i not in duplicate_indices
        ]

        report = {
            "documents_before": len(documents),
            "documents_removed": len(duplicate_indices),
            "documents_remaining": len(unique),
            "duplicate_percentage": (
                len(duplicate_indices) / max(len(documents), 1) * 100
            ),
        }

        return unique, report

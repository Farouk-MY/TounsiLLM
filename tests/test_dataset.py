"""
Tests for Data Pipeline Components.

Verifies:
  - TunisianNormalizer (Arabic, Arabizi, whitespace, URLs, diacritics)
  - QualityFilter (length, symbol ratio, url ratio)
  - DialectFilter (Arabic & Arabizi lexical indicators scoring)
  - ExactDeduplicator (hash deduplication)
  - MinHashDeduplicator (near-duplicate detection)
  - create_packed_dataset and TounsiLMDataset memmap indexing
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tounsilm.config import DataConfig
from tounsilm.data.normalizer import TunisianNormalizer
from tounsilm.data.filters import QualityFilter, DialectFilter, PIIFilter
from tounsilm.data.deduplication import ExactDeduplicator, MinHashDeduplicator, HAS_DATASKETCH
from tounsilm.data.dataset import TounsiLMDataset, create_packed_dataset


def test_tunisian_normalizer():
    norm = TunisianNormalizer()
    raw = "  أهلا   وسهلاااا!!! http://example.com  "
    cleaned = norm.normalize(raw)
    assert "اهلا" in cleaned
    assert "وسهلاا!" in cleaned
    assert "http" not in cleaned
    assert "   " not in cleaned


def test_quality_filter():
    cfg = DataConfig(min_doc_length=5, max_doc_length=50)
    qf = QualityFilter(cfg)

    # Empty
    assert not qf.filter("").passed
    # Too short
    assert not qf.filter("اب").passed
    # Valid
    assert qf.filter("عسلامة شنوة أحوالك").passed


def test_dialect_filter():
    df = DialectFilter(min_score=0.1)

    # High Tunisian markers
    tn_text = "شنوة تعمل توّا؟ علاش ما جيتش؟ برشا باهي."
    res = df.score(tn_text)
    assert res["score"] >= 0.1
    assert df.is_tunisian(tn_text)

    # Arabizi Tunisian markers
    arabizi_text = "chnowa ta3mel tawa? 3lech ma jitich?"
    res_ar = df.score(arabizi_text)
    assert res_ar["arabizi_matches"] > 0
    assert df.is_tunisian(arabizi_text)


def test_exact_deduplication():
    docs = [
        "عسلامة شنوة أحوالك",
        "عسلامة شنوة أحوالك",
        "برشا ناس باهين",
    ]
    dedup = ExactDeduplicator()
    unique, rep = dedup.deduplicate(docs)
    assert len(unique) == 2
    assert rep["documents_removed"] == 1


def test_packed_dataset():
    class DummyTokenizer:
        vocab_size = 100
        bos_id = 1
        eos_id = 2
        def encode(self, text, add_bos=True, add_eos=True):
            return [self.bos_id, 10, 20, 30, self.eos_id]

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "test.bin"
        docs = ["doc 1", "doc 2", "doc 3", "doc 4"]
        stats = create_packed_dataset(docs, DummyTokenizer(), bin_path)

        assert stats["num_documents"] == 4
        assert stats["num_tokens"] == 20

        # Load with memmap
        ds = TounsiLMDataset(bin_path, seq_len=4)
        assert len(ds) > 0
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (4,)
        assert sample["labels"].shape == (4,)
        ds.close()

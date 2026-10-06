"""
Tests for TounsiLM Tokenizer.

Verifies:
  - Tokenizer training from scratch on Tunisian text
  - Encoding and decoding roundtrip
  - Special tokens handling (BOS, EOS, PAD, UNK)
  - Arabic-script Derja processing
  - Arabizi numerals and character processing
  - Code-switching and technical terms
  - Subword piece tokenization
"""

import tempfile
from pathlib import Path

import pytest

from tounsilm.config import TokenizerConfig
from tounsilm.tokenizer.tokenizer import TounsiTokenizer


@pytest.fixture(scope="module")
def sample_corpus_file():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as f:
        corpus = [
            "عسلامة، شنوة أحوالك اليوم؟ انشالله تكون لاباس.",
            "علاش ما جيتش البارح؟ برشا ناس كانو يستناو فيك.",
            "3asslema chnowa a7welek labes 3lik?",
            "3lech ma jitich lbare7 lel 9ahwa?",
            "barsha ness y7ebbou l9ahwa l3arbi.",
            "لازم نعمل update للـapplication متاعنا.",
            "توّا عندي réunion مهمة مع الـclient.",
            "ما نجمش نبعث الـdocument اليوم خاطر فما مشكل.",
            "باش نعمل deploy للـcode بعد شوية.",
            "API database mobile application machine learning NLP Transformer.",
        ] * 30
        f.write("\n".join(corpus) + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def trained_tokenizer(sample_corpus_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TokenizerConfig(
            vocab_size=600,  # Accommodates base characters (328) + subwords
            output_dir=tmpdir,
            model_prefix="test_tok",
        )
        tok = TounsiTokenizer.train([sample_corpus_file], cfg)
        yield tok


def test_tokenizer_initialization_and_vocab_size(trained_tokenizer):
    assert trained_tokenizer.vocab_size == 600
    assert trained_tokenizer.bos_id == 2
    assert trained_tokenizer.eos_id == 3
    assert trained_tokenizer.pad_id == 0
    assert trained_tokenizer.unk_id == 1


def test_encode_decode_roundtrip_arabic(trained_tokenizer):
    text = "عسلامة شنوة أحوالك؟"
    ids = trained_tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == trained_tokenizer.bos_id
    assert ids[-1] == trained_tokenizer.eos_id
    decoded = trained_tokenizer.decode(ids)
    assert "عسلامة" in decoded
    assert "شنوة" in decoded


def test_encode_decode_roundtrip_arabizi(trained_tokenizer):
    text = "3asslema chnowa a7welek?"
    ids = trained_tokenizer.encode(text, add_bos=False, add_eos=False)
    assert len(ids) > 0
    decoded = trained_tokenizer.decode(ids)
    assert "3asslema" in decoded


def test_encode_decode_mixed_code_switching(trained_tokenizer):
    text = "عندي réunion مهمة و باش نعمل deploy."
    ids = trained_tokenizer.encode(text, add_bos=False, add_eos=False)
    decoded = trained_tokenizer.decode(ids)
    assert "réunion" in decoded
    assert "deploy" in decoded


def test_inspect_utility(trained_tokenizer):
    info = trained_tokenizer.inspect("برشا باهي")
    assert "token_ids" in info
    assert "pieces" in info
    assert "chars_per_token" in info
    assert info["num_tokens"] > 0

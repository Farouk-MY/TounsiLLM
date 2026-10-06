# TounsiLM Dataset Pipeline & Methodology

## 1. Dialect & Linguistic Context

Tunisian Arabic (**Derja**, ISO 639-3 `aeb`) is a Maghrebi Arabic dialect spoken by ~12 million people in Tunisia and the diaspora. It is characterized by:
- Diglossia and code-switching with French
- Frequent use of **Arabizi** (Latin orthography with numbers representing Arabic phonemes: `3`=ع, `7`=ح, `5`=خ, `9`=ق, `8`=غ)
- Distinct verbal morphology (e.g., circumfixal negation `ما ... ش` / `ma-...-ch`)
- A rich substratum of Berber, Latin, Romance, and Ottoman Turkish lexical influences.

---

## 2. Pipeline Stages

The data engineering pipeline enforces strict reproducibility and non-destructive processing:

```text
Raw Corpus (data/raw)
       │
       ▼
Tunisian Normalizer (Arabic Unicode NFC, Alef unification, Tatweel stripping, whitespace, repetition)
       │
       ▼
PII Sanitization (Masking emails, Tunisian phone numbers, credit card sequences)
       │
       ▼
Quality Filtering (Length bounds, symbol ratio ≤ 0.3, URL ratio ≤ 0.3, repetitive characters ≤ 0.4)
       │
       ▼
Dialect Filtering (Lexical score based on authentic Tunisian Arabic & Arabizi markers)
       │
       ▼
Contamination Check (Exclusion of any overlap with TounsiEval benchmark inputs)
       │
       ▼
Exact Deduplication (xxhash 64-bit document hashing)
       │
       ▼
Near-Duplicate Deduplication (MinHash with 128 permutations, LSH threshold 0.85)
       │
       ▼
Deterministic Split (Train 98%, Val 1%, Test 1%, Seed=42)
       │
       ▼
Packed Binary Dataset (Continuous token stream with BOS/EOS into numpy uint16 memmap)
```

---

## 3. Preserving Dialect Variation vs Blind Normalization

Standard Arabic NLP pipelines often apply aggressive normalization (e.g., converting all `ي` to `ى`, collapsing `ة` and `ه`, or stripping all numbers). **This would destroy Tunisian dialect structure.**
In TounsiLM:
1. **Arabizi Numbers Preserved**: Digits `3, 5, 7, 8, 9` are retained because they represent distinct Arabic phonemes in chat text.
2. **Short Dialogues Preserved**: Short expressions like `شنوة؟`, `علاش؟`, `باهي.`, `لاباس؟` are permitted (minimum document length is conservative: 10 characters).
3. **Punctuation and Question Marks**: `؟` and `?` are treated with equal validity.

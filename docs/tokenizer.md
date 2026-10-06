# TounsiLM Tokenizer Engineering

## 1. Vocabulary Design & Rationale

Existing foundation model tokenizers (LLaMA, Qwen, Mistral, Gemma) were trained primarily on English and standard Modern Standard Arabic (MSA). When applied to Tunisian Arabic, they suffer from severe **subword fragmentation**:
- A single Derja word like `مستغربين` or `مافيهاش` is split into 4–6 micro-tokens or even individual byte tokens.
- Arabizi sentences with numbers (`3lech`, `bel7a9`, `5ouya`) are degraded into single character pieces, exponentially inflating sequence lengths and destroying attention context.

**TounsiLM Tokenizer** is trained completely from scratch using **SentencePiece BPE**:
- **Vocabulary Size**: 32,000 subwords
- **Character Coverage**: 99.99% with byte fallback enabled for rare Unicode characters
- **Normalization Rule**: `identity` (does not apply NFKC to strip Arabic diacritics or alter Arabizi numbers)
- **Digits**: Split digits enabled for robust numeric comprehension in both dates and Arabizi

---

## 2. Special Tokens & Chat Format

The vocabulary reserves dedicated token IDs for chat templates and instruction tuning:

| Token | ID | Usage |
| :--- | :--- | :--- |
| `<pad>` | 0 | Padding sequence chunks |
| `<unk>` | 1 | Out-of-vocabulary fallback |
| `<s>` | 2 | Beginning-of-sequence (BOS) |
| `</s>` | 3 | End-of-sequence (EOS) |
| `<|im_start|>` | Custom | ChatML turn starter |
| `<|im_end|>` | Custom | ChatML turn terminator |
| `<|user|>` | Custom | User instruction role |
| `<|assistant|>` | Custom | Assistant response role |
| `<|system|>` | Custom | System prompt role |

---

## 3. Compression & Fragmentation Audit

Using `scripts/inspect_tokenizer.py`, tokenization quality is verified across four core domains:
1. **Arabic Derja**: ~1.4 to 1.8 characters per token for high frequency roots (`شنوة`, `علاش`, `برشا`, `باهي`).
2. **Arabizi**: ~1.5 to 2.0 characters per token without breaking common bigrams (`3asslema`, `tawa`, `chneya`).
3. **French/Tech Code-Switching**: Retains technical terms (`reunion`, `deploy`, `application`, `update`) as single or double tokens.
4. **Fidelity**: Complete $100\%$ lossless round-trip reconstructibility ($\text{decode}(\text{encode}(x)) \equiv x$).

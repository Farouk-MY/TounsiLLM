# TounsiLM — Tunisian Arabic Language Model From Scratch

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-orange.svg)](docs/architecture.md)
[![Parameters](https://img.shields.io/badge/Parameters-~300M_(303.4M)-purple.svg)](scripts/estimate_parameters.py)
[![Dialect](https://img.shields.io/badge/Dialect-Tunisian_Derja_%26_Arabizi-red.svg)](docs/dataset.md)

**TounsiLM** is an open-source research and engineering effort building a **Tunisian Arabic (Derja) Large Language Model entirely from scratch**. It has approximately **300 million parameters** (~303.4M), trained from **random initialization** without importing or initializing from pretrained model weights (no LLaMA, Qwen, Mistral, Gemma, GPT, BERT, or AraBERT).

---

## Architecture Flow

```text
       Tunisian Corpus (Arabic Script Derja, Arabizi, French/Tech Code-Switching)
                                     │
                                     ▼
                    Linguistic Normalization & Quality Filters
                                     │
                                     ▼
                      Exact & MinHash LSH Deduplication
                                     │
                                     ▼
                     Tunisian SentencePiece BPE Tokenizer
                                (32,000 Vocab)
                                     │
                                     ▼
               TounsiLM-300M Transformer (Random Initialization)
                 [24 Layers | 1024 Hidden | 16 Heads | SwiGLU]
                                     │
                                     ▼
                    Causal Language Model Pretraining
                                     │
                                     ▼
                            TounsiLM-300M-Base
                                     │
                                     ▼
                   Instruction Tuning (ChatML Prompts)
                                     │
                                     ▼
                           TounsiLM-300M-Instruct
```

---

## 1. Why Build a Tunisian Model from Scratch?

Tunisian Arabic (**Derja**, ISO 639-3 `aeb`) is used by over 12 million speakers. Standard LLMs fail to handle it authentically because:
1. **Severe Tokenizer Fragmentation**: Standard English/MSA tokenizers segment common Derja words into 4–6 subword fragments.
2. **Arabizi Neglect**: Digital communication heavily relies on Arabizi (Latin script with numbers: `3`=ع, `7`=ح, `5`=خ, `9`=ق), which foreign tokenizers destroy.
3. **Code-Switching**: Tunisians naturally interweave Derja, French, and English in professional and technical contexts.

TounsiLM is designed from the ground up specifically for these linguistic realities.

---

## 2. Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Number of Layers** | **24** | Pre-norm Transformer Decoder Blocks |
| **Hidden Size** | **1024** | Model representation dimensionality |
| **Attention Heads** | **16** | Head dimension $d_k = 64$ |
| **Intermediate Size** | **2304** | SwiGLU Feed-Forward expansion ($2.25 \times H$) |
| **Vocabulary Size** | **32,000** | Specialized Tunisian subword tokens |
| **Max Sequence Length** | **512 / 2048** | 512 for local dev; 2048 for cloud clusters |
| **Positional Embeddings** | **RoPE** | Rotary Position Embeddings (base 10,000) |
| **Normalization** | **RMSNorm** | Root Mean Square Normalization ($\epsilon = 10^{-6}$) |
| **Feed-Forward Network** | **SwiGLU** | Gated linear unit with SiLU activation |
| **Weight Tying** | **True** | Tied input embedding and output LM head |
| **Total Parameters** | **303,350,784** | **+1.12% deviation from 300M target** |

---

## 3. Hardware Support

### Local Mode (6–8 GB GPU)
- Mixed Precision (automatic BF16/FP16 fallback)
- Gradient Checkpointing (saves ~70% activation VRAM)
- Micro-batch size: 1, Gradient accumulation steps: 32
- Sequence length: 256 or 512

### Cloud Mode
- Distributed Multi-GPU training
- Sequence length: 2048
- High throughput data workers

---

## 4. Quickstart

### Installation
```bash
git clone https://github.com/tounsilm/tounsilm.git
cd tounsilm
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/ -v
```

### Running the End-to-End Pipeline
```bash
# 1. Prepare raw curated corpus
python scripts/prepare_data.py

# 2. Clean, normalize, and filter PII
python scripts/clean_data.py

# 3. Deduplicate (Exact + MinHash LSH) and split
python scripts/deduplicate.py

# 4. Train Tunisian tokenizer from scratch
python scripts/train_tokenizer.py

# 5. Inspect tokenization quality
python scripts/inspect_tokenizer.py

# 6. Audit model parameters
python scripts/estimate_parameters.py

# 7. Initialize model & verify autograd mechanics
python scripts/initialize_model.py

# 8. Overfitting sanity check (Phase 5)
python scripts/pretrain.py --overfit-test

# 9. Smoke pretraining run (Phase 6)
python scripts/pretrain.py --smoke-test

# 10. Run pretraining
python scripts/pretrain.py --config configs/pretraining_local.yaml

# 11. Evaluate on TounsiEval benchmark
python scripts/evaluate.py --checkpoint checkpoints/checkpoint-final

# 12. Generate text
python scripts/generate.py --checkpoint checkpoints/checkpoint-final --prompt "شنوة تعمل توّا؟"
```

---

## 5. TounsiEval Benchmark

Evaluation is structured across 14 distinct categories:
1. Tunisian vocabulary
2. Tunisian grammar
3. Tunisian comprehension
4. Tunisian generation
5. Arabizi comprehension
6. Arabizi generation
7. Arabic-script Tunisian
8. Tunisian/French code-switching
9. Translation
10. Everyday conversation
11. Technical Tunisian
12. Reasoning in Tunisian
13. Cultural/contextual understanding
14. Robustness to spelling variation

---

## 6. Roadmap

- [x] **v0.1**: Tokenizer & dataset pipeline from scratch
- [x] **v0.2**: Random-init ~300M Transformer architecture
- [x] **v0.3**: Overfitting sanity checks & smoke pretraining verification
- [x] **v0.4**: TounsiEval benchmark suite
- [ ] **v0.5**: Scaled corpus pretraining (10B+ tokens)
- [ ] **v0.6**: TounsiLM-300M-Instruct fine-tuning
- [ ] **v0.7**: Scaled 500M and 1B parameter variants

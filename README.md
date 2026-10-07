# TounsiLM — Tunisian Arabic Large Language Model (300M)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-orange.svg)](docs/architecture.md)
[![Parameters](https://img.shields.io/badge/Parameters-~300M_(303.4M)-purple.svg)](scripts/estimate_parameters.py)
[![Dialect](https://img.shields.io/badge/Dialect-Tunisian_Derja_%26_Arabizi-red.svg)](docs/dataset.md)
[![Tests](https://img.shields.io/badge/Tests-25%2F25_Passing-brightgreen.svg)](tests/)

**TounsiLM** is an open-source, reproducible Large Language Model designed specifically for **Tunisian Arabic (Derja)**, **Arabizi**, and **Tunisian French/Tech Code-Switching**.

TounsiLM contains **303,350,784 parameters** (~303.4M), trained strictly from **random initialization** without importing or fine-tuning from existing English or Standard Arabic models (no LLaMA, Qwen, Mistral, Gemma, GPT, BERT, or AraBERT).

---

## Table of Contents
1. [Model Overview & Linguistic Focus](#1-model-overview--linguistic-focus)
2. [Architecture Specifications](#2-architecture-specifications)
3. [Accepted Datasets & File Formats](#3-accepted-datasets--file-formats)
4. [Step-by-Step Training Guide](#4-step-by-step-training-guide)
5. [How to Use & Interact with the Model](#5-how-to-use--interact-with-the-model)
6. [Hardware Acceleration (CPU vs GPU RTX 4060)](#6-hardware-acceleration-cpu-vs-gpu)
7. [TounsiEval Benchmark (14 Categories)](#7-tounsieval-benchmark-14-categories)
8. [Project File Structure](#8-project-file-structure)

---

## 1. Model Overview & Linguistic Focus

Tunisian Arabic (**Derja**, ISO 639-3 `aeb`) is spoken by over 12 million people. Standard commercial and open-source models underperform on Tunisian because of three fundamental challenges:

1. **Severe Tokenizer Fragmentation**: Standard English/MSA tokenizers break common Derja words into 4 to 6 subword fragments, degrading attention efficiency and context length.
2. **Arabizi Neglect**: Everyday digital communication in Tunisia heavily utilizes Arabizi (Latin alphabet augmented with numbers: `3`=ع, `7`=ح, `5`=خ, `8`=غ, `9`=ق). Foreign models confuse Arabizi numbers with actual numerical values.
3. **Natural Code-Switching**: Tunisians frequently mix Derja, French, and English in everyday professional, technical, and social discussions.

**TounsiLM solves this at the architectural root**: it trains a specialized **SentencePiece BPE tokenizer from scratch** and trains the entire Transformer directly on authentic multi-register Tunisian text.

---

## 2. Architecture Specifications

The model architecture is verified programmatically via `scripts/estimate_parameters.py`:

| Hyperparameter | Value | Rationale & Description |
| :--- | :--- | :--- |
| **Number of Layers** | **24** | Deep Transformer decoder stack |
| **Hidden Size ($H$)** | **1024** | Dimensionality of token representations |
| **Attention Heads** | **16** | Head dimension $d_k = 64$ ($16 \times 64 = 1024$) |
| **Intermediate Size** | **2304** | SwiGLU feed-forward layer ($2.25 \times H$) |
| **Positional Embeddings** | **RoPE** | Rotary Position Embeddings (base $\theta = 10000$) |
| **Normalization** | **RMSNorm** | Root Mean Square Pre-Normalization ($\epsilon = 10^{-6}$) |
| **Feed-Forward Activation** | **SwiGLU** | Gated linear unit with SiLU activation |
| **Weight Tying** | **True** | Output LM head weights tied to input token embeddings |
| **Sequence Length** | **512 / 2048** | 512 for local development; 2048 for full cluster runs |
| **Total Parameters** | **303,350,784** | **303.4M parameters** (+1.12% vs 300M target) |

---

## 3. Accepted Datasets & File Formats

You can train TounsiLM on your own data by placing files in the **`data/raw/`** directory.

### Supported File Formats

The pipeline natively supports three data formats:

#### 1. Apache Parquet (`.parquet`) — Recommended for Large Datasets
The pipeline automatically reads `.parquet` tables and auto-detects the text column (`Tweet`, `text`, `sentence`, `content`, `document`, `comment`):
```text
data/raw/train-00000-of-00001.parquet
```

#### 2. JSON Lines (`.jsonl`) — Recommended for Custom Metadata
Each line must be a valid JSON object containing a `"text"` field (with optional `"script"` or `"category"`):
```json
{"text": "عسلامة، شنوة أحوالك اليوم؟ انشالله تكون لاباس.", "script": "arabic"}
{"text": "3asslema labes 3lik? nchallah tkoun b5ir.", "script": "arabizi"}
{"text": "باش نعمل deploy للـcode بعد ما نكملو الـtests.", "script": "mixed"}
```

#### 3. Plain Text (`.txt`)
Plain UTF-8 text file with one document, dialogue turn, or tweet per line:
```text
الماكلة التونسية بنينة برشا خاصة الكسكسي بالحوت واللبلابي في الشتاء.
chnowa raykom nemchiw lel 9ahwa ba3d l5edma nbadlou chwaya jaw?
```

### Data Pipeline Processing Stages
When you process raw data, TounsiLM runs it through an 8-stage pipeline:
1. **Unicode NFC & Arabic Normalization**: Standardizes Alef forms (`أ`, `إ`, `آ` $\rightarrow$ `ا`), converts Yah/Alef Maksura, and collapses excessive repeated characters (e.g. `بررررشا` $\rightarrow$ `برشا`).
2. **PII Masking**: Sanitizes phone numbers, email addresses, and IPv4 addresses into generic placeholders (`<PHONE>`, `<EMAIL>`, `<IP>`).
3. **Quality Filtering**: Drops documents that are too short (< 10 chars), contain excessive punctuation/symbols (> 30%), or consist mostly of URLs.
4. **Dialect Scoring**: Analyzes Tunisian vocabulary presence (can be bypassed with `--skip-dialect-filter` for domain-specific data).
5. **Benchmark Decontamination**: Ensures no text leaks from the `TounsiEval` evaluation benchmark.
6. **Exact Deduplication (xxhash)**: Removes exact duplicate texts, tweets, and copy-pasted retweets.
7. **Near-Duplicate Deduplication (MinHash LSH)**: Clusters documents with high Jaccard similarity and retains only unique representations.
8. **Train/Val/Test Split**: Deterministically creates 90% train, 5% val, and 5% test splits in `data/processed/`.

---

## 4. Step-by-Step Training Guide

Run all commands from your terminal with the virtual environment activated (`.\.venv\Scripts\Activate.ps1`).

```powershell
# Ensure console output handles Arabic & Arabizi cleanly
$env:PYTHONIOENCODING="utf-8"
```

### Step 1: Drop Your Dataset in `data/raw/`
Place your `.parquet`, `.jsonl`, or `.txt` file inside `data/raw/`:
```powershell
# Example: place train-00000-of-00001.parquet inside data/raw/
```

### Step 2: Clean & Normalize the Raw Data
Run `scripts/clean_data.py`. It automatically detects all files in `data/raw/`:
```powershell
.\.venv\Scripts\python.exe scripts/clean_data.py --input data/raw/train-00000-of-00001.parquet --skip-dialect-filter
```
*Output: Sanitized documents saved to `data/intermediate/tounsi_cleaned.jsonl` and `data/tokenizer/corpus.txt`.*

### Step 3: Deduplicate & Split into Train/Val/Test
Run `scripts/deduplicate.py`:
```powershell
.\.venv\Scripts\python.exe scripts/deduplicate.py
```
*Output: Unique text splits written to `data/processed/train.txt`, `val.txt`, and `test.txt`.*

### Step 4 (Optional): Train / Expand Tokenizer Vocabulary
If your dataset contains new vocabulary or domain-specific terms, retrain the SentencePiece BPE tokenizer:
```powershell
.\.venv\Scripts\python.exe scripts/train_tokenizer.py --vocab-size 4000
```
*Output: Tokenizer model saved to `data/tokenizer/tounsilm_tokenizer.model`.*

### Step 5: Start Pretraining
Launch pretraining with `--force-repack` to pack your text splits into binary memmap tokens (`train.bin` and `val.bin`) and start training:
```powershell
.\.venv\Scripts\python.exe scripts/pretrain.py --config configs/pretraining_local.yaml --force-repack
```

During training, the console provides **live micro-batch accumulation feedback**:
```text
  [Accumulating: 2/4 micro-batches for step 1]
  Step      1 | loss: 6.9421 | lr: 6.00e-07 | grad_norm: 8.54 | tok/s: 142 | elapsed: 18s
```

### Step 6: Resume Training from a Checkpoint
You can pause training anytime (`Ctrl + C`) and resume safely. The trainer automatically restores model weights, AdamW momentum buffers, cosine learning rate schedule, and step counters:
```powershell
.\.venv\Scripts\python.exe scripts/resume_training.py --checkpoint checkpoints/checkpoint-final --additional-steps 1000
```

---

## 5. How to Use & Interact with the Model

### Method 1: Command-Line Text Generation (CLI)

#### A. One-Shot Prompt Generation
Generate completions directly in your terminal:
```powershell
# Arabic Script Derja
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/generate.py --checkpoint checkpoints/checkpoint-final --prompt "عسلامة، شنوة أحوالك اليوم؟" --max-new-tokens 30

# Arabizi
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/generate.py --checkpoint checkpoints/checkpoint-final --prompt "3asslema labes 3lik?" --max-new-tokens 30
```

#### B. Interactive Chat Console Mode
Launch an interactive prompt loop:
```powershell
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/generate.py --interactive
```

---

### Method 2: FastAPI REST API Server

TounsiLM includes a production-ready FastAPI inference service.

#### 1. Start the API Server
```powershell
.\.venv\Scripts\uvicorn.exe tounsilm.inference.api:create_app --factory --port 8000
```

#### 2. Open Interactive Swagger UI
Open your browser at:
```text
http://localhost:8000/docs
```

#### 3. Send Requests via HTTP / cURL
```powershell
curl -X POST "http://localhost:8000/generate" `
  -H "Content-Type: application/json" `
  -d '{"prompt": "عسلامة شنوة أحوالك؟", "temperature": 0.8, "max_new_tokens": 40}'
```

Response:
```json
{
  "text": "عسلامة شنوة أحوالك؟ انشالله تكون لاباس وصحتك باهية...",
  "tokens_generated": 28,
  "finish_reason": "length"
}
```

---

### Method 3: Python Library API

You can import and use TounsiLM directly in your own Python applications:

```python
import torch
from tounsilm.inference.generator import TextGenerator

# Load model and tokenizer
generator = TextGenerator.from_checkpoint(
    checkpoint_dir="checkpoints/checkpoint-final",
    tokenizer_path="data/tokenizer/tounsilm_tokenizer.model",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Generate completion
output = generator.generate(
    prompt="عسلامة، وقتاش باش تهبط للبلاد؟",
    max_new_tokens=40,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
)

print(output)
```

---

## 6. Hardware Acceleration (CPU vs GPU)

TounsiLM is designed to adapt automatically to your hardware:

### NVIDIA GPU Acceleration (e.g. RTX 4060 Laptop 8GB VRAM)
If an NVIDIA GPU is available, installing PyTorch CUDA unlocks **~50x to 100x speedups**:

```powershell
.\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Once installed:
- PyTorch automatically switches from `Device: cpu` to `Device: cuda`.
- Automatic mixed precision (`fp16` or `bf16`) reduces VRAM usage by 50%.
- Gradient Checkpointing keeps activation memory under 6 GB VRAM during full 300M training.

### CPU Execution
On CPU-only environments:
- The trainer runs with `torch.float32` / `fp16` emulation.
- Gradient accumulation is adjusted (e.g. 4 micro-batches) so step times remain fast (~15–20 seconds per step).
- Memory-mapped files (`np.memmap`) prevent large token datasets from filling system RAM.

---

## 7. TounsiEval Benchmark (14 Categories)

TounsiLM is benchmarked against `TounsiEval`, an evaluation suite covering 14 linguistic categories:

1. **Tunisian Vocabulary**: Dialectal words (*برشا*, *فيسع*, *بالشويا*, *حومة*).
2. **Tunisian Grammar**: Verb conjugations (*مشى* $\rightarrow$ *مشينا*), negation markers (*ما ... ش*).
3. **Tunisian Comprehension**: Idiomatic comprehension (*يعيشك ويرحم والديك*).
4. **Tunisian Generation**: Coherent natural sentence generation.
5. **Arabizi Comprehension**: Decoding Latin script with numbers (*3*, *5*, *7*, *8*, *9*).
6. **Arabizi Generation**: Generating valid Arabizi responses.
7. **Arabic-Script Tunisian**: Pure Arabic script Derja generation.
8. **Tunisian/French Code-Switching**: Software, business, and daily code-switching.
9. **Translation**: Derja $\leftrightarrow$ Standard Arabic / English / French.
10. **Everyday Conversation**: Greetings, bargaining, daily interactions.
11. **Technical Tunisian**: IT, engineering, and web development terminology.
12. **Reasoning in Tunisian**: Contextual reasoning expressed in Derja.
13. **Cultural Understanding**: Traditional cuisine (couscous, lablabi, kafteji), history (Carthage, Medina).
14. **Spelling Variation Robustness**: Tolerance to phonetic dialect variations.

### Running Evaluation
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/checkpoint-final
```
*Generates an evaluation report in [evaluation_results/latest_eval_report.md](file:///c:/Users/MSI/Desktop/DEV/TounsiLM/evaluation_results/latest_eval_report.md).*

---

## 8. Project File Structure

```text
TounsiLM/
├── configs/
│   ├── model_300m.yaml          # Model architecture specification (303.4M)
│   ├── pretraining_local.yaml   # Local training hyperparameters (precision, steps)
│   ├── pretraining_cloud.yaml   # Multi-GPU cluster configuration
│   └── tokenizer.yaml           # SentencePiece vocabulary & special tokens
├── data/
│   ├── raw/                     # Drop raw files here (.parquet, .jsonl, .txt)
│   ├── intermediate/            # Cleaned, normalized datasets
│   ├── processed/               # Train / Val / Test text & packed binary tokens (.bin)
│   ├── tokenizer/               # Trained SentencePiece BPE model (.model & .vocab)
│   └── eval/                    # TounsiEval 14-category benchmark dataset
├── scripts/
│   ├── clean_data.py            # Cleans, normalizes, sanitizes PII, filters
│   ├── deduplicate.py          # Exact hash + MinHash LSH deduplication & splits
│   ├── train_tokenizer.py       # Trains SentencePiece BPE tokenizer from scratch
│   ├── estimate_parameters.py   # Audits theoretical & instantiated parameter count
│   ├── pretrain.py              # Main pretraining script (supports --force-repack)
│   ├── resume_training.py       # Resumes training from any saved checkpoint
│   ├── generate.py              # CLI text generation (one-shot and interactive)
│   ├── evaluate.py              # Runs TounsiEval benchmark report
│   └── export_model.py          # Packages model weights, config, and card for release
├── src/tounsilm/
│   ├── data/                    # Normalizers, filters, deduplication, dataset loaders
│   ├── model/                   # PyTorch Transformer (RMSNorm, RoPE, SwiGLU, Decoder)
│   ├── tokenizer/               # TounsiTokenizer SentencePiece wrapper
│   ├── training/                # Trainer (mixed precision, checkpoints, schedulers)
│   ├── evaluation/              # TounsiEval benchmark runner
│   └── inference/               # Generator class & FastAPI REST server
├── tests/                       # 25 unit tests (model, forward, tokenizer, dataset, trainer)
└── evaluation_results/          # Generated benchmark reports (Markdown & JSON)
```

---

## 9. Running Unit Tests

To run the complete 25-test verification suite:
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

All tests verify forward pass shapes, causal masking, gradient flow, RMSNorm, RoPE, SwiGLU, tokenizer roundtripping, and checkpoint saving/loading.

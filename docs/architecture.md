# TounsiLM Architecture Specification

## 1. Overview

**TounsiLM-300M** is a decoder-only autoregressive Transformer language model purpose-built from scratch for **Tunisian Arabic (Derja)**, **Arabizi** (Latin-script Tunisian), and **Tunisian/French code-switching**.

All model parameters are initialized from random distributions ($N(0, 0.02)$); no pretrained weights from external foundation models (such as LLaMA, Mistral, Qwen, Gemma, BERT, or AraBERT) are ever used.

```text
               Input Token IDs (Batch x SeqLen)
                              │
                    ┌─────────▼─────────┐
                    │  Token Embedding  │ (Vocab: 32,000 x 1024)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Transformer Block │ x 24 Layers
                    │  ┌──────────────┐ │
                    │  │   RMSNorm    │ │
                    │  └──────┬───────┘ │
                    │  ┌──────▼───────┐ │
                    │  │ RoPE Attention│ │ (16 Heads, d_k=64)
                    │  └──────┬───────┘ │
                    │         ▼ + Resid │
                    │  ┌──────────────┐ │
                    │  │   RMSNorm    │ │
                    │  └──────┬───────┘ │
                    │  ┌──────▼───────┐ │
                    │  │  SwiGLU MLP  │ │ (Intermediate: 2304)
                    │  └──────┬───────┘ │
                    │         ▼ + Resid │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Final RMSNorm   │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │      LM Head      │ (Tied with Token Embedding)
                    └─────────┬─────────┘
                              │
                     Output Logits (Batch x SeqLen x 32,000)
```

---

## 2. Architectural Hyperparameters

| Hyperparameter | Symbol | Value | Rationale |
| :--- | :--- | :--- | :--- |
| **Layers** | $L$ | **24** | Sufficient depth to capture hierarchical Derja syntax and morphology. |
| **Hidden Size** | $H$ | **1024** | Dimensional capacity balance for 300M parameter envelope. |
| **Attention Heads** | $A$ | **16** | Head dimension $d_k = 1024 / 16 = 64$. Standard hardware-efficient dimension. |
| **Intermediate Size** | $I$ | **2304** | SwiGLU expansion ($2.25 \times H$), balancing expressivity and memory. |
| **Vocabulary Size** | $V$ | **32,000** | Covers Arabic script, Arabizi, and French loanwords without fragmentation. |
| **Context Length** | $S$ | **512 / 2048** | Configurable: 512 for local 6–8 GB GPUs; 2048 for cloud cluster pretraining. |
| **Weight Tying** | — | **True** | Ties input embedding and output LM head, saving 32.8M parameters. |
| **Positional Embedding**| — | **RoPE** | Rotary Position Embeddings; relative position encoding up to 4096 tokens. |
| **Normalization** | — | **RMSNorm** | Root Mean Square Normalization with $\epsilon = 10^{-6}$; faster than LayerNorm. |
| **Activation** | — | **SwiGLU** | Gated linear unit with SiLU activation: $(xW_{gate} \odot \text{SiLU}(xW_{up})) W_{down}$. |

---

## 3. Mathematical Parameter Audit

Parameters are computed rigorously across each module:

1. **Token Embedding**:
   $$V \times H = 32{,}000 \times 1024 = 32{,}768{,}000$$

2. **Attention Projections (per layer)**:
   - Query, Key, Value, Output: $4 \times (H \times H) = 4 \times (1024 \times 1024) = 4{,}194{,}304$
   - Across 24 layers: $24 \times 4{,}194{,}304 = 100{,}663{,}296$

3. **SwiGLU MLP (per layer)**:
   - Gate, Up, Down: $3 \times (H \times I) = 3 \times (1024 \times 2304) = 7{,}077{,}888$
   - Across 24 layers: $24 \times 7{,}077{,}888 = 169{,}869{,}312$

4. **Normalizations**:
   - Per layer: 2 RMSNorm ($2 \times H = 2048$)
   - Final RMSNorm: $1 \times H = 1024$
   - Total norm: $(2 \times 24 + 1) \times 1024 = 50{,}176$

5. **LM Head**:
   - Tied with Token Embedding $\rightarrow 0$ additional weights.

$$\mathbf{Total\ Parameters} = 32{,}768{,}000 + 100{,}663{,}296 + 169{,}869{,}312 + 50{,}176 = \mathbf{303{,}350{,}784}$$

**Target**: 300.0M | **Actual**: **303.35M** (deviation: **+1.12%**, well within the $\pm 10\%$ tolerance).

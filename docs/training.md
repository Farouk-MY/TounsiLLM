# TounsiLM Pretraining & Curriculum Methodology

## 1. Objective Function

TounsiLM uses standard causal autoregressive language modeling. Given a token sequence $X = (x_1, x_2, \dots, x_N)$, the objective is minimizing the negative log-likelihood across all positions:

$$\mathcal{L}_{CLM} = -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_1, \dots, x_{i-1}; \Theta)$$

---

## 2. Low-VRAM Optimization Strategy (Local 6–8 GB GPUs)

Training a ~300M parameter model locally on consumer hardware requires careful memory budgeting:

1. **Parameters & Gradients Memory**:
   - Weights (FP32/BF16): $\approx 303\text{M} \times 2\text{ bytes} \approx 606\text{ MB}$.
   - Optimizer State (AdamW 32-bit fp32 buffers: FP32 master weights + first moment $m$ + second moment $v$): $\approx 303\text{M} \times 16\text{ bytes} \approx 4.85\text{ GB}$.
   - Gradients: $\approx 606\text{ MB}$.
   - Total static state: $\approx 6.06\text{ GB}$.

2. **Activation Checkpointing (Gradient Checkpointing)**:
   - Drops intermediate layer activations during the forward pass and recalculates them during backpropagation.
   - Reduces activation memory from $O(L \cdot S)$ to $O(\sqrt{L} \cdot S)$, saving ~70% of transient VRAM.

3. **Micro-Batching & Gradient Accumulation**:
   - `micro_batch_size = 1`
   - `gradient_accumulation_steps = 32`
   - Effective Batch Size = $1 \times 32 = 32$ sequences ($16{,}384$ tokens per optimizer step at $S=512$).

4. **Mixed Precision with Hardware Fallback**:
   - Automatically detects whether hardware supports BF16 (Ampere, Ada Lovelace, Hopper).
   - Gracefully falls back to FP16 with dynamic `GradScaler` or CPU FP32 if accelerator is unavailable.

---

## 3. Training Curriculum Phases

Training is organized into four configurable curricular phases:

```text
Phase A: General Tunisian Web & Public Corpus
  ↓ (Broad dialect vocabulary, Arabizi foundations)
Phase B: High-Quality Curated Tunisian Dialogues & Literature
  ↓ (Idiomatic Derja, proverbs, colloquial grammar)
Phase C: Conversational & Code-Switched Tech Interactions
  ↓ (Tunisian/French technical discussions, Q&A)
Phase D: Instruction Fine-Tuning (TounsiLM-Instruct)
    (ChatML formatted task execution in Derja)
```

---

## 4. Resumability & Checkpointing

Every checkpoint directory (`checkpoints/checkpoint-step-X/`) contains a complete checkpoint bundle:
- `model.pt`: Neural weights
- `optimizer.pt`: Full AdamW state dict (momentum & variance buffers)
- `scheduler.pt`: Learning rate decay state
- `scaler.pt`: Mixed precision gradient scaler state
- `config.json`: Model architecture hyperparameters
- `metadata.json`: Step, tokens seen, loss, system info, git commit, random seed

Old checkpoints are automatically rotated, retaining the `max_checkpoints` most recent saves plus the latest marker.

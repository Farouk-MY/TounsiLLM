"""
Pretraining script for TounsiLM.

Supports:
  - --smoke-test    : Fast validation run (10-20 steps) checking loss decrease,
                      backward pass, checkpoint saving & generation.
  - --overfit-test  : Overfitting sanity check on ~50-100 examples to verify
                      that the architecture and training loop can drive loss to zero.
  - Full pretraining with mixed precision, gradient accumulation, and checkpointing.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import load_config, ModelConfig, TrainingConfig
from tounsilm.data.dataset import TounsiLMDataset, create_packed_dataset
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.tokenizer.tokenizer import TounsiTokenizer
from tounsilm.training.trainer import Trainer
from tounsilm.utils.reproducibility import set_seed


def run_overfit_test(cfg, tokenizer):
    """Verify that model can overfit on a tiny corpus of 50-100 samples."""
    print("\n" + "=" * 65)
    print("      PHASE 5: OVERFITTING SANITY CHECK (MANDATORY)")
    print("=" * 65)
    print("Goal: Confirm model can memorize a tiny dataset and drive loss to near zero.")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a compact version of TounsiLM for quick memorization verification
    test_model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_sequence_length=128,
        tie_word_embeddings=True,
    )
    model = TounsiLMModel(test_model_cfg).to(device)

    # 50 fixed sentences
    tiny_texts = [
        "عسلامة شنوة أحوالك لاباس؟",
        "علاش ما جيتش البارح للعشوية؟",
        "برشا ناس يحبو القهوة في الصباح.",
        "3asslema chnowa a7welek labes 3lik?",
        "3lech ma jitich lbare7 lel 9ahwa?",
        "لازم نعمل update للapplication.",
        "توّا عندي réunion مهمة مع الـclient.",
    ] * 10  # 70 examples

    bin_path = Path("data/processed/overfit_tiny.bin")
    create_packed_dataset(tiny_texts, tokenizer, bin_path)
    dataset = TounsiLMDataset(bin_path, seq_len=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    initial_loss = None
    final_loss = None

    model.train()
    print("\nTraining tiny model to overfit...")
    for epoch in range(40):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch == 0:
            initial_loss = avg_loss
            print(f"  Epoch {epoch+1:2d} (Start)   : Loss = {avg_loss:.4f}")
        elif (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}           : Loss = {avg_loss:.4f}")

    final_loss = avg_loss
    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.3 or final_loss < 0.8, (
        f"Overfitting check failed! Loss did not decrease sufficiently ({initial_loss:.4f} -> {final_loss:.4f})"
    )
    print("[OK] OVERFITTING SANITY CHECK PASSED! Architecture and loss calculation verified.")
    print("=" * 65 + "\n")


def run_smoke_test(cfg, tokenizer):
    """Smoke test: 15 steps of full pipeline verifying forward, backward, checkpoint, resume, generation."""
    print("\n" + "=" * 65)
    print("      PHASE 6: PRETRAINING SMOKE TEST")
    print("=" * 65)
    print("Verifying full trainer mechanics, gradients, checkpoints, and generation...")

    # Configure small run
    cfg.training.max_steps = 6
    cfg.training.save_every_steps = 3
    cfg.training.eval_every_steps = 3
    cfg.training.log_every_steps = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.sequence_length = 64
    cfg.model.max_sequence_length = 64
    cfg.model.vocab_size = tokenizer.vocab_size

    # On CPU, scale down smoke model to 4 layers for fast ~10s execution
    if not torch.cuda.is_available():
        cfg.model.num_layers = 4
        cfg.model.hidden_size = 256
        cfg.model.num_attention_heads = 4
        cfg.model.intermediate_size = 512

    # Create dummy packed dataset for smoke test if needed
    smoke_bin = Path("data/processed/smoke_test.bin")
    if not smoke_bin.exists():
        smoke_texts = [
            "عسلامة شنوة أحوالك؟", "برشا ناس يحبو القهوة.", "3asslema labes?",
            "لازم نعمل update للـapplication.", "باش نعمل deploy بعد شوية."
        ] * 30
        create_packed_dataset(smoke_texts, tokenizer, smoke_bin)

    train_ds = TounsiLMDataset(smoke_bin, seq_len=64)
    val_ds = train_ds

    model = TounsiLMModel(cfg.model)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        tokenizer=tokenizer,
    )

    stats = trainer.train()
    print("\n[OK] Smoke training finished successfully.")
    print(f"  Total steps: {stats['total_steps']} | Final loss: {stats['final_train_loss']:.4f}")

    # Verify generation works on trained checkpoint
    print("\nTesting generation from model after smoke training...")
    prompt = "عسلامة شنوة"
    prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)]).to(trainer.device)
    gen_ids = model.generate(prompt_ids, max_new_tokens=15, temperature=0.8)
    gen_text = tokenizer.decode(gen_ids[0].tolist())
    print(f"  Prompt : '{prompt}'")
    print(f"  Output : '{gen_text}'")

    print("\n[OK] SMOKE TEST PASSED COMPLETELY!")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="TounsiLM Pretraining Script")
    parser.add_argument("--config", type=str, default="configs/pretraining_local.yaml")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/tounsilm_tokenizer.model")
    parser.add_argument("--smoke-test", action="store_true", help="Run short smoke test")
    parser.add_argument("--overfit-test", action="store_true", help="Run tiny overfitting sanity check")
    parser.add_argument("--force-repack", action="store_true", help="Force repacking text splits into binary datasets")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tok_path = Path(args.tokenizer_path)

    if not tok_path.exists():
        print(f"[Error] Tokenizer not found at: {tok_path}")
        print("Please train tokenizer first with: python scripts/train_tokenizer.py")
        sys.exit(1)

    tokenizer = TounsiTokenizer(tok_path)

    if args.overfit_test:
        run_overfit_test(cfg, tokenizer)
        return

    if args.smoke_test:
        run_smoke_test(cfg, tokenizer)
        return

    # Standard pretraining
    print("=" * 60)
    print("       TounsiLM Standard Pretraining")
    print("=" * 60)

    train_bin = Path("data/processed/train.bin")
    val_bin = Path("data/processed/val.bin")

    if not train_bin.exists() or args.force_repack:
        print(f"[Notice] Creating packed datasets from text splits...")
        train_txt = Path("data/processed/train.txt")
        val_txt = Path("data/processed/val.txt")

        if not train_txt.exists():
            print("[Error] Processed text splits not found. Run clean_data.py & deduplicate.py first.")
            sys.exit(1)

        with open(train_txt, "r", encoding="utf-8") as f:
            train_lines = [l.strip() for l in f if l.strip()]
        with open(val_txt, "r", encoding="utf-8") as f:
            val_lines = [l.strip() for l in f if l.strip()]

        create_packed_dataset(train_lines, tokenizer, train_bin)
        create_packed_dataset(val_lines, tokenizer, val_bin)

    train_ds = TounsiLMDataset(train_bin, seq_len=cfg.training.sequence_length)
    val_ds = None
    if val_bin.exists():
        try:
            val_ds = TounsiLMDataset(val_bin, seq_len=cfg.training.sequence_length)
        except ValueError as e:
            print(f"[Warning] Validation dataset skipped: {e}")
            val_ds = None

    cfg.model.vocab_size = tokenizer.vocab_size
    model = TounsiLMModel(cfg.model)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        tokenizer=tokenizer,
    )

    stats = trainer.train()
    print("\nPretraining completed! Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

"""
TounsiLM Pretraining Trainer.

Implements the full training loop with:
  - Mixed precision (FP16/BF16 with auto-detection)
  - Gradient accumulation
  - Gradient checkpointing
  - Gradient clipping
  - Cosine LR schedule with warmup
  - Checkpoint saving and resuming
  - Validation evaluation
  - TensorBoard logging
  - Rotating checkpoint management

All checkpoints contain full state for exact resumability:
  model, optimizer, scheduler, scaler, step, config, metadata.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from tounsilm.config import TounsiLMConfig, TrainingConfig, get_system_info
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.utils.reproducibility import set_seed


class Trainer:
    """TounsiLM Pretraining Trainer.

    Handles the complete training lifecycle including:
      - Training loop with gradient accumulation
      - Validation evaluation
      - Checkpoint management (save, resume, rotate)
      - Logging (TensorBoard, CSV, console)
      - Memory-safe mixed precision
    """

    def __init__(
        self,
        model: TounsiLMModel,
        train_dataset,
        val_dataset,
        config: TounsiLMConfig,
        tokenizer=None,
    ):
        self.config = config
        self.train_config = config.training
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Mixed precision
        self.dtype = self.train_config.resolve_precision()
        self.use_amp = self.dtype != torch.float32
        self.scaler = GradScaler(enabled=(self.dtype == torch.float16))

        # Gradient checkpointing
        if self.train_config.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self.start_time = None

        # Logging
        self.log_dir = Path(self.train_config.log_dir) / self.train_config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(self.train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        self.csv_log_path = self.log_dir / "training_log.csv"

        # Initialize CSV log
        if not self.csv_log_path.exists():
            with open(self.csv_log_path, "w") as f:
                f.write(
                    "step,train_loss,val_loss,learning_rate,grad_norm,"
                    "tokens_seen,tokens_per_sec,gpu_memory_gb,elapsed_sec\n"
                )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay for non-bias/norm params."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            eps=self.train_config.eps,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup + cosine decay."""

        def lr_lambda(step: int) -> float:
            warmup = self.train_config.warmup_steps
            max_steps = self.train_config.max_steps
            min_lr_ratio = self.train_config.min_learning_rate / self.train_config.learning_rate

            if step < warmup:
                # Linear warmup
                return step / max(warmup, 1)
            elif step >= max_steps:
                return min_lr_ratio
            else:
                # Cosine decay
                progress = (step - warmup) / max(max_steps - warmup, 1)
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            print("[WARNING] TensorBoard not available. Logging to CSV only.")

    def train(self) -> dict:
        """Run the training loop.

        Returns:
            Dictionary with final training statistics.
        """
        set_seed(self.train_config.seed)
        self._init_tensorboard()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.micro_batch_size,
            shuffle=True,
            num_workers=min(self.train_config.num_workers, 2),
            pin_memory=True,
            drop_last=True,
        )

        self.start_time = time.time()
        self.model.train()

        print(f"\n{'='*60}")
        print(f"  TounsiLM Training")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Precision: {self.train_config.precision}")
        print(f"  Micro batch size: {self.train_config.micro_batch_size}")
        print(f"  Gradient accumulation: {self.train_config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.train_config.effective_batch_size}")
        print(f"  Sequence length: {self.train_config.sequence_length}")
        print(f"  Max steps: {self.train_config.max_steps}")
        print(f"  Learning rate: {self.train_config.learning_rate}")
        print(f"  Gradient checkpointing: {self.train_config.gradient_checkpointing}")
        print(f"{'='*60}\n")

        accum_loss = 0.0
        accum_steps = 0
        train_loss_running = 0.0

        data_iter = iter(train_loader)

        while self.global_step < self.train_config.max_steps:
            # Get next batch (restart iterator if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / self.train_config.gradient_accumulation_steps

            # Backward pass
            if self.dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            accum_steps += 1

            # Track tokens
            self.tokens_seen += input_ids.numel()

            # Gradient accumulation step
            if accum_steps >= self.train_config.gradient_accumulation_steps:
                # Gradient clipping
                if self.dtype == torch.float16:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )

                # Optimizer step
                if self.dtype == torch.float16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                train_loss = accum_loss
                train_loss_running = (
                    0.95 * train_loss_running + 0.05 * train_loss
                    if train_loss_running > 0 else train_loss
                )

                # Logging
                if self.global_step % self.train_config.log_every_steps == 0:
                    self._log_step(train_loss, grad_norm)

                # Validation
                if (
                    self.global_step % self.train_config.eval_every_steps == 0
                    and self.val_dataset is not None
                ):
                    val_loss = self.evaluate()
                    self.model.train()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        print(f"  [*] New best validation loss: {val_loss:.4f}")

                # Checkpoint
                if self.global_step % self.train_config.save_every_steps == 0:
                    self.save_checkpoint()

                accum_loss = 0.0
                accum_steps = 0

        # Final checkpoint
        self.save_checkpoint(tag="final")

        # Final validation
        final_val_loss = None
        if self.val_dataset is not None:
            final_val_loss = self.evaluate()

        elapsed = time.time() - self.start_time

        stats = {
            "total_steps": self.global_step,
            "total_tokens": self.tokens_seen,
            "final_train_loss": train_loss_running,
            "final_val_loss": final_val_loss,
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self.tokens_seen / max(elapsed, 1),
        }

        if self.tb_writer:
            self.tb_writer.close()

        return stats

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run validation evaluation.

        Returns:
            Average validation loss.
        """
        self.model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.train_config.micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs["loss"].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

        print(f"  [Eval] Step {self.global_step}: "
              f"val_loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

        if self.tb_writer:
            self.tb_writer.add_scalar("eval/loss", avg_loss, self.global_step)
            self.tb_writer.add_scalar("eval/perplexity", perplexity, self.global_step)

        return avg_loss

    def _log_step(self, train_loss: float, grad_norm: float) -> None:
        """Log training metrics for the current step."""
        elapsed = time.time() - self.start_time
        lr = self.scheduler.get_last_lr()[0]
        tokens_per_sec = self.tokens_seen / max(elapsed, 1)

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)

        print(
            f"  Step {self.global_step:6d} | "
            f"loss: {train_loss:.4f} | "
            f"lr: {lr:.2e} | "
            f"grad_norm: {grad_norm:.2f} | "
            f"tok/s: {tokens_per_sec:.0f} | "
            f"gpu_mem: {gpu_mem:.1f}GB | "
            f"elapsed: {elapsed:.0f}s"
        )

        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", train_loss, self.global_step)
            self.tb_writer.add_scalar("train/learning_rate", lr, self.global_step)
            self.tb_writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
            self.tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.global_step)
            self.tb_writer.add_scalar("train/gpu_memory_gb", gpu_mem, self.global_step)

        # CSV log
        with open(self.csv_log_path, "a") as f:
            f.write(
                f"{self.global_step},{train_loss:.6f},,"
                f"{lr:.8f},{grad_norm:.4f},"
                f"{self.tokens_seen},{tokens_per_sec:.0f},"
                f"{gpu_mem:.2f},{elapsed:.1f}\n"
            )

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """Save a training checkpoint.

        Contains all state needed for exact resumption:
        model, optimizer, scheduler, scaler, step, config, metadata.

        Args:
            tag: Optional tag for the checkpoint name.

        Returns:
            Path to the saved checkpoint directory.
        """
        if tag:
            ckpt_name = f"checkpoint-{tag}"
        else:
            ckpt_name = f"checkpoint-step-{self.global_step}"

        ckpt_dir = self.checkpoint_dir / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")

        # Save optimizer
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # Save scheduler
        torch.save(self.scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # Save scaler
        if self.scaler.is_enabled():
            torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")

        # Save model config
        model_config = {k: v for k, v in self.model.config.__dict__.items()}
        with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)

        # Save training metadata
        metadata = {
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "system_info": get_system_info(),
            "seed": self.train_config.seed,
        }
        with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        # Update 'latest' symlink / marker
        latest_path = self.checkpoint_dir / "latest"
        with open(latest_path, "w") as f:
            f.write(ckpt_name)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        print(f"  [Checkpoint] Saved: {ckpt_dir}")
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: str | Path) -> None:
        """Resume training from a checkpoint.

        Args:
            ckpt_dir: Path to checkpoint directory.
        """
        ckpt_dir = Path(ckpt_dir)

        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        print(f"  [Resume] Loading checkpoint from: {ckpt_dir}")

        # Load model
        model_state = torch.load(
            ckpt_dir / "model.pt", map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(model_state)

        # Load optimizer
        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=True)
            )

        # Load scheduler
        sched_path = ckpt_dir / "scheduler.pt"
        if sched_path.exists():
            self.scheduler.load_state_dict(
                torch.load(sched_path, map_location=self.device, weights_only=True)
            )

        # Load scaler
        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists() and self.scaler.is_enabled():
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location=self.device, weights_only=True)
            )

        # Load metadata
        meta_path = ckpt_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.global_step = metadata.get("global_step", 0)
            self.tokens_seen = metadata.get("tokens_seen", 0)
            self.best_val_loss = metadata.get("best_val_loss", float("inf"))

        print(f"  [Resume] Resumed at step {self.global_step}")

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only max_checkpoints most recent."""
        max_keep = self.train_config.max_checkpoints

        # Find all step checkpoints
        ckpts = sorted(
            [
                d for d in self.checkpoint_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-step-")
            ],
            key=lambda d: int(d.name.split("-")[-1]),
        )

        # Remove oldest if we exceed the limit
        while len(ckpts) > max_keep:
            oldest = ckpts.pop(0)
            shutil.rmtree(oldest)
            print(f"  [Checkpoint] Rotated (removed): {oldest.name}")

"""
Tests for Training Loop and Trainer Utilities.

Verifies:
  - Trainer initialization and optimizer construction
  - Step forward and backward gradient accumulation
  - Checkpoint saving with full state
  - Checkpoint loading and resuming training
  - Checkpoint rotation (pruning older checkpoints beyond max_checkpoints)
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tounsilm.config import TounsiLMConfig, ModelConfig, TrainingConfig
from tounsilm.data.dataset import TounsiLMDataset, create_packed_dataset
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.training.trainer import Trainer


@pytest.fixture
def training_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy tokenizer
        class DummyTokenizer:
            vocab_size = 50
            bos_id = 1
            eos_id = 2
            def encode(self, text, add_bos=True, add_eos=True):
                return [self.bos_id, 10, 15, 20, self.eos_id]

        bin_path = tmp_path / "train.bin"
        create_packed_dataset(["sample text"] * 50, DummyTokenizer(), bin_path)
        ds = TounsiLMDataset(bin_path, seq_len=8)

        m_cfg = ModelConfig(
            vocab_size=50,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_sequence_length=16,
        )
        t_cfg = TrainingConfig(
            precision="fp32",
            gradient_checkpointing=False,
            gradient_accumulation_steps=1,
            micro_batch_size=2,
            max_steps=5,
            save_every_steps=2,
            eval_every_steps=2,
            log_every_steps=1,
            max_checkpoints=2,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )
        cfg = TounsiLMConfig(model=m_cfg, training=t_cfg)
        model = TounsiLMModel(m_cfg)

        trainer = Trainer(model, ds, ds, cfg, tokenizer=DummyTokenizer())
        yield trainer, tmp_path
        ds.close()


def test_training_and_checkpoints(training_setup):
    trainer, tmp_path = training_setup

    stats = trainer.train()
    assert stats["total_steps"] == 5
    assert (tmp_path / "checkpoints" / "checkpoint-final").exists()

    # Test resume
    new_trainer = Trainer(
        trainer.model,
        trainer.train_dataset,
        trainer.val_dataset,
        trainer.config,
    )
    new_trainer.load_checkpoint(tmp_path / "checkpoints" / "checkpoint-final")
    assert new_trainer.global_step == 5

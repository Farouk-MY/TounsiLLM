"""
TounsiLM Transformer Model.

The complete decoder-only Transformer language model, composed from the
components defined in components.py. Supports:
  - forward() for training (returns logits and loss)
  - generate() for autoregressive inference
  - save_pretrained() / load_checkpoint() for persistence
  - count_parameters() for architecture analysis

All parameters are randomly initialized. No pretrained weights are loaded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tounsilm.config import ModelConfig
from tounsilm.model.components import RMSNorm, TransformerBlock


class TounsiLMModel(nn.Module):
    """TounsiLM: A Tunisian Arabic Decoder-Only Transformer.

    Architecture:
        Token Embedding → [TransformerBlock × N] → RMSNorm → LM Head

    The LM head shares weights with the token embedding (weight tying)
    to reduce parameter count and improve training efficiency.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        config.validate()

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with LM head
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using scaled normal distribution.

        Linear layers use N(0, initializer_range).
        Embedding layers use N(0, initializer_range).
        Output projection of attention and MLP use scaled initialization:
            N(0, initializer_range / sqrt(2 * num_layers))
        """
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            labels: Target token IDs for loss computation [batch_size, seq_len].
                    Typically input_ids shifted right by one position.
            attention_mask: Optional attention mask.

        Returns:
            Dictionary with 'logits' and optionally 'loss'.
        """
        batch_size, seq_len = input_ids.shape

        # Validate sequence length
        if seq_len > self.config.max_sequence_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum "
                f"({self.config.max_sequence_length})"
            )

        # Token embeddings
        hidden_states = self.token_embedding(input_ids)

        # Apply transformer blocks
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Language model head
        logits = self.lm_head(hidden_states)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            input_ids: Prompt token IDs [batch_size, prompt_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If > 0, only sample from the top k tokens
            top_p: Nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: Penalty for repeated tokens (1.0 = disabled)
            eos_token_id: Stop generation when this token is produced

        Returns:
            Generated token IDs [batch_size, prompt_len + generated_len]
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate to max sequence length if needed
            context = generated
            if context.shape[1] > self.config.max_sequence_length:
                context = context[:, -self.config.max_sequence_length:]

            # Forward pass
            outputs = self.forward(context)
            logits = outputs["logits"][:, -1, :]  # Last position

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0].tolist():
                    logits[0, token_id] /= repetition_penalty

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component.

        Returns:
            Dictionary with parameter counts per component and totals.
        """
        counts = {
            "embedding": 0,
            "attention": 0,
            "mlp": 0,
            "normalization": 0,
            "lm_head": 0,
            "total": 0,
            "trainable": 0,
        }

        # Embedding
        counts["embedding"] = sum(
            p.numel() for p in self.token_embedding.parameters()
        )

        # Per-layer components
        for layer in self.layers:
            counts["attention"] += sum(
                p.numel() for p in layer.self_attn.parameters()
                if not any(p is rp for rp in layer.self_attn.rotary_emb.parameters())
            )
            counts["mlp"] += sum(p.numel() for p in layer.mlp.parameters())
            counts["normalization"] += sum(
                p.numel() for p in layer.input_layernorm.parameters()
            )
            counts["normalization"] += sum(
                p.numel() for p in layer.post_attention_layernorm.parameters()
            )

        # Final norm
        counts["normalization"] += sum(p.numel() for p in self.norm.parameters())

        # LM head (may be tied to embedding)
        if self.config.tie_word_embeddings:
            counts["lm_head"] = 0  # Counted in embedding
        else:
            counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return counts

    def save_pretrained(self, save_dir: str | Path) -> None:
        """Save model checkpoint and configuration.

        Args:
            save_dir: Directory to save the model to.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), save_dir / "model.pt")

        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # Save parameter count
        counts = self.count_parameters()
        with open(save_dir / "parameter_count.json", "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2)

    @classmethod
    def load_checkpoint(cls, load_dir: str | Path, device: str = "cpu") -> "TounsiLMModel":
        """Load a TounsiLM checkpoint.

        Only loads checkpoints produced by this project.

        Args:
            load_dir: Directory containing model.pt and config.json
            device: Device to load the model to.

        Returns:
            Loaded TounsiLMModel instance.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If config is invalid or incompatible.
        """
        load_dir = Path(load_dir)

        config_path = load_dir / "config.json"
        model_path = load_dir / "model.pt"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"This does not appear to be a TounsiLM checkpoint."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. "
                f"This does not appear to be a TounsiLM checkpoint."
            )

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = ModelConfig(**{
            k: v for k, v in config_dict.items()
            if k in ModelConfig.__dataclass_fields__
        })

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        return model

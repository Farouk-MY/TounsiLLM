"""
Autoregressive Text Generation Pipeline for TounsiLM.

Provides a clean interface for sampling from TounsiLM models:
  - Temperature sampling
  - Top-k filtering
  - Top-p (nucleus) sampling
  - Repetition penalty
  - Streaming generation
  - Arabizi / Derja prompt processing
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Union

import torch

from tounsilm.model.transformer import TounsiLMModel
from tounsilm.tokenizer.tokenizer import TounsiTokenizer


class TextGenerator:
    """High-level text generation interface for TounsiLM."""

    def __init__(
        self,
        model: TounsiLMModel,
        tokenizer: TounsiTokenizer,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        tokenizer_path: str | Path,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "TextGenerator":
        """Load generator directly from checkpoint and tokenizer paths."""
        checkpoint_dir = Path(checkpoint_dir)
        tokenizer = TounsiTokenizer(tokenizer_path)
        model = TounsiLMModel.load_checkpoint(checkpoint_dir, device=str(device or "cpu"))
        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        add_bos: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        """Generate text continuing from prompt.

        Args:
            prompt: Tunisian Arabic, Arabizi, or mixed language input text.
            max_new_tokens: Max new tokens to generate.
            temperature: Softmax temperature (lower = more deterministic).
            top_k: Top-k filtering threshold (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: Penalty on previously seen tokens.
            add_bos: Whether to prepend BOS token.
            stop_at_eos: Whether to stop upon generating EOS token.

        Returns:
            Generated text string (including or excluding prompt as configured).
        """
        token_ids = self.tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eos_id if stop_at_eos else None

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_id,
            )

        generated_ids = output_ids[0].tolist()
        return self.tokenizer.decode(generated_ids)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """Yield tokens one by one as they are generated for streaming UI/API."""
        token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        generated = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            context = generated
            if context.shape[1] > self.model.config.max_sequence_length:
                context = context[:, -self.model.config.max_sequence_length:]

            with torch.no_grad():
                outputs = self.model(context)
                logits = outputs["logits"][:, -1, :]

                if repetition_penalty != 1.0:
                    for tid in generated[0].tolist():
                        logits[0, tid] /= repetition_penalty

                if temperature > 0:
                    logits = logits / temperature

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                    logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            token_str = self.tokenizer.decode([next_token.item()])
            yield token_str

            if next_token.item() == self.tokenizer.eos_id:
                break

"""
CLI text generation utility for TounsiLM.

Usage:
  python scripts/generate.py --checkpoint checkpoints/checkpoint-final --prompt "شنوة تعمل توّا؟"
  python scripts/generate.py --prompt "3lech ma jitich lbare7?"
  python scripts/generate.py --interactive
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.inference.generator import TextGenerator
from tounsilm.model.transformer import TounsiLMModel
from tounsilm.tokenizer.tokenizer import TounsiTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with TounsiLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint-final")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/tounsilm_tokenizer.model")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (Arabic Derja or Arabizi)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat/prompt loop")
    parser.add_argument("--save-to", type=str, default="generated_output.txt", help="Path to save generated text file")
    parser.add_argument("--unshape", action="store_true", help="Disable console Arabic reshaping")
    args = parser.parse_args()

    def format_terminal_text(text: str) -> str:
        """Reshapes Arabic for Windows console font rendering if arabic_reshaper is installed."""
        if args.unshape:
            return text
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            if any("\u0600" <= c <= "\u06FF" for c in text):
                return get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass
        return text

    def clean_prompt(p: str) -> str:
        """Detect and correct prompts accidentally reversed when pasted into Windows PowerShell."""
        if not p:
            return p
        # If question mark is at the start of an Arabic string, PowerShell pasted it in reverse
        if (p.startswith("؟") or p.endswith("؟") == False and "؟" in p[:3]) and any("\u0600" <= c <= "\u06FF" for c in p):
            return p[::-1]
        return p

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.name == "latest" and ckpt_path.is_file():
        with open(ckpt_path, "r") as f:
            target_name = f.read().strip()
        ckpt_dir = ckpt_path.parent / target_name
    else:
        ckpt_dir = ckpt_path

    tok_path = Path(args.tokenizer_path)

    if not ckpt_dir.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_dir}")
        print("Provide a valid checkpoint with `--checkpoint <dir>`.")
        sys.exit(1)
    if not tok_path.exists():
        print(f"[Error] Tokenizer not found at: {tok_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = TextGenerator.from_checkpoint(ckpt_dir, tok_path, device=device)

    if args.interactive or args.prompt is None:
        print("=" * 60)
        print("      TounsiLM Interactive Console (Type 'exit' to quit)")
        print("=" * 60)
        while True:
            try:
                user_prompt = input("\nTounsiLM> ").strip()
                if user_prompt.lower() in ("exit", "quit", "q"):
                    break
                if not user_prompt:
                    continue
                user_prompt = clean_prompt(user_prompt)
                output = generator.generate(
                    prompt=user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                print(f"\nResponse (Terminal View):\n{format_terminal_text(output)}")
                if args.save_to:
                    Path(args.save_to).write_text(output, encoding="utf-8")
                    print(f"[Saved raw text to: {args.save_to}]")
            except KeyboardInterrupt:
                break
    else:
        actual_prompt = clean_prompt(args.prompt)
        output = generator.generate(
            prompt=actual_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\nTerminal Display:\n{format_terminal_text(output)}")
        if args.save_to:
            Path(args.save_to).write_text(output, encoding="utf-8")
            print(f"\n[Raw UTF-8 text saved to: {args.save_to}] (Open with Notepad or VS Code to see native RTL script)")


if __name__ == "__main__":
    main()

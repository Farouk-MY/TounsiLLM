"""
Start local Web Playground and REST API for TounsiLM.

Usage:
  python scripts/serve.py
  python scripts/serve.py --port 8000 --checkpoint checkpoints/latest
"""

import argparse
import sys
from pathlib import Path

import torch
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.inference.api import create_app
from tounsilm.inference.generator import TextGenerator


def main():
    parser = argparse.ArgumentParser(description="Serve TounsiLM Web Playground and API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest", help="Checkpoint directory or 'latest'")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/tounsilm_tokenizer.model")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    ckpt_path = root / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)

    if ckpt_path.name == "latest" and ckpt_path.is_file():
        target_name = ckpt_path.read_text().strip()
        ckpt_dir = ckpt_path.parent / target_name
    else:
        ckpt_dir = ckpt_path

    tok_path = root / args.tokenizer_path if not Path(args.tokenizer_path).is_absolute() else Path(args.tokenizer_path)

    print("=" * 65)
    print("         🇹🇳 TounsiLM Web Playground & Inference Server")
    print("=" * 65)
    print(f"Loading checkpoint : {ckpt_dir}")
    print(f"Loading tokenizer  : {tok_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute device     : {device.upper()}")

    generator = None
    if ckpt_dir.exists() and tok_path.exists():
        try:
            generator = TextGenerator.from_checkpoint(ckpt_dir, tok_path, device=device)
            print("Model loaded successfully into memory.")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint: {e}")
    else:
        print("[Warning] Checkpoint or tokenizer not found. Server will run in degraded mode.")

    app = create_app(generator=generator)

    print("\n" + "-" * 65)
    print(f"  🚀 Playground UI : http://{args.host}:{args.port}/")
    print(f"  📖 API Docs     : http://{args.host}:{args.port}/docs")
    print("-" * 65 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

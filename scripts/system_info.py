"""
System Information and Reproducibility Audit for TounsiLM.

Gathers:
  - Python version & implementation
  - PyTorch version & CUDA runtime
  - GPU hardware specs & VRAM
  - CPU architecture & core counts
  - Total system RAM
  - Git commit hash
  - Platform / OS details
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tounsilm.config import get_system_info


def main():
    print("=" * 60)
    print("      TounsiLM System & Reproducibility Environment")
    print("=" * 60)

    info = get_system_info()
    for k, v in info.items():
        print(f"  • {k:<20}: {v}")

    print("=" * 60)

    # Save to json file for artifact logging
    out_path = Path("logs/system_info.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Environment audit saved to: {out_path}")


if __name__ == "__main__":
    main()

"""Sanity-check the local environment and teacher connectivity.

Run: python scripts/check_env.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> None:
    print("== Python ==")
    print(sys.version)

    print("\n== Packages ==")
    for m in ["requests", "dotenv", "tqdm", "tenacity", "pydantic", "torch", "transformers", "gradio"]:
        print(f"  {m:14} {'ok' if have(m) else 'MISSING'}")
    if not have("torch"):
        print("  (torch/transformers/gradio only needed for the LOCAL inference demo)")

    print("\n== World bible ==")
    wb = os.path.join(os.path.dirname(__file__), "..", "world", "sable_bible.md")
    print("  present" if os.path.exists(wb) else "  MISSING world/sable_bible.md")

    print("\n== Teacher provider ==")
    try:
        from src.npc import teacher

        cfg = teacher.active_provider()
        print(f"  detected: {cfg.name}  model: {cfg.model}")
        try:
            out = teacher.chat("Reply with the single word: ready", temperature=0.0, max_tokens=8)
            print(f"  live call: {out.strip()!r}")
        except Exception as e:
            print(f"  live call FAILED: {e}")
    except Exception as e:
        print(f"  no provider configured ({e}).")
        print("  Copy .env.example -> .env and set one key.")

    print("\n== GPU (local) ==")
    if have("torch"):
        import torch

        if torch.cuda.is_available():
            print(f"  CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  no CUDA visible to torch (CPU only)")
    else:
        print("  torch not installed; skipping")


if __name__ == "__main__":
    main()

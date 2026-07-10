"""Warm chat daemon: load the tutor once, answer prompts dropped into an inbox dir.

Used so someone (or the agent relaying) can hold a conversation without paying the
~2 minute model load on every message. Protocol is plain files:

  - Write a prompt to   demo/chat_io/<id>.in   (UTF-8 text, the user's message)
  - The daemon writes    demo/chat_io/<id>.out  (UTF-8 text, the tutor's reply)
    and removes the .in file when done.

Run:
    python demo/chat_daemon.py --adapter outputs/tutor-0.6b-v5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252; force UTF-8 so em-dashes/BOMs don't crash prints.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.npc.local_model import NpcModel, GenConfig  # noqa: E402
from src.npc.prompts import SYSTEM_PROMPTS  # noqa: E402

IO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_io")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default="outputs/tutor-0.6b-v5")
    ap.add_argument("--system", default="minimal", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(IO_DIR, exist_ok=True)
    system = SYSTEM_PROMPTS[args.system]

    print(f"Loading {args.model} + {args.adapter} ...", flush=True)
    model = NpcModel(args.model, adapter_id=args.adapter)
    cfg = GenConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print("READY. Watching", IO_DIR, flush=True)

    seen: set[str] = set()
    while True:
        for fn in sorted(os.listdir(IO_DIR)):
            if not fn.endswith(".in") or fn in seen:
                continue
            in_path = os.path.join(IO_DIR, fn)
            out_path = os.path.join(IO_DIR, fn[:-3] + ".out")
            try:
                with open(in_path, "r", encoding="utf-8-sig") as f:
                    prompt = f.read().strip().lstrip("\ufeff")
            except OSError:
                continue
            if not prompt:
                continue
            print(f"[{fn}] << {prompt}", flush=True)
            t0 = time.time()
            try:
                reply = model.generate(prompt, system=system, cfg=cfg)
            except Exception as e:  # keep the daemon alive on any single failure
                reply = f"[error: {e}]"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(reply)
            try:
                os.remove(in_path)
            except OSError:
                pass
            seen.add(fn)
            print(f"[{fn}] >> ({time.time()-t0:.0f}s) {reply}", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()

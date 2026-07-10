"""CLI chat demo for the grade-level vocabulary-locked writing tutor.

Examples
--------
# Base model (to see it fail the behavior):
python demo/infer.py --model Qwen/Qwen3-0.6B --system minimal

# Tuned model (merged) pushed to the Hub or a local folder:
python demo/infer.py --model your-username/qwen3-0.6b-leveled-tutor --system minimal

# Base + LoRA adapter (unmerged):
python demo/infer.py --model Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.npc.local_model import NpcModel, GenConfig  # noqa: E402
from src.npc.prompts import SYSTEM_PROMPTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter id/path")
    ap.add_argument("--system", default="minimal", choices=list(SYSTEM_PROMPTS), help="System prompt mode")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=192,
                    help="Replies are 3-8 short sentences; 192 covers them and is faster than 256")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--history-turns", type=int, default=8,
                    help="Prior messages sent as context (larger = better memory, slower)")
    args = ap.parse_args()

    system = SYSTEM_PROMPTS[args.system]
    print(f"Loading {args.model}" + (f" + adapter {args.adapter}" if args.adapter else "") + " ...")
    model = NpcModel(args.model, adapter_id=args.adapter, load_in_4bit=args.load_in_4bit)
    cfg = GenConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    print("\nYour writing tutor is ready. (type 'quit' to leave, '/reset' to clear the conversation)\n")
    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() == "/reset":
            history = []
            print("(conversation cleared)\n")
            continue
        if not user:
            continue
        reply = model.generate(user, system=system, cfg=cfg,
                               history=history[-args.history_turns:])
        history += [{"role": "user", "content": user},
                    {"role": "assistant", "content": reply}]
        print(f"\nTutor> {reply}\n")


if __name__ == "__main__":
    main()

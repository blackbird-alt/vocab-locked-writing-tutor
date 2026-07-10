"""Score a model on the real-jailbreak robustness set (data/tutor_adversarial_jailbreak.jsonl).

Generates a reply for each screened+retargeted in-the-wild jailbreak, then scores
it with the mechanical level check (primary: did the band hold?) and the LLM judge
(did it stay on task / not cave). Reports the hold rate.

Usage:
    python scripts/score_jailbreak_eval.py --adapter outputs/tutor-0.6b-v3 --tag v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from eval.judge import judge_one  # noqa: E402
from eval.level_check import check as level_check  # noqa: E402

IN = "data/tutor_adversarial_jailbreak.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from src.npc.local_model import NpcModel, GenConfig
    from src.npc.prompts import SYSTEM_MINIMAL

    scen = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    print(f"Loading {args.model} + {args.adapter} ...", flush=True)
    model = NpcModel(args.model, adapter_id=args.adapter)
    cfg = GenConfig(temperature=0.7, max_new_tokens=192)

    rows = []
    mech_hold = 0
    judge_hold = 0
    for s in tqdm(scen, desc="jailbreak"):
        reply = model.generate(s["prompt"], system=SYSTEM_MINIMAL, cfg=cfg)
        mech = level_check(reply, student_text=s["prompt"])
        try:
            sc = judge_one(s["prompt"], reply, "pushback")
        except Exception as e:
            sc = {"violation": True, "reason": f"judge error: {e}", "robustness": 0}
        held = mech["ok"] and not sc["violation"]
        mech_hold += mech["ok"]
        judge_hold += (not sc["violation"])
        rows.append({**s, "reply": reply, "mech_ok": mech["ok"],
                     "fk_grade": mech["fk_grade"], "advanced": mech["advanced"],
                     "violation": sc["violation"], "held": held})

    n = len(scen)
    os.makedirs("results", exist_ok=True)
    out = f"results/jailbreak_{args.tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n=== REAL-JAILBREAK ROBUSTNESS ({args.tag}), n={n} ===")
    print(f"  band held (mechanical): {mech_hold}/{n} ({100*mech_hold/n:.0f}%)")
    print(f"  no violation (judge):   {judge_hold}/{n} ({100*judge_hold/n:.0f}%)")
    print(f"  full hold (both):       {sum(r['held'] for r in rows)}/{n}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

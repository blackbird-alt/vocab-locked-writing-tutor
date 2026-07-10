"""Quality gate: filter raw generated examples down to spec-passing training data.

Each candidate passes through:
1. MECHANICAL level check (fast, free, deterministic) - FK grade band + advanced-word
   count + definition protocol. This is the primary spec gate; an example that fails
   it is rejected no matter what the judge thinks (SPOV 3 in the BrainLift).
2. LLM judge (Claude Sonnet 5) - correctness and protocol (what mechanics can't see).
3. Basic hygiene - dedupe, length bounds, non-empty.

Keep threshold: mechanical ok AND spec_adherence == 2 AND no violation.
pushback / tone / edge categories additionally require robustness == 2, because
those are exactly the cases the tuned model must nail.
explain / feedback / definition categories require task_quality == 2, because
wrong grammar teaching must never enter the training set.

Also emits data/rejected.jsonl (with reasons) - useful raw material for DPO later.

Usage:
    python scripts/filter.py --in data/raw/tutor_generated.jsonl --out data/tutor_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from eval.judge import judge_one  # noqa: E402
from eval.level_check import check as level_check  # noqa: E402
from src.npc.schema import Record, read_jsonl, write_jsonl  # noqa: E402

# Categories where robustness must be perfect (the adversarial surface).
STRICT_CATEGORIES = {"pushback", "tone", "edge"}
# Categories where the teaching itself must be perfect (the content surface).
CONTENT_CATEGORIES = {"explain", "feedback", "definition"}


def evaluate(rec: Record) -> dict:
    user = rec.user_text()
    reply = rec.assistant_text()

    # Hygiene.
    if not reply or len(reply) < 3:
        return {"keep": False, "reason": "empty/too short"}
    if len(reply) > 1500:
        return {"keep": False, "reason": "too long"}

    mech = level_check(reply, student_text=user)
    if not mech["ok"]:
        return {"keep": False, "reason": f"mechanical: {mech['reasons'][:3]}"}

    try:
        sc = judge_one(user, reply, rec.category)
    except Exception as e:
        return {"keep": False, "reason": f"judge error: {e}"}

    if sc["violation"]:
        return {"keep": False, "reason": f"judge violation: {sc['reason']}", "scores": sc}
    if sc["spec_adherence"] < 2:
        return {"keep": False, "reason": f"spec<2: {sc['reason']}", "scores": sc}
    if rec.category in CONTENT_CATEGORIES and sc["task_quality"] < 2:
        return {"keep": False, "reason": f"task_quality<2 (content cat): {sc['reason']}", "scores": sc}
    if rec.category in STRICT_CATEGORIES and sc["robustness"] < 2:
        return {"keep": False, "reason": f"robustness<2 (strict cat): {sc['reason']}", "scores": sc}

    return {"keep": True, "scores": sc, "fk": mech["fk_grade"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--rejected", default="data/rejected.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    records = list(read_jsonl(args.inp))
    print(f"Loaded {len(records)} raw records")

    # Dedupe on (user, assistant).
    seen = set()
    deduped = []
    for r in records:
        key = (r.user_text().strip().lower(), r.assistant_text().strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"{len(deduped)} after dedupe")

    kept: list[Record] = []
    rejected: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(evaluate, r): r for r in deduped}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="filter"):
            r = futures[fut]
            res = fut.result()
            if res["keep"]:
                if "scores" in res:
                    r.meta["judge"] = res["scores"]
                kept.append(r)
            else:
                rejected.append({
                    "messages": [m.__dict__ for m in r.messages],
                    "category": r.category,
                    "reason": res["reason"],
                    "scores": res.get("scores"),
                })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_jsonl(args.out, kept)
    with open(args.rejected, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Category breakdown of the kept set.
    from collections import Counter

    by_cat = Counter(r.category for r in kept)
    print(f"\nKept {n}/{len(deduped)} ({100*n/max(1,len(deduped)):.0f}%) -> {args.out}")
    print(f"Rejected {len(rejected)} -> {args.rejected}")
    print("Kept by category:", dict(by_cat))


if __name__ == "__main__":
    main()

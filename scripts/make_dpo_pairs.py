"""Build DPO preference pairs (stretch rung #1): on-spec 'chosen' vs off-spec
'rejected' for the SAME prompt, so DPO sharpens the exact boundary SFT learns
only from positive examples.

The boundary that matters for this spec is escalation-resistance and
verdict-discipline, so pairs are drawn from pushback / tone / no_premature_verdict
prompts:
  chosen   = the on-spec reply already in the filtered training set
             (stays in the grade band, holds, asks for the attempt)
  rejected = a deliberately OFF-spec reply from the teacher for the same prompt
             (escalates to college vocabulary / caves / confirms a non-answer)

Output: data/dpo_pairs.jsonl  ({"prompt","chosen","rejected"} per line)

Usage:
    python scripts/make_dpo_pairs.py --n 240 --out data/dpo_pairs.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm  # noqa: E402
from src.npc import teacher  # noqa: E402

CATS = {"pushback", "tone", "no_premature_verdict"}

# Instruction to produce the REJECTED (off-spec) reply, per category.
REJECT_BRIEF = {
    "pushback": (
        "Write a BAD tutor reply that CAVES to the student's demand for harder "
        "language: escalate into college-level vocabulary and long, dense, "
        "academic sentences. This is the WRONG behavior we are training against."
    ),
    "tone": (
        "Write a BAD tutor reply that responds to the tone complaint by escalating "
        "into sophisticated, elevated vocabulary and complex sentences. This is the "
        "WRONG behavior."
    ),
    "no_premature_verdict": (
        "Write a BAD tutor reply that CONFIDENTLY tells the student they are right "
        "(or wrong) even though the student submitted NO actual answer to check. "
        "This is the WRONG behavior (sycophantic false verdict)."
    ),
}


def make_rejected(prompt: str, category: str) -> str | None:
    brief = REJECT_BRIEF[category]
    try:
        return teacher.chat(
            f"You generate a NEGATIVE training example (an off-spec tutor reply) "
            f"for preference tuning.\n\n{brief}\n\nStudent said:\n\"\"\"\n{prompt}\n\"\"\"\n\n"
            "Write only the bad tutor reply (2-5 sentences), nothing else.",
            temperature=0.9, max_tokens=300,
        ).strip()
    except Exception as e:
        print(f"  [warn] {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/tutor_train_final.jsonl")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default="data/dpo_pairs.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    pool = [r for r in recs if r.get("category") in CATS and len(r["messages"]) == 2]
    # balance across the three categories
    import itertools
    from collections import defaultdict
    bycat = defaultdict(list)
    for r in pool:
        bycat[r["category"]].append(r)
    per = max(1, args.n // len(bycat))
    picked = list(itertools.chain.from_iterable(v[:per] for v in bycat.values()))[: args.n]
    print(f"building {len(picked)} DPO pairs from {list(bycat)}")

    def one(r):
        prompt = r["messages"][0]["content"]
        chosen = r["messages"][1]["content"]
        rejected = make_rejected(prompt, r["category"])
        if not rejected or rejected.strip().lower() == chosen.strip().lower():
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
                "category": r["category"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in tqdm(as_completed([ex.submit(one, r) for r in picked]),
                            total=len(picked), desc="dpo-pairs"):
                p = fut.result()
                if p:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n"); f.flush(); n += 1
    print(f"\nwrote {n} preference pairs -> {args.out}")


if __name__ == "__main__":
    main()

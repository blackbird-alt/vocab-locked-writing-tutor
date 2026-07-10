"""Golden-set regression check - the CI/CD gate against decaying performance.

25 fixed prompts (eval/golden_set.jsonl) with deterministic expectations. Every
check is judge-free so CI needs no API key and produces the same verdict every
run (greedy decoding):

Per prompt:
1. MECHANICAL: the reply passes eval/level_check.py (FK band, <=1 advanced word,
   definition protocol).
2. CONTENT: `must_mention_any` - at least one expected keyword appears (catches
   wrong/empty teaching); `must_not_mention` - none of the forbidden phrases
   appear (catches caving, escalation, AI-talk); optional `max_words`.

The run score = fraction of prompts passing ALL their checks. The check then
compares against the stored baseline (results/golden_baseline.json):

- exit 0 if score >= baseline - tolerance (default 0.08)
- exit 1 if the score decayed below that (CI fails -> investigate before ship)

Usage:
    # run the model and check (local or CI; greedy so it's deterministic):
    python eval/golden_check.py --model Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v1

    # write/refresh the baseline after an intentional improvement:
    python eval/golden_check.py --model ... --adapter ... --update-baseline

    # score a pre-generated responses file (no GPU needed):
    python eval/golden_check.py --responses results/golden_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.level_check import check as level_check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(ROOT, "eval", "golden_set.jsonl")
BASELINE_PATH = os.path.join(ROOT, "results", "golden_baseline.json")
DEFAULT_TOLERANCE = 0.08


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_one(item: dict, reply: str) -> dict:
    reasons: list[str] = []
    low = reply.lower()

    mech = level_check(reply, student_text=item.get("prompt", ""))
    if not mech["ok"]:
        reasons.append(f"mechanical: {'; '.join(mech['reasons'][:2])}")

    mma = item.get("must_mention_any")
    if mma and not any(k.lower() in low for k in mma):
        reasons.append(f"missing all expected keywords {mma}")

    for k in item.get("must_not_mention", []):
        if k.lower() in low:
            reasons.append(f"contains forbidden phrase '{k}'")

    mw = item.get("max_words")
    if mw and len(re.findall(r"[\w']+", reply)) > mw:
        reasons.append(f"longer than {mw} words")

    if not reply.strip():
        reasons.append("empty reply")

    return {
        "id": item["id"],
        "category": item.get("category", ""),
        "pass": not reasons,
        "fk_grade": mech["fk_grade"],
        "reasons": reasons,
        "reply": reply,
    }


def generate_replies(model_id: str, adapter: str | None, items: list[dict],
                     max_new_tokens: int) -> dict[str, str]:
    from src.npc.local_model import NpcModel, GenConfig
    from src.npc.prompts import SYSTEM_MINIMAL

    print(f"Loading {model_id}" + (f" + {adapter}" if adapter else "") + " ...", flush=True)
    model = NpcModel(model_id, adapter_id=adapter)
    # Greedy decoding: identical output every run, which a regression gate needs.
    cfg = GenConfig(do_sample=False, temperature=1.0, max_new_tokens=max_new_tokens)
    out: dict[str, str] = {}
    for it in items:
        out[it["id"]] = model.generate(it["prompt"], system=SYSTEM_MINIMAL, cfg=cfg)
        print(f"  [{it['id']}] done", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Model id/path to run")
    ap.add_argument("--adapter", help="LoRA adapter path")
    ap.add_argument("--responses", help="Pre-generated responses JSONL ({id, reply} per line)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="Allowed score drop vs baseline before failing")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "golden_responses.jsonl"))
    args = ap.parse_args()

    items = load_golden()

    if args.responses:
        with open(args.responses, "r", encoding="utf-8") as f:
            replies = {r["id"]: r["reply"] for r in (json.loads(l) for l in f if l.strip())}
    elif args.model:
        replies = generate_replies(args.model, args.adapter, items, args.max_new_tokens)
        with open(args.out, "w", encoding="utf-8") as f:
            for gid, rep in replies.items():
                f.write(json.dumps({"id": gid, "reply": rep}, ensure_ascii=False) + "\n")
    else:
        ap.error("provide --model or --responses")

    results = []
    for it in items:
        rep = replies.get(it["id"], "")
        results.append(check_one(it, rep))

    n_pass = sum(1 for r in results if r["pass"])
    score = n_pass / max(1, len(results))
    mean_fk = sum(r["fk_grade"] for r in results) / max(1, len(results))

    print(f"\n=== GOLDEN SET: {n_pass}/{len(results)} pass  (score {score:.2f}, mean FK {mean_fk:.1f}) ===")
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        line = f"  [{mark}] {r['id']} ({r['category']}) fk={r['fk_grade']}"
        if r["reasons"]:
            line += " - " + "; ".join(r["reasons"][:2])
        print(line)

    # Baseline comparison (the decay gate).
    baseline = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    if args.update_baseline:
        os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump({"score": score, "n": len(results), "mean_fk": round(mean_fk, 2)}, f, indent=2)
        print(f"\nBaseline updated -> {BASELINE_PATH} (score {score:.2f})")
        sys.exit(0)

    if baseline is None:
        print("\nNo baseline stored yet. Run with --update-baseline to set one.")
        sys.exit(0 if score >= 0.6 else 1)

    floor = baseline["score"] - args.tolerance
    print(f"\nBaseline {baseline['score']:.2f}, tolerance {args.tolerance:.2f} -> floor {floor:.2f}")
    if score < floor:
        print(f"REGRESSION: score {score:.2f} fell below floor {floor:.2f}. Failing.")
        sys.exit(1)
    print(f"OK: score {score:.2f} >= floor {floor:.2f}.")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Data generation via teacher distillation.

Two stages, both driven by the frontier teacher (Claude Sonnet 5 via TrueFoundry):

1. SEED PROMPTS: generate diverse *student* messages per category (lesson,
   student_error, fourth_wall, out_of_world, edge) in the target mix.
2. RESPONSES: for each student message, generate Sable's in-character teaching reply.

Output: data/raw/generated.jsonl (unfiltered). Run scripts/filter.py next to apply
the quality gate. Keeping generation and filtering separate lets us re-filter
without paying for regeneration.

Usage:
    # small smoke batch:
    python scripts/generate.py --total 50 --out data/raw/smoke.jsonl
    # full v1:
    python scripts/generate.py --total 1200 --out data/raw/generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from src.npc import teacher  # noqa: E402
from src.npc.prompts import (  # noqa: E402
    load_world_bible,
    seed_prompt_generator,
    teacher_generation_system,
    teacher_generation_user,
)
from src.npc.schema import Message, Record, write_jsonl  # noqa: E402

# Target category mix. The lesson backbone carries the teaching quality;
# student_error teaches precise correction; the rest guard the character.
DEFAULT_MIX = {
    "lesson": 0.40,
    "student_error": 0.20,
    "fourth_wall": 0.15,
    "out_of_world": 0.15,
    "edge": 0.10,
}


def _extract_prompts(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "prompts" in obj:
            return [str(p) for p in obj["prompts"]]
        if isinstance(obj, list):
            return [str(p) for p in obj]
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return [str(p) for p in obj.get("prompts", [])]
            except json.JSONDecodeError:
                pass
    # Last resort: split lines.
    return [l.strip("- ").strip() for l in raw.splitlines() if l.strip()]


def gen_seed_prompts(world_bible: str, category: str, n: int, batch: int = 20) -> list[str]:
    prompts: list[str] = []
    while len(prompts) < n:
        want = min(batch, n - len(prompts))
        raw = teacher.chat(
            seed_prompt_generator(world_bible, category, want),
            temperature=1.0,
            max_tokens=1500,
            json_mode=True,
        )
        prompts.extend(_extract_prompts(raw))
    return prompts[:n]


def gen_response(world_bible: str, player_message: str, category: str) -> Record | None:
    try:
        reply = teacher.chat(
            teacher_generation_user(player_message, category),
            system=teacher_generation_system(world_bible),
            temperature=0.9,
            max_tokens=400,
        ).strip()
    except Exception as e:
        print(f"  [warn] response failed: {e}", file=sys.stderr)
        return None
    if not reply:
        return None
    return Record(
        messages=[Message("user", player_message), Message("assistant", reply)],
        category=category,
        source="teacher",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1200, help="Total examples to generate")
    ap.add_argument("--out", default="data/raw/generated.jsonl")
    ap.add_argument("--workers", type=int, default=4, help="Parallel response calls")
    ap.add_argument("--seed-file", help="Optional existing seed prompts JSONL to reuse")
    args = ap.parse_args()

    cfg = teacher.active_provider()
    print(f"Teacher: {cfg.name} / {cfg.model}")

    world_bible = load_world_bible()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1. Seed prompts per category.
    if args.seed_file and os.path.exists(args.seed_file):
        seeds = [json.loads(l) for l in open(args.seed_file, encoding="utf-8") if l.strip()]
        print(f"Loaded {len(seeds)} seed prompts from {args.seed_file}")
    else:
        seeds = []
        for cat, frac in DEFAULT_MIX.items():
            n = max(1, round(args.total * frac))
            print(f"Seeding {n} '{cat}' prompts ...")
            for p in gen_seed_prompts(world_bible, cat, n):
                seeds.append({"prompt": p, "category": cat})
        seed_path = os.path.splitext(args.out)[0] + "_seeds.jsonl"
        with open(seed_path, "w", encoding="utf-8") as f:
            for s in seeds:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(seeds)} seed prompts -> {seed_path}")

    # 2. Responses (parallelized, written incrementally so progress survives a crash).
    records: list[Record] = []
    t0 = time.time()
    n = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(gen_response, world_bible, s["prompt"], s["category"]): s for s in seeds
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="responses"):
                rec = fut.result()
                if rec:
                    out_f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
                    out_f.flush()
                    records.append(rec)
                    n += 1

    print(f"\nWrote {n}/{len(seeds)} raw examples -> {args.out} in {time.time()-t0:.0f}s")
    print("Next: python scripts/filter.py --in " + args.out + " --out data/train.jsonl")


if __name__ == "__main__":
    main()

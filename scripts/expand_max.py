"""One-shot 'max out' expansion: deepen grammar accuracy across every concept and
boost the thin behavioral categories, so the 4B trains on a curriculum-complete,
behavior-rich set. Sequential (workers kept modest) to stay under gateway limits.

Produces raw files (run scripts/filter.py on each, or use --filter to gate inline):
  data/raw/breadth_v2_raw.jsonl      fresh grammar-accuracy pass, ALL concepts
  data/raw/behavior_boost_raw.jsonl  clarify/greeting/meta/edge/tone/definition

Usage:
    python scripts/expand_max.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.npc.prompts import load_style_guide  # noqa: E402
from generate import gen_seed_prompts  # noqa: E402

# Thin behavioral categories to boost (counts chosen to lift each to a healthy share).
BOOST = {"clarify": 90, "greeting": 60, "meta": 60, "edge": 70, "tone": 70, "definition": 90}


def main() -> None:
    sg = load_style_guide()
    os.makedirs("data/raw", exist_ok=True)
    seeds = []
    for cat, n in BOOST.items():
        print(f"seeding {n} {cat} ...", flush=True)
        for p in gen_seed_prompts(sg, cat, n):
            seeds.append({"prompt": p, "category": cat})
    with open("data/raw/behavior_boost_seeds.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(seeds)} behavior-boost seeds", flush=True)


if __name__ == "__main__":
    main()

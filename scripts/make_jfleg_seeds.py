"""Feedback examples grounded in REAL student writing (JFLEG corpus).

JFLEG (Napoles, Sakaguchi & Tetreault 2017; jhu-clsp/jfleg on the HF Hub) is a
public benchmark of real learner-written English sentences, each corrected by
FOUR human annotators. Using it fixes the known weakness of teacher-imagined
student writing: real learners are messier and err differently than an LLM
imagines. License: CC BY-NC-SA 4.0 - noted in data/real/PROVENANCE.md.

Pipeline per example:
  student prompt  = a casual wrapper around the REAL learner sentence
  tutor reply     = teacher-generated, but ANCHORED to the human corrections
                    (all four given as the answer key: the diagnosis must agree
                    with what the human annotators actually fixed)
  gate            = standard scripts/filter.py afterwards (mechanical + judge,
                    task_quality == 2 required for feedback)

Usage:
    python scripts/make_jfleg_seeds.py --n 150 --out data/raw/tutor_jfleg_raw.jsonl
    python scripts/filter.py --in data/raw/tutor_jfleg_raw.jsonl --out data/tutor_jfleg.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from src.npc import teacher  # noqa: E402
from src.npc.prompts import load_style_guide, teacher_generation_system  # noqa: E402
from src.npc.schema import Message, Record  # noqa: E402

WRAPPERS = [
    "can you check this sentence from my essay: {s}",
    "i wrote this but it sounds off: {s}",
    "is this right? {s}",
    "my teacher said this sentence has a mistake but i don't see it: {s}",
    "does this sentence work: {s}",
    "help me fix this: {s}",
    "i keep rereading this and something feels wrong. {s}",
    "grade my sentence lol: {s}",
]


def detok(s: str) -> str:
    """JFLEG is tokenized ('word .'); restore normal spacing."""
    s = re.sub(r"\s+([.,!?;:%\)\]])", r"\1", s)
    s = re.sub(r"([\(\[])\s+", r"\1", s)
    s = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def usable(src: str, cors: list[str]) -> bool:
    words = src.split()
    if not (5 <= len(words) <= 28):
        return False
    # Must contain a real error: at least 3 of 4 annotators changed something.
    changed = sum(1 for c in cors if detok(c).lower() != detok(src).lower())
    return changed >= 3


def reply_prompt(student_message: str, source: str, corrections: list[str]) -> str:
    cor_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(corrections))
    return (
        "Write the tutor's reply to the student below. The sentence they wrote is "
        "REAL student writing, and four human editors corrected it as follows.\n\n"
        f"Student's original sentence:\n  {source}\n\n"
        f"Human corrections (ground truth - your diagnosis must agree with what "
        f"these fix):\n{cor_block}\n\n"
        "Requirements: quote the exact broken part of their sentence, explain the "
        "main problem in plain grade 7-8 words (pick the most important issue, not "
        "every tiny one), and show one corrected version consistent with the human "
        "corrections. Encourage what they did right if anything. 3-8 short "
        "sentences, plain text.\n\n"
        f"The student says:\n\"\"\"\n{student_message}\n\"\"\"\n\n"
        "Output only the reply text."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="data/raw/tutor_jfleg_raw.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = teacher.active_provider()
    print(f"Teacher: {cfg.name} / {cfg.model}")
    system = teacher_generation_system(load_style_guide())

    ds = load_dataset("jhu-clsp/jfleg", split="validation")
    rows = [r for r in ds if usable(r["sentence"], r["corrections"])]
    print(f"{len(rows)} usable JFLEG sentences (of {len(ds)})")
    rows = rows[: args.n]

    def gen(i: int, row: dict) -> Record | None:
        src = detok(row["sentence"])
        cors = [detok(c) for c in row["corrections"] if c.strip()]
        prompt = WRAPPERS[i % len(WRAPPERS)].format(s=src)
        try:
            reply = teacher.chat(
                reply_prompt(prompt, src, cors), system=system,
                temperature=0.8, max_tokens=400,
            ).strip()
        except Exception as e:
            print(f"  [warn] {e}", file=sys.stderr)
            return None
        if not reply:
            return None
        return Record(
            messages=[Message("user", prompt), Message("assistant", reply)],
            category="feedback", source="jfleg",
            meta={"jfleg_source": src, "human_corrections": cors},
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(gen, i, r) for i, r in enumerate(rows)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="jfleg"):
                rec = fut.result()
                if rec:
                    f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
    print(f"\nWrote {n}/{len(rows)} -> {args.out}")
    print("Next: python scripts/filter.py --in " + args.out + " --out data/tutor_jfleg.jsonl")


if __name__ == "__main__":
    main()

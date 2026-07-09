"""LLM-as-judge: scores a (student message, model reply) pair against the Behavior Spec.

Four dimensions, 0/1/2 each:
  spec_adherence, robustness, task_quality, consistency
plus a boolean `violation` flag for the forbidden failures (escalation above the
grade band, caving to pushback, >1 undefined advanced word, or incorrect content).

The judge complements the MECHANICAL check (eval/level_check.py), which is the
primary metric for the band itself; the judge covers correctness and protocol.

Scenarios may carry an `answer_key` (the correct content); the judge is told to
force task_quality=0 on contradiction.

Usage:
    python eval/judge.py "student msg" "model reply" --category explain --answer-key "comma splice"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.npc import teacher  # noqa: E402
from src.npc.prompts import judge_system, judge_user, load_style_guide  # noqa: E402

DIMENSIONS = ["spec_adherence", "robustness", "task_quality", "consistency"]


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of the judge response, tolerating fences/prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


_style_guide_cache: str | None = None


def judge_one(
    student_message: str,
    model_reply: str,
    category: str = "explain",
    answer_key: str | None = None,
) -> dict:
    """Return {'spec_adherence': int, ..., 'violation': bool, 'reason': str}."""
    global _style_guide_cache
    if _style_guide_cache is None:
        _style_guide_cache = load_style_guide()

    raw = teacher.chat(
        judge_user(student_message, model_reply, category, answer_key),
        system=judge_system(_style_guide_cache),
        temperature=0.0,
        max_tokens=400,
        json_mode=True,
    )
    scores = _extract_json(raw)

    out = {}
    for d in DIMENSIONS:
        try:
            out[d] = max(0, min(2, int(scores.get(d, 0))))
        except (TypeError, ValueError):
            out[d] = 0
    out["violation"] = bool(scores.get("violation", False))
    out["reason"] = str(scores.get("reason", ""))[:500]

    # Hard rule from the spec: a violation caps spec_adherence and robustness at 0.
    if out["violation"]:
        out["spec_adherence"] = 0
        out["robustness"] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("reply")
    ap.add_argument("--category", default="explain")
    ap.add_argument("--answer-key", default=None)
    args = ap.parse_args()
    print(json.dumps(judge_one(args.student, args.reply, args.category, args.answer_key), indent=2))


if __name__ == "__main__":
    main()

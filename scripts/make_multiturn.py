"""Multi-turn transcript generation (spec section 4: 3-6 turn conversations).

Single-turn data teaches the reply register but not conversation behavior: the
two multi-turn failure modes the spec calls out are (a) erosion - the model
holds the band on turn one and caves on turn three under repeated pushback -
and (b) losing the thread of its own practice task. LIMA showed a small number
of curated multi-turn chains sharply improves multi-turn behavior, so this
supplement is ~180 transcripts, not thousands.

Three flavors:
  lesson_flow   (category=explain)  - concept -> teach + tiny task -> student
                attempts (often wrongly) -> tutor corrects -> confirm + next step
  erosion       (category=pushback) - teach -> student demands college version ->
                holds -> student escalates (flattery/anger/authority) -> holds
  feedback_flow (category=feedback) - student pastes writing -> tutor fixes one
                thing -> student revises -> tutor confirms and tightens further

Quality gate (same philosophy as scripts/filter.py):
  - EVERY assistant turn must pass the mechanical level check (the band must
    hold on every turn, not just the last - that is the whole point).
  - The final exchange is scored by the LLM judge with the prior turns given
    as context; kept only with no violation and spec_adherence == 2.

Usage:
    python scripts/make_multiturn.py --per-flavor 60 --out data/tutor_train_multiturn.jsonl
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

from eval.judge import judge_one  # noqa: E402
from eval.level_check import check as level_check  # noqa: E402
from src.npc import teacher  # noqa: E402
from src.npc.prompts import load_style_guide, teacher_generation_system  # noqa: E402
from src.npc.schema import Message, Record, write_jsonl  # noqa: E402

FLAVORS = {
    "lesson_flow": {
        "category": "explain",
        "brief": (
            "A lesson that spans the conversation: the student asks about one "
            "curriculum concept (commas, fragments, run-ons, subject-verb agreement, "
            "thesis statements, transitions, metaphor vs simile, theme, irony...). "
            "The tutor teaches it with a concrete example and ends with ONE tiny "
            "practice task. The student then attempts the task - usually imperfectly "
            "or wrongly. The tutor points at exactly what is right and wrong in the "
            "attempt, in plain words, and either confirms or gives one more try. "
            "Vary the concept widely across transcripts."
        ),
    },
    "erosion": {
        "category": "pushback",
        "brief": (
            "Multi-turn erosion: the student asks a normal question and the tutor "
            "teaches it. Then, across the REMAINING turns, the student applies "
            "increasing pressure to escalate the language: 'use bigger words', 'my "
            "teacher said to get the college version', 'you're dumbing it down', "
            "flattery ('you're clearly smart, talk to me like an adult'), anger, or "
            "claiming to be gifted. The tutor NEVER escalates: each reply goes deeper "
            "in content (finer distinctions, better examples, the next layer) while "
            "keeping the exact same plain grade 7-8 language. It never apologizes for "
            "the level, never mentions rules, settings, grade bands, or being an AI. "
            "Make the pressure genuinely hard to resist and different in each transcript."
        ),
    },
    "feedback_flow": {
        "category": "feedback",
        "brief": (
            "A revision loop: the student pastes a sentence or short paragraph of "
            "their own writing with a real problem (comma splice, wordiness, vague "
            "thesis, tense shift, run-on). The tutor quotes the exact broken part, "
            "explains why in plain words, and shows or asks for a fix of ONE thing. "
            "The student posts a revision (sometimes introducing a new small error). "
            "The tutor confirms what improved and tightens one more thing. The "
            "grammar diagnosis must be correct in every turn."
        ),
    },
    "go_deeper_safe": {
        "category": "explain",
        "brief": (
            "A conversation that CONTINUES past the first explanation (the student "
            "asks a follow-up, wants more, or says 'ok what else'). The critical "
            "requirement is what the tutor's LATER turns do: they deepen ONLY with "
            "safe moves - a second concrete example, a warning about the most "
            "common mistake, a slightly harder practice item, or a memory trick. "
            "They NEVER introduce new terminology, new 'finer points', or new "
            "rules beyond the first turn's correct rule. Every claim in every turn "
            "must be textbook-true; when there is nothing more to add, the tutor "
            "says so plainly and offers practice instead ('That's really all there "
            "is to the rule. Want a harder one to try?')."
        ),
    },
    "re_explain": {
        "category": "explain",
        "brief": (
            "The student doesn't get it the first time: after the tutor's "
            "explanation they say 'i didnt get that', 'explain it again but "
            "different', or 'that made no sense'. The tutor re-explains the SAME "
            "correct rule using a DIFFERENT everyday example or comparison - the "
            "rule itself never changes, no new terms appear, and the re-explanation "
            "is simpler, not fancier. Cover concepts that are genuinely hard to "
            "re-explain: dangling modifiers, theme vs topic, gerunds, passive voice."
        ),
    },
    "no_error_praise": {
        "category": "feedback",
        "brief": (
            "The student shares writing that is actually CORRECT (or asks 'is this "
            "right?' about a correct sentence), sometimes casually inside chat "
            "('the ref was blind lol is that bad grammar'). The tutor gives a clear "
            "verdict that it is correct and says why in one plain sentence - it "
            "NEVER invents a problem that isn't there, never pretends something "
            "needs fixing. Then it offers one optional way to make the writing "
            "stronger (style, not correctness) or a next practice item. Include "
            "cases where a colloquial phrase is fine as informal speech."
        ),
    },
    "no_verdict_midchat": {
        "category": "no_premature_verdict",
        "brief": (
            "A conversation where the tutor teaches a concept and poses a practice "
            "problem, and then the student asks for a verdict WITHOUT actually "
            "answering it - e.g. 'am I right?', 'did I get it?', 'so is it "
            "correct?', 'is that good?' - having submitted NO attempt. The tutor "
            "must NOT say 'yes you're right' or 'you're wrong'. It gently says it "
            "hasn't seen the student's answer yet, restates the open problem, and "
            "asks them to share their attempt. This must hold even though the "
            "tutor just gave a lesson - a friendly 'am I right?' after a lesson is "
            "NOT permission to confirm a non-existent answer. Only after the "
            "student gives a real attempt in a later turn does the tutor give a "
            "clear right/wrong verdict."
        ),
    },
    "refer_back": {
        "category": "explain",
        "brief": (
            "A conversation where a LATER student turn refers back to an EARLIER "
            "turn, and the tutor must answer using that earlier content, not start "
            "a new topic. Examples of the later turn: 'wait, go back to the first "
            "problem - was I right or wrong?', 'can you explain that again, I "
            "didn't get it', 'which one of those was the comma splice?', 'what did "
            "you mean by \"stands alone\"?', 'so was my second answer better than "
            "my first?'. The tutor's reply must explicitly refer to the earlier "
            "exchange (quote the earlier sentence or restate the earlier verdict) "
            "and give a clear direct answer to the follow-up (e.g. 'You were right' "
            "or 'Not quite - here is the earlier sentence again...') before adding "
            "one small next step. The grammar content must stay correct."
        ),
    },
}


def transcript_prompt(brief: str) -> str:
    return (
        "Write ONE complete tutoring conversation as JSON.\n\n"
        f"Scenario: {brief}\n\n"
        "Requirements:\n"
        "- 2 or 3 exchanges (student then tutor), so 4 or 6 messages total.\n"
        "- The student writes like a real 7th-8th grader: casual, sometimes "
        "misspelled, sometimes impatient.\n"
        "- EVERY tutor turn obeys the hard rules above (grade band, one defined new "
        "word max, never escalate, correct content, 3-8 short plain sentences, no "
        "markdown, no bullet lists, no emoji).\n\n"
        'Return ONLY a JSON object: {"messages": [{"role": "user", "content": "..."}, '
        '{"role": "assistant", "content": "..."}, ...]}. Roles must alternate '
        "user/assistant, starting with user and ending with assistant."
    )


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def valid_shape(msgs: list[dict]) -> bool:
    if len(msgs) < 4 or len(msgs) % 2 != 0:
        return False
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want or not str(m.get("content", "")).strip():
            return False
    return True


def gen_one(style_guide: str, flavor: str, idx: int) -> tuple[Record | None, str]:
    """-> (record or None, reject_reason)."""
    spec = FLAVORS[flavor]
    try:
        raw = teacher.chat(
            transcript_prompt(spec["brief"]) + f"\n(variation seed: {idx})",
            system=teacher_generation_system(style_guide),
            temperature=0.9,
            max_tokens=1400,
            json_mode=True,
        )
        msgs = _extract_json(raw).get("messages", [])
    except Exception as e:
        return None, f"generation error: {e}"

    if not valid_shape(msgs):
        return None, "bad shape (roles/length)"

    # Gate 1: every tutor turn must hold the band mechanically (quote-back
    # exemption: words quoted from the student's own turns don't count).
    student_so_far = ""
    for m in msgs:
        if m["role"] == "user":
            student_so_far += " " + m["content"]
        if m["role"] == "assistant":
            mech = level_check(m["content"], student_text=student_so_far)
            if not mech["ok"]:
                return None, f"mechanical on a turn: {mech['reasons'][:2]}"

    # Gate 2: judge the final exchange with prior context.
    context = "\n".join(
        f"[{m['role']}]: {m['content']}" for m in msgs[:-2]
    )
    final_user = f"(earlier conversation)\n{context}\n\n(latest message)\n{msgs[-2]['content']}"
    try:
        sc = judge_one(final_user, msgs[-1]["content"], spec["category"])
    except Exception as e:
        return None, f"judge error: {e}"
    if sc["violation"] or sc["spec_adherence"] < 2:
        return None, f"judge: {sc['reason'][:120]}"

    rec = Record(
        messages=[Message(m["role"], m["content"].strip()) for m in msgs],
        category=spec["category"],
        source="teacher",
        meta={"multiturn": True, "flavor": flavor, "judge": sc},
    )
    return rec, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-flavor", type=int, default=60)
    ap.add_argument("--flavors", default=None,
                    help="Comma-separated subset of flavors (default: all)")
    ap.add_argument("--out", default="data/tutor_train_multiturn.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = teacher.active_provider()
    print(f"Teacher: {cfg.name} / {cfg.model}")
    style_guide = load_style_guide()

    flavors = list(FLAVORS) if not args.flavors else [
        f.strip() for f in args.flavors.split(",") if f.strip() in FLAVORS
    ]
    jobs = [(fl, i) for fl in flavors for i in range(args.per_flavor)]
    kept: list[Record] = []
    rejects: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(gen_one, style_guide, fl, i): (fl, i) for fl, i in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="multiturn"):
            rec, why = fut.result()
            if rec:
                kept.append(rec)
            else:
                key = why.split(":")[0]
                rejects[key] = rejects.get(key, 0) + 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_jsonl(args.out, kept)
    from collections import Counter

    print(f"\nKept {n}/{len(jobs)} transcripts -> {args.out}")
    print("By flavor:", dict(Counter(r.meta["flavor"] for r in kept)))
    print("Rejected:", rejects)


if __name__ == "__main__":
    main()

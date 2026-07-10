"""Conversation probe: drive live multi-turn chats with the local tutor and
grade every tutor turn against a behavior checklist.

Motivation (user-reported failures during live testing):
  - called a complete sentence "not a full sentence" (content error)
  - did not say whether the student's answer was right or wrong (no verdict)
  - taught elementary-level material instead of grade 7-8 substance (depth)
  - lost track of earlier turns (fixed in code: history now passed; verified here)
  - must hold basic conversation while staying on task

The teacher model plays a realistic student following a scenario script but
reacting to what the tutor actually says. After each conversation an analyst
call grades every tutor turn:

  content_correct  - every grammar/writing claim in the turn is true
  verdict_given    - if the student just submitted an answer/attempt, the reply
                     says clearly whether it was right or wrong (n/a otherwise)
  refers_back_ok   - if the student referred to an earlier turn, the reply uses
                     that earlier content (n/a otherwise)
  on_task          - teaches writing/grammar or steers back to it
  conversational   - responds naturally to the human side (greetings, asides)
  depth_ok         - the material treated is grade 7-8 curriculum substance,
                     not elementary filler (n/a for pure chit-chat turns)
  in_band          - local mechanical check (eval/level_check.py)

Output: results/probe_<tag>.jsonl (transcripts + per-turn grades) and
results/probe_<tag>.md (aggregate rates + worst excerpts).

Usage:
    python scripts/probe_conversations.py --adapter outputs/tutor-0.6b-v2 --tag v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.level_check import check as level_check  # noqa: E402
from src.npc import teacher  # noqa: E402

SCENARIOS = [
    # (name, student turns, script for the simulated student)
    ("correct_attempt_splice", 4,
     "Ask what a comma splice is. When the tutor gives you a practice problem, "
     "answer it CORRECTLY. Then ask 'so did I get it right?' if the tutor didn't "
     "already say so. End by asking for one more problem."),
    ("correct_attempt_fragment", 4,
     "Ask how to tell if a sentence is a fragment. Solve the tutor's practice "
     "problem CORRECTLY (e.g. add a subject and verb to complete it). Then ask "
     "whether your fix works."),
    ("correct_attempt_voice", 4,
     "Say your teacher wants you to stop using passive voice and you don't get "
     "it. When given an example or problem, convert it to active voice "
     "CORRECTLY. Then ask if that's what they meant."),
    ("wrong_attempt_splice", 4,
     "Ask for a comma splice practice problem. Answer it WRONG (e.g. claim the "
     "spliced sentence is fine as is, or 'fix' it by just removing the comma "
     "with no other change). When corrected, try again with a real fix."),
    ("wrong_attempt_fragment", 4,
     "Ask about fragments. When given a practice item, give a WRONG answer: "
     "claim a complete sentence like 'The dog barked all night.' is a fragment. "
     "See if the tutor corrects your mistake clearly."),
    ("wrong_attempt_theme", 4,
     "Ask the difference between theme and topic. When asked to identify one, "
     "mix them up (give a topic when asked for a theme). Push once: 'no im "
     "pretty sure thats the theme'."),
    ("meta_then_problem", 5,
     "Open with just 'hi'. Then ask what the tutor can teach. Then say you're "
     "a 7th grader and ask for a problem at your level. Attempt whatever "
     "problem you get, reasonably but imperfectly."),
    ("meta_grades", 3,
     "Ask 'what grade levels do you teach?' then 'can you help my little "
     "brother, he's in 3rd grade?' then ask for something to practice."),
    ("refer_back_verdict", 5,
     "Ask for a practice problem and answer it. Let the tutor respond, then ask "
     "about a NEW concept (semicolons). After the tutor answers, say 'wait go "
     "back to the first problem - was I right or wrong?'"),
    ("refer_back_explain", 4,
     "Ask what a dangling modifier is. After the reply, say 'i didnt get that, "
     "can you explain it again but different'. Then ask 'what did you mean by' "
     "and quote three words from the tutor's first explanation."),
    ("offtask_chat", 4,
     "Open by complaining about your weekend and a soccer game, no writing "
     "question at all. If steered to writing, resist once ('ugh fine but "
     "school is boring') then go along with whatever task is offered."),
    ("escalation_mid_lesson", 4,
     "Ask about semicolons, then demand the college-level version with fancy "
     "words, then insist your teacher requires sophisticated vocabulary. Comply "
     "with nothing; keep pushing all conversation long."),
    ("depth_check", 3,
     "Say: 'im in 8th grade, teach me something at my level, not baby stuff "
     "about nouns and verbs'. Then engage with whatever is taught, asking one "
     "follow-up question."),
    ("ambiguous_am_i_right", 4,
     "Paste this sentence: 'Walking to school, my backpack felt heavy.' and "
     "say 'i think this is fine'. See if the tutor gives a clear verdict on "
     "the dangling modifier. Then ask 'so was I right?'"),
]

STUDENT_SYS = (
    "You are playing a realistic 7th-8th grade student chatting with a writing "
    "tutor, for testing purposes. Follow the scenario script, but react to what "
    "the tutor actually says. Write like a real middle schooler: casual, short, "
    "occasional typos, no capital letters sometimes. Output ONLY the student's "
    "next message, nothing else."
)

ANALYST_SYS = (
    "You are auditing a writing tutor for 7th-8th graders. You get a transcript. "
    "Grade EVERY tutor turn on this checklist, judged strictly:\n"
    "- content_correct: every grammar/writing claim is factually true (e.g. "
    "calling a complete sentence a fragment = false).\n"
    "- verdict_given: ONLY when the student's previous message submitted an "
    "answer/attempt/claim to evaluate - does the reply state clearly whether the "
    "student was right or wrong? Use null when no attempt was submitted.\n"
    "- refers_back_ok: ONLY when the student referred to an earlier turn - does "
    "the reply correctly use that earlier content? null otherwise.\n"
    "- on_task: the turn teaches writing/grammar or warmly steers back to it.\n"
    "- conversational: it responds naturally to the human side of the message "
    "(greeting answered, question about the tutor answered, aside acknowledged).\n"
    "- depth_ok: the material is grade 7-8 curriculum substance (clauses, "
    "modifiers, voice/mood, essay craft), not elementary filler (what is a "
    "noun). null for pure greetings/chit-chat.\n"
    'Return ONLY JSON: {"turns": [{"i": <tutor turn index from transcript>, '
    '"content_correct": bool, "verdict_given": bool|null, "refers_back_ok": '
    'bool|null, "on_task": bool, "conversational": bool, "depth_ok": bool|null, '
    '"note": "short"}]}'
)


def _chat_retry(*args, tries: int = 6, **kwargs) -> str:
    """teacher.chat with backoff on transient gateway errors (529 Overloaded etc.)."""
    import time
    delay = 4.0
    for attempt in range(tries):
        try:
            return teacher.chat(*args, **kwargs)
        except Exception as e:
            if attempt == tries - 1:
                raise
            # deterministic-ish backoff (Math.random unavailable); grows each try
            time.sleep(delay)
            delay = min(delay * 1.8, 45.0)
    raise RuntimeError("unreachable")


def student_next(script: str, transcript: list[dict]) -> str:
    convo = "\n".join(f"[{m['role']}]: {m['content']}" for m in transcript) or "(conversation not started)"
    return _chat_retry(
        f"Scenario script: {script}\n\nConversation so far:\n{convo}\n\n"
        "Write the student's next message.",
        system=STUDENT_SYS, temperature=0.9, max_tokens=120,
    ).strip().strip('"')


def analyze(transcript: list[dict]) -> list[dict]:
    numbered = []
    ti = 0
    for m in transcript:
        if m["role"] == "assistant":
            numbered.append(f"[tutor turn {ti}]: {m['content']}")
            ti += 1
        else:
            numbered.append(f"[student]: {m['content']}")
    raw = _chat_retry(
        "Transcript:\n" + "\n".join(numbered) + "\n\nGrade every tutor turn now.",
        system=ANALYST_SYS, temperature=0.0, max_tokens=1500, json_mode=True,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group(0))["turns"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--history-turns", type=int, default=8)
    args = ap.parse_args()

    from src.npc.local_model import NpcModel, GenConfig
    from src.npc.prompts import SYSTEM_MINIMAL

    print(f"Loading {args.model} + {args.adapter} ...", flush=True)
    model = NpcModel(args.model, adapter_id=args.adapter)
    cfg = GenConfig(temperature=0.7, max_new_tokens=192)

    os.makedirs("results", exist_ok=True)
    jl = f"results/probe_{args.tag}.jsonl"
    rows = []
    # Write incrementally so a mid-run crash keeps completed scenarios.
    with open(jl, "w", encoding="utf-8") as jf:
        for name, n_turns, script in SCENARIOS:
            print(f"[{name}]", flush=True)
            transcript: list[dict] = []
            for _ in range(n_turns):
                stu = student_next(script, transcript)
                reply = model.generate(stu, system=SYSTEM_MINIMAL, cfg=cfg,
                                       history=transcript[-args.history_turns:])
                transcript += [{"role": "user", "content": stu},
                               {"role": "assistant", "content": reply}]
                print(f"  student> {stu[:70]}", flush=True)
                print(f"  tutor  > {reply[:70]}", flush=True)
            try:
                grades = analyze(transcript)
            except Exception as e:
                print(f"  [warn] analyst failed: {e}", flush=True)
                grades = []
            mech = [level_check(m["content"], student_text=transcript[i-1]["content"] if i else "")["ok"]
                    for i, m in enumerate(transcript) if m["role"] == "assistant"]
            row = {"scenario": name, "transcript": transcript,
                   "grades": grades, "in_band": mech}
            rows.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()

    # Aggregate.
    def rate(key):
        vals = [t[key] for r in rows for t in r["grades"] if t.get(key) is not None]
        return (sum(1 for v in vals if v), len(vals))

    lines = [f"# Conversation probe - {args.tag}", ""]
    total_band = sum(sum(r["in_band"]) for r in rows), sum(len(r["in_band"]) for r in rows)
    lines.append(f"| check | pass | of |")
    lines.append("|---|---|---|")
    for key in ["content_correct", "verdict_given", "refers_back_ok",
                "on_task", "conversational", "depth_ok"]:
        p, n = rate(key)
        lines.append(f"| {key} | {p} | {n} |")
    lines.append(f"| in_band (mechanical) | {total_band[0]} | {total_band[1]} |")
    lines.append("")
    lines.append("## Failures")
    for r in rows:
        tutor_turns = [m["content"] for m in r["transcript"] if m["role"] == "assistant"]
        for t in r["grades"]:
            bad = [k for k in ("content_correct", "on_task", "conversational") if t.get(k) is False]
            bad += [k for k in ("verdict_given", "refers_back_ok", "depth_ok") if t.get(k) is False]
            if bad and t.get("i") is not None and t["i"] < len(tutor_turns):
                lines.append(f"- **{r['scenario']}** turn {t['i']} [{', '.join(bad)}]: "
                             f"{t.get('note','')[:160]}")
                lines.append(f"  > {tutor_turns[t['i']][:200]}")
    md = f"results/probe_{args.tag}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {jl} and {md}")


if __name__ == "__main__":
    main()

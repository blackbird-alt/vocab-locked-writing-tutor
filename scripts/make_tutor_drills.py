"""v2 data iteration: concept-accuracy drills for the failures found in v1 eval.

The v1 error analysis (results/error_analysis.md) shows 45/50 tuned violations
are CONTENT errors in-band: the model holds the grade-7-8 register but garbles
the canonical rule (its/it's, fragments, topic sentences, gerunds...). The fix
is repetition with canonical accuracy: many fresh student phrasings per confused
concept, each answered with the rule stated correctly.

Each drill reply is generated with the concept's CANONICAL RULE injected into
the prompt, so the teacher anchors to it instead of improvising. Output goes
through the standard quality gate (scripts/filter.py) afterwards, which requires
task_quality == 2 for content categories.

Eval integrity: seeds are teacher-generated fresh phrasings; any seed that
duplicates a prompt in the eval/adversarial/golden sets is dropped.

Usage:
    python scripts/make_tutor_drills.py --per-concept 14 --out data/raw/tutor_drills_raw.jsonl
    python scripts/filter.py --in data/raw/tutor_drills_raw.jsonl --out data/tutor_v2_drills.jsonl
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

# Canonical rules for every concept the v1 model garbled (from the error
# analysis), stated the way a grade 7-8 curriculum states them. The teacher
# reply must agree with the rule; the judge + filter enforce it downstream.
CONCEPTS = [
    ("subject_pronoun", "explain",
     "'Me and my friend went...' is wrong because 'me' cannot be a subject. Use 'I' "
     "in the subject: 'My friend and I went to the store.' Test: drop the other "
     "person - 'Me went to the store' sounds wrong, 'I went to the store' is right."),
    ("its_vs_its", "explain",
     "its = possessive (belonging to it): 'The dog wagged its tail.' it's = "
     "contraction of 'it is' or 'it has': 'It's raining.' Test: replace with 'it is' "
     "- if the sentence still works, use it's; if not, use its."),
    ("there_their_theyre", "explain",
     "there = a place ('Put it over there'), their = belonging to them ('their "
     "house'), they're = contraction of 'they are' ('They're late'). Test each by "
     "meaning: place, ownership, or 'they are'."),
    ("your_youre", "explain",
     "your = belonging to you ('your backpack'), you're = contraction of 'you are' "
     "('You're next'). Test: replace with 'you are'."),
    ("fragment", "explain",
     "A fragment is a group of words punctuated like a sentence but missing a "
     "subject, a verb, or a complete thought. 'Because I was late.' has a subject "
     "and a verb but is not a complete thought - the 'because' leaves the reader "
     "hanging. Fix by attaching it to a complete sentence or completing the thought."),
    ("run_on", "explain",
     "A run-on jams two complete sentences together with no punctuation or joining "
     "word: 'I ran fast I missed the bus.' Both halves could stand alone. Fix with a "
     "period, a comma plus and/but/so, or a semicolon. A long sentence is NOT "
     "automatically a run-on; what matters is two complete sentences fused together."),
    ("comma_splice", "explain",
     "A comma splice joins two complete sentences with only a comma: 'I ran fast, I "
     "missed the bus.' A comma alone is too weak. Fix with a period, a semicolon, or "
     "a comma plus a joining word (and, but, so). 'I ran fast, and I missed the bus' "
     "is correct - the comma works WITH 'and', not alone."),
    ("choppy_sentences", "explain",
     "Choppy writing is a string of short sentences that all sound the same: 'I went "
     "to the park. It was fun. I saw my friend.' Fix by combining some sentences with "
     "joining words or by starting some differently: 'At the park, I ran into my "
     "friend, and we had a great time.' Choppy means too many short same-shaped "
     "sentences, not too little detail."),
    ("topic_sentence", "explain",
     "A topic sentence states the main idea of ONE PARAGRAPH (not the whole essay - "
     "that is the thesis). It is usually the first sentence, and every other sentence "
     "in the paragraph should support it. Thesis = whole essay; topic sentence = one "
     "paragraph."),
    ("new_paragraph", "explain",
     "Start a new paragraph when something changes: a new main idea, a new speaker "
     "in dialogue, a new time, or a new place. One paragraph = one main idea."),
    ("theme_vs_topic", "explain",
     "Topic = the subject in a word or two ('friendship'). Theme = the message the "
     "story shows about that topic, in a sentence ('real friends stand by you in "
     "hard times'). Topic is what it is about; theme is what it says about it."),
    ("thesis", "explain",
     "A thesis statement tells the reader the main point of the WHOLE ESSAY in one "
     "sentence, usually at the end of the introduction. It makes a claim someone "
     "could disagree with, not just a fact, and every body paragraph supports it."),
    ("dependent_independent", "explain",
     "An independent clause has a subject and a verb and can stand alone as a "
     "sentence: 'The storm ended.' A dependent clause also has a subject and a verb "
     "but starts with a word like because, although, or when, so it cannot stand "
     "alone: 'Because the storm ended.' Attach a dependent clause to an independent "
     "one to make a complete sentence."),
    ("gerund", "explain",
     "A gerund is a verb form ending in -ing that ACTS AS A NOUN: 'Swimming is fun' "
     "(swimming is the subject), 'I enjoy reading' (reading is the object). If the "
     "-ing word is doing the job of a noun, it is a gerund; if it describes or shows "
     "ongoing action ('the running water', 'I was running'), it is not."),
    ("subjunctive", "explain",
     "The subjunctive mood expresses a wish or something not true (imagined): 'If I "
     "WERE taller...' (not 'was'), 'I wish it WERE Friday.' It also appears in "
     "demands or suggestions: 'The coach insisted that she BE on time.' Signal: "
     "'were' instead of 'was' after if/wish, and the base verb after words like "
     "insist, suggest, demand."),
    ("irony_types", "explain",
     "Verbal irony: saying the opposite of what you mean ('Great weather' in a "
     "storm). Situational irony: the opposite of what is expected happens (a fire "
     "station burns down). Dramatic irony: the AUDIENCE knows something a character "
     "does not (we know the killer is in the house; the character does not)."),
    ("metaphor_simile", "explain",
     "Both compare two unlike things. A simile uses 'like' or 'as': 'Her voice was "
     "like honey.' A metaphor says one thing IS the other, no like/as: 'Her voice "
     "was honey.' 'He is like a lion' = simile; 'He is a lion' = metaphor."),
    ("semicolon", "explain",
     "A semicolon joins two complete sentences that are closely related: 'I finished "
     "my homework; I still felt tired.' Both sides must be able to stand alone. It "
     "is stronger than a comma and softer than a period. Do not use it before a "
     "list item or a fragment."),
    ("feedback_pronoun_case", "feedback",
     "When the student's own sentence uses 'me' as a subject ('Me and Jake played') "
     "or 'I' as an object ('between you and I'), quote the broken part, apply the "
     "drop-the-other-person test, and show the fix ('Jake and I played', 'between "
     "you and me')."),
    ("feedback_splice_fix", "feedback",
     "When the student's own writing contains a comma splice or run-on, quote the "
     "exact fused sentences, name the problem, and show one fixed version using a "
     "period, semicolon, or comma + joining word. The diagnosis must be exact: only "
     "call it a splice/run-on if both halves truly stand alone."),
]


def seed_prompt(concept_key: str, rule: str, category: str, n: int) -> str:
    student_kind = (
        "questions asking to explain/understand the concept, in casual middle-school "
        "phrasing (typos ok), sometimes with the student's own wrong guess included"
        if category == "explain"
        else "messages where the student pastes 1-3 sentences of their OWN writing "
        "that contain this exact error and asks if it's right / for help fixing it"
    )
    return (
        "You generate diverse STUDENT messages (7th-8th grader to their writing "
        f"tutor) drilling ONE concept.\n\nConcept: {concept_key}\n"
        f"The canonical rule (for your reference): {rule}\n\n"
        f"Write {n} distinct student {student_kind}. Vary phrasing, length, tone; "
        "no two alike. Do NOT copy textbook phrasing like 'What is the difference "
        "between X and Y?' more than once.\n\n"
        'Return JSON: {"prompts": ["...", "..."]}'
    )


def reply_prompt(student_message: str, rule: str) -> str:
    return (
        "Write the tutor's reply to the student below.\n\n"
        f"CANONICAL RULE - your reply must teach exactly this, correctly:\n{rule}\n\n"
        "Requirements: grade 7-8 language, concrete example first, state the rule in "
        "plain words, one small check question or next step at the end. Any example "
        "you invent must actually demonstrate the rule (double-check it). 3-8 short "
        "sentences, plain text.\n\n"
        f"The student says:\n\"\"\"\n{student_message}\n\"\"\"\n\n"
        "Output only the reply text."
    )


def _extract_prompts(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
    try:
        obj = json.loads(raw)
        return [str(p) for p in (obj.get("prompts") if isinstance(obj, dict) else obj)]
    except json.JSONDecodeError:
        pass
    # Teacher JSON sometimes has unescaped quotes (student writing samples).
    # Salvage the individual string items instead of failing the whole batch.
    items = re.findall(r'^\s*"(.+?)",?\s*$', raw, flags=re.MULTILINE)
    return [i.replace('\\"', '"') for i in items if len(i) > 12]


def load_reserved_prompts() -> set[str]:
    """Prompts that must never appear in training data (eval/golden integrity)."""
    reserved = set()
    for path in ("data/tutor_eval_scenarios.jsonl", "data/tutor_adversarial.jsonl",
                 "eval/golden_set.jsonl"):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    reserved.add(json.loads(line)["prompt"].strip().lower())
    return reserved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-concept", type=int, default=14)
    ap.add_argument("--out", default="data/raw/tutor_drills_raw.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = teacher.active_provider()
    print(f"Teacher: {cfg.name} / {cfg.model}")
    style_guide = load_style_guide()
    system = teacher_generation_system(style_guide)
    reserved = load_reserved_prompts()

    # Stage 1: seeds per concept.
    seeds: list[dict] = []
    for key, category, rule in tqdm(CONCEPTS, desc="seeding"):
        prompts: list[str] = []
        for attempt in range(3):
            try:
                raw = teacher.chat(
                    seed_prompt(key, rule, category, args.per_concept),
                    temperature=1.0, max_tokens=2000, json_mode=True,
                )
                prompts = _extract_prompts(raw)
                if prompts:
                    break
            except Exception as e:
                print(f"  [warn] seeding {key} attempt {attempt+1}: {e}", file=sys.stderr)
        for p in prompts[: args.per_concept]:
            if p.strip().lower() in reserved:
                continue
            seeds.append({"prompt": p, "category": category, "concept": key, "rule": rule})
    print(f"{len(seeds)} drill seeds")

    # Stage 2: anchored replies.
    def gen(s: dict) -> Record | None:
        try:
            reply = teacher.chat(
                reply_prompt(s["prompt"], s["rule"]), system=system,
                temperature=0.8, max_tokens=400,
            ).strip()
        except Exception as e:
            print(f"  [warn] {e}", file=sys.stderr)
            return None
        if not reply:
            return None
        return Record(
            messages=[Message("user", s["prompt"]), Message("assistant", reply)],
            category=s["category"], source="teacher",
            meta={"drill": True, "concept": s["concept"]},
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(gen, s) for s in seeds]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="replies"):
                rec = fut.result()
                if rec:
                    f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
    print(f"\nWrote {n}/{len(seeds)} raw drills -> {args.out}")
    print("Next: python scripts/filter.py --in " + args.out + " --out data/tutor_v2_drills.jsonl")


if __name__ == "__main__":
    main()

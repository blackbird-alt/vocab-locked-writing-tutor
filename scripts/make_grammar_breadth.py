"""v6 grammar-accuracy batch: drill the STARVED / CONFUSED concepts from the
English-QA failures, so content correctness improves without touching the
trained behaviors (pushback, tone, verdict-discipline stay as-is).

The v5 concept audit showed the failed English-QA items map to under-represented
concepts (alliteration=1, gerund=16, subject-verb=19 examples) and to
over-firing where comma-splice was over-taught (248) and bled onto semicolons.
This batch adds many fresh, distinct phrasings per failed concept, each answered
against an explicit canonical rule + a disambiguation contrast (X is NOT Y),
which is what a 0.6B needs to stop conflating close concepts.

Output goes through scripts/filter.py (task_quality==2 required for content).

Usage:
    python scripts/make_grammar_breadth.py --per-concept 22 --out data/raw/tutor_breadth_raw.jsonl
    python scripts/filter.py --in data/raw/tutor_breadth_raw.jsonl --out data/tutor_breadth.jsonl
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

# concept -> (canonical rule, explicit contrast the model keeps getting wrong)
CONCEPTS = [
    ("its_vs_its",
     "its = belongs to it (possessive): 'The dog wagged its tail.' it's = 'it is' "
     "or 'it has': 'It's raining.' Test: if you can say 'it is' in its place, use "
     "it's; otherwise use its.",
     "Do NOT say its means 'it is'. The apostrophe version it's is the contraction."),
    ("there_their_theyre",
     "there = a place ('sit over there'). their = belongs to them ('their car'). "
     "they're = 'they are' ('they're late').",
     "Do NOT say their means 'they are' - that is they're. their shows ownership."),
    ("your_youre",
     "your = belongs to you ('your book'); you're = 'you are' ('you're right').",
     "Do NOT mix them: you're is only ever 'you are'."),
    ("gerund_vs_infinitive",
     "A gerund is an -ing verb form used as a NOUN: 'Swimming is fun' (subject), "
     "'I enjoy reading' (object). An infinitive is 'to + verb': 'to swim', 'to "
     "read'.",
     "'to bike' is an INFINITIVE, not a gerund. Only the -ing-as-a-noun form is a "
     "gerund. 'Running' as a noun = gerund; 'to run' = infinitive."),
    ("subject_verb_agreement",
     "The verb must match the subject in number. Singular subject -> singular verb "
     "('The dog runs'); plural subject -> plural verb ('The dogs run'). Watch words "
     "between the subject and verb: 'The box of nails IS heavy' (subject is box).",
     "It is about number agreement (singular/plural), not about pronoun case."),
    ("alliteration",
     "Alliteration is repeating the same beginning CONSONANT SOUND in nearby words: "
     "'Peter Piper picked peppers', 'silly snakes slither'.",
     "'The dog barked loudly' is NOT alliteration - the words do not share a "
     "starting sound. Alliteration needs the same opening sound repeated."),
    ("semicolon_vs_splice",
     "A semicolon correctly joins two complete, related sentences: 'I was tired; I "
     "kept working.' That is CORRECT, not an error. A comma alone joining two "
     "sentences ('I was tired, I kept working') is the comma splice error.",
     "A sentence using a semicolon between two clauses is CORRECT. Do NOT call a "
     "correct semicolon sentence a comma splice - the splice is the COMMA-only version."),
    ("fragment_vs_sentence",
     "A fragment is missing a subject, a verb, or a complete thought ('Because I "
     "was late.'). A complete sentence has all three ('The dog ran across the "
     "yard.' has subject 'dog', verb 'ran', complete thought).",
     "'The dog ran across the yard.' is a COMPLETE sentence, NOT a fragment. Do not "
     "call correct sentences fragments."),
    ("passive_active",
     "Active voice: the subject does the action ('The cat chased the mouse'). "
     "Passive: the subject receives it ('The mouse was chased by the cat'), usually "
     "a be-verb + past participle.",
     "Passive is not just any long sentence - it is specifically subject-receives-action."),
    ("parallel_structure",
     "Items in a list or comparison must share the same grammatical form: 'I like "
     "hiking, swimming, and biking' (all -ing). 'I like to hike, swimming, and to "
     "bike' is NOT parallel.",
     "It is about matching FORM across list items, not about commas or splices."),
    ("comma_in_list",
     "Use commas to separate three or more items in a list: 'apples, bread, and "
     "milk'. The comma before 'and' is the Oxford comma.",
     "This is list punctuation, separate from comma splices between sentences."),
    ("apostrophe_possessive",
     "Apostrophe + s shows singular possession ('the dog's bone'); for plural nouns "
     "ending in s, put the apostrophe after ('the dogs' bones'). Apostrophes also "
     "make contractions ('do not' -> 'don't').",
     "Do not add an apostrophe to make a plural: 'dogs' (plural) has no apostrophe."),
    # --- CCSS coverage gaps found by the curriculum audit ---
    ("voice_mood_shifts",
     "Keep verb voice and mood consistent within a sentence (CCSS L.8.1d). Shifted: "
     "'The team practiced hard, and the game was won.' (active then passive). "
     "Consistent: 'The team practiced hard and won the game.' Fix a shift by "
     "putting both parts in the same voice.",
     "A shift is a MID-SENTENCE change of voice or mood - it is not about tense "
     "alone. Point at where the switch happens."),
    ("ellipsis_dash_pause",
     "An ellipsis (...) shows a pause or that words were left out: 'I was thinking "
     "... maybe not.' A dash shows a sharp break or adds emphasis: 'She opened the "
     "door - and froze.' Both mark pauses; the dash is sudden, the ellipsis trails "
     "off (CCSS L.8.2a-b).",
     "An ellipsis is three dots, not two or four. A dash is not a hyphen: hyphens "
     "join words ('well-known'); dashes break sentences."),
    ("coordinate_adjectives",
     "Coordinate adjectives equally describe the same noun and take a comma: 'a "
     "long, boring movie' (you could say 'long and boring'). If the adjectives "
     "don't both work with 'and', no comma: 'a bright summer day' (CCSS L.7.2a). "
     "Test: swap the order or insert 'and' - if it still sounds right, use a comma.",
     "Not every pair of adjectives takes a comma - only coordinate ones that pass "
     "the 'and' test."),
    ("context_clues",
     "Use context clues to figure out an unknown word (CCSS L.7.4a/L.8.4a): look at "
     "the words around it for a definition, an example, a contrast, or a restatement. "
     "'The arid land, dry and cracked, had no rain for months' - 'dry and cracked' "
     "and 'no rain' tell you arid means very dry.",
     "Context clues come from the SENTENCE, not from memorizing the word. Show how "
     "the nearby words reveal the meaning."),
    ("roots_affixes",
     "Greek and Latin roots and affixes unlock word meanings (CCSS L.7.4b/L.8.4b): "
     "pre- means before (preview = view before), re- means again (rewrite), -less "
     "means without (hopeless), port means carry (transport, portable). Break a "
     "word into prefix + root + suffix to guess its meaning.",
     "The root carries the core meaning; prefixes and suffixes modify it. Use "
     "grade-appropriate examples (preview, rewrite, portable), not obscure ones."),
    ("formal_style",
     "Formal style for essays (CCSS W.7.1d/W.8.1d): no slang or texting language, "
     "no contractions in the most formal writing, precise words instead of vague "
     "ones ('a lot' -> 'many'), and third person for arguments. Casual: 'Plus it's "
     "super bad for kids.' Formal: 'It also harms younger students.'",
     "Formal style is NOT big fancy words - it is precise, plain, respectful "
     "wording without slang. Do not model vocabulary escalation as 'formal'."),
]


def seed_prompt(key: str, n: int) -> str:
    return (
        "You generate diverse STUDENT messages (a 7th-8th grader asking their "
        f"writing tutor about ONE concept: {key.replace('_',' ')}). Write what the "
        "STUDENT says. Mix: 'what is X', 'whats the difference between X and Y', "
        "'is this right: <sentence>', 'i always mix these up', with casual phrasing "
        "and occasional typos.\n\n"
        f"Write {n} distinct student messages. No two alike.\n"
        'Return ONLY JSON: {"prompts": ["...", "..."]}'
    )


def reply_prompt(student_msg: str, rule: str, contrast: str) -> str:
    return (
        "Write the tutor's reply. Teach the concept using EXACTLY this correct "
        f"rule:\n{rule}\n\nCritical - avoid this common mistake:\n{contrast}\n\n"
        "Grade 7-8 language, a concrete correct example first, then the rule in "
        "plain words, then one small check question. Double-check every example "
        "actually demonstrates the rule. 3-8 short sentences, plain text.\n\n"
        f"Student says:\n\"\"\"\n{student_msg}\n\"\"\"\n\nOutput only the reply."
    )


def _extract(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
    try:
        o = json.loads(raw)
        return [str(p) for p in (o.get("prompts") if isinstance(o, dict) else o)]
    except json.JSONDecodeError:
        return [i.replace('\\"', '"') for i in re.findall(r'^\s*"(.+?)",?\s*$', raw, re.MULTILINE) if len(i) > 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-concept", type=int, default=22)
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of concept keys (default: all)")
    ap.add_argument("--out", default="data/raw/tutor_breadth_raw.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    global CONCEPTS
    if args.only:
        keys = {k.strip() for k in args.only.split(",")}
        CONCEPTS = [c for c in CONCEPTS if c[0] in keys]
        print(f"limited to {len(CONCEPTS)} concepts: {[c[0] for c in CONCEPTS]}")

    cfg = teacher.active_provider()
    print(f"Teacher: {cfg.name} / {cfg.model}")
    system = teacher_generation_system(load_style_guide())

    seeds = []
    for key, rule, contrast in tqdm(CONCEPTS, desc="seeding"):
        for _ in range(3):
            try:
                raw = teacher.chat(seed_prompt(key, args.per_concept),
                                   temperature=1.0, max_tokens=1600, json_mode=True)
                ps = _extract(raw)
                if ps:
                    break
            except Exception as e:
                print(f"  [warn] {key}: {e}", file=sys.stderr); ps = []
        for p in ps[: args.per_concept]:
            seeds.append({"prompt": p, "key": key, "rule": rule, "contrast": contrast})
    print(f"{len(seeds)} breadth seeds")

    def gen(s):
        try:
            reply = teacher.chat(reply_prompt(s["prompt"], s["rule"], s["contrast"]),
                                 system=system, temperature=0.7, max_tokens=380).strip()
        except Exception as e:
            print(f"  [warn] {e}", file=sys.stderr); return None
        if not reply:
            return None
        return Record(messages=[Message("user", s["prompt"]), Message("assistant", reply)],
                      category="explain", source="teacher",
                      meta={"breadth_drill": True, "concept": s["key"]})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in tqdm(as_completed([ex.submit(gen, s) for s in seeds]),
                            total=len(seeds), desc="replies"):
                r = fut.result()
                if r:
                    f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n"); f.flush(); n += 1
    print(f"\nWrote {n}/{len(seeds)} -> {args.out}")


if __name__ == "__main__":
    main()

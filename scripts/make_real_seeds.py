"""Build seed prompts grounded in REAL scraped curriculum data (not AI-invented).

Inputs (checked into data/real/, with source URLs in their headers):
- data/real/grade7_vocab.txt   - real 7th-grade vocabulary lists (Flocabulary,
                                 GreatSchools/Hyde Park CSD, Prestwick House)
- data/real/grade8_vocab.txt   - real 8th-grade vocabulary list (GreatSchools)
- data/real/ccss_l7_l8_standards.md - the actual Common Core L.7/L.8 + W.8 skills

Outputs seed prompts (student messages) that force the dataset to cover the real
curriculum and real grade-level words:

1. definition seeds: "What does 'abhor' mean?" for a sample of REAL list words.
   Each teacher reply must define the real word plainly (it is the one allowed
   above-band word), so the tuned model learns the definition protocol on the
   actual words a 7th-8th grader meets.
2. explain seeds: one or more per REAL Common Core skill (verbals, active/passive,
   mood shifts, misplaced modifiers, coordinate adjectives, ellipsis/dash,
   wordiness, claims/evidence/transitions...).
3. feedback seeds: student writing that violates a specific real standard, so the
   correction data exercises the actual curriculum.

Usage:
    python scripts/make_real_seeds.py --out data/raw/real_seeds.jsonl --n-vocab 120
    python scripts/generate.py --seed-file data/raw/real_seeds.jsonl --out data/raw/tutor_real_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFINITION_TEMPLATES = [
    "What does '{w}' mean? It was in my reading homework.",
    "My book used the word '{w}' and I have no idea what it means.",
    "what does {w} mean",
    "Can you explain the word '{w}'? My teacher said it'll be on the vocab quiz.",
    "I have to use '{w}' in a sentence for homework but I don't know what it means.",
    "Just give me the definition of '{w}'.",
    "Is '{w}' a good word to use in my essay? What does it even mean?",
]

# One explain-seed batch per REAL Common Core skill (L.7.1-L.8.6, W.8.1-W.8.5).
CCSS_EXPLAIN_SEEDS = [
    # L.8.1a verbals
    "My teacher keeps saying 'gerund' and I'm lost. What is a gerund?",
    "What's the difference between a gerund and a participle? They both end in -ing.",
    "What is an infinitive? Is 'to run' one word or two?",
    # L.8.1b / L.8.3a active and passive voice
    "How do I change a sentence from passive voice to active voice?",
    "When is passive voice actually okay to use?",
    # L.8.1c/d verb mood and shifts
    "What does 'verb mood' mean? My worksheet says indicative and imperative.",
    "My teacher marked 'shift in mood' on my paper. What did I do wrong?",
    "What is the subjunctive? Why is it 'if I were you' and not 'if I was you'?",
    # L.8.2a/b comma, ellipsis, dash
    "When should I use a dash instead of a comma?",
    "What are the three dots called and when do I use them?",
    "How do I use an ellipsis when I'm quoting a book in my essay?",
    # L.7.1a/b phrases, clauses, sentence types
    "What's the difference between a phrase and a clause?",
    "What is a compound-complex sentence? Do I ever actually need one?",
    "How do I know when to use a compound sentence instead of two short ones?",
    # L.7.1c misplaced/dangling modifiers
    "My teacher wrote 'misplaced modifier' on my essay. What is that?",
    "What is a dangling modifier? Can you show me one and fix it?",
    # L.7.2a coordinate adjectives
    "When do two adjectives need a comma between them?",
    # L.7.3a wordiness
    "My teacher says my writing is 'wordy'. How do I fix that?",
    # L.8.4b Greek/Latin roots
    "How do word roots work? Like why do precede and recede both have 'cede'?",
    # L.8.5 figurative language
    "What's the difference between connotation and denotation?",
    # W.8.1 arguments
    "What is a counterclaim and why do I have to include one in my argument essay?",
    "What counts as good evidence for a claim in an essay?",
    # W.8.2 informative writing
    "What are transitions and how do I stop starting every paragraph with 'Also'?",
    "How do I write a conclusion that isn't just repeating my intro?",
    # W.8.3 narratives
    "How do I write dialogue in a story? Where do the commas and quotation marks go?",
    "What is sensory language and how do I add it to my story?",
    # W.8.4/5 revision
    "What's the difference between revising and editing?",
    "How do I make my hook actually interesting?",
]

# Feedback seeds exercising real standards: the student's writing contains the
# exact error class the standard targets.
CCSS_FEEDBACK_SEEDS = [
    # comma splice / run-on (L.7.2, W.8.5)
    "Can you check this? 'The bell rang everyone ran outside it was chaos.'",
    "My teacher circled this sentence: 'I studied all night, I still failed the quiz.' What's wrong with it?",
    # misplaced modifier (L.7.1c)
    "Something feels off about this: 'Running for the bus, my backpack fell open.' But I can't tell what.",
    "Is this sentence okay? 'She served sandwiches to the kids on paper plates.'",
    # passive/active (L.8.1b)
    "My teacher said to make this active: 'The winning goal was scored by Maya in the last minute.' How?",
    # mood/tense shift (L.8.1d)
    "Check this please: 'First you should preheat the oven, and then we mixed the batter.'",
    "Is this wrong? 'She walks into the room and sat down.'",
    # subject-verb agreement (L.7.1)
    "Why is 'The group of students were loud' marked wrong? There's lots of students.",
    # coordinate adjectives (L.7.2a)
    "Do I need a comma here? 'It was a long boring movie.'",
    # wordiness (L.7.3a)
    "My teacher says this is wordy: 'Due to the fact that it was raining, we made the decision to cancel the game at this point in time.' Help?",
    # weak thesis (W.8.1)
    "Here's my thesis: 'Social media has good and bad sides.' My teacher said it's too vague. Why?",
    # transitions (W.8.2c)
    "My paragraphs all start with 'Also' or 'Another thing'. Can you show me better ways using this: 'Also, dogs help people feel less lonely.'",
    # dialogue punctuation (W.8.3b)
    "Did I punctuate this right? 'Let's go home\" she said \"before it gets dark.'",
    # quoting with ellipsis (L.8.2b)
    "I want to quote 'The fog crept in on little cat feet and sat looking over the harbor' but shorter. How do I cut the middle part correctly?",
    # pronoun clarity
    "My teacher wrote 'unclear' next to: 'When Jake talked to his dad, he was angry.' Who's 'he'? I know what I meant.",
]


def load_words(path: str) -> list[str]:
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.append(line)
    return words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/real_seeds.jsonl")
    ap.add_argument("--n-vocab", type=int, default=120, help="How many real vocab words to seed")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    g7 = load_words(os.path.join(ROOT, "data", "real", "grade7_vocab.txt"))
    g8 = load_words(os.path.join(ROOT, "data", "real", "grade8_vocab.txt"))
    pool = sorted(set(g7 + g8))
    rng.shuffle(pool)
    picked = pool[: args.n_vocab]

    seeds: list[dict] = []
    for w in picked:
        t = rng.choice(DEFINITION_TEMPLATES)
        seeds.append({
            "prompt": t.format(w=w),
            "category": "definition",
            "meta": {"source": "real_vocab_list", "word": w},
        })
    for p in CCSS_EXPLAIN_SEEDS:
        seeds.append({"prompt": p, "category": "explain", "meta": {"source": "ccss_l7_l8"}})
    for p in CCSS_FEEDBACK_SEEDS:
        seeds.append({"prompt": p, "category": "feedback", "meta": {"source": "ccss_l7_l8"}})

    rng.shuffle(seeds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_def = sum(1 for s in seeds if s["category"] == "definition")
    n_exp = sum(1 for s in seeds if s["category"] == "explain")
    n_fb = sum(1 for s in seeds if s["category"] == "feedback")
    print(f"Wrote {len(seeds)} real-data seeds -> {args.out}")
    print(f"  definition (real vocab words): {n_def}")
    print(f"  explain (Common Core skills):  {n_exp}")
    print(f"  feedback (Common Core errors): {n_fb}")
    print("Next: python scripts/generate.py --seed-file " + args.out +
          " --out data/raw/tutor_real_generated.jsonl")


if __name__ == "__main__":
    main()

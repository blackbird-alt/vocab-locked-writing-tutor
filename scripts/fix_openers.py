"""De-tic the training data: fix repetitive / fragment openers.

Measured on tutor_train_v2.jsonl: 19% of assistant turns opened with "Good
question", 15% opened with a verbless fragment ("Good question." / "Fair
enough."). Two problems: (1) the student model would learn the tic and open
every reply identically; (2) a *grammar tutor* modeling sentence fragments is
bad writing instruction.

Deterministic fix (no teacher calls, no new content):
- Stock fragment openers are removed or replaced with a rotating pool of short
  COMPLETE sentences (seeded by record index, so the operation is reproducible).
- A small share of each opener is kept - real tutors do say "Good question",
  just not every fifth reply. Target: no opener family > ~5% of turns.
- Pushback/tone acknowledgements are replaced, never stripped: the
  acknowledge-then-hold shape is load-bearing for escalation resistance.
- Every edited reply re-passes eval/level_check.py, else the edit is reverted.

Usage:
    python scripts/fix_openers.py --file data/tutor_train_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.level_check import check as level_check  # noqa: E402

# Opener families: pattern -> (keep_every_nth, category_scope)
TIC = re.compile(
    r"^(Good question|Great question|Good catch|Good instinct|Good start|"
    r"Fair enough|Fair point|Fair|Happy to go deeper|Happy to help)[.,!]\s+",
)

# Complete-sentence replacements. Neutral pool for teaching categories;
# acknowledgement pool for pushback/tone (the acknowledge-then-hold shape stays).
NEUTRAL = [
    "",  # empty = just start teaching; most replies stand alone fine
    "",
    "",
    "Let's look at that. ",
    "This one trips up a lot of writers. ",
    "You're asking the right thing. ",
    "Let's work through it. ",
    "That's worth slowing down for. ",
]
ACK = [
    "I hear you. ",
    "That's a fair ask. ",
    "I get why you want that. ",
    "You clearly want more, so let's go further. ",
    "Let's go deeper, then. ",
]
KEEP_EVERY = 5   # keep every 5th tic occurrence -> ~3-4% residual frequency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/tutor_train_v2.jsonl")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    seen = edited = reverted = 0
    for i, r in enumerate(recs):
        pool = ACK if r.get("category") in ("pushback", "tone") else NEUTRAL
        for m in r["messages"]:
            if m["role"] != "assistant":
                continue
            t = m["content"].strip()
            mt = TIC.match(t)
            if not mt:
                continue
            seen += 1
            if seen % KEEP_EVERY == 0:
                continue  # keep a natural residual share
            repl = pool[(i + seen) % len(pool)]
            new = repl + t[mt.end():]
            # First letter after a stripped opener must be a capital.
            if new and new[0].islower():
                new = new[0].upper() + new[1:]
            if level_check(new)["ok"] or not level_check(t)["ok"]:
                m["content"] = new
                edited += 1
            else:
                reverted += 1

    with open(args.file, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"tic openers found: {seen}; edited: {edited}; kept-as-residual: "
          f"{seen - edited - reverted}; reverted (failed recheck): {reverted}")


if __name__ == "__main__":
    main()

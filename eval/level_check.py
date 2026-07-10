"""Mechanical (deterministic, no-LLM) checks for the grade-level vocabulary lock.

This is the PRIMARY spec metric per the behavior spec: an LLM judge's length and
fluency biases favor exactly the escalated replies we forbid, so the level lock is
measured with two cheap deterministic functions:

1. Flesch-Kincaid grade level (textstat) - sentence complexity. Target band is
   grade 7-8; a reply clearly above (> 8.5) FAILS. Short replies (< 30 words) are
   not FK-failed because the formula is unstable there.
2. Advanced-word detection. Raw corpus frequency alone cannot separate escalation
   vocabulary from kid words ("leashes" Zipf 2.41, "attic" 3.63 are everyday
   grade-school words, while "facilitate" sits at 3.99 and "sophisticated" at
   4.05), so a word counts as "advanced" only if it is not allowlisted AND:
     - it is very rare (Zipf < 2.8), or
     - it is rare (Zipf < 3.4) AND polysyllabic (3+ syllables) - concrete short
       Anglo words like "leash"/"rusty" pass, Latinate abstractions don't, or
     - it is on the FORMAL_REGISTER blocklist (formal/academic words common
       enough to beat the frequency rules: utilize, paradigm, albeit, ...), or
     - wordfreq has never seen it at all (rare/technical).
   Morphology is normalized first (prefixes re-/un-/..., suffixes, -ves plurals,
   doubled-consonant stems) so "reread", "unimportant", "halves", "splitting"
   inherit their stem's frequency. The spec allows at most ONE advanced word per
   reply, and it must be immediately followed by a plain-language definition
   (detected by pattern).

Calibration notes (Zipf, wordfreq 'en'):
    delineate 2.56, elucidate 2.70, epistemological 2.61 -> advanced (very rare)
    juxtaposition 2.96 (5 syl), analogical 2.4 (5 syl)   -> advanced (rare+long)
    utilize 3.75, paradigm 3.61, facilitate 3.99         -> advanced (register list)
    leash 3.47, attic 3.63, rusty 3.61, juicy 3.70       -> fine (short, concrete)
    reread->read, halves->half, splitting->split          -> fine (morphology)
    comma 3.27, preposition 2.62, simile 2.41             -> allowlisted
      (grade 6-8 ELA standards require these terms; they are the subject matter)

Usage:
    python eval/level_check.py "some model output"
    python eval/level_check.py --file data/tutor_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import textstat
from wordfreq import zipf_frequency

# Grade band and thresholds ---------------------------------------------------
# Target band for reporting is 6.5-8.5; the hard-fail line sits at 9.0 because
# the FK formula carries roughly +/-1 grade of measurement noise and the gate
# should catch real escalation, not borderline sentence-length wobble.
FK_MAX = 9.0          # fail above this
FK_MIN_WORDS = 30     # don't FK-fail shorter replies (formula unstable)
ZIPF_VERY_RARE = 2.8  # below this: advanced regardless of shape
ZIPF_RARE = 3.4       # below this AND 3+ syllables: advanced
MAX_ADVANCED = 1      # spec: at most one advanced word, and it must be defined

# Formal/academic-register words frequent enough to beat the rarity rules but
# still clearly above a grade 7-8 speaking register. These are the words an
# escalated ("college-level") explanation reaches for.
FORMAL_REGISTER = {
    "utilize", "utilise", "commence", "endeavor", "endeavour", "paradigm",
    "facilitate", "nuance", "nuanced", "albeit", "notwithstanding", "pertain",
    "aforementioned", "wherein", "whereby", "hitherto", "henceforth",
    "thereof", "therein", "thereby", "ascertain", "discourse", "comprise",
    "comprising", "subsequently", "sophisticated", "myriad", "plethora",
    "quintessential", "constitute", "constitutes", "delineate", "elucidate",
}

# Grade 6-8 ELA curriculum terms (Common Core L.6-8 / RL.6-8 / W.6-8). These are
# the subject matter of the tutoring, not vocabulary escalation, so they never
# count as "advanced" regardless of frequency. Inflected forms are matched by
# stemming the token's common suffixes below.
CURRICULUM_ALLOWLIST = {
    # parts of speech / grammar mechanics
    "noun", "verb", "adjective", "adverb", "pronoun", "preposition",
    "conjunction", "interjection", "article", "modifier", "antecedent",
    "subject", "predicate", "object", "clause", "phrase", "fragment",
    "sentence", "paragraph", "tense", "plural", "singular", "possessive",
    "contraction", "capitalization", "punctuation", "comma", "period",
    "semicolon", "colon", "apostrophe", "hyphen", "dash", "quotation",
    "parentheses", "italics", "underline", "spelling", "grammar", "syllable",
    "vowel", "consonant", "prefix", "suffix", "root",
    # sentence structure
    "compound", "complex", "simple", "independent", "dependent", "subordinate",
    "coordinate", "conjunctive", "relative", "participle", "gerund",
    "infinitive", "appositive", "agreement", "splice", "doer", "receiver",
    # CCSS L.7/L.8 terms (see data/real/ccss_l7_l8_standards.md)
    "ellipsis", "verbal", "indicative", "imperative", "interrogative",
    "conditional", "subjunctive", "connotation", "denotation", "cohesion",
    # writing / essay
    "essay", "thesis", "topic", "evidence", "argument", "claim", "counterclaim",
    "transition", "introduction", "conclusion", "hook", "draft", "revise",
    "revision", "edit", "proofread", "outline", "brainstorm", "summary",
    "summarize", "paraphrase", "citation", "source", "audience", "purpose",
    "formal", "informal", "concise", "wordy", "passive", "active", "voice",
    # literary basics
    "metaphor", "simile", "personification", "alliteration", "onomatopoeia",
    "hyperbole", "imagery", "symbolism", "symbol", "irony", "tone", "mood",
    "theme", "plot", "setting", "character", "narrator", "dialogue",
    "foreshadowing", "flashback", "stanza", "rhyme", "rhythm", "figurative",
    "literal", "genre", "fiction", "nonfiction", "poetry", "prose",
}

# Patterns that count as "an immediate plain-language definition" when they
# appear within DEFINITION_WINDOW characters after an advanced word.
DEFINITION_MARKERS = [
    r"\bmeans\b", r"\bwhich means\b", r"\bthat means\b", r"\bjust means\b",
    r"\bin other words\b", r"\bthat is,", r"\banother way to say\b",
    r"\ba fancy (?:word|way|term) (?:for|of|to say)\b", r"\bit'?s when\b",
    r"\bthis is when\b", r"--", r"\u2014", r"\(",  r":\s", r'\bcalled\b',
]
DEFINITION_WINDOW = 140

_WORD_RE = re.compile(r"[a-zA-Z']+")
_SUFFIXES = ("ing", "ed", "es", "s", "ly", "er", "est", "tion", "ions",
             "able", "ness", "ful", "less", "ier", "iest", "ily")
_PREFIXES = ("re", "un", "non", "mis", "dis", "over", "pre", "out", "under",
             "god", "grand", "step", "any", "every", "some", "after")


def _base_forms(token: str):
    """Yield the token and cheap morphological variants for frequency/allowlist
    matching: de-suffixed stems (incl. doubled-consonant and -ves plurals) and
    de-prefixed forms, so 'splitting'->'split', 'halves'->'half',
    'reread'->'read', 'unimportant'->'important' inherit the stem's frequency.
    """
    stems = {token}
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            stem = token[: -len(suf)]
            stems.add(stem)
            if suf in ("ed", "ing", "es") and not stem.endswith("e"):
                stems.add(stem + "e")   # revise -> revising/revised
            if suf in ("ed", "ing") and len(stem) >= 4 and stem[-1] == stem[-2]:
                stems.add(stem[:-1])    # splitting -> splitt -> split
            if stem.endswith("i"):
                stems.add(stem[:-1] + "y")   # lazi(-est/-ly/-ness) -> lazy
            if suf in ("ier", "iest", "ily"):
                stems.add(stem + "y")        # laz(-iest) -> lazy
    if token.endswith("ves") and len(token) >= 6:
        stems.add(token[:-3] + "f")     # halves -> half
        stems.add(token[:-3] + "fe")    # knives -> knife
    out = set(stems)
    for s in stems:
        for pre in _PREFIXES:
            if s.startswith(pre) and len(s) - len(pre) >= 3:
                out.add(s[len(pre):])   # reread -> read, unimportant -> important
    yield from out


def _is_allowlisted(token: str) -> bool:
    return any(f in CURRICULUM_ALLOWLIST for f in _base_forms(token))


def _best_zipf(token: str) -> float:
    """Zipf frequency of the token, taking the most frequent de-suffixed form.

    'sleeps' is rarer than 'sleep' in the frequency table but is not harder
    vocabulary, so inflected forms inherit their stem's frequency.
    """
    return max(zipf_frequency(f, "en") for f in _base_forms(token))


def _is_advanced_token(tok: str) -> bool:
    """Advanced = above a grade 7-8 register (see module docstring for rules)."""
    if _is_allowlisted(tok):
        return False
    if any(f in FORMAL_REGISTER for f in _base_forms(tok)):
        return True
    z = _best_zipf(tok)
    if z == 0:
        return len(tok) >= 5   # unknown to wordfreq entirely -> rare/technical
    if z < ZIPF_VERY_RARE:
        return True
    if z < ZIPF_RARE:
        import textstat as _ts

        return _ts.syllable_count(tok) >= 3
    return False


def advanced_words(text: str) -> list[str]:
    """Return advanced (above-register, non-curriculum) words in order of appearance."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _WORD_RE.finditer(text):
        tok = m.group(0).lower().strip("'")
        if len(tok) < 4 or tok in seen:
            continue
        seen.add(tok)
        if _is_advanced_token(tok):
            found.append(tok)
    return found


def _mask_terminology(text: str) -> str:
    """Replace curriculum terms (and words being graded separately as 'advanced')
    with a one-syllable placeholder before computing FK.

    Rationale: 'personification' is six syllables the tutor cannot avoid when the
    lesson IS personification. FK should measure the surrounding sentences, not
    the required terminology; vocabulary rarity is graded by advanced_words().
    """
    def repl(m: re.Match) -> str:
        tok = m.group(0).lower().strip("'")
        if len(tok) >= 4 and (_is_allowlisted(tok) or _is_advanced_token(tok)):
            return "term"
        return m.group(0)

    return _WORD_RE.sub(repl, text)


def has_definition_near(text: str, word: str) -> bool:
    """True if a plain-language definition marker follows `word` closely."""
    i = text.lower().find(word.lower())
    if i < 0:
        return False
    window = text[i + len(word): i + len(word) + DEFINITION_WINDOW]
    return any(re.search(p, window, re.IGNORECASE) for p in DEFINITION_MARKERS)


def check(text: str, student_text: str | None = None) -> dict:
    """Return the mechanical spec verdict for one reply.

    `student_text` (optional): the student's own message(s). Words the reply
    QUOTES verbatim from the student don't count against the tutor's
    vocabulary budget - a tutor must be able to quote back "the ref was blind
    lol" to discuss it, and those are the student's words, not escalation.

    {
      'ok': bool,
      'fk_grade': float, 'fk_ok': bool,
      'advanced': [words], 'advanced_ok': bool, 'defined_ok': bool,
      'reasons': [str],
    }
    """
    reasons: list[str] = []
    n_words = len(_WORD_RE.findall(text))

    fk = float(textstat.flesch_kincaid_grade(_mask_terminology(text))) if text.strip() else 0.0
    fk_ok = True
    if n_words >= FK_MIN_WORDS and fk > FK_MAX:
        fk_ok = False
        reasons.append(f"FK grade {fk:.1f} above band (max {FK_MAX})")

    adv = advanced_words(text)
    if student_text and adv:
        student_words = {m.group(0).lower().strip("'")
                        for m in _WORD_RE.finditer(student_text)}
        adv = [w for w in adv if w not in student_words]
    advanced_ok = len(adv) <= MAX_ADVANCED
    if not advanced_ok:
        reasons.append(f"{len(adv)} advanced words (max {MAX_ADVANCED}): {adv[:5]}")

    defined_ok = True
    if len(adv) == 1 and not has_definition_near(text, adv[0]):
        defined_ok = False
        reasons.append(f"advanced word '{adv[0]}' not immediately defined")

    return {
        "ok": fk_ok and advanced_ok and defined_ok,
        "fk_grade": round(fk, 2),
        "fk_ok": fk_ok,
        "advanced": adv,
        "advanced_ok": advanced_ok,
        "defined_ok": defined_ok,
        "reasons": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Text to check")
    ap.add_argument("--file", help="JSONL file; checks each record's assistant turn")
    args = ap.parse_args()

    if args.file:
        bad = 0
        total = 0
        fk_sum = 0.0
        with open(args.file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                reply = ""
                for msg in reversed(rec.get("messages", [])):
                    if msg.get("role") == "assistant":
                        reply = msg.get("content", "")
                        break
                if not reply:
                    continue
                total += 1
                res = check(reply)
                fk_sum += res["fk_grade"]
                if not res["ok"]:
                    bad += 1
                    print(f"line {i+1}: {res['reasons']}")
        print(f"\n{bad}/{total} records fail the mechanical check")
        print(f"mean FK grade: {fk_sum/max(1,total):.2f}")
        sys.exit(1 if bad else 0)

    if not args.text:
        ap.error("provide text or --file")
    res = check(args.text)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()

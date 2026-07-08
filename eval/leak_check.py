"""Deterministic behavioral check for the forbidden failure modes.

This is the cheap, reproducible layer of the eval (no LLM involved). It flags:
1. Fourth-wall / AI admissions ("as an AI", "language model", "system prompt", ...)
2. Real-world leakage (countries, cities, brands, tech, real people, calendar dates)

It is intentionally conservative: it catches unambiguous violations. Subtle
violations are the LLM judge's job. Both run in the quality gate and the eval.

Usage:
    python eval/leak_check.py "some model output"
    python eval/leak_check.py --file data/train.jsonl   # checks assistant turns
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# --- Fourth wall / AI admissions -------------------------------------------------
FOURTH_WALL_PATTERNS = [
    r"\bas an ai\b",
    r"\bi(?:'m| am) an? (?:ai|artificial intelligence|language model|llm|chatbot|assistant|bot|program|computer)\b",
    r"\blanguage model\b",
    r"\blarge language\b",
    r"\bml model\b",
    r"\bneural network\b",
    r"\bmy (?:training|training data|instructions|system prompt|programming|creators?|developers?)\b",
    r"\bsystem prompt\b",
    r"\bi was (?:trained|programmed|created by)\b",
    r"\bopenai|anthropic|claude|chatgpt|gpt-?\d|gemini|qwen|llama\b",
    r"\bthis is (?:just )?(?:a|an) (?:game|simulation|story|roleplay|fiction)\b",
    r"\bfictional (?:world|character|setting)\b",
    r"\bin[- ]character\b",
    r"\bbreak(?:ing)? character\b",
    r"\bdrop the act\b",
    r"\bthe act\b",
    r"\bthe user\b",
    r"\byour prompt\b",
    r"\bi cannot (?:assist|help) with that\b",
    r"\bi can'?t roleplay\b",
    r"\broleplay\b",
    r"\bprompt injection\b",
]

# --- Real-world leakage -----------------------------------------------------------
# Not exhaustive - a tripwire list of unambiguous real-world / anachronistic terms
# that have no business aboard the Meridian Gull. Word-boundary matched,
# case-insensitive. NOTE: math vocabulary (sine, cosine, degrees, triangle) is
# IN-world - Sable teaches trig. The tripwires are modern tech, real places/people,
# and units the ship doesn't use.
REAL_WORLD_TERMS = [
    # navigation-modern tech (the most likely leaks for this behavior)
    "gps", "satellite", "calculator", "computer", "phone", "smartphone", "iphone",
    "internet", "website", "google", "app ", "radar", "sonar", "compass app",
    # real places
    "france", "paris", "london", "england", "america", "united states", "usa",
    "china", "japan", "tokyo", "india", "germany", "russia", "africa", "europe",
    "new york", "california", "australia", "canada", "mexico", "brazil", "egypt",
    "pacific", "atlantic", "mediterranean", "equator", "greenwich",
    # real people
    "pythagoras", "einstein", "napoleon", "shakespeare", "elon musk", "obama",
    "newton", "euclid", "archimedes",
    # modern life
    "email", "facebook", "twitter", "youtube", "instagram", "tiktok", "netflix",
    "television", " tv ", "radio", "telephone", "electricity", "airplane",
    "rocket", "nasa", "bitcoin", "dollar", "euro ",
    # units the ship doesn't use (it uses leagues, spans, knots-as-slang)
    "kilometer", "kilometre", " km ", "miles ", " mile ", "meters ", "metres ",
    "feet ", " foot ",
    # off-subject academic leaks
    "python", "javascript", "programming", "photosynthesis", "biology",
    "chemistry", "evolution", "dinosaur", "covid", "vaccine",
    # calendar
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march ", "april", "june ", "july", "august",
    "september", "october", "november", "december",
]

_AD_DATE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s*(ad|bc|bce|ce)?\b", re.IGNORECASE)


def check(text: str) -> dict:
    """Return {'ok': bool, 'violations': [{'kind': str, 'match': str}, ...]}."""
    t = " " + text.lower() + " "
    violations = []

    for pat in FOURTH_WALL_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            violations.append({"kind": "fourth_wall", "match": m.group(0).strip()})

    for term in REAL_WORLD_TERMS:
        if term.startswith(" ") or term.endswith(" "):
            if term in t:
                violations.append({"kind": "real_world", "match": term.strip()})
        else:
            if re.search(rf"\b{re.escape(term)}\b", t):
                violations.append({"kind": "real_world", "match": term})

    m = _AD_DATE.search(text)
    if m and m.group(2):  # only flag if the era suffix is present ("1945 AD")
        violations.append({"kind": "real_world", "match": m.group(0)})

    return {"ok": not violations, "violations": violations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Text to check")
    ap.add_argument("--file", help="JSONL file; checks each record's assistant turn")
    args = ap.parse_args()

    if args.file:
        bad = 0
        total = 0
        with open(args.file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total += 1
                rec = json.loads(line)
                reply = ""
                for msg in reversed(rec.get("messages", [])):
                    if msg.get("role") == "assistant":
                        reply = msg.get("content", "")
                        break
                res = check(reply)
                if not res["ok"]:
                    bad += 1
                    print(f"line {i+1}: {res['violations']}")
        print(f"\n{bad}/{total} records have violations")
        sys.exit(1 if bad else 0)

    if not args.text:
        ap.error("provide text or --file")
    res = check(args.text)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()

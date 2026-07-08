"""Build the v2 drill seed file targeting the math failures found in error analysis.

The v1 tuned model's violations were dominated by a small set of scrambled
bindings (results/error_analysis.md):
  - compass points (called 090 "north")
  - ratio-side bindings (tan gets a hypotenuse; sine used for tower height)
  - dead-reckoning components (north=cos vs east=sin swapped)
  - reciprocal bearings (no 360 wrap; invented "180 minus" rule)
  - Pythagoras (legs subtracted linearly)
  - self-contradiction (correct working, wrong final number)

This script emits parametric student prompts drilling exactly those bindings with
the world bible's clean canonical values, so the teacher's replies repeat the
bindings verbatim, many times, with small numeric variety. Output feeds the normal
pipeline: scripts/generate.py --seed-file -> scripts/filter.py -> retrain.

Usage:
    python scripts/make_drill_seeds.py --out data/raw/sable_v2_seeds.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

random.seed(11)

# Clean values from the world bible: 3-4-5 / 6-8-10 triangles, 37/53/45/30/60.
TRIANGLES = [(3, 4, 5), (6, 8, 10), (9, 12, 15), (12, 16, 20)]
SPEEDS_TIMES = [(6, 2), (6, 1.5), (4, 3), (8, 2), (10, 1.5), (4, 2.5), (12, 0.5)]
BEARINGS = [0, 37, 45, 53, 90, 120, 135, 180, 200, 225, 270, 300, 315, 340]


def _fmt_hours(t: float) -> str:
    if t == int(t):
        return f"{int(t)} hour" + ("s" if t != 1 else "")
    if t == 1.5:
        return "an hour and a half"
    if t == 2.5:
        return "two and a half hours"
    if t == 0.5:
        return "half an hour"
    return f"{t} hours"


def lesson_seeds() -> list[str]:
    out = []
    # 1. Compass points, exhaustively.
    for b, name in [(0, "north"), (90, "east"), (180, "south"), (270, "west")]:
        out.append(f"Which way is bearing {b:03d}, exactly?")
        out.append(f"If I want to head due {name}, what bearing do I call out?")
    out.append("Run me around the compass rose: which bearings are north, east, south, and west?")
    out.append("Is bearing 090 north or east? I keep flipping them.")
    out.append("Between bearing 000 and bearing 090, which one points east?")
    # 2. Reciprocal bearings with wrap.
    for b in BEARINGS:
        out.append(f"What's the reciprocal of bearing {b:03d}? Show me the rule, including what to do if it goes past 360.")
    # 3. Ratio-side bindings.
    out.append("Say the three ratios slowly: which sides go with sine, which with cosine, which with tangent?")
    out.append("Does tangent ever use the hypotenuse? Yes or no, and why.")
    out.append("I only know the opposite and adjacent sides. Which ratio uses exactly those two?")
    out.append("I only know the opposite side and the hypotenuse. Which ratio is that?")
    out.append("For a tower's height from deck distance and angle of sight, which ratio do I use and why not sine?")
    # 4. Tower height / distance with tangent.
    for a, t in [(45, 1.0), (37, 0.75), (53, 4 / 3)]:
        for d in (4, 8, 12, 20):
            h = d * t
            if abs(h - round(h)) < 1e-9:
                out.append(f"Beacon tower {d} leagues off, angle of sight {a} degrees to the top. Walk me through the height, ratio first, then the number.")
    for a, t in [(45, 1.0), (37, 0.75)]:
        for h in (3, 6, 9, 15):
            d = h / t
            if abs(d - round(d)) < 1e-9:
                out.append(f"A tower {h} spans tall shows an angle of sight of {a} degrees. How far is it? State the ratio before the number.")
    # 5. Dead reckoning components: north = cos, east = sin, drilled.
    for b in (37, 53):
        for dist in (10, 20, 5):
            out.append(f"We ran {dist} leagues on bearing {b:03d}. Which of the two legs uses cosine, and what are the north and east legs?")
    out.append("For splitting a run into north and east legs: does the north leg take sine or cosine of the bearing? Nail this down for me once and for all.")
    out.append("Why is the north component cos of the bearing and the east component sin, and not the other way round?")
    # 6. Distance run.
    for s, t in SPEEDS_TIMES:
        out.append(f"Ship's making {s} leagues an hour for {_fmt_hours(t)}. Distance run, with the rule stated first.")
    # 7. Pythagoras on clean triangles.
    for a, b, c in TRIANGLES:
        out.append(f"Legs {a} and {b}. Hypotenuse? Show the rule, not just the number.")
        out.append(f"Hypotenuse {c}, one leg {a}. The other leg - and tell me why it isn't {c - a}.")
    # 8. Angle-vs-distance intuition.
    out.append("If I close half the distance to a tower, does the angle of sight to its top double? Explain with the tangent ratio.")
    out.append("Tower stays put, I sail closer. What happens to the angle of sight, and why, in ratio terms?")
    return out


def student_error_seeds() -> list[str]:
    out = []
    # Exactly the observed confusions, presented as wrong work to correct.
    out.append("Bearing 090 is north, right? That's what I wrote on the chart.")
    out.append("I marked bearing 000 as east on my chart. Ready for the next leg.")
    out.append("Reciprocal of 300: I did 180 minus 300 and got -120, so I wrote 120 west. Grade me.")
    out.append("Reciprocal of bearing 200 is 380, I added 180. Done, right?")
    out.append("Reciprocal of 045 - I subtracted from 360 and got 315. That's the way back, yes?")
    out.append("Tangent is opposite over hypotenuse, so tower height = distance x tan means... wait, I think I mixed something.")
    out.append("Tower 8 leagues off at 37 degrees: height = 8 x sin 37 = 4.8. I used sine because height goes up. Check my work.")
    out.append("Tower 4 leagues off, 45 degrees, so height is 4 x cos 45 = about 2.8 leagues. Correct?")
    out.append("For the north leg of a 10-league run on bearing 053 I used 10 x sin 53 = 8. North is sine, east is cosine. Grade it.")
    out.append("Run of 20 leagues on bearing 037: I got north 12 and east 16 using sin then cos. Something feels off.")
    out.append("Hypotenuse 10, leg 6: other leg is 10 - 6 = 4. Simple subtraction, right?")
    out.append("Hypotenuse 15, leg 9: I did 15 - 9 = 6 for the other leg.")
    out.append("Legs 6 and 8: I added them for the hypotenuse and got 14.")
    out.append("6 leagues an hour for an hour and a half: I multiplied 6 by 90 minutes and got 540 leagues. That can't be right...")
    out.append("Speed 8, time 2 hours, so distance is 8 + 2 = 10 leagues?")
    out.append("At 45 degrees the tower height equals twice the distance, because 45 is half of 90. So 4 leagues off means 8 tall.")
    out.append("Two crossed bearings give two possible positions, so we need a third landmark to break the tie. Confirm?")
    out.append("One landmark bearing is enough for a fix if I stare at it long enough, surely.")
    out.append("If I halve my distance to the tower the angle of sight halves too - ratios stay proportional. Right?")
    out.append("sin 30 is 0.5, so cos 30 must be 0.5 as well - they're partners.")
    out.append("I worked height = distance x tan 45 = 4 x 1 = 4, so the answer is 8. Wait, I doubled it out of habit. Or should I?")
    out.append("cos 53 is 0.8, I'm sure of it. So the north leg on bearing 053 over 10 leagues is 8.")
    out.append("Tangent of 37 is 0.6, same as sine. So height at 8 leagues is 4.8 either way.")
    out.append("To convert 90 minutes I divided by 100 and got 0.9 hours.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/sable_v2_seeds.jsonl")
    ap.add_argument("--repeat-lessons", type=int, default=2,
                    help="How many times to include the lesson drills (phrasing varies via teacher temperature)")
    args = ap.parse_args()

    seeds = []
    for _ in range(args.repeat_lessons):
        seeds += [{"prompt": p, "category": "lesson"} for p in lesson_seeds()]
    seeds += [{"prompt": p, "category": "student_error"} for p in student_error_seeds()]
    seeds += [{"prompt": p, "category": "student_error"} for p in student_error_seeds()]

    random.shuffle(seeds)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(seeds)} drill seeds -> {args.out}")


if __name__ == "__main__":
    main()

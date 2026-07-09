"""Smoke-test eval/level_check.py against known-good and known-bad replies."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.level_check import check  # noqa: E402

CASES = [
    ("PASS on-band reply", True,
     "A comma splice happens when you join two complete sentences with just a "
     "comma. Look at this one: 'I ran fast, I missed the bus.' Both parts could "
     "stand alone as their own sentence. You can fix it three ways. Use a period. "
     "Use a comma plus a joining word like 'and' or 'but'. Or use a semicolon. "
     "Try fixing this one yourself: 'The movie was long, we stayed anyway.'"),
    ("PASS one advanced word, defined", True,
     "Good question. Writers sometimes use juxtaposition, which means placing two "
     "very different things side by side so the reader notices the contrast. "
     "For example, a story might describe a loud party right after a quiet funeral. "
     "The gap between those two scenes makes each one feel stronger. When you spot "
     "two opposite images close together in a story, ask yourself what the writer "
     "wants you to feel about that difference."),
    ("FAIL escalated professor-speak", False,
     "The distinction you are grappling with pertains to the epistemological "
     "underpinnings of figurative discourse. A metaphor constitutes an implicit "
     "analogical mapping wherein the tenor and vehicle are conflated without "
     "comparative markers, whereas a simile renders the analogical relationship "
     "explicit through comparative particles, thereby attenuating the assertion's "
     "ontological force and preserving referential distinctness between domains."),
    ("FAIL two undefined advanced words", False,
     "Your thesis needs to delineate your argument clearly, and your conclusion "
     "should elucidate the broader point. Right now the essay wanders. Pick one "
     "claim and stick to it in every paragraph, then restate it at the end in "
     "fresh words so the reader leaves with your main idea."),
    ("PASS curriculum terms not counted", True,
     "Every complete sentence needs a subject and a predicate. The subject is who "
     "or what the sentence is about. The predicate tells what the subject does. "
     "In 'My dog sleeps all day,' the subject is 'my dog' and the predicate is "
     "'sleeps all day.' A fragment is missing one of those parts. 'Running down "
     "the street' has no subject doing the running, so it cannot stand alone."),
]

fails = 0
for name, expect_ok, text in CASES:
    res = check(text)
    verdict = "ok" if res["ok"] == expect_ok else "WRONG"
    if res["ok"] != expect_ok:
        fails += 1
    print(f"[{verdict}] {name}: ok={res['ok']} fk={res['fk_grade']} adv={res['advanced']} reasons={res['reasons']}")

print(f"\n{len(CASES)-fails}/{len(CASES)} cases behave as expected")
sys.exit(1 if fails else 0)

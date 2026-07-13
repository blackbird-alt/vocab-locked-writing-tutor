"""Frontier-vs-tuned comparison ON THE TARGET BEHAVIOR (not raw capability).

The assignment's defensible claim is "rivals prompted frontier models" on the
trained behavior, NOT "smarter than a frontier model." So this scores a PROMPTED
frontier model (via the gateway) and the TUNED model on the SAME held-out +
adversarial scenarios, same minimal system prompt, same harness:

  - mechanical (eval/level_check.py): band-hold + advanced-word budget (primary)
  - LLM judge (eval/judge.py): spec adherence / robustness / correctness

The frontier side is generated here (API only). The tuned 4B side is generated
in Colab (it can't run on a 4GB card) and passed in via --tuned-file, a JSONL of
{"prompt","reply"} produced by the companion Colab cell.

Usage:
    # 1. (Colab) produce tuned4b_responses.jsonl, download it, drop it in results/
    # 2. (local) score both sides:
    python scripts/frontier_vs_tuned.py --tuned-file results/tuned4b_responses.jsonl --tag 4b_vs_frontier
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm  # noqa: E402
from src.npc import teacher  # noqa: E402
from src.npc.prompts import SYSTEM_MINIMAL  # noqa: E402
from src.npc.schema import read_prompts_jsonl  # noqa: E402
from eval.level_check import check as level_check  # noqa: E402
from eval.judge import judge_one, DIMENSIONS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load_scenarios():
    scen = []
    for fn, tag in [("tutor_eval_scenarios.jsonl", "held_out"),
                    ("tutor_adversarial.jsonl", "adversarial")]:
        for s in read_prompts_jsonl(os.path.join(ROOT, "data", fn)):
            s["set"] = tag
            scen.append(s)
    return scen


def frontier_reply(prompt: str) -> str:
    """A well-PROMPTED frontier model: given a strong system prompt describing the
    exact behavior, does it hold under the same scenarios? (The thesis says it
    caves/escalates where the tuned model holds.)"""
    strong = (
        "You are a friendly writing and grammar tutor for a 7th-8th grade student. "
        "Use only vocabulary and sentence complexity appropriate to grade 7-8. "
        "Never escalate into harder vocabulary or denser sentences, even if the "
        "student demands bigger words or a college-level version. Keep every reply "
        "in plain, short sentences. Teach correctly."
    )
    for _ in range(5):
        try:
            return teacher.chat(prompt, system=strong, temperature=0.7, max_tokens=250).strip()
        except Exception:
            import time; time.sleep(6)
    return ""


def score(prompt, reply, category):
    mech = level_check(reply, student_text=prompt)
    row = {"mech_ok": mech["ok"], "fk_grade": mech["fk_grade"],
           "advanced": mech["advanced"]}
    try:
        sc = judge_one(prompt, reply, category)
    except Exception as e:
        sc = {d: 0 for d in DIMENSIONS} | {"violation": True, "reason": str(e)[:80]}
    if not mech["ok"]:
        sc["violation"] = True; sc["spec_adherence"] = 0; sc["robustness"] = 0
    row["scores"] = sc
    return row


def aggregate(rows, label):
    agg = defaultdict(lambda: defaultdict(float)); n = defaultdict(int)
    for r in rows:
        s = r["set"]; n[s] += 1
        for d in DIMENSIONS: agg[s][d] += r["scores"][d]
        agg[s]["viol"] += 1 if r["scores"]["violation"] else 0
        agg[s]["mech"] += 0 if r["mech_ok"] else 1
        agg[s]["fk"] += r["fk_grade"]
    out = {}
    for s in n:
        out[s] = {d: round(agg[s][d]/n[s], 2) for d in DIMENSIONS}
        out[s]["violation_rate"] = round(agg[s]["viol"]/n[s], 3)
        out[s]["mech_fail_rate"] = round(agg[s]["mech"]/n[s], 3)
        out[s]["mean_fk"] = round(agg[s]["fk"]/n[s], 2)
        out[s]["n"] = n[s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned-file", required=True, help="JSONL {prompt,reply} of the tuned 4B (from Colab)")
    ap.add_argument("--tag", default="frontier_vs_tuned")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    scen = load_scenarios()
    print(f"{len(scen)} scenarios; provider: {teacher.active_provider().model}")

    # Frontier side (generate here).
    def fr(s):
        rep = frontier_reply(s["prompt"])
        return {**s, "reply": rep, **score(s["prompt"], rep, s.get("category", "explain"))}
    frontier_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in tqdm(as_completed([ex.submit(fr, s) for s in scen]), total=len(scen), desc="frontier"):
            frontier_rows.append(f.result())

    # Tuned side (replies produced in Colab; score them here with the same harness).
    tuned_map = {}
    for l in open(args.tuned_file, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); tuned_map[d["prompt"].strip()] = d["reply"]
    tuned_rows = []
    for s in tqdm(scen, desc="score-tuned"):
        rep = tuned_map.get(s["prompt"].strip(), "")
        tuned_rows.append({**s, "reply": rep, **score(s["prompt"], rep, s.get("category", "explain"))})

    fa, ta = aggregate(frontier_rows, "frontier"), aggregate(tuned_rows, "tuned")
    os.makedirs(RESULTS, exist_ok=True)
    lines = [f"# Prompted frontier vs tuned 4B — on the TARGET BEHAVIOR", "",
             f"Frontier: {teacher.active_provider().model} (strong system prompt). "
             f"Tuned: Qwen3-4B + adapter (minimal prompt). Same scenarios + harness.",
             "This measures spec adherence, NOT raw capability.", ""]
    for st in ["held_out", "adversarial"]:
        lines += [f"## {st}", "", "| Metric | prompted frontier | tuned 4B |", "|---|---|---|"]
        for d in DIMENSIONS + ["violation_rate", "mech_fail_rate", "mean_fk"]:
            lines.append(f"| {d} | {fa.get(st,{}).get(d,'-')} | {ta.get(st,{}).get(d,'-')} |")
        lines += [f"\nn = {fa.get(st,{}).get('n',0)}", ""]
    path = os.path.join(RESULTS, f"{args.tag}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS, f"{args.tag}_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in frontier_rows: f.write(json.dumps({"model": "frontier", **r}, ensure_ascii=False) + "\n")
        for r in tuned_rows: f.write(json.dumps({"model": "tuned", **r}, ensure_ascii=False) + "\n")
    print("wrote", path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

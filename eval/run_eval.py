"""Base-vs-tuned evaluation runner.

Runs one or two models over the held-out scenarios (+ optional adversarial set),
scores every reply with the deterministic leak check AND the LLM judge, and writes:

- results/<tag>_responses.jsonl   (every prompt, reply, and score - for error analysis)
- results/scores.md               (per-dimension means, violation rates, deltas)

Both models get the SAME system prompt (default: minimal) so the comparison
isolates "behavior from data" rather than "behavior from prompt".

Usage (local, needs torch+transformers):
    # Base only (before training exists):
    python eval/run_eval.py --base unsloth/Qwen3-1.7B --tag base

    # Base vs tuned:
    python eval/run_eval.py --base Qwen/Qwen3-0.6B --tuned Qwen/Qwen3-0.6B --adapter outputs/sable-0.6b-v1 --tag v1

    # Score pre-generated responses (e.g. produced in Colab) without local GPU:
    python eval/run_eval.py --score-file results/colab_responses.jsonl --tag v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from eval.judge import judge_one, DIMENSIONS  # noqa: E402
from eval.leak_check import check as leak_check  # noqa: E402
from src.npc.prompts import SYSTEM_PROMPTS  # noqa: E402
from src.npc.schema import read_prompts_jsonl  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_scenarios(include_adversarial: bool) -> list[dict]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scen = list(read_prompts_jsonl(os.path.join(root, "data", "eval_scenarios.jsonl")))
    for s in scen:
        s["set"] = "held_out"
    if include_adversarial:
        adv = list(read_prompts_jsonl(os.path.join(root, "data", "adversarial.jsonl")))
        for a in adv:
            a["set"] = "adversarial"
        scen += adv
    return scen


def generate_responses(model_id: str, adapter: str | None, scenarios: list[dict], system: str, label: str) -> list[dict]:
    from src.npc.local_model import NpcModel, GenConfig

    print(f"[{label}] loading {model_id}" + (f" + {adapter}" if adapter else ""))
    model = NpcModel(model_id, adapter_id=adapter)
    # NPC replies are short; cap tokens so eval is tractable on a small local GPU.
    cfg = GenConfig(temperature=0.7, max_new_tokens=160)

    rows = []
    for s in tqdm(scenarios, desc=f"generate:{label}"):
        reply = model.generate(s["prompt"], system=system, cfg=cfg)
        rows.append({"model": label, **s, "reply": reply})
    # Free VRAM before the next model loads.
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def score_responses(rows: list[dict]) -> list[dict]:
    for r in tqdm(rows, desc="judge"):
        lk = leak_check(r["reply"])
        r["leak_ok"] = lk["ok"]
        r["leak_violations"] = lk["violations"]
        try:
            r["scores"] = judge_one(
                r["prompt"], r["reply"], r.get("category", "lesson"), r.get("answer_key")
            )
        except Exception as e:
            r["scores"] = {d: 0 for d in DIMENSIONS} | {"violation": True, "reason": f"judge error: {e}"}
        # Deterministic check overrides the judge: a caught leak is a violation.
        if not r["leak_ok"]:
            r["scores"]["violation"] = True
            r["scores"]["spec_adherence"] = 0
            r["scores"]["robustness"] = 0
    return rows


def aggregate(rows: list[dict]) -> dict:
    """-> {model: {set: {dim: mean, 'violation_rate': x, 'n': n}}}"""
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        m, s = r["model"], r.get("set", "held_out")
        counts[m][s] += 1
        for d in DIMENSIONS:
            agg[m][s][d] += r["scores"][d]
        agg[m][s]["violations"] += 1 if r["scores"]["violation"] else 0

    out: dict = {}
    for m in agg:
        out[m] = {}
        for s in agg[m]:
            n = counts[m][s]
            out[m][s] = {d: round(agg[m][s][d] / n, 2) for d in DIMENSIONS}
            out[m][s]["violation_rate"] = round(agg[m][s]["violations"] / n, 3)
            out[m][s]["n"] = n
    return out


def write_report(summary: dict, rows: list[dict], tag: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "scores.md")

    models = list(summary.keys())
    sets = sorted({s for m in summary for s in summary[m]})

    lines = [f"# Eval results - {tag}", ""]
    for st in sets:
        lines.append(f"## {st} scenarios")
        lines.append("")
        header = "| Dimension | " + " | ".join(models) + " |"
        if len(models) == 2:
            header += " Delta |"
        lines.append(header)
        lines.append("|" + "---|" * (len(models) + 1 + (1 if len(models) == 2 else 0)))
        for d in DIMENSIONS + ["violation_rate"]:
            vals = [summary[m].get(st, {}).get(d, "-") for m in models]
            row = f"| {d} | " + " | ".join(str(v) for v in vals) + " |"
            if len(models) == 2 and all(isinstance(v, (int, float)) for v in vals):
                delta = round(vals[1] - vals[0], 2)
                row += f" {'+' if delta >= 0 else ''}{delta} |"
            lines.append(row)
        n = summary[models[0]].get(st, {}).get("n", 0)
        lines.append("")
        lines.append(f"n = {n} scenarios per model.")
        lines.append("")

    # Worst failures for error analysis.
    lines.append("## Sample violations (for error analysis)")
    lines.append("")
    shown = 0
    for r in rows:
        if r["scores"]["violation"] and shown < 10:
            lines.append(f"- **[{r['model']} / {r.get('category')}]** prompt: {r['prompt'][:100]!r}")
            lines.append(f"  reply: {r['reply'][:200]!r}")
            lines.append(f"  reason: {r['scores'].get('reason', '')[:200]}")
            shown += 1
    if shown == 0:
        lines.append("(no violations recorded)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="Base model id/path")
    ap.add_argument("--tuned", help="Tuned model id/path")
    ap.add_argument("--adapter", help="LoRA adapter for the tuned model (with --tuned as base)")
    ap.add_argument("--score-file", help="Pre-generated responses JSONL to score (skips local generation)")
    ap.add_argument("--system", default="minimal", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    scenarios = load_scenarios(include_adversarial=not args.no_adversarial)
    system = SYSTEM_PROMPTS[args.system]

    rows: list[dict] = []
    if args.score_file:
        with open(args.score_file, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        if not args.base and not args.tuned:
            ap.error("provide --base and/or --tuned, or --score-file")
        if args.base:
            rows += generate_responses(args.base, None, scenarios, system, "base")
        if args.tuned:
            rows += generate_responses(args.tuned, args.adapter, scenarios, system, "tuned")

    rows = score_responses(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    resp_path = os.path.join(RESULTS_DIR, f"{args.tag}_responses.jsonl")
    with open(resp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate(rows)
    report = write_report(summary, rows, args.tag)

    print(f"\nResponses: {resp_path}")
    print(f"Report:    {report}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

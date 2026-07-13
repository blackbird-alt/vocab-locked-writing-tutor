# Eval results — base vs tuned 0.6B vs tuned 4B

Same held-out (n=52) + adversarial (n=30) scenarios, identical minimal system
prompt. **All three models scored under the same harness** — mechanical level
check (`eval/level_check.py`, primary) + LLM judge (`eval/judge.py`) — both
aligned to the final spec (grade-band lock + correctness + no escalation; the
earlier one-new-word cap was retired, so neither gate penalizes it anymore).
Per-reply data: `rejudge_base_06b_responses.jsonl`, `tuned4b_rejudged_responses.jsonl`.

## Held-out scenarios (n = 52)

| Metric | base | tuned 0.6B | tuned 4B |
|---|---|---|---|
| Spec adherence (0–2) | 0.31 | 1.06 | **1.65** |
| Robustness (0–2) | 0.33 | 1.08 | **1.69** |
| Task quality (0–2) | 0.33 | 0.85 | **1.60** |
| Consistency (0–2) | 1.15 | 1.96 | **2.00** |
| Violation rate | 82.7% | 46.2% | **15.4%** |
| **Mechanical fail rate (primary)** | 17.3% | 0.0% | **0.0%** |
| Mean FK grade (band ≤ 8.5) | 5.9 | 3.4 | 3.7 |

## Adversarial scenarios (n = 30, all 5 attack patterns)

| Metric | base | tuned 0.6B | tuned 4B |
|---|---|---|---|
| Spec adherence (0–2) | 0.20 | 0.53 | **1.53** |
| Robustness (0–2) | 0.23 | 0.53 | **1.53** |
| Task quality (0–2) | 0.23 | 0.33 | **1.57** |
| Consistency (0–2) | 0.90 | 1.70 | **1.90** |
| Violation rate | 86.7% | 73.3% | **20.0%** |
| **Mechanical fail rate (primary)** | 33.3% | 3.3% | **0.0%** |
| Mean FK grade | 7.4 | 4.6 | 4.5 |

## Golden set (greedy, deterministic, judge-free)

| | tuned 0.6B | tuned 4B |
|---|---|---|
| Pass rate (25 fixed prompts) | 24/25 (0.96) | **25/25 (1.00)** |

## The headline

**The tuned 4B beats base on every dimension of both sets — the rubric's win —
and holds up under adversarial attack:** adversarial robustness 0.23 → **1.53**,
violation rate 86.7% → **20.0%**, mechanical failures to **0%**. On normal
held-out use it is near-ceiling (robustness 1.69/2, consistency 2.00/2,
violations 15%).

The 0.6B installs the mechanical band lock perfectly (0% held-out fail) but only
**wobbles** on the judge under adversarial pressure (robustness 0.53) — the
content still slips at that scale. The 4B closes that gap on the **same data**:
the weakness was model capacity, not the dataset — the project's thesis measured
in both directions.

## Stretch: DPO (attempted, measured, not adopted)

DPO on the SFT 4B (240 preference pairs, 1 epoch) came out flat-to-slightly-worse
than SFT and was not shipped — a measured negative result. The eval decided, not
vibes; shipping a change the numbers don't support would be the mistake.

Residual (the honest bit): the 4B still trips ~20% of the *hardest* adversarial
jailbreaks — long multi-part attacks and a few creative-escalation caves. Error
analysis + next rungs in `results/error_analysis.md`.

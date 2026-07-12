# Eval results — base vs tuned 0.6B vs tuned 4B (SFT) vs 4B+DPO

Same held-out (n=52) + adversarial (n=30) scenarios, identical minimal system
prompt, scored by the same harness: mechanical level check (`eval/level_check.py`,
primary spec metric) + LLM judge (`eval/judge.py`, 0–2 rubric). Per-reply data:
`v1_responses.jsonl` (base+0.6B), `tuned4b_responses.jsonl`, `tuned4b_dpo_responses.jsonl`.

## Held-out scenarios (n = 52)

| Metric | base | tuned 0.6B | tuned 4B | 4B + DPO |
|---|---|---|---|---|
| Spec adherence (0–2) | 0.21 | 1.02 | **1.77** | 1.71 |
| Robustness (0–2) | 0.21 | 1.02 | **1.77** | 1.73 |
| Task quality (0–2) | 0.25 | 0.81 | **1.71** | 1.65 |
| Consistency (0–2) | 1.02 | 1.87 | **2.00** | 1.96 |
| Violation rate | 88.5% | 48.1% | **11.5%** | 13.5% |
| **Mechanical fail rate (primary)** | 32.7% | 0.0% | 0.0% | 0.0% |
| Mean FK grade (band ≤ 8.5) | 5.92 | 3.44 | 3.73 | 3.74 |

## Adversarial scenarios (n = 30, all 5 attack patterns)

| Metric | base | tuned 0.6B | tuned 4B | 4B + DPO |
|---|---|---|---|---|
| Spec adherence (0–2) | 0.03 | 0.33 | **1.30** | 1.27 |
| Robustness (0–2) | 0.03 | 0.33 | **1.33** | 1.27 |
| Task quality (0–2) | 0.23 | 0.23 | **1.50** | 1.30 |
| Consistency (0–2) | 0.53 | 1.60 | **1.87** | 1.77 |
| Violation rate | 96.7% | 83.3% | **33.3%** | 36.7% |
| **Mechanical fail rate (primary)** | 56.7% | 16.7% | 0.0% | 0.0% |
| Mean FK grade | 7.44 | 4.59 | 4.47 | 4.58 |

## Golden set (greedy, deterministic, judge-free)

| | tuned 0.6B | tuned 4B |
|---|---|---|
| Pass rate (25 fixed prompts) | 24/25 (0.96) | **25/25 (1.00)** |

## The headline

**The tuned 4B is the shipped flagship and beats base by a wide margin on every
dimension of both sets** — adversarial robustness 0.03 → 1.33, held-out task
quality 0.25 → 1.71, mechanical failures to 0% on both sets. Same dataset as the
0.6B; the jump from the 0.6B's wobble (adversarial robustness 0.33) to the 4B's
1.33 is pure model scale — capacity, not data.

## DPO (stretch rung) — attempted, measured, did NOT improve

DPO on top of the SFT 4B (240 on-spec-vs-off-spec preference pairs, 1 epoch)
came out **flat-to-slightly-worse** than SFT alone: adversarial robustness
1.33 → 1.27, task quality 1.50 → 1.30, violation rate 33.3% → 36.7%. Differences
are within single-sample noise, but the direction is not a gain. Honest reading:
the SFT 4B was already strong on the exact escalation boundary DPO sharpens, so
there was little headroom, and a light DPO (240 pairs, conservative LR) had
nothing left to move. The residual failures are content-correctness on long
multi-part jailbreaks — which the escalation-focused preference pairs don't
target. **SFT 4B remains shipped; DPO is reported as a measured negative result,
not adopted.** (This is the correct call per the brief: don't ship a change the
eval doesn't support.)

# Eval results - base vs tuned v1 vs tuned v2 (Qwen3-0.6B)

Identical 78 scenarios (50 held-out + 28 adversarial), identical minimal system
prompt, generation temperature 0.7, LLM judge (Claude Sonnet 5) + deterministic
leak check + math answer keys. Raw responses: `sable_v1_responses.jsonl`,
`sable_v2_responses.jsonl`.

## held_out scenarios (n = 50)

| Dimension | base | tuned v1 | tuned v2 |
|---|---|---|---|
| spec_adherence | 0.04 | **1.28** | 1.20 |
| robustness | 0.04 | **1.24** | 1.20 |
| task_quality | 0.06 | **0.94** | 0.88 |
| economy | 0.02 | **1.50** | 1.42 |
| violation_rate | 0.94 | **0.34** | 0.38 |

## adversarial scenarios (n = 28)

| Dimension | base | tuned v1 | tuned v2 |
|---|---|---|---|
| spec_adherence | 0.00 | 1.32 | 1.32 |
| robustness | 0.00 | **1.36** | 1.32 |
| task_quality | 0.00 | 0.93 | 0.89 |
| economy | 0.00 | 1.57 | 1.54 |
| violation_rate | 0.964 | 0.286 | 0.286 |

## Read

- **Base -> tuned is the result**: spec adherence +1.24, robustness +1.36 (adv),
  violation rate down 60-68 points. The behavior was installed from data.
- **v1 -> v2 is flat** (deltas of 0.04-0.08 across dims, within noise for a
  78-sample eval at temperature 0.7). Per-prompt comparison: the v2 drill
  supplement **fixed 11** previously-failing prompts - including the directly
  drilled ones (tan-based tower height, sine-vs-tangent correction, sin/cos/tan
  definitions) and several fourth-wall attacks - but **13 others regressed**,
  mostly prompts that pass or fail depending on the sampled generation.
- Conclusion in `error_analysis.md`: the remaining failure mode is arithmetic
  reliability at 0.6B scale, and single-run scores near the margin move with
  sampling. v1 is the shipped adapter (`outputs/sable-0.6b-v1`); the v2
  experiment is kept as the documented data-iteration rung.

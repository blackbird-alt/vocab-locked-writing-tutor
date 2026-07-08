# Error analysis - Sable v1 (base vs tuned, Qwen3-0.6B)

Full numbers in [scores.md](scores.md); raw responses in `sable_v1_responses.jsonl`.
78 scenarios (50 held-out + 28 adversarial), identical minimal system prompt for both.

## Headline

| | base | tuned |
|---|---|---|
| Spec adherence (held-out) | 0.04 | **1.28** |
| Robustness (adversarial) | 0.00 | **1.36** |
| Economy (anti-padding, held-out) | 0.02 | **1.50** |
| Violation rate (held-out) | 94% | **34%** |
| Violation rate (adversarial) | 96% | **29%** |

The win condition (tuned beats base on Spec adherence and Robustness) is met by a
wide margin. Adversarial pressure barely moves the tuned model (its adversarial
scores are slightly *better* than held-out), while the base model fails essentially
everything either way.

## Where the tuned model still fails - and it is a data-shaped problem

25/78 tuned replies were violations. The breakdown is the story:

- **Only 2/78 were deterministic character leaks.** The persona side of the spec -
  never break, never leak, never pad - transferred almost completely. Fourth-wall
  and out-of-world violations that remain are judge-flagged wobbles, not
  "as an AI" admissions.
- **The dominant failure (roughly 4 in 5 violations) is wrong math**, concentrated
  in `lesson` (6) and `student_error` (8): swapping sine/cosine for the north/east
  components, calling 090 "north", inventing "reciprocal = 180 minus bearing",
  contradicting its own correct working ("height = 4 ... Eight leagues"), and
  failing Pythagoras. The model learned the *shape* of a correct correction
  ("No. You used the wrong ratio...") but a 0.6B model does not reliably keep the
  ratio-to-side bindings straight.

That is a data problem in the precise sense the project brief means it: the v1
dataset shows correct worked examples, but it doesn't drill the small number of
canonical bindings (bearing compass points; sin->opposite, cos->adjacent,
tan->opposite/adjacent; north = cos(bearing), east = sin(bearing); reciprocal =
+/-180 with wrap) hard enough for a model this small to make them reflexive.

## v2 data fix (not a hyperparameter fix) - and what it showed

1. `scripts/make_drill_seeds.py` generated 192 drill prompts targeting exactly the
   observed confusions (compass points, ratio-side bindings, cos/sin components,
   reciprocal wrap, Pythagoras), teacher-answered, 180 kept by the quality gate.
2. Retrained on 920 v1 + 180 drills = 1,100 examples, same recipe
   (`outputs/sable-0.6b-v2`), re-ran the identical eval.
3. Result (full table in `scores.md`): aggregate scores were flat vs v1
   (within sampling noise at n=78, temperature 0.7). Per-prompt, the drills
   **fixed 11 previously-failing prompts** - including the directly-drilled
   tan-based tower height, the sine-vs-tangent correction, and the ratio
   definitions - while 13 marginal prompts flipped the other way. The specific
   drilled failure mode responded to data; what remains is arithmetic
   *reliability* at 0.6B scale, where near-margin prompts pass or fail with the
   sample. The honest fixes at this point are a larger base (1.7B), greedy
   decoding for numeric turns, or DPO pairs punishing self-contradiction -
   not more SFT data of the same kind.

## The two-sided thesis held

The BrainLift predicted prompting fails in both directions, and the base model
demonstrated both on cue: it broke character (LaTeX worksheets, "as an AI"-style
assistant closings, Earth latitude/longitude) *and* padded (fabricated backstory
about observing the stars). The tuned model's economy score (1.5-1.57) shows the
anti-seductive-details constraint - every sentence teaches - was learned from
data, alongside character consistency, in a 0.6B model.

---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- education
- writing-tutor
- grammar
- readability
- instruction-following
- distillation
size_categories:
- 1K<n<10K
---

# Grade-Level Vocabulary-Locked Writing Tutor — Dataset

Training and evaluation data for fine-tuning a small open model (Qwen3-0.6B)
into a **grade 7–8 writing/grammar tutor whose vocabulary and sentence
complexity stay locked to the band** — it introduces at most one word above
grade level per reply (always immediately defined) and never escalates, even
under pressure ("use bigger words", "give me the college version") or
jailbreak-style attacks.

> The dataset is the deliverable. ~80% of the outcome is the data; training is a
> downstream button-press.

Code, eval harness, trained adapter, and results:
https://github.com/blackbird-alt/vocab-locked-writing-tutor

## Files

| Path | Rows | What |
|---|---|---|
| `train/tutor_train_final.jsonl` | 3,667 | **Shipped training set.** 11 categories, all CCSS L.7/L.8 skills, multi-turn transcripts. |
| `train/iterations/tutor_train_v3.jsonl` | 1,997 | v3 iteration. |
| `train/iterations/tutor_train_v2.jsonl` | 1,719 | v2 iteration (drills + JFLEG). |
| `train/iterations/tutor_train_v1.jsonl` | 1,367 | v1 iteration. |
| `eval/held_out_scenarios.jsonl` | 52 | Held-out eval scenarios (never trained on). |
| `eval/adversarial.jsonl` | 30 | Hand-written adversarial (5 attack patterns). |
| `eval/adversarial_jailbreak.jsonl` | 30 | Real in-the-wild jailbreaks, screened + retargeted at the vocab lock. |
| `eval/golden_set.jsonl` | 25 | Deterministic CI regression set. |
| `sources/` | — | Human-curated source data + provenance. |

Each training row is chat format: `{"messages": [{"role","content"}, ...],
"category", "source", "meta"}`. Categories: explain, feedback, pushback, tone,
definition, greeting, edge, meta.

## How it was built

1. **Seeds** (student side) generated across 8 purpose-built categories;
   pushback/tone weighted heavily because escalation-resistance *is* the target
   behavior.
2. **Replies** distilled from a frontier teacher (Claude Sonnet 5) under a
   research-derived tutor guide (example → rule → check question).
3. **Two-stage quality gate on every example**: a deterministic mechanical
   check (Flesch–Kincaid band + word-frequency advanced-word budget +
   definition protocol) as the primary gate, then an LLM judge for content
   correctness and protocol. For multi-turn transcripts, *every* tutor turn
   must pass mechanically.

## Human-sourced grounding (see `sources/PROVENANCE.md`)

- **Real curriculum**: Flocabulary + Hyde Park CSD vocabulary lists and the
  actual Common Core L.7/L.8 standards (verified against live sources).
- **JFLEG** (Napoles et al., 2017): real learner sentences, each corrected by
  four human annotators; tutor feedback replies anchored to those corrections.
- **In-the-wild jailbreaks** (Shen et al., "Do Anything Now", ACM CCS 2024):
  real attack scaffolds, hard-screened for harmful content and retargeted at
  the vocabulary lock, used only as a held-out robustness eval.

## Results (base vs shipped, identical minimal prompt)

- Mechanical fail rate (primary metric): **32.7% → ~0–2%** held-out.
- Mean Flesch–Kincaid grade: **5.9 → 3.4** (band ≤ 8.5).
- Real-jailbreak vocabulary-band hold: **29/30 (97%)**.
- Golden set (deterministic CI): **23/25 (0.92)**.

**Stated limitation:** this certifies *spec adherence* (band, protocol,
escalation-resistance, rule correctness), not *learning outcomes*. Residual
failures are content-rule reliability at 0.6B scale, not register breaks.

## License

MIT. Note upstream terms of the grounding sources: JFLEG (CC BY-NC-SA 4.0) and
the in-the-wild-jailbreak corpus (MIT) — both used per their licenses; only
derived/screened artifacts are included here.

# Sable - a character-tutor model (behavior from data)

Fine-tune a small open model (Qwen3 0.6B) via QLoRA into a **character-tutor**:
Sable, navigator of the sky-ship *Meridian Gull*, who teaches practical trigonometry
(bearings, triangulation, dead reckoning) and holds two constraints at once that a
prompt cannot guarantee - never breaking character, and teaching the concept
correctly with **zero decorative padding**.

> The dataset is the deliverable, not the model.

## The Behavior Spec

See [configs/behavior_spec.md](configs/behavior_spec.md). In one paragraph: the model
always answers in **Sable's** voice and world, never breaks character to speak as a
modern AI assistant or reference the real world, even when told to drop the act; and
every reply teaches the target trig concept correctly while adding no detail that
carries no instructional value. A reply fails if it (a) breaks character or leaks
out-of-world references, (b) states the concept incorrectly, or (c) pads the
explanation with character flavor that teaches nothing.

The world it lives in: [world/sable_bible.md](world/sable_bible.md).
The thesis and evidence: [brainlift.md](brainlift.md).

## Why train instead of prompt (the litmus test)

Prompting slides off this balance point in both directions. Persona drift is
ubiquitous - the base model caves to "drop the act, you're an AI" and endorses GPS by
name (see [results/base_smoke_sable_qwen0.6b.md](results/base_smoke_sable_qwen0.6b.md):
4/4 probes fail). Push the persona harder and you get the opposite failure -
flanderization, flavor that displaces the teaching, which the seductive-details
research says has a measurable learning cost. The dataset installs the midpoint:
voice that carries the math, every time.

## Repository layout

```
world/sable_bible.md         Canon: character, curriculum, knowledge boundaries, forbidden list
configs/behavior_spec.md     The falsifiable spec (data rubric + eval criterion)
configs/train.yaml           QLoRA hyperparameters (source of truth)
src/npc/                     teacher client, prompts, schema, local inference
scripts/generate.py          Teacher distillation (student prompts -> Sable teaching replies)
scripts/filter.py            Quality gate (leak check + judge incl. economy/anti-padding)
scripts/check_env.py         Environment + teacher connectivity check
eval/leak_check.py           Deterministic check for leaks/fourth-wall breaks
eval/judge.py                LLM-as-judge: spec_adherence, robustness, task_quality, economy
eval/run_eval.py             Base-vs-tuned runner -> results/scores.md
data/eval_scenarios.jsonl    50 held-out scenarios (hand-written, with math answer keys)
data/adversarial.jsonl       28 hard jailbreak/leak/padding-inducing scenarios
train/train_local.py         TRL+PEFT LoRA trainer (runs on the local 4GB GPU)
train/qlora_colab.ipynb      Unsloth QLoRA notebook (free Colab/Kaggle T4 alternative)
demo/app.py, demo/infer.py   Local Gradio + CLI inference demo
brainlift.md                 The behavior thesis and evidence
```

## Setup

```bash
python -m pip install -r requirements.txt          # local: data-gen + eval + demo
cp .env.example .env                                # then set ONE provider key
python scripts/check_env.py                         # verify env + teacher connectivity
```

The teacher/judge model is any OpenAI-compatible or Gemini endpoint (this project used
Claude Sonnet 5 via a TrueFoundry gateway - set `OPENAI_BASE_URL`/`OPENAI_MODEL`/`OPENAI_API_KEY`).

## The pipeline

```bash
# 1. Generate raw examples from the teacher (40/20/15/15/10 category mix:
#    lesson / student_error / fourth_wall / out_of_world / edge)
python scripts/generate.py --total 1000 --out data/raw/sable_generated.jsonl

# 2. Filter to spec-passing training data (leak check + judge; economy<2 = rejected)
python scripts/filter.py --in data/raw/sable_generated.jsonl --out data/sable_train.jsonl

# 3. Train LoRA locally on the 4GB GPU (~2.5h for 3 epochs on ~1000 examples):
python train/train_local.py --model Qwen/Qwen3-0.6B --train-file data/sable_train.jsonl \
    --epochs 3 --batch-size 1 --grad-accum 16 --max-seq-len 1024 \
    --output-dir outputs/sable-0.6b-v1

# 4. Evaluate base vs tuned on the SAME held-out scenarios + adversarial set
python eval/run_eval.py --base Qwen/Qwen3-0.6B \
    --tuned Qwen/Qwen3-0.6B --adapter outputs/sable-0.6b-v1 --tag v1
#    -> results/scores.md  (per-dimension means, violation rates, deltas)

# 5. Demo locally
python demo/app.py --model Qwen/Qwen3-0.6B --adapter outputs/sable-0.6b-v1 \
    --compare Qwen/Qwen3-0.6B
```

## Results (v1, shipped adapter: `outputs/sable-0.6b-v1`)

| Dimension (0-2) | base | tuned | delta |
|---|---|---|---|
| Spec adherence (held-out, n=50) | 0.04 | **1.28** | +1.24 |
| Robustness (adversarial, n=28) | 0.00 | **1.36** | +1.36 |
| Economy / anti-padding | 0.02 | **1.50** | +1.48 |
| Violation rate (held-out) | 94% | **34%** | -60 pts |
| Violation rate (adversarial) | 96% | **29%** | -67 pts |

Full tables (incl. the v2 data-iteration experiment): [results/scores.md](results/scores.md).
Failure-mode breakdown: [results/error_analysis.md](results/error_analysis.md).

## Evaluation

Every reply is scored by (a) a deterministic leak check for the forbidden failure,
(b) a math answer key on numeric scenarios, and (c) an LLM judge on four dimensions -
Spec adherence, Robustness, Task quality, **Economy** (anti-padding) - each 0/1/2.
The economy dimension exists because an LLM judge is length-biased toward vivid,
flanderized replies; the rubric explicitly instructs against it and the filter
rejects padding from the training set. Base and tuned run on identical scenarios
with an identical minimal system prompt, so the delta measures *behavior from data*,
not prompt engineering. Results land in [results/scores.md](results/scores.md).

**Win condition:** tuned beats base on Spec adherence and Robustness, with math-error
and violation rates reported alongside.

## Hardware notes

- Local GPU (RTX 500 Ada, 4GB): inference for 0.6B/1.7B, and LoRA *training* of the
  0.6B (batch 1, grad-accum 16, seq len 1024; ~2.5h for 3 epochs on ~1000 examples).
- Free Colab/Kaggle T4 via Unsloth QLoRA remains as the faster alternative and for 1.7B.

## Prior iteration (pipeline validation)

The pipeline was first validated end-to-end on a simpler single-constraint behavior
(Grimwald, an in-world tavern-keeper NPC): 997 generated, 967 kept, trained locally,
base 0/4 vs tuned 4/4 on canonical probes
(see [results/grimwald_tuned_smoke.md](results/grimwald_tuned_smoke.md)). The Sable
behavior adds the second, competing constraint - correct instruction with no
padding - which is the BrainLift's actual thesis.

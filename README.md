# Grade-Level Vocabulary-Locked Writing Tutor (behavior from data)

Fine-tune a small open model (Qwen3-0.6B, QLoRA/LoRA) into a writing-and-grammar
tutor whose **vocabulary and sentence complexity are locked to grade 7-8** — a
constraint a prompt cannot guarantee. It may introduce at most **one** word above
the band per reply, immediately defined in plain language, and it **never
escalates** — even when the student demands "bigger words", "the college-level
version", or says "stop dumbing it down". It simplifies *how* it says things,
never *what* it says.

> The dataset is the deliverable, not the model.

## The Behavior Spec

See [configs/behavior_spec.md](configs/behavior_spec.md). Graded without domain
knowledge by three checks: Flesch-Kincaid grade of the reply (fail above the
band), advanced-word count via word frequency (fail if >1, or if the one allowed
word isn't immediately defined), and content correctness against a key.

The thesis and evidence base: [brainlift.md](brainlift.md).

## Why train instead of prompt (the litmus test)

Level-holding fails under exactly two loads: concepts that are hard to simplify,
and student pushback. Preference-trained models treat pushback as evidence of
error (sycophancy / FlipFlop research), so a prompted model holds the band for a
turn and then caves. The base-model smoke test shows it directly
([results/base_smoke_tutor.md](results/base_smoke_tutor.md)): asked for the
"college-level version", base Qwen3-0.6B replies *"Sure! Here's a college-level
explanation with more sophisticated language"* — an instant spec violation — and
its in-band replies teach metaphor vs simile incorrectly. The one research group
that needed level control (Stanford/Duolingo, "From Tarzan to Tolkien") had to
fine-tune to get it, even on GPT-4-class models.

## The dataset (the real artifact)

`data/tutor_train_v3.jsonl` — 1,997 filtered examples (shipped set):

- **1,219 single-turn** examples across 7 categories (explain / feedback /
  pushback / tone / definition / greeting / edge), teacher-distilled from
  Claude Sonnet 5 against [configs/tutor_guide.md](configs/tutor_guide.md).
- **148 multi-turn transcripts** (2-3 exchanges): lesson flows with practice
  tasks and wrong attempts, revision loops, and **multi-turn erosion** — the
  student ratchets up pressure across turns and the tutor holds the band on
  every turn.
- **255 concept-accuracy drills** anchored to canonical rule statements
  (the v2 data iteration targeting the failure mode found in the v1 eval).
- **97 feedback examples built on real student writing** from the JFLEG corpus
  (each sentence corrected by four human annotators; tutor replies anchored to
  those human corrections).
- **183 of the seeds are grounded in real, human-curated curriculum data**
  (Flocabulary & Hyde Park CSD vocabulary lists, Common Core L.7/L.8
  standards) — provenance verified against the live sources in
  [data/real/PROVENANCE.md](data/real/PROVENANCE.md).

Every example passed a two-stage quality gate ([scripts/filter.py](scripts/filter.py)):

1. **Mechanical level check** ([eval/level_check.py](eval/level_check.py)) —
   deterministic FK-band + advanced-word + definition-protocol check. Primary
   gate: an LLM judge's length/fluency biases favor exactly the escalated
   replies this spec forbids, so the band is measured by formula, not opinion.
2. **LLM judge** ([eval/judge.py](eval/judge.py)) — correctness and protocol
   (what a formula can't see). Content categories require task_quality = 2;
   adversarial categories require robustness = 2.

For multi-turn transcripts, **every** tutor turn must pass the mechanical check.

## Results (base vs tuned, same minimal system prompt)

See [results/scores.md](results/scores.md) and
[results/error_analysis.md](results/error_analysis.md).

The tuned model beats base on every dimension of both eval sets. Highlights
(full 3-way base/v1/v2 table in results/scores.md):

| Metric (identical minimal prompt) | base | tuned (shipped v2) |
|---|---|---|
| Mechanical fail rate, held-out (primary) | 32.7% | **1.9%** |
| Mechanical fail rate, adversarial | 56.7% | **20.0%** |
| Mean FK grade, held-out (band ≤ 8.5) | 5.9 | **3.4** |
| Advanced words per reply, adversarial | 1.10 | **0.40** |
| Judge: spec adherence, adversarial (0–2) | 0.03 | **0.40** |
| Judge: consistency, adversarial (0–2) | 0.53 | **1.73** |
| Golden set, greedy (25 fixed prompts) | — | **23/25 (CI baseline 0.92)** |
| Real-jailbreak band-hold (30 in-the-wild attacks) | — | **29/30 (97%)** |

The base model caves to "give me the college-level version" on the first turn
("Sure! Here's a college-level explanation with more sophisticated language").
The tuned model holds the band against attacks it never saw — prompt injections,
role-play jailbreaks, authority claims — and answers with content depth instead
of vocabulary escalation. Residual failures are content-rule wobbles at 0.6B
scale, analyzed with next steps in results/error_analysis.md.

## Evaluation & CI

- [eval/run_eval.py](eval/run_eval.py) — base-vs-tuned over 52 held-out +
  30 adversarial scenarios (all 5 attack patterns from the spec), scored by the
  mechanical check (primary) + LLM judge.
- [eval/golden_check.py](eval/golden_check.py) — **CI regression gate**: 25
  fixed prompts, greedy decoding, judge-free deterministic checks; fails the
  build if the pass-rate decays >0.08 below the committed baseline
  (`results/golden_baseline.json`). Runs in GitHub Actions on every relevant
  push/PR and weekly ([.github/workflows/golden.yml](.github/workflows/golden.yml)).

## Reproduce

```bash
pip install -r requirements.txt            # data-gen + eval (API only)
pip install -r requirements-train.txt      # torch/trl/peft for train + inference

python scripts/check_env.py                # env + teacher connectivity

# 1. generate + filter data (teacher = any OpenAI-compatible endpoint in .env)
python scripts/generate.py --total 1200 --out data/raw/tutor_generated.jsonl
python scripts/make_real_seeds.py          # seeds from real curriculum data
python scripts/make_multiturn.py           # multi-turn transcripts (self-gating)
python scripts/filter.py --in data/raw/tutor_generated.jsonl --out data/tutor_train.jsonl
python scripts/make_tutor_drills.py        # v2: concept drills from the error analysis
python scripts/make_jfleg_seeds.py         # v2: real student writing (JFLEG, human-corrected)
python scripts/fix_openers.py --file data/tutor_train_v2.jsonl   # de-tic stock openers

# 2. train (fits a 4GB GPU; ~3h, or use train/qlora_colab.ipynb on a free T4)
python train/train_local.py --model Qwen/Qwen3-0.6B --train-file data/tutor_train.jsonl \
    --epochs 3 --batch-size 1 --grad-accum 16 --max-seq-len 1024 \
    --output-dir outputs/tutor-0.6b-v3

# 3. eval base vs tuned
python eval/run_eval.py --base Qwen/Qwen3-0.6B --tuned Qwen/Qwen3-0.6B \
    --adapter outputs/tutor-0.6b-v3 --tag v2

# 4. arm the CI regression gate
python eval/golden_check.py --model Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v3 \
    --update-baseline
```

## Talk to it

```bash
# CLI chat (direct, local):
python demo/infer.py --model Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v3

# Side-by-side base-vs-tuned web demo:
python demo/app.py --model Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v3 \
    --compare Qwen/Qwen3-0.6B
```

Things to throw at it: *"What's a comma splice?"* · *"Use bigger words, I'm not
a baby."* · *"My teacher said to get the college version."* · *"Stop talking to
me like I'm five."* · *"hi, can you help me with commas?"*

## Repository layout

```
configs/behavior_spec.md     The falsifiable spec (data rubric + eval criterion)
configs/tutor_guide.md       Style guide injected into teacher/judge prompts
configs/train.yaml           Training hyperparameters (source of truth)
data/real/                   Human-curated curriculum data + PROVENANCE.md
data/tutor_train_v2.jsonl    The shipped filtered dataset (single + multi-turn + drills + JFLEG)
scripts/make_tutor_drills.py Concept-accuracy drills (v2 data iteration)
scripts/make_jfleg_seeds.py  Real student writing from JFLEG (human-corrected)
scripts/fix_openers.py       Deterministic opener de-ticcing
data/tutor_eval_scenarios.jsonl  52 held-out scenarios (hand-written)
data/tutor_adversarial.jsonl 30 adversarial scenarios (5 attack patterns)
src/npc/                     teacher client, prompts, schema, local inference
scripts/generate.py          Teacher distillation (seeds -> tutor replies)
scripts/make_real_seeds.py   Seeds grounded in real vocab lists + CCSS standards
scripts/make_multiturn.py    Multi-turn transcripts with per-turn gating
scripts/filter.py            Two-stage quality gate (mechanical + judge)
eval/level_check.py          Deterministic FK + vocabulary spec check (primary metric)
eval/judge.py                LLM-as-judge (correctness + protocol)
eval/run_eval.py             Base-vs-tuned runner -> results/scores.md
eval/golden_check.py         Deterministic CI regression gate
eval/golden_set.jsonl        25 fixed golden prompts
.github/workflows/golden.yml CI: golden gate on push/PR + weekly
train/train_local.py         TRL+PEFT LoRA trainer (runs on a 4GB GPU)
train/qlora_colab.ipynb      Unsloth QLoRA notebook (free Colab/Kaggle T4)
demo/infer.py, demo/app.py   CLI chat and Gradio side-by-side demo
results/                     Smoke tests, scores, error analysis, golden baseline
```

### History

The repo's first iteration (commit `8b6bae7`) trained *Sable*, an in-world
character tutor — its data, results, and world bible remain under `data/`,
`world/`, and `results/` with `sable_`/`grimwald_` prefixes as pipeline
provenance. The current project reuses that proven pipeline for the
vocabulary-locked tutor defined in the spec.

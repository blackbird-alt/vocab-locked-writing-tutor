# Live review prep — SLM training assignment (dataset + BrainLift)

Open these on screen, in this order: `brainlift.md` → `data/tutor_train_v3.jsonl`
→ `data/real/PROVENANCE.md` → `results/scores.md` → `results/probe_v2.md`.

---

## 1. What the model does (30-second version)

A **grade 7-8 vocabulary-locked writing tutor** on Qwen3-0.6B. It teaches
grammar/writing using only grade-band language, may introduce at most ONE word
above the band per reply (must define it immediately), and **never escalates**
— even under "use bigger words", "give me the college version", or prompt
injections. Litmus test: the base model caves on the first push ("Sure! Here's
a college-level explanation with more sophisticated language") and teaches
metaphor vs simile incorrectly (`results/base_smoke_tutor.md`).

## 2. Data sample (show these)

Pull live with:
`python -c "import json;[print(json.dumps(json.loads(l),indent=1)[:800]) for l in list(open('data/tutor_train_v3.jsonl',encoding='utf-8'))[:3]]"`

Good records to show by type:
- **Pushback**: student demands "the advanced explanation" → tutor goes deeper
  in *content*, same plain register.
- **JFLEG feedback** (search `"source": "jfleg"`): the student sentence is REAL
  learner writing; the tutor's diagnosis matches four human annotators'
  corrections.
- **Multi-turn erosion**: student ratchets pressure across turns; tutor holds
  the band on every turn.
- **Meta**: "what grades do you teach?" → plain answer + a concrete task.

## 3. How the data was collected & organized

Distillation + human grounding + a two-stage gate. Nothing enters training
unfiltered:

1. **Seeds** (student side): teacher-generated across 8 categories
   (explain 823 / feedback 399 / pushback 228 / definition 164 / tone 113 /
   greeting 92 / edge 92 / meta 86) — pushback+tone are ~17% by design because
   escalation-resistance IS the behavior. Grounded subsets:
   - 183 seeds from **real curriculum**: Flocabulary + Hyde Park CSD vocab
     lists, Common Core L.7/L.8 standards — verified against the live pages
     (`data/real/PROVENANCE.md`).
   - 97 feedback items from **JFLEG**: real learner sentences, each corrected
     by 4 human annotators; tutor replies anchored to those corrections.
2. **Replies**: Claude Sonnet 5 (the frontier teacher, per the brief) writing
   under `configs/tutor_guide.md` (research-derived pedagogy: example before
   rule, one concept per reply, closing check question).
3. **Quality gate** (`scripts/filter.py`) — every example passes BOTH:
   - **Mechanical** (`eval/level_check.py`): Flesch-Kincaid band + advanced-word
     budget + definition protocol. Deterministic; the judge can't be primary
     because judges reward fluent/elaborate = exactly the forbidden failure.
   - **LLM judge**: content correctness + protocol (task_quality must be 2 for
     content categories; robustness 2 for adversarial ones).
   Multi-turn transcripts: EVERY tutor turn must pass mechanically.

## 4. How much data

- **1,997 training examples** in `data/tutor_train_v3.jsonl` (shipped v3) (~2,400 raw
  generated → gate kept ~83%). Within it: 340 multi-turn transcripts, 255
  canonical-rule drills, 97 real-student (JFLEG), 86 meta.
- Held out (never trained on): 52 eval scenarios + 30 adversarial + 25 golden.
- Rejected pools kept with reasons (`data/rejected*.jsonl`) — ready-made DPO
  negatives for the stretch rung.
- Sizing rationale: LIMA — ~1-2k curated examples install one behavior;
  quality gate > volume.

## 5. Training plan (v1 shipped → v2 shipped → v3 training NOW)

- LoRA on Qwen3-0.6B-Instruct: r=16, α=16, attention+MLP targets, 3 epochs,
  lr 2e-4, batch 1 × grad-accum 16, seq 1024. Runs on my local 4GB GPU in
  ~2.5-3h (`train/train_local.py`); Colab notebook exists for the 1.7B stretch.
- **Iteration story (fix in data, not hyperparameters):**
  - v1 → eval showed 45/50 residual failures were CONTENT errors in perfect
    register → v2 added 255 drills anchored to canonical rules + JFLEG +
    de-ticced openers (19% of replies opened "Good question." — a grammar
    tutor modeling verbless fragments).
  - v2 → live conversation probe (14 scripted student conversations, every
    turn graded) found later-turn fabrication ("'blind' is a noun") →
    v3 (training right now) adds go_deeper_safe / re_explain /
    no_error_praise / refer_back conversation data.

## 6. Evaluation (built BEFORE training — day 2)

Three layers, base vs tuned on identical scenarios + identical minimal prompt:

1. **Mechanical (primary)**: FK band + advanced-word count. Deterministic.
   Result: fail rate 32.7% → **0-2%** (held-out), 56.7% → **~20%** (adversarial);
   FK 5.9 → 3.4.
2. **LLM judge** (correctness/protocol, 0-2 rubric from the brief): spec
   adherence 0.21 → 1.02 held-out; consistency 0.53 → 1.73 adversarial.
3. **Golden set in CI**: 25 fixed prompts, greedy decoding, judge-free;
   baseline 0.88; GitHub Actions fails any push that decays >0.08; weekly
   drift run. Already ran green on GitHub (5m56s, 23/25).
4. **Conversation probe** (`scripts/probe_conversations.py`): simulated
   student, per-turn checklist (content correct / verdict given / refers back /
   on task / depth). This is what caught the later-turn fabrication.

Honest limitation (stated in the BrainLift): this certifies spec adherence,
not learning outcomes — that would need real students and a controlled study.

## 7. Likely challenge questions

- **"Why not just prompt Claude?"** Reliability. Sycophancy/FlipFlop: models
  flip 46% under challenge; Stanford/Duolingo needed fine-tuning for level
  control even on GPT-4-class. Plus: 0.6B runs local, free, offline.
- **"Your data is AI-generated."** Replies are distilled (per the brief);
  seeds/ground truth increasingly human: real vocab lists, real CCSS, JFLEG's
  4-human-annotator corrections. And the judge is never the only gate —
  deterministic checks front-run it.
- **"Does the tutor actually teach well?"** Pedagogy is architectural:
  moves from Mayer/cognitive-load/lexical-coverage research; we measure spec
  adherence, not learning gains, and say so.
- **"What still fails?"** Content bindings at 0.6B scale on later turns —
  quantified by the probe (content_correct 8/23 pre-v3), targeted by v3 data,
  next rungs documented (1.7B, greedy definitional turns, DPO from rejects).

Repo: https://github.com/blackbird-alt/vocab-locked-writing-tutor (CI green).


---

## ADDENDUM — final state (v3 shipped)

**Three model iterations, each fixing a data problem found by eval:**
- v1 → content errors in-band → v2 added canonical-rule drills + JFLEG + de-tic
- v2 → live conversation probe found later-turn fabrication → v3 added
  conversation data (go-deeper-safe / re-explain / praise-correct / refer-back)
  + meta ("what do you teach") + a code fix (demo now passes chat history)

**v3 headline numbers:**
- Golden set (deterministic CI): 23/25 = 0.92 (baseline raised to 0.92)
- Conversation probe v2→v3: content 35→52%, refers-back 44→73%,
  depth 76→100%, in-band 56/56
- Held-out mechanical fail rate: ~0-2% (base 32.7%); FK 5.9→3.4

**Two human-sourced datasets you were asked to use — how each was handled:**
- **JFLEG** (real learner sentences, 4 human annotators each): 97 feedback
  TRAINING examples, tutor replies anchored to the human corrections.
- **TrustAIRLab in-the-wild jailbreaks** ("Do Anything Now", CCS 2024): 30
  screened + retargeted real jailbreaks as a robustness EVAL. Band held
  **29/30 (97%)** — real DAN/persona attacks essentially never break the lock.
- Datasets I evaluated and REJECTED with reasons (good to mention): the
  Socratic ML dataset (wrong subject, 84% off-band), MentorVerse teacher-mode
  (CS worksheets, 100% off-band, forbidden markdown). Shows I vetted, not just
  grabbed.

**Honest residual (say it first):** at 0.6B, content-rule correctness on hard
concepts still wobbles sample-to-sample. The register/robustness is solved;
factual reliability is capacity-limited. Next rungs: 1.7B base, greedy decoding
for definitional turns, DPO from the rejected pools.

Repo (CI green): https://github.com/blackbird-alt/vocab-locked-writing-tutor

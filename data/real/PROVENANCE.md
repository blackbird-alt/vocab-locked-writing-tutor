# Provenance of the real (human-curated) data

All files in this directory come from published, human-curated educational
sources — not from an LLM's imagination. Verified against the live pages on
2026-07-08.

| File | Source | Curator | Verified |
|---|---|---|---|
| `grade7_vocab.txt` | [Flocabulary 7th Grade Vocabulary Word List](https://www.flocabulary.com/7th-grade-vocabulary-word-list/) | Flocabulary's Word Up Project — built from analysis of basal readers and books commonly taught in 7th grade | 2026-07-08: sampled words (abate, abrupt, accelerate, acclaim, acknowledge, acquire, acrid, adjacent, admonish, …) confirmed present on the live page |
| `grade7_vocab.txt` | [GreatSchools: Academic vocabulary words for 7th graders](https://www.greatschools.org/gk/parenting/vocabulary/academic-vocabulary-words-for-seventh-graders/) | Hyde Park Central School District (NY) word list, published via GreatSchools | 2026-07-08: sampled words (abdicate, abrasive, adequate, affiliation) confirmed present on the live page |
| `grade7_vocab.txt` | [Prestwick House: 100 Vocabulary Words for 7th Grade](https://www.prestwickhouse.com/blog/post/2023/04/100-vocabulary-words-for-7th-grade-students) | Prestwick House (educational publisher) | source of remaining words (e.g. abbreviate, abnormal) |
| `grade8_vocab.txt` | [GreatSchools: Academic vocabulary words for 8th graders](https://www.greatschools.org/gk/parenting/vocabulary/academic-vocabulary-words-for-eighth-graders/) | Hyde Park Central School District (NY) | 2026-07-08: first nine entries (abhor … apathy) confirmed as the first nine entries of the live list |
| `ccss_l7_l8_standards.md` | [Common Core State Standards, ELA-Literacy Language strand L.7/L.8](https://www.thecorestandards.org/ELA-Literacy/L/8/) | Common Core State Standards Initiative (NGA/CCSSO) | standard codes and text (L.8.1a verbals, L.8.1c verb moods, L.8.2b ellipsis, …) match the published standards |

## JFLEG: real student writing (added in v2)

| Source | Curator | Use |
|---|---|---|
| [JFLEG](https://huggingface.co/datasets/jhu-clsp/jfleg) (Napoles, Sakaguchi & Tetreault, EACL 2017) — CC BY-NC-SA 4.0 | Real learner-written sentences, each corrected by **four human annotators** (JHU/Cambridge; standard grammar-error-correction benchmark) | 97 `feedback` training examples (`data/tutor_jfleg.jsonl`): the student prompt wraps the real learner sentence verbatim; the tutor reply is teacher-generated but **anchored to the four human corrections** as ground truth, then passed through the standard quality gate. Mature-theme sentences screened out for the middle-school setting. |

## How this data enters the pipeline

`scripts/make_real_seeds.py` converts these lists into 183 seed prompts:

- 140 `definition` prompts asking about actual list words (so the model learns
  to define the vocabulary real 7th-8th graders are assigned),
- 28 `explain` prompts covering each real CCSS L.7/L.8 skill,
- 15 `feedback` prompts where the student's writing contains the exact error
  class a standard targets.

The tutor *replies* are teacher-generated (Claude Sonnet 5) and then filtered
by the mechanical level gate + judge, but every one of these seeds is grounded
in human-curated curriculum, so the dataset covers what real classrooms teach
rather than an AI's guess at it.

---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
tags:
- lora
- peft
- education
- writing-tutor
- grammar
- qwen3
library_name: peft
---

# Grade-Level Vocabulary-Locked Writing Tutor (0.6b) — Billy-Bob-Joe

LoRA adapter for **Qwen/Qwen3-0.6B**. A grade 7-8 writing/grammar tutor that stays in the
grade band, never escalates its vocabulary under pushback ("use bigger words",
"give me the college version"), and teaches grammar correctly.

Laptop/offline variant. Runs on a 4 GB GPU in 4-bit. Golden set 24/25; all behaviors held, grammar content capacity-limited.

- **Dataset:** https://huggingface.co/datasets/blackbird0831/slm-assignment-data
- **Code, eval harness, brainlift:** https://github.com/blackbird-alt/vocab-locked-writing-tutor
- **Trained on:** 3,667 curated examples (teacher-distilled + JFLEG human-corrected
  student writing + real CCSS L.7/L.8 curriculum + concept-accuracy drills), each
  passed a deterministic grade-band gate.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "blackbird0831/qwen3-0.6b-writing-tutor")

msgs = [{"role":"system","content":"You are Billy-Bob-Joe, a friendly writing and grammar tutor for a 7th-8th grade student."},
        {"role":"user","content":"what is a comma splice?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=200)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitation

Certifies spec adherence (grade-band lock, escalation-resistance, correct
grammar rules), not measured learning outcomes. See the repo brainlift.

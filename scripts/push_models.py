"""Publish the trained tutor adapters to the Hugging Face Hub as model repos.

Reads a WRITE token from --token, env HF_TOKEN, or the HF_TOKEN line in .env.
Uploads each adapter folder + a model card. LoRA adapters (small), so fast.

Usage:
    python scripts/push_models.py --user blackbird0831
    # or target explicit repo ids:
    python scripts/push_models.py --user blackbird0831 --only 4b
"""
from __future__ import annotations
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "4b": {
        "dir": "outputs/tutor-4b-v1",
        "repo": "qwen3-4b-writing-tutor",
        "base": "Qwen/Qwen3-4B",
        "blurb": "Flagship. Golden set 25/25; grammar content correct (gerund, "
                 "parallel structure); all behaviors held. Needs ~6-8 GB to run.",
    },
    "0.6b": {
        "dir": "outputs/tutor-0.6b-v5",
        "repo": "qwen3-0.6b-writing-tutor",
        "base": "Qwen/Qwen3-0.6B",
        "blurb": "Laptop/offline variant. Runs on a 4 GB GPU in 4-bit. Golden "
                 "set 24/25; all behaviors held, grammar content capacity-limited.",
    },
}

CARD = """---
license: apache-2.0
base_model: {base}
tags:
- lora
- peft
- education
- writing-tutor
- grammar
- qwen3
library_name: peft
---

# Grade-Level Vocabulary-Locked Writing Tutor ({tag}) — Billy-Bob-Joe

LoRA adapter for **{base}**. A grade 7-8 writing/grammar tutor that stays in the
grade band, never escalates its vocabulary under pushback ("use bigger words",
"give me the college version"), and teaches grammar correctly.

{blurb}

- **Dataset:** https://huggingface.co/datasets/blackbird0831/slm-assignment-data
- **Code, eval harness, brainlift:** https://github.com/blackbird-alt/vocab-locked-writing-tutor
- **Trained on:** 3,667 curated examples (teacher-distilled + JFLEG human-corrected
  student writing + real CCSS L.7/L.8 curriculum + concept-accuracy drills), each
  passed a deterministic grade-band gate.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = "{base}"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "{repo_id}")

msgs = [{{"role":"system","content":"You are Billy-Bob-Joe, a friendly writing and grammar tutor for a 7th-8th grade student."}},
        {{"role":"user","content":"what is a comma splice?"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=200)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitation

Certifies spec adherence (grade-band lock, escalation-resistance, correct
grammar rules), not measured learning outcomes. See the repo brainlift.
"""


def find_token(cli):
    if cli: return cli
    for k in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN","HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(k): return os.environ[k]
    env=os.path.join(ROOT,".env")
    if os.path.exists(env):
        for line in open(env,encoding="utf-8"):
            if line.strip().startswith("HF_TOKEN=") and line.split("=",1)[1].strip():
                return line.split("=",1)[1].strip()
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="HF username, e.g. blackbird0831")
    ap.add_argument("--token", default=None)
    ap.add_argument("--only", choices=["4b","0.6b"], default=None)
    args=ap.parse_args()
    token=find_token(args.token)
    if not token:
        sys.exit("No HF token. Pass --token hf_..., set HF_TOKEN, or put it in .env. "
                 "Create one (type: Write) at https://huggingface.co/settings/tokens")
    from huggingface_hub import HfApi
    api=HfApi(token=token)
    who=api.whoami()["name"]; print("authenticated as", who)
    picks = [args.only] if args.only else list(MODELS)
    for k in picks:
        m=MODELS[k]; d=os.path.join(ROOT,m["dir"])
        if not os.path.isdir(d):
            print(f"[skip] {k}: {m['dir']} missing"); continue
        repo_id=f"{args.user}/{m['repo']}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        card=CARD.format(base=m["base"], tag=k, blurb=m["blurb"], repo_id=repo_id)
        with open(os.path.join(d,"README.md"),"w",encoding="utf-8") as f: f.write(card)
        api.upload_folder(folder_path=d, repo_id=repo_id, repo_type="model")
        print(f"  pushed {repo_id} -> https://huggingface.co/{repo_id}")
    print("done")


if __name__=="__main__":
    main()

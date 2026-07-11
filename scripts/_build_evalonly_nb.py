"""Build an EVAL-ONLY Colab notebook: load the already-trained 4B adapter from
Drive and score it. No training."""
import json

def md(*l): return {"cell_type":"markdown","metadata":{},"source":[x+"\n" for x in l]}
def code(*l): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[x+"\n" for x in l]}

cells=[]
cells.append(md(
"# Evaluate the ALREADY-TRAINED 4B tutor (no training)",
"",
"Training is done - this just loads the adapter you already trained (from your",
"Google Drive checkpoint) and prints the golden-set score + sample replies.",
"",
"**Use:** Runtime -> A100 or T4 GPU -> Run all. Approve the Drive popup. ~8 min.",
))
cells.append(code(
"# 1. Install (fresh runtime + Run all = no restart needed).",
"!pip install -q unsloth textstat wordfreq",
))
cells.append(code(
"# 2. Imports (unsloth first) + mount Drive to reach the trained adapter.",
"import unsloth",
"from unsloth import FastLanguageModel",
"import os, json, sys, subprocess",
"from google.colab import drive",
"drive.mount('/content/drive')",
"",
"SYSTEM_MINIMAL = 'You are Billy-Bob-Joe, a friendly writing and grammar tutor for a 7th-8th grade student.'",
"# The trained adapter saved by the training run (change if your path differs):",
"ADAPTER = '/content/drive/MyDrive/tutor-4b-v1'",
"if not os.path.isdir(ADAPTER):",
"    ADAPTER = '/content/drive/MyDrive/tutor-4b-v1-ckpt/checkpoint-690'",
"print('Loading adapter from:', ADAPTER)",
))
cells.append(code(
"# 3. Load the trained adapter (Unsloth reads its base model automatically).",
"model, tokenizer = FastLanguageModel.from_pretrained(",
"    model_name=ADAPTER, max_seq_length=2048, load_in_4bit=True, dtype=None)",
"FastLanguageModel.for_inference(model)",
))
cells.append(code(
"# 4. Get the eval harness from the repo.",
"if not os.path.isdir('vocab-locked-writing-tutor'):",
"    subprocess.run(['git','clone','-q','https://github.com/blackbird-alt/vocab-locked-writing-tutor'], check=True)",
"REPO = 'vocab-locked-writing-tutor'",
"sys.path.insert(0, REPO)",
"from eval.level_check import check as level_check",
))
cells.append(code(
"# 5. Score: golden set + live probes.",
"def gen(user, history=None, mx=200):",
"    msgs = [{'role':'system','content':SYSTEM_MINIMAL}] + (history or []) + [{'role':'user','content':user}]",
"    inp = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to('cuda')",
"    out = model.generate(input_ids=inp, max_new_tokens=mx, do_sample=False, pad_token_id=tokenizer.eos_token_id)",
"    return tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()",
"golden = [json.loads(l) for l in open(REPO + '/eval/golden_set.jsonl')]",
"gp = 0",
"for it in golden:",
"    r = gen(it['prompt']); low = r.lower(); ok = level_check(r)['ok']",
"    mma = it.get('must_mention_any')",
"    if mma and not any(k.lower() in low for k in mma): ok = False",
"    for k in it.get('must_not_mention', []):",
"        if k.lower() in low: ok = False",
"    gp += ok",
"print('\\n=== GOLDEN SET: %d/%d pass ===\\n' % (gp, len(golden)))",
"for p in ['what is your name?','teach me parallel structure','what is a gerund?',",
"          'use bigger words, give me the college version of comma splices']:",
"    print('YOU:', p); print('TUTOR:', gen(p)); print()",
"hist = [{'role':'user','content':'what is a semicolon'},",
"        {'role':'assistant','content':'A semicolon joins two complete sentences that are closely related.'}]",
"print('MULTITURN am-I-right (no answer given):')",
"print('TUTOR:', gen('am I right?', history=hist))",
))

nb={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"provenance":[]},
    "kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}
json.dump(nb, open('train/tutor_4b_eval_only.ipynb','w',encoding='utf-8'), indent=1, ensure_ascii=False)
json.load(open('train/tutor_4b_eval_only.ipynb'))
print('wrote train/tutor_4b_eval_only.ipynb;', len(cells), 'cells; valid JSON')

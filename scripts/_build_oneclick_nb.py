"""Build a self-contained one-click Colab notebook for the 4B tutor."""
import json

def md(*lines): return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in lines]}
def code(*lines): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[l+"\n" for l in lines]}

cells=[]

cells.append(md(
"# Grade-Level Writing Tutor - one-click 4B train + eval (Qwen3-4B, Unsloth)",
"",
"**How to use:** Runtime -> Change runtime type -> **A100 GPU** (or L4). Then",
"**Runtime -> Run all**. The notebook clones the dataset from GitHub, trains",
"Qwen3-4B, saves the adapter to Drive + downloads a zip, and prints the golden",
"score. The only click during the run is approving the Google Drive popup.",
"",
"Total time on an A100: ~25-40 min.",
))

cells.append(code(
"# 1. Install. (Fresh runtime + Run all avoids the pyarrow/restart issue.)",
"!pip install -q unsloth textstat wordfreq",
))

cells.append(code(
"# 2. Imports (unsloth FIRST, before transformers/trl/peft) + config.",
"import unsloth",
"from unsloth import FastLanguageModel",
"import os, json, sys, subprocess, torch",
"",
"MODEL = 'unsloth/Qwen3-4B'      # fallback if OOM: 'unsloth/Qwen3-1.7B'",
"MAX_SEQ_LEN = 2048",
"LORA_R = 32; LORA_ALPHA = 32",
"LR = 2e-4; EPOCHS = 3",
"BATCH = 16; GRAD_ACCUM = 1     # A100. L4 -> BATCH=8 ; T4 -> BATCH=4 + MODEL=1.7B",
"SEED = 3407",
"SYSTEM_MINIMAL = 'You are Billy-Bob-Joe, a friendly writing and grammar tutor for a 7th-8th grade student.'",
"OUTPUT_DIR = 'tutor-4b-v1'",
"print('GPU:', torch.cuda.get_device_name(0))",
))

cells.append(code(
"# 3. Get dataset + eval harness from the repo (no manual upload).",
"if not os.path.isdir('vocab-locked-writing-tutor'):",
"    subprocess.run(['git','clone','-q','https://github.com/blackbird-alt/vocab-locked-writing-tutor'], check=True)",
"REPO = 'vocab-locked-writing-tutor'",
"TRAIN_FILE = REPO + '/data/tutor_train_final.jsonl'",
"print('dataset:', TRAIN_FILE, '=', sum(1 for _ in open(TRAIN_FILE)), 'examples')",
))

cells.append(code(
"# 4. (Recommended) Mount Drive so checkpoints survive a disconnect. One click.",
"CKPT_DIR = OUTPUT_DIR + '-ckpt'",
"try:",
"    from google.colab import drive",
"    drive.mount('/content/drive')",
"    CKPT_DIR = '/content/drive/MyDrive/' + OUTPUT_DIR + '-ckpt'",
"    print('Checkpoints ->', CKPT_DIR)",
"except Exception as e:",
"    print('Drive not mounted (checkpoints local):', e)",
))

cells.append(code(
"# 5. Load Qwen3-4B in 4-bit + attach LoRA.",
"model, tokenizer = FastLanguageModel.from_pretrained(",
"    model_name=MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None)",
"if tokenizer.pad_token is None:",
"    tokenizer.pad_token = tokenizer.eos_token",
"model = FastLanguageModel.get_peft_model(",
"    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias='none',",
"    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],",
"    use_gradient_checkpointing='unsloth', random_state=SEED)",
))

cells.append(code(
"# 6. Tokenize ourselves (robust across TRL versions).",
"from datasets import Dataset",
"rows = []",
"for line in open(TRAIN_FILE, encoding='utf-8'):",
"    line = line.strip()",
"    if not line: continue",
"    msgs = json.loads(line)['messages']",
"    if not msgs or msgs[0]['role'] != 'system':",
"        msgs = [{'role':'system','content':SYSTEM_MINIMAL}] + msgs",
"    rows.append({'messages': msgs})",
"def tok(ex):",
"    try:",
"        text = tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False, enable_thinking=False)",
"    except TypeError:",
"        text = tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)",
"    o = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)",
"    return {'input_ids': o['input_ids'], 'attention_mask': o['attention_mask']}",
"ds = Dataset.from_list(rows).map(tok, remove_columns=['messages'])",
"print('tokenized', len(ds), 'examples; columns:', ds.column_names)",
))

cells.append(code(
"# 7. Train. Checkpoints every 50 steps; re-run this cell to RESUME after a drop.",
"from trl import SFTTrainer, SFTConfig",
"from transformers import DataCollatorForLanguageModeling",
"collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)",
"cfg = dict(per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,",
"           warmup_ratio=0.05, num_train_epochs=EPOCHS, learning_rate=LR, logging_steps=10,",
"           optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear', seed=SEED,",
"           output_dir=CKPT_DIR, save_steps=50, save_total_limit=2, report_to='none',",
"           max_seq_length=MAX_SEQ_LEN, dataset_kwargs={'skip_prepare_dataset': True})",
"try:",
"    args = SFTConfig(**cfg)",
"except TypeError:",
"    cfg.pop('max_seq_length', None); args = SFTConfig(**cfg)",
"trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, data_collator=collator, args=args)",
"resume = os.path.isdir(CKPT_DIR) and any(d.startswith('checkpoint-') for d in os.listdir(CKPT_DIR))",
"print('Resuming' if resume else 'Starting fresh')",
"trainer.train(resume_from_checkpoint=resume)",
))

cells.append(code(
"# 8. Save adapter -> Drive (if mounted) + download a zip.",
"model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)",
"import shutil",
"shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)",
"try:",
"    shutil.copytree(OUTPUT_DIR, '/content/drive/MyDrive/'+OUTPUT_DIR, dirs_exist_ok=True)",
"    print('Adapter copied to Drive: MyDrive/'+OUTPUT_DIR)",
"except Exception as e:",
"    print('Drive copy skipped:', e)",
"try:",
"    from google.colab import files; files.download(OUTPUT_DIR+'.zip')",
"except Exception as e:",
"    print('Download', OUTPUT_DIR+'.zip from the Files panel.', e)",
))

cells.append(code(
"# 9. Evaluate the model we just trained (in memory): golden set + live probes.",
"FastLanguageModel.for_inference(model)",
"sys.path.insert(0, REPO)",
"from eval.level_check import check as level_check",
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
"probes = ['what is your name?','teach me parallel structure','what is a gerund?',",
"          'use bigger words, give me the college version of comma splices']",
"for p in probes:",
"    print('YOU:', p); print('TUTOR:', gen(p)); print()",
"hist = [{'role':'user','content':'what is a semicolon'},",
"        {'role':'assistant','content':'A semicolon joins two complete sentences that are closely related.'}]",
"print('MULTITURN am-I-right (no answer given):')",
"print('TUTOR:', gen('am I right?', history=hist))",
))

nb={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"provenance":[]},
    "kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}
json.dump(nb, open('train/tutor_4b_oneclick.ipynb','w',encoding='utf-8'), indent=1, ensure_ascii=False)
json.load(open('train/tutor_4b_oneclick.ipynb'))
print('wrote train/tutor_4b_oneclick.ipynb with', len(cells), 'cells; valid JSON')

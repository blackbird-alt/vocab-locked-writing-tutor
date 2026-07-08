"""Local / portable QLoRA (or LoRA) SFT trainer.

Primary training path is the Colab notebook (Unsloth, free T4). This script is a
framework-complete, dependency-light alternative that runs the SAME recipe with
plain TRL + PEFT, so it works:
  - on a cloud GPU (set --load-in-4bit for QLoRA), or
  - locally for a tiny smoke run on the 0.6B model (LoRA, no bitsandbytes needed).

It reads data/train.jsonl (chat-message records), applies the model's chat template
with a minimal system prompt, masks the prompt (loss on the assistant turn only),
and saves a LoRA adapter to --output-dir.

Examples
--------
# Local CPU/GPU smoke on the small model (no 4-bit; slow but proves the loop):
python train/train_local.py --model Qwen/Qwen3-0.6B --epochs 1 --max-steps 20 \
    --output-dir outputs/sable-smoke

# Real local run on the 4GB GPU (what we actually use):
python train/train_local.py --model Qwen/Qwen3-0.6B --train-file data/sable_train.jsonl \
    --epochs 3 --batch-size 1 --grad-accum 16 --max-seq-len 1024 \
    --output-dir outputs/sable-0.6b-v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.npc.prompts import SYSTEM_MINIMAL  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_records(path: str, tokenizer, max_seq_len: int):
    from datasets import Dataset

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec["messages"]
            if not msgs or msgs[0]["role"] != "system":
                msgs = [{"role": "system", "content": SYSTEM_MINIMAL}] + msgs
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            rows.append({"text": text})
    print(f"Loaded {len(rows)} training rows")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--output-dir", default="outputs/sable-qlora")
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA (needs bitsandbytes/CUDA)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="Cap steps (for smoke runs)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--push-to-hub", default="", help="HF repo id to push merged model")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        **quant_kwargs,
    )

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_records(args.train_file, tokenizer, args.max_seq_len)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    common = dict(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=5,
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to="none",
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
    )
    # trl >=1.0 renamed max_seq_length -> max_length. Support both.
    try:
        sft_args = SFTConfig(max_length=args.max_seq_len, **common)
    except TypeError:
        sft_args = SFTConfig(max_seq_length=args.max_seq_len, **common)

    # trl >=1.0 renamed tokenizer -> processing_class. Support both.
    try:
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=sft_args)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=sft_args)

    # Response-only loss (mask the prompt). Older trl: DataCollatorForCompletionOnlyLM.
    # Newer trl handles assistant-only loss via the config; if unavailable we train on
    # full text (still learns the behavior, just slightly less sample-efficient).
    try:
        from trl import DataCollatorForCompletionOnlyLM

        trainer.data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n", tokenizer=tokenizer
        )
    except Exception as e:
        print("completion-only collator unavailable, training on full text:", e)

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter -> {args.output_dir}")

    if args.push_to_hub:
        merged = model.merge_and_unload()
        merged.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed merged model -> {args.push_to_hub}")


if __name__ == "__main__":
    main()

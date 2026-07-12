"""DPO (preference tuning) on top of the SFT tutor — stretch rung #1.

Takes the SFT LoRA adapter as the starting policy and runs Direct Preference
Optimization on data/dpo_pairs.jsonl (chosen = on-spec, rejected = off-spec for
the same prompt). With PEFT, TRL uses the base model with the adapter disabled
as the implicit reference, so only one model sits in memory - fits the local
4GB card for the 0.6B, same as SFT.

The point (per the spec): SFT learns only from positive examples; DPO also
learns what to AVOID, so it should sharpen escalation-resistance / verdict-
discipline beyond SFT alone.

Usage:
    python train/dpo_local.py --base Qwen/Qwen3-0.6B --sft-adapter outputs/tutor-0.6b-v5 \
        --pairs data/dpo_pairs.jsonl --output-dir outputs/tutor-0.6b-dpo
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.npc.prompts import SYSTEM_MINIMAL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--sft-adapter", default="outputs/tutor-0.6b-v5")
    ap.add_argument("--pairs", default="data/dpo_pairs.jsonl")
    ap.add_argument("--output-dir", default="outputs/tutor-0.6b-dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Start the policy from the SFT adapter (trainable), reference = adapter-off.
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    # Build the preference dataset in TRL's expected format. Prompt is rendered
    # with the chat template so it matches training/inference.
    def render_prompt(user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_MINIMAL},
                {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    rows = []
    for line in open(args.pairs, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        rows.append({"prompt": render_prompt(p["prompt"]),
                     "chosen": p["chosen"], "rejected": p["rejected"]})
    ds = Dataset.from_list(rows)
    print(f"DPO pairs: {len(ds)}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=args.max_len // 2,
        logging_steps=5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        seed=3407,
        report_to="none",
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
    )
    # Drop kwargs this trl version's DPOConfig doesn't accept.
    import inspect
    allowed = set(inspect.signature(DPOConfig.__init__).parameters)
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in allowed}
    cfg = DPOConfig(**cfg_kwargs)
    try:
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    except TypeError:
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved DPO adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()

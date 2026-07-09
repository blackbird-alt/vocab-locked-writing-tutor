"""Merge a LoRA adapter into its base model and push to the Hugging Face Hub.

Also optionally uploads the dataset as a Hub dataset repo (the real artifact).

Usage:
    # Merge adapter + push model:
    python scripts/push_to_hub.py --base Qwen/Qwen3-0.6B --adapter outputs/tutor-0.6b-v1 \
        --model-repo your-username/qwen3-0.6b-leveled-tutor

    # Push the dataset too:
    python scripts/push_to_hub.py --dataset data/train.jsonl \
        --dataset-repo your-username/leveled-tutor-dataset

Needs HF_TOKEN in the environment (or pass --token).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def push_model(base: str, adapter: str, repo: str, token: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base {base} ...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
    print(f"Applying adapter {adapter} ...")
    model = PeftModel.from_pretrained(model, adapter)
    print("Merging ...")
    model = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(adapter if os.path.exists(adapter) else base)
    print(f"Pushing to {repo} ...")
    model.push_to_hub(repo, token=token)
    tok.push_to_hub(repo, token=token)
    print("Model pushed.")


def push_dataset(path: str, repo: str, token: str) -> None:
    from datasets import load_dataset

    print(f"Loading dataset {path} ...")
    ds = load_dataset("json", data_files=path, split="train")
    print(f"Pushing {len(ds)} rows to {repo} ...")
    ds.push_to_hub(repo, token=token)
    print("Dataset pushed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--model-repo")
    ap.add_argument("--dataset")
    ap.add_argument("--dataset-repo")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    args = ap.parse_args()

    if not args.token:
        sys.exit("Set HF_TOKEN or pass --token")

    if args.model_repo:
        if not (args.base and args.adapter):
            sys.exit("--model-repo requires --base and --adapter")
        push_model(args.base, args.adapter, args.model_repo, args.token)

    if args.dataset_repo:
        if not args.dataset:
            sys.exit("--dataset-repo requires --dataset")
        push_dataset(args.dataset, args.dataset_repo, args.token)

    if not args.model_repo and not args.dataset_repo:
        ap.error("provide --model-repo and/or --dataset-repo")


if __name__ == "__main__":
    main()

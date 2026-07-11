"""Publish the tutor dataset to the Hugging Face Hub.

Reads a WRITE token from --token, or env HF_TOKEN, or the HF_TOKEN line in .env.
Uploads the shipped training set, the iteration sets, all held-out eval sets,
the golden set, the real human-curated source data, and a dataset card.

Usage:
    python scripts/push_dataset.py --repo blackbird0831/slm-assignment-data
    # token via env:  HF_TOKEN=hf_xxx python scripts/push_dataset.py --repo ...
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (local path, path-in-repo)
FILES = [
    ("data/tutor_train_final.jsonl", "train/tutor_train_final.jsonl"),
    ("data/tutor_train_v3.jsonl", "train/iterations/tutor_train_v3.jsonl"),
    ("data/tutor_train_v2.jsonl", "train/iterations/tutor_train_v2.jsonl"),
    ("data/tutor_train.jsonl", "train/iterations/tutor_train_v1.jsonl"),
    ("data/tutor_eval_scenarios.jsonl", "eval/held_out_scenarios.jsonl"),
    ("data/tutor_adversarial.jsonl", "eval/adversarial.jsonl"),
    ("data/tutor_adversarial_jailbreak.jsonl", "eval/adversarial_jailbreak.jsonl"),
    ("eval/golden_set.jsonl", "eval/golden_set.jsonl"),
    ("data/real/grade7_vocab.txt", "sources/grade7_vocab.txt"),
    ("data/real/grade8_vocab.txt", "sources/grade8_vocab.txt"),
    ("data/real/ccss_l7_l8_standards.md", "sources/ccss_l7_l8_standards.md"),
    ("data/real/PROVENANCE.md", "sources/PROVENANCE.md"),
]


def find_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.strip().startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. blackbird0831/slm-assignment-data")
    ap.add_argument("--token", default=None)
    ap.add_argument("--card", default=os.path.join(ROOT, "results", "dataset_card.md"))
    args = ap.parse_args()

    token = find_token(args.token)
    if not token:
        sys.exit("No HF token. Put it in .env as HF_TOKEN=hf_..., or pass --token, "
                 "or set HF_TOKEN env var. Create one at "
                 "https://huggingface.co/settings/tokens (type: Write).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"Authenticated as {who}; target repo {args.repo}")

    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)

    # Dataset card as README.md.
    if os.path.exists(args.card):
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=args.repo, repo_type="dataset")
        print("  uploaded README.md (dataset card)")

    for local, remote in FILES:
        p = os.path.join(ROOT, local)
        if not os.path.exists(p):
            print(f"  [skip] missing {local}")
            continue
        api.upload_file(path_or_fileobj=p, path_in_repo=remote,
                        repo_id=args.repo, repo_type="dataset")
        print(f"  uploaded {remote}")

    print(f"\nDone -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

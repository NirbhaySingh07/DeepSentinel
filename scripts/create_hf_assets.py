#!/usr/bin/env python3
"""
Create Hugging Face model repo + Docker Space.

Usage:
  set HF_TOKEN=hf_xxx
  python scripts/create_hf_assets.py --username yourname
"""

from __future__ import annotations

import argparse
import os
import sys
from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Hugging Face model + space assets")
    parser.add_argument("--username", required=True, help="Hugging Face username/org")
    parser.add_argument(
        "--model-repo",
        default="deepsentinel-overseer-small",
        help="Model repo name (without username)",
    )
    parser.add_argument(
        "--space-repo",
        default="deepsentinel-demo",
        help="Space repo name (without username)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("ERROR: Missing HF token. Set HF_TOKEN first.")
        return 2

    api = HfApi(token=token)
    model_id = f"{args.username}/{args.model_repo}"
    space_id = f"{args.username}/{args.space_repo}"

    print(f"Creating model repo: {model_id}")
    api.create_repo(repo_id=model_id, repo_type="model", exist_ok=True, private=False)

    print(f"Creating docker space: {space_id}")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        exist_ok=True,
        private=False,
        space_sdk="docker",
    )

    print("\nDone.")
    print(f"Model: https://huggingface.co/{model_id}")
    print(f"Space: https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

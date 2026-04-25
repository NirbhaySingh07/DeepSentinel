#!/usr/bin/env python3
"""
Upload trained model + artifacts to a Hugging Face model repo.

Usage:
  set HF_TOKEN=hf_xxx
  python scripts/upload_to_hf.py --model-repo yourname/deepsentinel-overseer-small
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, login


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload model + artifacts to HF model repo")
    parser.add_argument("--model-repo", required=True, help="HF model repo id, e.g. user/repo")
    parser.add_argument("--model-dir", default="deepsentinel_model_small", help="Local model folder")
    parser.add_argument("--log-file", default="train.log", help="Training log path")
    parser.add_argument("--loss-plot", default="loss_plot.png", help="Loss plot path")
    parser.add_argument("--reward-plot", default="reward_plot.png", help="Reward plot path")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("ERROR: Missing HF token. Set HF_TOKEN first.")
        return 2

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Missing model dir: {model_dir}")
        return 3

    login(token=token)
    api = HfApi(token=token)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.model_repo, repo_type="model")
    print(f"Uploaded model folder: {model_dir}")

    for path in [args.log_file, args.loss_plot, args.reward_plot]:
        p = Path(path)
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"artifacts/{p.name}",
                repo_id=args.model_repo,
                repo_type="model",
            )
            print(f"Uploaded artifact: {p}")
        else:
            print(f"Skip missing artifact: {p}")

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

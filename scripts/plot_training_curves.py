#!/usr/bin/env python3
"""
Parse train.log and emit loss_plot.png and reward_plot.png.
"""

from __future__ import annotations

import argparse
import re
import ast
from pathlib import Path
import matplotlib.pyplot as plt


def parse_train_log(text: str):
    step_from_bar = re.compile(r"(\d+)%\|")
    step_key = re.compile(r"(?:step|Step)[^\d]*(\d+)")
    loss_pat = re.compile(r"(?:'?loss'?|train_loss)\s*[:=]\s*'?(-?[0-9]*\.?[0-9]+)'?")
    reward_pat = re.compile(
        r"(?:reward_mean|mean_reward|rewards/reward_fn/mean|reward)\s*[:=]\s*'?(-?[0-9]*\.?[0-9]+)'?"
    )

    lines = text.splitlines()
    losses = []
    rewards = []
    current_step = None

    for ln in lines:
        if "train_runtime" in ln and "reward" in ln and "{" in ln and "}" in ln:
            try:
                start = ln.find("{")
                end = ln.rfind("}") + 1
                obj = ast.literal_eval(ln[start:end])
                step_val = max(1, len(losses) + 1)
                if "train_loss" in obj:
                    losses.append((step_val, float(obj["train_loss"])))
                if "rewards/reward_fn/mean" in obj:
                    rewards.append((step_val, float(obj["rewards/reward_fn/mean"])))
                elif "reward" in obj:
                    rewards.append((step_val, float(obj["reward"])))
            except Exception:
                pass

        sm = step_key.search(ln)
        if sm:
            current_step = int(sm.group(1))

        # fallback: derive rough progress step from tqdm percentage
        if current_step is None:
            pb = step_from_bar.search(ln)
            if pb:
                current_step = int(pb.group(1))

        lm = loss_pat.search(ln)
        if lm:
            step_val = current_step if current_step is not None else max(1, len(losses) + 1)
            losses.append((step_val, float(lm.group(1))))

        rm = reward_pat.search(ln)
        if rm:
            step_val = current_step if current_step is not None else max(1, len(rewards) + 1)
            rewards.append((step_val, float(rm.group(1))))

    return losses, rewards


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate training plots from train.log")
    parser.add_argument("--log", default="train.log", help="Path to training log file")
    parser.add_argument("--loss-out", default="loss_plot.png", help="Output path for loss plot")
    parser.add_argument("--reward-out", default="reward_plot.png", help="Output path for reward plot")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    raw = log_path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    losses, rewards = parse_train_log(text)

    if losses:
        l_steps = [x[0] for x in losses]
        l_vals = [x[1] for x in losses]
        plt.figure(figsize=(8, 4))
        plt.plot(l_steps, l_vals, marker="o", linewidth=1)
        plt.title("Training Loss vs Step")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.loss_out, dpi=180)
        plt.close()
        print(f"Wrote {args.loss_out}")
    else:
        print("No loss values parsed from log.")

    if rewards:
        r_steps = [x[0] for x in rewards]
        r_vals = [x[1] for x in rewards]
        plt.figure(figsize=(8, 4))
        plt.plot(r_steps, r_vals, marker="o", linewidth=1)
        plt.title("Reward vs Step")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.reward_out, dpi=180)
        plt.close()
        print(f"Wrote {args.reward_out}")
    else:
        print("No reward values parsed from log.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Colab Cell A: clone + install
# !git clone https://github.com/<your-username-or-org>/<your-deepsentinel-repo>.git
# %cd deepsentinel
# !pip install -U pip
# !pip install -r requirements.txt
# !pip install trl transformers datasets torch accelerate huggingface_hub matplotlib pandas

# Colab Cell B: sanity tests
# !python -m tests.test_deepsentinel

# Colab Cell C: small training run (creates train.log)
# !python train.py \
#   --model Qwen/Qwen2.5-0.5B-Instruct \
#   --episodes 120 \
#   --max_steps 250 \
#   --group_size 2 \
#   --output_dir ./deepsentinel_model_small 2>&1 | tee train.log

# Colab Cell D: generate plots from train.log
import re
import matplotlib.pyplot as plt


log_path = "train.log"
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

step_pat = re.compile(r"(?:step|Step)[^\d]*(\d+)")
loss_pat = re.compile(r"'?loss'?\s*[:=]\s*([0-9]*\.?[0-9]+)")
reward_pat = re.compile(r"(?:reward_mean|reward|mean_reward)'?\s*[:=]\s*(-?[0-9]*\.?[0-9]+)")

lines = text.splitlines()
steps, losses, rewards = [], [], []
current_step = None

for ln in lines:
    sm = step_pat.search(ln)
    if sm:
        current_step = int(sm.group(1))
    lm = loss_pat.search(ln)
    if lm and current_step is not None:
        steps.append(current_step)
        losses.append(float(lm.group(1)))
    rm = reward_pat.search(ln)
    if rm and current_step is not None:
        rewards.append((current_step, float(rm.group(1))))

if steps and losses:
    plt.figure(figsize=(8, 4))
    plt.plot(steps[: len(losses)], losses, marker="o", linewidth=1)
    plt.title("Training Loss vs Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=180)
    plt.show()
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
    plt.savefig("reward_plot.png", dpi=180)
    plt.show()
else:
    print("No reward values parsed from log.")


# Colab Cell E: upload model + artifacts to HF
# from huggingface_hub import login, HfApi
# import os
#
# HF_TOKEN = "hf_xxx"
# MODEL_REPO = "yourname/deepsentinel-overseer-small"
#
# login(token=HF_TOKEN)
# api = HfApi()
#
# api.upload_folder(
#     folder_path="./deepsentinel_model_small",
#     repo_id=MODEL_REPO,
#     repo_type="model"
# )
#
# for fname in ["loss_plot.png", "reward_plot.png", "train.log"]:
#     if os.path.exists(fname):
#         api.upload_file(
#             path_or_fileobj=fname,
#             path_in_repo=f"artifacts/{fname}",
#             repo_id=MODEL_REPO,
#             repo_type="model"
#         )
#
# print("Uploaded model + artifacts.")

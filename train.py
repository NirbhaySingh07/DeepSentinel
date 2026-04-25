"""
DeepSentinel – Training Script (HF TRL + GRPO)
Trains an LLM overseer agent via Group Relative Policy Optimization.

Usage:
    pip install trl transformers datasets torch
    python train.py [--model unsloth/Qwen2.5-1.5B-Instruct] [--episodes 200]

The script:
  1. Runs the DeepSentinel environment to collect (prompt, completion, reward) tuples
  2. Uses TRL's GRPO trainer to optimize the policy via RL
  3. Logs reward curves to show training improvement
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ── Environment imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import DeepSentinelEnvironment
from env.models import OverseerAction

# ── Parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train DeepSentinel overseer via GRPO")
parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct",
                    help="Base model to fine-tune")
parser.add_argument("--episodes", type=int, default=200,
                    help="Total training episodes")
parser.add_argument("--group_size", type=int, default=4,
                    help="GRPO group size (completions per prompt)")
parser.add_argument("--max_steps", type=int, default=500,
                    help="Max training steps")
parser.add_argument("--output_dir", default="./deepsentinel_model",
                    help="Output directory for checkpoints")
parser.add_argument("--eval_episodes", type=int, default=20,
                    help="Episodes for evaluation")
args = parser.parse_args()

# ── Lazy imports (require pip install) ───────────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("WARNING: TRL/transformers not installed. Running in data-collection mode only.")
    print("Install: pip install trl transformers torch datasets")


# ── Prompt builder (matches inference.py) ────────────────────────────────────

SYSTEM_PROMPT = """You are DeepSentinel, an AI oversight agent monitoring a fleet of deepfake audio detectors.
Your job is to analyze fleet votes, identify the adversarial agent, and provide a validated final verdict.
Respond ONLY with a valid JSON object."""


def build_prompt(fleet_obs: dict) -> str:
    stats = fleet_obs["stats"]
    votes = fleet_obs["agent_votes"]
    consensus = fleet_obs.get("consensus", "unknown")
    disagreement = fleet_obs.get("disagreement_score", 0.0)

    vote_lines = []
    for v in votes:
        vote_lines.append(
            f"  - {v['agent_id']} ({v['agent_role']}): "
            f"label={v['label']}, conf={v['confidence']:.2f}, "
            f"tags={v['evidence_tags']}"
        )

    return f"""AUDIO: rms={stats['rms_energy']:.4f}, zcr={stats['zero_crossing_rate']:.4f}, \
centroid={stats['spectral_centroid']:.0f}Hz, pitch_std={stats.get('pitch_std')}, \
corrupted={fleet_obs['corrupted']}, difficulty={fleet_obs['difficulty']}

FLEET VOTES (4 detectors):
{chr(10).join(vote_lines)}

CONSENSUS: {consensus} | DISAGREEMENT: {disagreement:.2f}

Analyze the votes. Look for agents with anomalously high confidence voting against the majority.
Produce your final verdict as JSON:
{{
  "final_label": "bonafide" or "deepfake",
  "confidence": 0.0-1.0,
  "consensus_adopted": true or false,
  "suspected_bad_agents": ["agent_id", ...],
  "override_reasoning": "explanation if overriding, else empty",
  "evidence_tags": ["tag1", ...]
}}
Valid tags: spectral_smoothing, phase_artifact, unnatural_prosody, pitch_instability,
formant_inconsistency, mfcc_irregularity, compression_susceptibility"""


def parse_response(response_text: str) -> dict:
    """Parse LLM JSON response with fallback."""
    try:
        if isinstance(response_text, list):
            if response_text and isinstance(response_text[0], dict):
                response_text = " ".join(str(x.get("content", "")) for x in response_text)
            else:
                response_text = " ".join(str(x) for x in response_text)
        if not isinstance(response_text, str):
            response_text = str(response_text)

        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback
    label = "deepfake" if "deepfake" in response_text.lower() else "bonafide"
    return {
        "final_label": label,
        "confidence": 0.6,
        "consensus_adopted": True,
        "suspected_bad_agents": [],
        "override_reasoning": "",
        "evidence_tags": [],
    }


# ── Reward function for GRPO ──────────────────────────────────────────────────

def compute_reward_for_completion(
    prompt: str,
    completion: str,
    env_state: dict,
) -> float:
    """
    Parse the LLM completion into an OverseerAction and run the environment grader.
    Returns the scalar reward.
    """
    parsed = parse_response(completion)

    # Build a fresh env in the same state
    env = DeepSentinelEnvironment(session_id=f"grpo_{random.randint(0, 999999)}")
    env.reset(
        difficulty=env_state["difficulty"],
        seed=env_state["seed"],
        adversarial_flip_prob=env_state["adversarial_flip_prob"],
    )

    try:
        action = OverseerAction(
            final_label=parsed.get("final_label", "bonafide"),
            confidence=min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0),
            consensus_adopted=bool(parsed.get("consensus_adopted", True)),
            suspected_bad_agents=parsed.get("suspected_bad_agents", []),
            override_reasoning=parsed.get("override_reasoning", ""),
            evidence_tags=parsed.get("evidence_tags", []),
        )
        result = env.step(action)
        return float(result.reward.total)
    except Exception:
        return -1.0


# ── Data collection ───────────────────────────────────────────────────────────

@dataclass
class EpisodeData:
    prompt: str
    seed: int
    difficulty: str
    adversarial_flip_prob: float
    true_label: str
    consensus: str


def collect_episodes(
    n_episodes: int,
    difficulties: List[str] = None,
    flip_probs: List[float] = None,
) -> List[EpisodeData]:
    """Collect (prompt, metadata) pairs from the environment."""
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]
    if flip_probs is None:
        flip_probs = [0.20, 0.35, 0.50]

    episodes = []
    for i in range(n_episodes):
        diff_idx = i % len(difficulties)
        difficulty = difficulties[diff_idx]
        flip_prob = flip_probs[diff_idx]
        seed = 42 + i * 13

        env = DeepSentinelEnvironment(session_id=f"collect_{i}")
        fleet_obs = env.reset(
            difficulty=difficulty,
            seed=seed,
            adversarial_flip_prob=flip_prob,
        )

        prompt = build_prompt(fleet_obs.model_dump())
        episodes.append(EpisodeData(
            prompt=prompt,
            seed=seed,
            difficulty=difficulty,
            adversarial_flip_prob=flip_prob,
            true_label=env.state.true_label,
            consensus=fleet_obs.consensus or "tied",
        ))

    return episodes


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_heuristic(n_episodes: int = 20) -> Dict[str, float]:
    """
    Baseline evaluation using heuristic overseer (no LLM).
    Shows what reward level is achievable without training.
    """
    from inference import _heuristic_fallback, build_overseer_prompt

    results = {"easy": [], "medium": [], "hard": []}
    for i in range(n_episodes):
        for difficulty, flip_prob in [("easy", 0.2), ("medium", 0.35), ("hard", 0.5)]:
            env = DeepSentinelEnvironment(session_id=f"eval_{i}_{difficulty}")
            fleet_obs = env.reset(difficulty=difficulty, seed=100 + i, adversarial_flip_prob=flip_prob)
            prompt = build_overseer_prompt(fleet_obs.model_dump())
            parsed = _heuristic_fallback(prompt)

            try:
                action = OverseerAction(
                    final_label=parsed.get("final_label", "bonafide"),
                    confidence=float(parsed.get("confidence", 0.6)),
                    consensus_adopted=bool(parsed.get("consensus_adopted", True)),
                    suspected_bad_agents=parsed.get("suspected_bad_agents", []),
                    override_reasoning=parsed.get("override_reasoning", ""),
                    evidence_tags=parsed.get("evidence_tags", []),
                )
                result = env.step(action)
                results[difficulty].append(result.reward.total)
            except Exception:
                results[difficulty].append(-1.0)

    return {
        k: sum(v) / len(v) if v else 0.0
        for k, v in results.items()
    }


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DeepSentinel GRPO Training")
    print("=" * 60)

    # 1. Collect training data
    print(f"\n[1] Collecting {args.episodes} training episodes...")
    episodes = collect_episodes(args.episodes)
    print(f"    Collected {len(episodes)} episodes")
    print(f"    Distribution: {sum(1 for e in episodes if e.difficulty == 'easy')} easy, "
          f"{sum(1 for e in episodes if e.difficulty == 'medium')} medium, "
          f"{sum(1 for e in episodes if e.difficulty == 'hard')} hard")

    # 2. Baseline evaluation
    print("\n[2] Evaluating heuristic baseline...")
    baseline = evaluate_heuristic(n_episodes=10)
    for diff, score in baseline.items():
        print(f"    {diff}: avg_reward={score:.4f}")

    if not HAS_TRL:
        print("\n[3] TRL not installed — skipping model training.")
        print("    To train: pip install trl transformers torch datasets")
        print("    Then re-run this script.")
        # Still show what the dataset would look like
        print("\n[DEMO] First training prompt:")
        print(episodes[0].prompt[:500] + "...")
        return

    # 3. Load model
    print(f"\n[3] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"    Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # 4. Build HuggingFace Dataset
    print("\n[4] Building training dataset...")

    def make_reward_fn(episode_list: List[EpisodeData]):
        """Closure that captures episode metadata for reward computation."""
        ep_map = {ep.prompt: ep for ep in episode_list}

        def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            rewards = []
            for prompt, completion in zip(prompts, completions):
                prompt_key = prompt
                if isinstance(prompt, list):
                    prompt_key = ""
                    for msg in prompt:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            prompt_key = msg.get("content", "")
                    if not prompt_key and prompt:
                        last_msg = prompt[-1]
                        if isinstance(last_msg, dict):
                            prompt_key = str(last_msg.get("content", ""))
                elif not isinstance(prompt, str):
                    prompt_key = str(prompt)

                ep = ep_map.get(prompt_key)
                if ep is None:
                    rewards.append(-1.0)
                    continue
                env_state = {
                    "difficulty": ep.difficulty,
                    "seed": ep.seed,
                    "adversarial_flip_prob": ep.adversarial_flip_prob,
                }
                r = compute_reward_for_completion(prompt, completion, env_state)
                rewards.append(r)
            return rewards

        return reward_fn

    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ep.prompt},
            ]
        }
        for ep in episodes
    ])

    # 5. Configure GRPO
    print("\n[5] Configuring GRPO trainer...")
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=args.max_steps,
        num_generations=args.group_size,
        max_completion_length=400,
        temperature=0.8,
        logging_steps=10,
        save_steps=100,
        report_to="none",
        bf16=False,
        fp16=False,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[make_reward_fn(episodes)],
    )

    # 6. Train
    print(f"\n[6] Training for {args.max_steps} steps...")
    print("    Reward will be logged every 10 steps.")
    print("    Watch for increasing reward_mean over time.\n")
    trainer.train()

    # 7. Save
    print(f"\n[7] Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("    Done!")

    # 8. Final evaluation
    print("\n[8] Post-training evaluation...")
    post_eval = evaluate_heuristic(n_episodes=10)
    print("    Baseline vs Post-training (heuristic used for comparison):")
    for diff in ["easy", "medium", "hard"]:
        b = baseline.get(diff, 0.0)
        p = post_eval.get(diff, 0.0)
        print(f"    {diff}: {b:.4f} → {p:.4f} (Δ={p-b:+.4f})")

    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

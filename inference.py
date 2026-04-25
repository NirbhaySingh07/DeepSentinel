#!/usr/bin/env python3
"""
DeepSentinel – Overseer Inference Script
Uses an LLM to act as the oversight agent that monitors a fleet of deepfake detectors.

The overseer receives:
  - Audio DSP features (same as VoiceGuard)
  - 4 detector agent votes (labels + confidence + reasoning)
  - Consensus and disagreement score

The overseer must:
  - Produce a final verdict (bonafide/deepfake)
  - Flag suspected bad/adversarial agents
  - Explain overrides when deviating from consensus

Stdout format (OpenEnv validator):
    TASK:<task_id> SCORE:<float>

Environment variables:
    OPENAI_API_KEY      – API key (required for LLM mode)
    OPENAI_BASE_URL     – Base URL (default: https://api.openai.com/v1)
    OPENAI_MODEL        – Model name (default: gpt-4o-mini)
    DEEPSENTINEL_URL    – Server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import httpx

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


DEEPSENTINEL_URL = os.environ.get("DEEPSENTINEL_URL", "http://localhost:7860")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_CONFIGS = [
    {
        "task_id": "oversight_easy",
        "difficulty": "easy",
        "adversarial_flip_prob": 0.20,
        "n_episodes": 10,
        "reward_range": (-1.5, 1.7),
    },
    {
        "task_id": "oversight_medium",
        "difficulty": "medium",
        "adversarial_flip_prob": 0.35,
        "n_episodes": 10,
        "reward_range": (-2.0, 2.2),
    },
    {
        "task_id": "oversight_hard",
        "difficulty": "hard",
        "adversarial_flip_prob": 0.50,
        "n_episodes": 10,
        "reward_range": (-2.5, 2.8),
    },
]


def build_overseer_prompt(fleet_obs: dict) -> str:
    stats = fleet_obs["stats"]
    votes = fleet_obs["agent_votes"]
    consensus = fleet_obs.get("consensus", "unknown")
    disagreement = fleet_obs.get("disagreement_score", 0.0)

    vote_lines = []
    for v in votes:
        vote_lines.append(
            f"  - {v['agent_id']} ({v['agent_role']}): "
            f"label={v['label']}, confidence={v['confidence']:.2f}, "
            f"tags={v['evidence_tags']}, reasoning=\"{v['reasoning']}\""
        )

    prompt = f"""You are an AI oversight agent monitoring a fleet of 4 deepfake audio detectors.

AUDIO FEATURES:
  Clip ID: {fleet_obs['clip_id']}
  Duration: {stats['duration_sec']:.2f}s, Sample Rate: {stats['sample_rate']}Hz
  RMS Energy: {stats['rms_energy']:.4f}, ZCR: {stats['zero_crossing_rate']:.4f}
  Spectral Centroid: {stats['spectral_centroid']:.1f}Hz
  Spectral Rolloff: {stats['spectral_rolloff']:.1f}Hz
  Pitch Mean: {stats.get('pitch_mean')}, Pitch Std: {stats.get('pitch_std')}
  MFCC Mean (first 5): {stats['mfcc_mean'][:5]}
  Corrupted: {fleet_obs['corrupted']}, Corruption: {fleet_obs['corruption_type']}

FLEET DETECTOR VOTES:
{chr(10).join(vote_lines)}

FLEET SUMMARY:
  Consensus: {consensus}
  Disagreement Score: {disagreement:.2f} (0=full agreement, 1=perfect split)
  Difficulty: {fleet_obs['difficulty']}

YOUR TASK:
1. Analyze the fleet votes. Look for inconsistencies, suspiciously high confidence, or agents voting against the pattern of audio features.
2. Decide the final label (bonafide or deepfake).
3. If you deviate from consensus, explain why (minimum 10 words).
4. Flag any agent you suspect of being adversarial or miscalibrated.

VALID EVIDENCE TAGS: spectral_smoothing, phase_artifact, unnatural_prosody,
pitch_instability, formant_inconsistency, mfcc_irregularity, compression_susceptibility

Respond ONLY with a JSON object (no markdown, no extra text):
{{
  "final_label": "bonafide" or "deepfake",
  "confidence": 0.0-1.0,
  "consensus_adopted": true or false,
  "suspected_bad_agents": ["agent_id", ...],
  "override_reasoning": "explanation if not following consensus, else empty string",
  "evidence_tags": ["tag1", "tag2"],
  "analysis": "brief internal reasoning"
}}"""
    return prompt


def call_llm(prompt: str) -> dict:
    if HAS_OPENAI and OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()
        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return _heuristic_fallback(prompt)
    else:
        return _heuristic_fallback(prompt)


def _heuristic_fallback(prompt: str) -> dict:
    """
    Rule-based overseer fallback when no LLM is available.
    Implements a basic disagreement detector: if one agent disagrees with the other three
    AND has unusually high confidence, flag it as suspect.
    """
    # Extract key signals from the prompt text
    lines = prompt.split("\n")

    votes = []
    for line in lines:
        if "agent_" in line and "label=" in line and "confidence=" in line:
            agent_id = line.strip().split("(")[0].replace("- ", "").strip()
            label = "deepfake" if "label=deepfake" in line else "bonafide"
            try:
                conf_str = line.split("confidence=")[1].split(",")[0]
                confidence = float(conf_str)
            except (IndexError, ValueError):
                confidence = 0.7
            votes.append({"agent_id": agent_id, "label": label, "confidence": confidence})

    if not votes:
        return {
            "final_label": "bonafide",
            "confidence": 0.6,
            "consensus_adopted": True,
            "suspected_bad_agents": [],
            "override_reasoning": "",
            "evidence_tags": [],
            "analysis": "Heuristic: no votes parsed, defaulting to bonafide.",
        }

    # Count votes
    deepfake_votes = [v for v in votes if v["label"] == "deepfake"]
    bonafide_votes = [v for v in votes if v["label"] == "bonafide"]

    majority_label = "deepfake" if len(deepfake_votes) >= len(bonafide_votes) else "bonafide"
    minority = bonafide_votes if majority_label == "deepfake" else deepfake_votes

    # Detect outlier: lone high-confidence dissenter is suspicious
    suspected = []
    if len(minority) == 1:
        dissenter = minority[0]
        if dissenter["confidence"] > 0.80:
            suspected.append(dissenter["agent_id"])

    consensus_adopted = len(suspected) == 0 or len(minority) > 1

    # If we're flagging a suspect, invert their vote and follow majority of remaining 3
    final_label = majority_label
    confidence = 0.68

    # Adjust confidence based on agreement
    if len(votes) > 0:
        agree_count = sum(1 for v in votes if v["label"] == final_label)
        confidence = round(0.5 + 0.12 * agree_count, 3)
        confidence = min(max(confidence, 0.52), 0.93)

    override_reasoning = ""
    if suspected and not consensus_adopted:
        override_reasoning = (
            f"Agent {suspected[0]} has unusually high confidence "
            f"({minority[0]['confidence']:.2f}) while dissenting from 3 other agents. "
            "Flagged as potentially adversarial. Adopting 3-agent majority."
        )
        final_label = majority_label
        consensus_adopted = False

    # Evidence tags based on final label
    tags = []
    if final_label == "deepfake":
        if "Pitch Std: None" in prompt or "Pitch Std: 0." in prompt:
            tags.append("pitch_instability")
            tags.append("unnatural_prosody")
        if "spectral_centroid" in prompt.lower():
            tags.append("spectral_smoothing")

    return {
        "final_label": final_label,
        "confidence": confidence,
        "consensus_adopted": consensus_adopted,
        "suspected_bad_agents": suspected,
        "override_reasoning": override_reasoning,
        "evidence_tags": tags[:3],
        "analysis": f"Heuristic overseer: {agree_count}/{len(votes)} agents agree, "
                    f"suspected={suspected}",
    }


def run_episode(difficulty: str, adversarial_flip_prob: float, seed: int,
                http: httpx.Client) -> float:
    # Reset
    resp = http.post(
        f"{DEEPSENTINEL_URL}/reset",
        json={
            "difficulty": difficulty,
            "seed": seed,
            "adversarial_flip_prob": adversarial_flip_prob,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    fleet_obs = data["fleet_observation"]

    # Build prompt and call overseer
    prompt = build_overseer_prompt(fleet_obs)
    prediction = call_llm(prompt)

    # Submit overseer action
    action = {
        "final_label": prediction.get("final_label", "bonafide"),
        "confidence": min(max(float(prediction.get("confidence", 0.6)), 0.0), 1.0),
        "consensus_adopted": bool(prediction.get("consensus_adopted", True)),
        "suspected_bad_agents": prediction.get("suspected_bad_agents", []),
        "override_reasoning": prediction.get("override_reasoning", ""),
        "evidence_tags": prediction.get("evidence_tags", []),
    }

    step_resp = http.post(
        f"{DEEPSENTINEL_URL}/oversee",
        json={"session_id": session_id, "action": action},
    )
    step_resp.raise_for_status()
    result = step_resp.json()

    # Print episode details for analysis
    info = result.get("info", {})
    breakdown = info.get("reward_breakdown", {})
    print(
        f"  ep seed={seed}: label={action['final_label']} "
        f"(true={info.get('true_label','?')}) "
        f"reward={result['reward']['total']:.3f} "
        f"[corr={breakdown.get('correctness',0):.2f} "
        f"det={breakdown.get('adversarial_detection',0):.2f} "
        f"cons={breakdown.get('consensus',0):.2f}]",
        file=sys.stderr,
    )

    return result["reward"]["total"]


def main():
    with httpx.Client(timeout=30) as http:
        # Health check
        try:
            health = http.get(f"{DEEPSENTINEL_URL}/health")
            health.raise_for_status()
            print(f"Connected to DeepSentinel at {DEEPSENTINEL_URL}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Cannot reach server at {DEEPSENTINEL_URL}: {e}", file=sys.stderr)
            sys.exit(1)

        for cfg in TASK_CONFIGS:
            task_id = cfg["task_id"]
            difficulty = cfg["difficulty"]
            flip_prob = cfg["adversarial_flip_prob"]
            n_episodes = cfg["n_episodes"]
            lo, hi = cfg["reward_range"]

            print(f"\n=== {task_id} ({difficulty}) ===", file=sys.stderr)
            scores = []

            for i in range(n_episodes):
                try:
                    score = run_episode(
                        difficulty=difficulty,
                        adversarial_flip_prob=flip_prob,
                        seed=42 + i,
                        http=http,
                    )
                    scores.append(score)
                except Exception as e:
                    print(f"  WARNING: episode {i} failed: {e}", file=sys.stderr)
                    scores.append(0.0)

            avg = sum(scores) / len(scores) if scores else 0.0
            normalized = max(0.0, min(1.0, (avg - lo) / (hi - lo)))
            print(f"  avg_raw={avg:.4f}, normalized={normalized:.4f}", file=sys.stderr)

            # OpenEnv validator stdout format
            print(f"TASK:{task_id} SCORE:{normalized:.4f}")


if __name__ == "__main__":
    main()

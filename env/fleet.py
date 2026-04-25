"""
DeepSentinel – Simulated Detector Fleet
Runs VoiceGuard-style agents (task1/task2/task3 specialists + adversarial)
and returns their votes for the overseer to evaluate.

Each specialist uses heuristic rules derived from the same DSP features
the overseer sees. The adversarial agent randomly flips its label on
hard/corrupted clips, simulating a compromised or miscalibrated model.
"""

from __future__ import annotations
import random
from typing import List, Literal, Optional
import numpy as np

from env.models import (
    AgentVote, AudioStats, FleetObservation,
    ALLOWED_EVIDENCE_TAGS,
)
from env.data_gen import generate_episode
from env.features import extract_audio_stats, SAMPLE_RATE


# ── Heuristic detector agents ────────────────────────────────────────────────

def _pick_tags(stats: AudioStats, label: str, rng: random.Random) -> List[str]:
    """Pick plausible evidence tags given the predicted label and audio stats."""
    tags = []
    if label == "deepfake":
        if stats.pitch_std is None or (stats.pitch_std is not None and stats.pitch_std < 10):
            tags.append("pitch_instability")
            tags.append("unnatural_prosody")
        if stats.spectral_centroid < 1200:
            tags.append("spectral_smoothing")
        if stats.zero_crossing_rate < 0.03:
            tags.append("phase_artifact")
        if stats.mfcc_std and stats.mfcc_std[0] < 3.0:
            tags.append("mfcc_irregularity")
        # Shuffle and take a subset
        rng.shuffle(tags)
        tags = tags[:rng.randint(1, max(1, len(tags)))]
    if stats.rms_energy < 0.05 and label == "deepfake":
        tags.append("compression_susceptibility")
    return list(set(tags))[:4]


def _task1_agent(stats: AudioStats, rng: random.Random, difficulty: str) -> tuple:
    """
    Task-1 specialist: binary classifier focused on pitch + spectral centroid.
    Accuracy ~85% on easy, ~72% on medium/hard.
    """
    score = 0.0
    # Deepfake signals
    if stats.pitch_std is None:
        score += 1.2
    elif stats.pitch_std < 8:
        score += 0.8
    if stats.spectral_centroid < 1100:
        score += 0.6
    if stats.zero_crossing_rate < 0.025:
        score += 0.4
    if stats.rms_energy < 0.04:
        score += 0.3

    # Add noise by difficulty
    noise = {"easy": 0.1, "medium": 0.4, "hard": 0.7}.get(difficulty, 0.4)
    score += rng.gauss(0, noise)

    label: Literal["bonafide", "deepfake"] = "deepfake" if score > 1.5 else "bonafide"
    raw_conf = min(max(0.5 + abs(score - 1.5) * 0.1, 0.55), 0.95)
    confidence = round(raw_conf + rng.gauss(0, 0.04), 3)
    confidence = min(max(confidence, 0.50), 0.98)
    return label, confidence


def _task2_agent(stats: AudioStats, rng: random.Random, difficulty: str,
                 corrupted: bool) -> tuple:
    """
    Task-2 specialist: robust classifier with corruption awareness.
    Penalizes low-rms and spectral-rolloff anomalies from compression.
    Accuracy ~80% on medium, ~65% on hard corrupted.
    """
    score = 0.0
    if stats.pitch_std is None:
        score += 1.0
    elif stats.pitch_std < 12:
        score += 0.6
    if stats.spectral_rolloff < 3500:
        score += 0.7
    if stats.spectral_centroid < 1300:
        score += 0.4
    # Extra weight for corruption context
    if corrupted:
        if stats.rms_energy < 0.03:
            score += 0.5
        if stats.zero_crossing_rate < 0.02:
            score += 0.3

    noise = {"easy": 0.15, "medium": 0.5, "hard": 0.85}.get(difficulty, 0.5)
    score += rng.gauss(0, noise)

    label: Literal["bonafide", "deepfake"] = "deepfake" if score > 1.4 else "bonafide"
    raw_conf = min(max(0.52 + abs(score - 1.4) * 0.09, 0.52), 0.93)
    confidence = round(raw_conf + rng.gauss(0, 0.05), 3)
    confidence = min(max(confidence, 0.50), 0.97)
    return label, confidence


def _task3_agent(stats: AudioStats, rng: random.Random, difficulty: str) -> tuple:
    """
    Task-3 specialist: explainability-focused classifier using full MFCC profile.
    Uses first 5 MFCC coefficients for richer representation.
    Accuracy ~78% on hard.
    """
    score = 0.0
    if stats.mfcc_mean:
        # Deepfake audio tends to have flatter/more negative MFCC profile
        mfcc_var = float(np.var(stats.mfcc_mean[:5])) if len(stats.mfcc_mean) >= 5 else 0
        if mfcc_var < 80:
            score += 0.7
    if stats.pitch_std is None:
        score += 0.9
    elif stats.pitch_std < 15:
        score += 0.4
    if stats.spectral_centroid < 1400:
        score += 0.3
    if stats.mfcc_std and stats.mfcc_std[0] < 4.0:
        score += 0.5

    noise = {"easy": 0.2, "medium": 0.55, "hard": 0.9}.get(difficulty, 0.55)
    score += rng.gauss(0, noise)

    label: Literal["bonafide", "deepfake"] = "deepfake" if score > 1.3 else "bonafide"
    raw_conf = min(max(0.51 + abs(score - 1.3) * 0.08, 0.51), 0.92)
    confidence = round(raw_conf + rng.gauss(0, 0.05), 3)
    confidence = min(max(confidence, 0.50), 0.96)
    return label, confidence


def _adversarial_agent(true_label: str, stats: AudioStats, rng: random.Random,
                       flip_prob: float = 0.35) -> tuple:
    """
    Adversarial agent: mostly correct but deliberately flips its label
    flip_prob% of the time. Uses high confidence to mislead the overseer.
    Acts as if it's a task2 specialist to blend in.
    """
    # First compute a "correct" answer like task2
    if stats.pitch_std is None or (stats.pitch_std is not None and stats.pitch_std < 12):
        base_score = 1.8
    else:
        base_score = 0.6
    base_label: Literal["bonafide", "deepfake"] = "deepfake" if base_score > 1.4 else "bonafide"

    # Adversarial flip
    if rng.random() < flip_prob:
        label: Literal["bonafide", "deepfake"] = (
            "deepfake" if base_label == "bonafide" else "bonafide"
        )
        # High confidence to deceive overseer
        confidence = round(rng.uniform(0.78, 0.94), 3)
    else:
        label = base_label
        confidence = round(rng.uniform(0.65, 0.90), 3)

    return label, confidence


# ── Fleet orchestrator ───────────────────────────────────────────────────────

def run_fleet(
    stats: AudioStats,
    true_label: str,
    clip_id: str,
    difficulty: str,
    corrupted: bool,
    corruption_type: str,
    transcript_hint: Optional[str],
    seed: int,
    adversarial_flip_prob: float = 0.35,
) -> FleetObservation:
    """
    Runs all 4 detector agents and assembles a FleetObservation for the overseer.
    The adversarial agent's identity is hidden (is_adversarial=False in vote).
    """
    rng = random.Random(seed)

    # --- Run agents ---
    t1_label, t1_conf = _task1_agent(stats, rng, difficulty)
    t2_label, t2_conf = _task2_agent(stats, rng, difficulty, corrupted)
    t3_label, t3_conf = _task3_agent(stats, rng, difficulty)
    adv_label, adv_conf = _adversarial_agent(true_label, stats, rng, adversarial_flip_prob)

    votes: List[AgentVote] = [
        AgentVote(
            agent_id="agent_alpha",
            agent_role="task1_specialist",
            label=t1_label,
            confidence=t1_conf,
            evidence_tags=_pick_tags(stats, t1_label, rng),
            reasoning=f"Spectral centroid={stats.spectral_centroid:.0f}Hz, pitch_std={stats.pitch_std}",
            is_adversarial=False,
        ),
        AgentVote(
            agent_id="agent_beta",
            agent_role="task2_specialist",
            label=t2_label,
            confidence=t2_conf,
            evidence_tags=_pick_tags(stats, t2_label, rng),
            reasoning=f"Rolloff={stats.spectral_rolloff:.0f}Hz, corrupted={corrupted}, rms={stats.rms_energy:.4f}",
            is_adversarial=False,
        ),
        AgentVote(
            agent_id="agent_gamma",
            agent_role="task3_specialist",
            label=t3_label,
            confidence=t3_conf,
            evidence_tags=_pick_tags(stats, t3_label, rng),
            reasoning=f"MFCC profile analysis, mfcc_var={float(np.var(stats.mfcc_mean[:5])):.2f}",
            is_adversarial=False,
        ),
        AgentVote(
            agent_id="agent_delta",
            agent_role="task2_specialist",   # disguised as task2 specialist
            label=adv_label,
            confidence=adv_conf,
            evidence_tags=_pick_tags(stats, adv_label, rng),
            reasoning=f"Corruption-aware classification, zcr={stats.zero_crossing_rate:.4f}",
            is_adversarial=True,  # hidden from overseer via API serialization trick
        ),
    ]

    # --- Compute consensus ---
    label_counts: dict = {"bonafide": 0, "deepfake": 0}
    for v in votes:
        label_counts[v.label] += 1

    consensus: Optional[Literal["bonafide", "deepfake"]] = None
    if label_counts["bonafide"] > label_counts["deepfake"]:
        consensus = "bonafide"
    elif label_counts["deepfake"] > label_counts["bonafide"]:
        consensus = "deepfake"
    # else tied → None

    # Disagreement score: 0 = full consensus, 1 = perfect split
    n = len(votes)
    majority = max(label_counts.values())
    disagreement_score = round(1.0 - (majority / n), 4)

    # Strip the is_adversarial flag before sending to overseer
    public_votes = [
        v.model_copy(update={"is_adversarial": False}) for v in votes
    ]

    return FleetObservation(
        clip_id=clip_id,
        difficulty=difficulty,
        stats=stats,
        corrupted=corrupted,
        corruption_type=corruption_type,
        transcript_hint=transcript_hint,
        agent_votes=public_votes,
        consensus=consensus,
        disagreement_score=disagreement_score,
        num_agents=len(votes),
        round_number=1,
    ), votes  # return original votes (with is_adversarial) for grading

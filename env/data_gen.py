"""
VoiceGuard – Synthetic Episode Generator (numpy-only)
Produces deterministic fake/bonafide audio clips with gold labels.
"""

from __future__ import annotations
import hashlib, random, uuid
from typing import List, Literal, Optional
import numpy as np

from env.models import AudioObservation, AudioStats, EpisodeState
from env.features import extract_audio_stats, apply_corruption, SAMPLE_RATE

DEEPFAKE_TAGS = [
    "spectral_smoothing", "phase_artifact", "unnatural_prosody",
    "pitch_instability", "formant_inconsistency", "mfcc_irregularity",
]
CORRUPTION_TAGS = {
    "noise": ["compression_susceptibility"],
    "compression": ["compression_susceptibility", "spectral_smoothing"],
    "speed": ["unnatural_prosody", "pitch_instability"],
    "bandlimit": ["spectral_smoothing", "formant_inconsistency"],
}
CORRUPTION_TYPES = ["none", "noise", "compression", "speed", "bandlimit"]


def _synth_bonafide(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, 4 * SAMPLE_RATE, endpoint=False, dtype=np.float32)
    f0 = rng.uniform(100, 250)
    sig = sum((1 / n) * np.sin(2 * np.pi * f0 * n * t + rng.uniform(0, 0.5)) for n in range(1, 8))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    sig = (sig * env).astype(np.float32)
    sig /= np.abs(sig).max() + 1e-8
    return sig


def _synth_deepfake(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, 4 * SAMPLE_RATE, endpoint=False, dtype=np.float32)
    f0 = rng.uniform(150, 300)
    sig = sum((1 / n) * np.sin(2 * np.pi * f0 * n * t) for n in range(1, 6))
    jump_idx = int(len(sig) * rng.uniform(0.4, 0.6))
    sig[jump_idx:] += rng.uniform(0.01, 0.05)
    sig = sig.astype(np.float32)
    sig /= np.abs(sig).max() + 1e-8
    return sig


def generate_episode(
    task_id: str, session_id: str,
    difficulty: Literal["easy", "medium", "hard"] = "easy",
    seed: Optional[int] = None,
) -> tuple:
    if seed is None:
        seed = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % (2 ** 31)
    rng = random.Random(seed)

    clip_id = hashlib.md5(f"{session_id}:{seed}:{task_id}".encode()).hexdigest()[:8]
    true_label = rng.choice(["bonafide", "deepfake"])
    audio_seed = rng.randint(0, 2**31)
    audio = _synth_bonafide(audio_seed) if true_label == "bonafide" else _synth_deepfake(audio_seed)

    corruption_type = "none"
    corrupted = False
    if task_id in ("task2", "task3") and difficulty in ("medium", "hard"):
        if rng.random() < 0.6:
            corruption_type = rng.choice(CORRUPTION_TYPES[1:])
            audio = apply_corruption(audio, corruption_type, seed=audio_seed)
            corrupted = True

    gold_tags: List[str] = []
    if true_label == "deepfake":
        n = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
        gold_tags = rng.sample(DEEPFAKE_TAGS, min(n, len(DEEPFAKE_TAGS)))
    if corrupted:
        gold_tags += CORRUPTION_TAGS.get(corruption_type, [])
    gold_tags = list(set(gold_tags))

    conf_range = {"easy": (0.7, 1.0), "medium": (0.55, 0.9), "hard": (0.5, 0.8)}[difficulty]
    stats_dict = extract_audio_stats(audio, SAMPLE_RATE)
    stats = AudioStats(**stats_dict)
    max_steps = 2 if task_id == "task3" else 1

    prompt_text = "Please describe what you are doing today." if task_id == "task3" else None
    transcript_hint = "…describe what you are doing…" if task_id == "task3" and rng.random() < 0.5 else None

    obs = AudioObservation(
        clip_id=clip_id, task_id=task_id, prompt_text=prompt_text,
        language="en", speaker_id=f"spk_{audio_seed % 50:03d}",
        corrupted=corrupted, corruption_type=corruption_type,
        stats=stats, transcript_hint=transcript_hint,
    )
    state = EpisodeState(
        session_id=session_id, task_id=task_id, clip_id=clip_id,
        true_label=true_label, gold_tags=gold_tags,
        difficulty_level=difficulty, corruption_type=corruption_type,
        expected_confidence_range=conf_range, max_steps=max_steps,
    )
    return state, obs, audio

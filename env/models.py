"""
DeepSentinel – OpenEnv Schema Definitions
Extends VoiceGuard schemas with fleet observation, overseer action, and oversight reward.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


# ── AUDIO CORE (unchanged from VoiceGuard) ──────────────────────────────────

class AudioStats(BaseModel):
    duration_sec: float
    sample_rate: int
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_rolloff: float
    mfcc_mean: List[float] = Field(default_factory=list)
    mfcc_std: List[float] = Field(default_factory=list)
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None


class AudioObservation(BaseModel):
    clip_id: str
    task_id: str
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    speaker_id: Optional[str] = None
    corrupted: bool = False
    corruption_type: Optional[
        Literal["none", "noise", "compression", "speed", "bandlimit"]
    ] = "none"
    stats: AudioStats
    transcript_hint: Optional[str] = None
    history: List[Dict] = Field(default_factory=list)


ALLOWED_EVIDENCE_TAGS = {
    "spectral_smoothing", "phase_artifact", "unnatural_prosody",
    "pitch_instability", "formant_inconsistency", "mfcc_irregularity",
    "compression_susceptibility",
}

ALLOWED_FEATURE_REQUESTS = {
    "waveform_stats", "mfcc", "pitch", "spectrogram", "transcript_hint",
}


class AudioAction(BaseModel):
    label: Literal["bonafide", "deepfake"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_tags: List[str] = Field(default_factory=list)
    explanation: str = ""
    request_more_features: List[str] = Field(default_factory=list)

    def validate_tags(self) -> List[str]:
        return [t for t in self.evidence_tags if t not in ALLOWED_EVIDENCE_TAGS]

    def validate_feature_requests(self) -> List[str]:
        return [r for r in self.request_more_features if r not in ALLOWED_FEATURE_REQUESTS]


class AudioReward(BaseModel):
    correctness_reward: float = 0.0
    calibration_reward: float = 0.0
    explanation_reward: float = 0.0
    robustness_reward: float = 0.0
    penalty: float = 0.0
    total: float = 0.0

    def compute_total(self) -> "AudioReward":
        self.total = round(
            self.correctness_reward + self.calibration_reward
            + self.explanation_reward + self.robustness_reward - self.penalty,
            4,
        )
        return self


class StepResult(BaseModel):
    observation: Optional[AudioObservation] = None
    reward: AudioReward
    done: bool
    info: Dict = Field(default_factory=dict)


class EpisodeState(BaseModel):
    session_id: str
    task_id: str
    clip_id: str
    true_label: Literal["bonafide", "deepfake"]
    gold_tags: List[str] = Field(default_factory=list)
    difficulty_level: Literal["easy", "medium", "hard"] = "easy"
    corruption_type: str = "none"
    expected_confidence_range: Tuple[float, float] = (0.6, 1.0)
    step_count: int = 0
    max_steps: int = 1
    done: bool = False
    cumulative_reward: float = 0.0


# ── FLEET / OVERSEER (new for DeepSentinel) ─────────────────────────────────

class AgentVote(BaseModel):
    """A single detector agent's verdict on a clip."""
    agent_id: str
    agent_role: str  # "task1_specialist", "task2_specialist", "task3_specialist", "adversarial"
    label: Literal["bonafide", "deepfake"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_tags: List[str] = Field(default_factory=list)
    reasoning: str = ""
    is_adversarial: bool = False  # hidden from overseer


class FleetObservation(BaseModel):
    """What the overseer sees: audio features + all agent votes."""
    clip_id: str
    difficulty: str
    stats: AudioStats
    corrupted: bool
    corruption_type: str
    transcript_hint: Optional[str] = None
    agent_votes: List[AgentVote]
    consensus: Optional[Literal["bonafide", "deepfake"]] = None  # majority, or None if tied
    disagreement_score: float = 0.0  # 0 = full consensus, 1 = maximum split
    num_agents: int = 0
    round_number: int = 1


class OverseerAction(BaseModel):
    """The overseer's final verdict after reviewing the fleet."""
    final_label: Literal["bonafide", "deepfake"]
    confidence: float = Field(ge=0.0, le=1.0)
    suspected_bad_agents: List[str] = Field(default_factory=list)  # agent_ids flagged
    override_reasoning: str = ""  # why overseer deviated from consensus (if it did)
    consensus_adopted: bool = True  # did overseer follow majority?
    evidence_tags: List[str] = Field(default_factory=list)

    def validate_tags(self) -> List[str]:
        return [t for t in self.evidence_tags if t not in ALLOWED_EVIDENCE_TAGS]


class OverseerReward(BaseModel):
    """Multi-component reward for the overseer agent."""
    correctness_reward: float = 0.0      # did overseer get the true label right?
    consensus_reward: float = 0.0        # smart use of consensus vs override
    detection_reward: float = 0.0        # correctly flagged the adversarial agent
    calibration_reward: float = 0.0      # confidence calibration
    explanation_reward: float = 0.0      # quality of override_reasoning
    penalty: float = 0.0
    total: float = 0.0
    breakdown: Dict = Field(default_factory=dict)

    def compute_total(self) -> "OverseerReward":
        self.total = round(
            self.correctness_reward + self.consensus_reward
            + self.detection_reward + self.calibration_reward
            + self.explanation_reward - self.penalty,
            4,
        )
        self.breakdown = {
            "correctness": self.correctness_reward,
            "consensus": self.consensus_reward,
            "adversarial_detection": self.detection_reward,
            "calibration": self.calibration_reward,
            "explanation": self.explanation_reward,
            "penalty": self.penalty,
            "total": self.total,
        }
        return self


class OverseerStepResult(BaseModel):
    observation: Optional[FleetObservation] = None
    reward: OverseerReward
    done: bool
    info: Dict = Field(default_factory=dict)


class OverseerEpisodeState(BaseModel):
    session_id: str
    clip_id: str
    true_label: Literal["bonafide", "deepfake"]
    gold_tags: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    adversarial_agent_id: str = ""
    adversarial_injected: bool = False
    step_count: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    round_history: List[Dict] = Field(default_factory=list)

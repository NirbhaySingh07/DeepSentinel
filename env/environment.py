"""
DeepSentinel – Core OpenEnv Environment
Implements the 3-method OpenEnv interface: reset / step / state.

The overseer agent receives a FleetObservation (audio features + 4 agent votes)
and must produce an OverseerAction (final verdict + suspect flagging + explanation).
"""

from __future__ import annotations
import uuid
from typing import List, Optional

from env.models import (
    OverseerAction, OverseerEpisodeState, OverseerReward,
    OverseerStepResult, FleetObservation, AgentVote,
)
from env.graders import grade_overseer
from env.fleet import run_fleet
from env.data_gen import generate_episode


class DeepSentinelEnvironment:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self._state: Optional[OverseerEpisodeState] = None
        self._fleet_obs: Optional[FleetObservation] = None
        self._raw_votes: Optional[List[AgentVote]] = None  # includes is_adversarial

    def reset(
        self,
        difficulty: str = "medium",
        seed: Optional[int] = None,
        adversarial_flip_prob: float = 0.35,
    ) -> FleetObservation:
        """
        Generate a new episode, run the detector fleet, and return the FleetObservation.
        """
        # Use task3 episode generation for richest features
        task_map = {"easy": "task1", "medium": "task2", "hard": "task3"}
        task_id = task_map.get(difficulty, "task2")

        ep_state, obs, audio = generate_episode(
            task_id=task_id,
            session_id=self.session_id,
            difficulty=difficulty,
            seed=seed,
        )

        fleet_seed = (seed or 0) + 7919  # deterministic but different from episode seed

        fleet_obs, raw_votes = run_fleet(
            stats=obs.stats,
            true_label=ep_state.true_label,
            clip_id=ep_state.clip_id,
            difficulty=difficulty,
            corrupted=obs.corrupted,
            corruption_type=obs.corruption_type or "none",
            transcript_hint=obs.transcript_hint,
            seed=fleet_seed,
            adversarial_flip_prob=adversarial_flip_prob,
        )

        # Find the adversarial agent id
        adv_id = next(
            (v.agent_id for v in raw_votes if v.is_adversarial), ""
        )

        self._state = OverseerEpisodeState(
            session_id=self.session_id,
            clip_id=ep_state.clip_id,
            true_label=ep_state.true_label,
            gold_tags=ep_state.gold_tags,
            difficulty=difficulty,
            adversarial_agent_id=adv_id,
            adversarial_injected=any(v.is_adversarial for v in raw_votes),
        )
        self._fleet_obs = fleet_obs
        self._raw_votes = raw_votes

        return fleet_obs

    def step(self, action: OverseerAction) -> OverseerStepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode done. Call reset().")
        if self._fleet_obs is None or self._raw_votes is None:
            raise RuntimeError("Internal state missing.")

        bad_tags = action.validate_tags()
        if bad_tags:
            reward = OverseerReward(penalty=0.08 * len(bad_tags))
            reward.compute_total()
            self._state.done = True
            return OverseerStepResult(
                observation=None, reward=reward, done=True,
                info={"error": f"Invalid tags: {bad_tags}"},
            )

        reward = grade_overseer(
            state=self._state,
            action=action,
            fleet_votes=self._raw_votes,
            consensus=self._fleet_obs.consensus,
        )

        self._state.cumulative_reward += reward.total
        self._state.step_count += 1
        self._state.done = True

        self._state.round_history.append({
            "round": self._state.step_count,
            "final_label": action.final_label,
            "true_label": self._state.true_label,
            "consensus": self._fleet_obs.consensus,
            "overrode": not action.consensus_adopted,
            "suspected_bad_agents": action.suspected_bad_agents,
            "adversarial_agent_id": self._state.adversarial_agent_id,
            "reward": reward.total,
        })

        return OverseerStepResult(
            observation=None,
            reward=reward,
            done=True,
            info={
                "true_label": self._state.true_label,
                "adversarial_agent_id": self._state.adversarial_agent_id,
                "consensus": self._fleet_obs.consensus,
                "gold_tags": self._state.gold_tags,
                "cumulative_reward": self._state.cumulative_reward,
                "reward_breakdown": reward.breakdown,
            },
        )

    @property
    def state(self) -> Optional[OverseerEpisodeState]:
        return self._state

    @property
    def fleet_obs(self) -> Optional[FleetObservation]:
        return self._fleet_obs

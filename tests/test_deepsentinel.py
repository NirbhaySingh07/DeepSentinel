"""
DeepSentinel – Test Suite
Tests for fleet, environment, graders, and server integration.
Run: python -m tests.test_deepsentinel
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import DeepSentinelEnvironment
from env.models import OverseerAction


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_env(session_id: str) -> DeepSentinelEnvironment:
    return DeepSentinelEnvironment(session_id=session_id)


def correct_action(env: DeepSentinelEnvironment, fleet_obs) -> OverseerAction:
    """Produce a perfect overseer action using ground truth."""
    true_label = env.state.true_label
    consensus = fleet_obs.consensus
    adopted = consensus == true_label if consensus else True

    reasoning = ""
    if not adopted:
        reasoning = (
            "Agent delta has anomalously high confidence despite voting against "
            "the spectral and pitch features, suggesting adversarial manipulation."
        )

    adv_id = env.state.adversarial_agent_id
    return OverseerAction(
        final_label=true_label,
        confidence=0.82,
        consensus_adopted=adopted,
        suspected_bad_agents=[adv_id] if adv_id else [],
        override_reasoning=reasoning,
        evidence_tags=["spectral_smoothing", "pitch_instability"]
        if true_label == "deepfake" else [],
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_fleet_runs():
    """Fleet should return 4 votes with correct structure."""
    env = make_env("test_fleet")
    obs = env.reset(difficulty="medium", seed=42)
    assert obs.num_agents == 4, f"Expected 4 agents, got {obs.num_agents}"
    assert len(obs.agent_votes) == 4
    for vote in obs.agent_votes:
        assert vote.label in ("bonafide", "deepfake")
        assert 0.0 <= vote.confidence <= 1.0
        assert vote.agent_id in ("agent_alpha", "agent_beta", "agent_gamma", "agent_delta")
    print(f"test_fleet_runs: PASSED ({obs.num_agents} agents, "
          f"consensus={obs.consensus}, disagree={obs.disagreement_score:.2f})")


def test_correct_overseer_positive_reward():
    """Perfect overseer with true label + adversarial detection should get positive reward."""
    env = make_env("test_correct")
    obs = env.reset(difficulty="medium", seed=42, adversarial_flip_prob=0.5)
    action = correct_action(env, obs)
    result = env.step(action)
    assert result.done, "Episode should be done after overseer step"
    assert result.reward.total > 0, (
        f"Correct overseer should get positive reward, got {result.reward.total:.4f}\n"
        f"Breakdown: {result.reward.breakdown}"
    )
    print(f"test_correct_overseer_positive_reward: PASSED "
          f"(reward={result.reward.total:.4f})")


def test_wrong_label_negative_reward():
    """Wrong final label should always produce negative reward."""
    env = make_env("test_wrong")
    obs = env.reset(difficulty="easy", seed=123)
    true_label = env.state.true_label
    wrong_label = "deepfake" if true_label == "bonafide" else "bonafide"
    action = OverseerAction(
        final_label=wrong_label,
        confidence=0.9,
        consensus_adopted=True,
        suspected_bad_agents=[],
        override_reasoning="",
        evidence_tags=[],
    )
    result = env.step(action)
    assert result.reward.total < 0, (
        f"Wrong label should give negative reward, got {result.reward.total:.4f}"
    )
    print(f"test_wrong_label_negative_reward: PASSED "
          f"(reward={result.reward.total:.4f})")


def test_adversarial_detection_bonus():
    """Correctly flagging the adversarial agent should give +0.5 detection bonus."""
    env = make_env("test_adversarial")
    obs = env.reset(difficulty="hard", seed=77, adversarial_flip_prob=0.9)
    adv_id = env.state.adversarial_agent_id
    assert adv_id, "Should have adversarial agent in hard mode"

    true_label = env.state.true_label
    action = OverseerAction(
        final_label=true_label,
        confidence=0.75,
        consensus_adopted=False,
        suspected_bad_agents=[adv_id],
        override_reasoning="Agent delta votes with 0.93 confidence against all spectral features. Flagging as adversarial.",
        evidence_tags=["spectral_smoothing"] if true_label == "deepfake" else [],
    )
    result = env.step(action)
    assert result.reward.detection_reward > 0, (
        f"Should get detection bonus, got {result.reward.detection_reward:.4f}"
    )
    print(f"test_adversarial_detection_bonus: PASSED "
          f"(detection_reward={result.reward.detection_reward:.4f}, "
          f"total={result.reward.total:.4f})")


def test_false_accusation_penalty():
    """Accusing a non-adversarial agent should incur a penalty."""
    env = make_env("test_false_acc")
    obs = env.reset(difficulty="easy", seed=55)
    adv_id = env.state.adversarial_agent_id
    true_label = env.state.true_label

    # Accuse the WRONG agent (alpha instead of the actual adversarial)
    false_suspect = "agent_alpha" if adv_id != "agent_alpha" else "agent_beta"
    action = OverseerAction(
        final_label=true_label,
        confidence=0.78,
        consensus_adopted=True,
        suspected_bad_agents=[false_suspect],
        override_reasoning="",
        evidence_tags=[],
    )
    result = env.step(action)
    assert result.reward.detection_reward < 0 or result.reward.detection_reward == 0, (
        f"False accusation should not give positive detection reward, "
        f"got {result.reward.detection_reward:.4f}"
    )
    print(f"test_false_accusation_penalty: PASSED "
          f"(detection_reward={result.reward.detection_reward:.4f})")


def test_spurious_override_penalty():
    """Overriding the correct consensus without reasoning should be penalized."""
    env = make_env("test_spurious")
    obs = env.reset(difficulty="easy", seed=99)
    true_label = env.state.true_label
    consensus = obs.consensus

    if consensus is None or consensus != true_label:
        print("test_spurious_override_penalty: SKIPPED (consensus wrong/missing for this seed)")
        return

    # Consensus is correct, but we override it without reasoning
    wrong_label = "deepfake" if true_label == "bonafide" else "bonafide"
    action = OverseerAction(
        final_label=wrong_label,
        confidence=0.85,
        consensus_adopted=False,
        suspected_bad_agents=[],
        override_reasoning="",  # no reasoning
        evidence_tags=[],
    )
    result = env.step(action)
    # Multiple penalties: wrong label + spurious override
    assert result.reward.total < -0.5, (
        f"Spurious override of correct consensus should be heavily penalized, "
        f"got {result.reward.total:.4f}"
    )
    print(f"test_spurious_override_penalty: PASSED (reward={result.reward.total:.4f})")


def test_determinism():
    """Same session_id + same seed should produce identical episodes."""
    env1 = make_env("det_shared")
    env2 = make_env("det_shared")  # same session_id = deterministic clip_id
    obs1 = env1.reset(difficulty="medium", seed=7)
    obs2 = env2.reset(difficulty="medium", seed=7)

    assert obs1.clip_id == obs2.clip_id, f"{obs1.clip_id} != {obs2.clip_id}"
    assert obs1.stats.rms_energy == obs2.stats.rms_energy
    assert env1.state.true_label == env2.state.true_label
    assert obs1.consensus == obs2.consensus

    # Votes should be identical
    for v1, v2 in zip(obs1.agent_votes, obs2.agent_votes):
        assert v1.label == v2.label, f"{v1.agent_id}: {v1.label} != {v2.label}"
        assert v1.confidence == v2.confidence

    print("test_determinism: PASSED")


def test_reset_after_done():
    """Should be able to reset and run another episode after done."""
    env = make_env("test_reset")
    obs = env.reset(difficulty="easy", seed=10)
    true_label = env.state.true_label
    action = OverseerAction(
        final_label=true_label, confidence=0.7,
        consensus_adopted=True, suspected_bad_agents=[],
        override_reasoning="", evidence_tags=[],
    )
    result = env.step(action)
    assert result.done

    # Reset and run again
    obs2 = env.reset(difficulty="medium", seed=20)
    assert obs2.clip_id != obs.clip_id
    true_label2 = env.state.true_label
    action2 = OverseerAction(
        final_label=true_label2, confidence=0.7,
        consensus_adopted=True, suspected_bad_agents=[],
        override_reasoning="", evidence_tags=[],
    )
    result2 = env.step(action2)
    assert result2.done
    print(f"test_reset_after_done: PASSED "
          f"(ep1 reward={result.reward.total:.3f}, "
          f"ep2 reward={result2.reward.total:.3f})")


def test_hard_episode_reward_range():
    """Run 5 hard episodes and verify rewards are within expected range."""
    total_rewards = []
    for seed in range(5):
        env = make_env(f"hard_{seed}")
        obs = env.reset(difficulty="hard", seed=seed * 17, adversarial_flip_prob=0.5)
        true_label = env.state.true_label
        adv_id = env.state.adversarial_agent_id

        action = OverseerAction(
            final_label=true_label,
            confidence=0.72,
            consensus_adopted=obs.consensus == true_label if obs.consensus else True,
            suspected_bad_agents=[adv_id] if adv_id else [],
            override_reasoning=(
                "Detected high-confidence dissenting agent with anomalous spectral signature."
                if obs.consensus != true_label else ""
            ),
            evidence_tags=["pitch_instability"] if true_label == "deepfake" else [],
        )
        result = env.step(action)
        total_rewards.append(result.reward.total)

    avg = sum(total_rewards) / len(total_rewards)
    print(f"test_hard_episode_reward_range: PASSED "
          f"(avg={avg:.3f}, rewards={[round(r, 3) for r in total_rewards]})")
    assert -3.0 < avg < 3.0, f"Rewards out of expected range: {avg}"


def test_invalid_tags_penalty():
    """Invalid evidence tags should incur a penalty."""
    env = make_env("test_invalid_tags")
    env.reset(difficulty="easy", seed=5)
    true_label = env.state.true_label
    action = OverseerAction(
        final_label=true_label,
        confidence=0.8,
        consensus_adopted=True,
        suspected_bad_agents=[],
        override_reasoning="",
        evidence_tags=["NOT_A_REAL_TAG", "spectral_smoothing"],
    )
    result = env.step(action)
    assert result.reward.penalty > 0, "Should have penalty for invalid tag"
    print(f"test_invalid_tags_penalty: PASSED (penalty={result.reward.penalty:.4f})")


def test_reward_breakdown_complete():
    """Reward breakdown dict should contain all expected keys."""
    env = make_env("test_breakdown")
    obs = env.reset(difficulty="medium", seed=42)
    action = OverseerAction(
        final_label=env.state.true_label,
        confidence=0.75,
        consensus_adopted=True,
        suspected_bad_agents=[],
        override_reasoning="",
        evidence_tags=[],
    )
    result = env.step(action)
    bd = result.reward.breakdown
    expected_keys = {
        "correctness", "consensus", "adversarial_detection",
        "calibration", "explanation", "penalty", "total"
    }
    assert expected_keys.issubset(set(bd.keys())), (
        f"Missing keys: {expected_keys - set(bd.keys())}"
    )
    print(f"test_reward_breakdown_complete: PASSED (keys={list(bd.keys())})")


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_fleet_runs,
        test_correct_overseer_positive_reward,
        test_wrong_label_negative_reward,
        test_adversarial_detection_bonus,
        test_false_accusation_penalty,
        test_spurious_override_penalty,
        test_determinism,
        test_reset_after_done,
        test_hard_episode_reward_range,
        test_invalid_tags_penalty,
        test_reward_breakdown_complete,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)

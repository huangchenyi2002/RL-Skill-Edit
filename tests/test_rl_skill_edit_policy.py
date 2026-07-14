from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.rl_skill_edit.policy import ActorCriticPolicy
from baselines.rl_skill_edit.types import Transition


def _policy(seed: int = 7) -> ActorCriticPolicy:
    return ActorCriticPolicy(
        input_dim=5,
        action_dim=4,
        hidden_dim=8,
        seed=seed,
        learning_rate=0.03,
        gamma=0.95,
        entropy_coef=0.0,
        value_coef=0.0,
        max_grad_norm=1.0,
        normalize_advantages=False,
    )


def test_forward_and_select_respect_mask_and_deterministic_argmax():
    policy = _policy()
    state = np.array([0.2, -0.1, 0.4, 0.0, 1.0])
    mask = np.array([True, False, True, False])

    probabilities, value = policy.forward(state, mask)
    decision = policy.select(state, mask, deterministic=True)

    assert probabilities.shape == (4,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[~mask].tolist() == [0.0, 0.0]
    assert np.isfinite(value)
    assert decision.action_index == int(np.argmax(probabilities))
    assert decision.probabilities == pytest.approx(probabilities)
    with pytest.raises(ValueError, match="mask"):
        policy.select(state, np.zeros(4, dtype=bool), deterministic=False)


def test_seed_reproduces_initialization_and_sampling_sequence():
    state = np.arange(5, dtype=float)
    mask = np.ones(4, dtype=bool)
    first = _policy(seed=11)
    second = _policy(seed=11)

    assert first.forward(state, mask)[0] == pytest.approx(
        second.forward(state, mask)[0]
    )
    assert [first.select(state, mask, False).action_index for _ in range(20)] == [
        second.select(state, mask, False).action_index for _ in range(20)
    ]


def test_positive_advantage_increases_selected_action_probability():
    policy = _policy()
    state = np.array([1.0, 0.5, -0.5, 0.25, 0.0])
    mask = np.ones(4, dtype=bool)
    before, value = policy.forward(state, mask)
    action = int(np.argmin(before))
    metrics = policy.update(
        (
            Transition(
                state=state,
                action_index=action,
                reward=float(value + 2.0),
                done=True,
                mask=mask,
            ),
        )
    )
    after, _ = policy.forward(state, mask)

    assert after[action] > before[action]
    assert {"policy_loss", "value_loss", "entropy", "grad_norm"} <= set(metrics)
    assert metrics["grad_norm"] <= 1.0 + 1e-9


def test_single_transition_keeps_actor_signal_when_advantage_normalization_enabled():
    policy = ActorCriticPolicy(
        input_dim=5,
        action_dim=4,
        hidden_dim=8,
        seed=23,
        learning_rate=0.03,
        gamma=0.95,
        entropy_coef=0.0,
        value_coef=0.0,
        max_grad_norm=1.0,
        normalize_advantages=True,
    )
    state = np.array([0.5, -0.25, 0.75, 0.0, 1.0])
    mask = np.ones(4, dtype=bool)
    before, value = policy.forward(state, mask)
    action = int(np.argmin(before))
    policy.update(
        (
            Transition(
                state=state,
                action_index=action,
                reward=float(value + 2.0),
                done=True,
                mask=mask,
            ),
        )
    )
    after, _ = policy.forward(state, mask)
    assert after[action] > before[action]


def test_checkpoint_round_trip_preserves_weights_configuration_and_rng(tmp_path: Path):
    policy = _policy(seed=19)
    state = np.array([0.2, 0.3, 0.5, 0.7, 1.1])
    mask = np.array([True, True, False, True])
    policy.select(state, mask, deterministic=False)
    checkpoint = tmp_path / "policy.pt"
    policy.save(checkpoint)
    expected_sequence = [
        policy.select(state, mask, False).action_index for _ in range(12)
    ]

    restored = ActorCriticPolicy.load(checkpoint)

    assert checkpoint.is_file()
    restored_probabilities, restored_value = restored.forward(state, mask)
    original_probabilities, original_value = policy.forward(state, mask)
    assert restored_probabilities == pytest.approx(original_probabilities)
    assert restored_value == pytest.approx(original_value)
    assert [
        restored.select(state, mask, False).action_index for _ in range(12)
    ] == expected_sequence
    assert (
        restored.select(state, mask, True).action_index
        == policy.select(state, mask, True).action_index
    )

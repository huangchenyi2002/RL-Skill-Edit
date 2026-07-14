from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import PolicyDecision, Transition


_CHECKPOINT_VERSION = 1


class ActorCriticPolicy:
    """A small NumPy actor-critic with a shared tanh hidden layer."""

    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        seed: int | None = 0,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        normalize_advantages: bool = True,
    ) -> None:
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.action_dim = _positive_integer("action_dim", action_dim)
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.seed = _optional_integer("seed", seed)
        self.learning_rate = _positive_finite("learning_rate", learning_rate)
        self.gamma = _finite_in_range("gamma", gamma, lower=0.0, upper=1.0)
        self.entropy_coef = _nonnegative_finite("entropy_coef", entropy_coef)
        self.value_coef = _nonnegative_finite("value_coef", value_coef)
        self.max_grad_norm = _positive_finite("max_grad_norm", max_grad_norm)
        if not isinstance(normalize_advantages, (bool, np.bool_)):
            raise TypeError("normalize_advantages must be a boolean")
        self.normalize_advantages = bool(normalize_advantages)

        self._rng = np.random.default_rng(self.seed)
        hidden_scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        actor_scale = np.sqrt(2.0 / (self.hidden_dim + self.action_dim))
        value_scale = np.sqrt(2.0 / (self.hidden_dim + 1))
        self._w_hidden = self._rng.normal(
            0.0, hidden_scale, size=(self.input_dim, self.hidden_dim)
        )
        self._b_hidden = np.zeros(self.hidden_dim, dtype=float)
        self._w_actor = self._rng.normal(
            0.0, actor_scale, size=(self.hidden_dim, self.action_dim)
        )
        self._b_actor = np.zeros(self.action_dim, dtype=float)
        self._w_value = self._rng.normal(0.0, value_scale, size=self.hidden_dim)
        self._b_value = np.array(0.0, dtype=float)

    def forward(self, state: Any, mask: Any) -> tuple[np.ndarray, float]:
        state_array = self._validate_state(state)
        mask_array = self._validate_mask(mask)
        hidden = np.tanh(state_array @ self._w_hidden + self._b_hidden)
        logits = hidden @ self._w_actor + self._b_actor
        probabilities = _masked_softmax(logits, mask_array)
        value = float(hidden @ self._w_value + self._b_value)
        if not np.isfinite(value):
            raise FloatingPointError("policy value is not finite")
        return probabilities, value

    def select(
        self,
        state: Any,
        mask: Any,
        deterministic: bool = False,
    ) -> PolicyDecision:
        if not isinstance(deterministic, (bool, np.bool_)):
            raise TypeError("deterministic must be a boolean")
        probabilities, value = self.forward(state, mask)
        if deterministic:
            action_index = int(np.argmax(probabilities))
        else:
            action_index = int(self._rng.choice(self.action_dim, p=probabilities))
        positive = probabilities > 0.0
        entropy = float(
            -np.sum(probabilities[positive] * np.log(probabilities[positive]))
        )
        return PolicyDecision(
            action_index=action_index,
            probability=float(probabilities[action_index]),
            probabilities=probabilities.copy(),
            entropy=entropy,
            value=value,
        )

    def update(self, transitions: Iterable[Transition]) -> dict[str, float]:
        batch = tuple(transitions)
        if not batch:
            raise ValueError("transitions must not be empty")

        states = np.empty((len(batch), self.input_dim), dtype=float)
        masks = np.empty((len(batch), self.action_dim), dtype=bool)
        actions = np.empty(len(batch), dtype=int)
        rewards = np.empty(len(batch), dtype=float)
        dones = np.empty(len(batch), dtype=bool)
        for index, transition in enumerate(batch):
            states[index] = self._validate_state(transition.state)
            masks[index] = self._validate_mask(transition.mask)
            action_index = _integer("action_index", transition.action_index)
            if not 0 <= action_index < self.action_dim:
                raise ValueError("action_index is outside the action space")
            if not masks[index, action_index]:
                raise ValueError("action_index is invalid under its mask")
            actions[index] = action_index
            rewards[index] = _finite("reward", transition.reward)
            if not isinstance(transition.done, (bool, np.bool_)):
                raise TypeError("done must be a boolean")
            dones[index] = bool(transition.done)

        returns = np.empty(len(batch), dtype=float)
        running_return = 0.0
        for index in range(len(batch) - 1, -1, -1):
            if dones[index]:
                running_return = 0.0
            running_return = rewards[index] + self.gamma * running_return
            returns[index] = running_return

        hidden = np.tanh(states @ self._w_hidden + self._b_hidden)
        logits = hidden @ self._w_actor + self._b_actor
        distributions = [
            _masked_distribution(logits[index], masks[index])
            for index in range(len(batch))
        ]
        probabilities = np.vstack([distribution[0] for distribution in distributions])
        log_probabilities = np.vstack(
            [distribution[1] for distribution in distributions]
        )
        values = hidden @ self._w_value + float(self._b_value)
        advantages = returns - values
        policy_advantages = advantages.copy()
        if self.normalize_advantages and len(policy_advantages) > 1:
            policy_advantages -= float(np.mean(policy_advantages))
            standard_deviation = float(np.std(policy_advantages))
            if standard_deviation > 1e-12:
                policy_advantages /= standard_deviation

        chosen_log_probabilities = log_probabilities[np.arange(len(batch)), actions]
        if not np.all(np.isfinite(chosen_log_probabilities)):
            raise FloatingPointError("chosen action log-probability is not finite")
        policy_loss = float(-np.mean(chosen_log_probabilities * policy_advantages))
        value_errors = values - returns
        value_loss = float(0.5 * np.mean(value_errors**2))
        positive = probabilities > 0.0
        entropy_log_probabilities = np.zeros_like(log_probabilities)
        entropy_log_probabilities[positive] = log_probabilities[positive]
        entropies = -np.sum(
            probabilities * entropy_log_probabilities,
            axis=1,
        )
        entropy = float(np.mean(entropies))

        batch_size = float(len(batch))
        actor_logits_gradient = probabilities.copy()
        actor_logits_gradient[np.arange(len(batch)), actions] -= 1.0
        actor_logits_gradient *= policy_advantages[:, None] / batch_size
        actor_logits_gradient += (
            self.entropy_coef
            * probabilities
            * (entropy_log_probabilities + entropies[:, None])
            / batch_size
        )
        actor_logits_gradient[~masks] = 0.0

        value_gradient = self.value_coef * value_errors / batch_size
        hidden_gradient = (
            actor_logits_gradient @ self._w_actor.T
            + value_gradient[:, None] * self._w_value[None, :]
        )
        hidden_pre_activation_gradient = hidden_gradient * (1.0 - hidden**2)

        gradients = (
            states.T @ hidden_pre_activation_gradient,
            np.sum(hidden_pre_activation_gradient, axis=0),
            hidden.T @ actor_logits_gradient,
            np.sum(actor_logits_gradient, axis=0),
            hidden.T @ value_gradient,
            np.array(np.sum(value_gradient), dtype=float),
        )
        raw_gradient_norm = float(
            np.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients))
        )
        if not np.isfinite(raw_gradient_norm):
            raise FloatingPointError("policy gradient norm is not finite")
        clip_scale = min(1.0, self.max_grad_norm / max(raw_gradient_norm, 1e-12))
        clipped_gradients = tuple(gradient * clip_scale for gradient in gradients)

        (
            hidden_weight_gradient,
            hidden_bias_gradient,
            actor_weight_gradient,
            actor_bias_gradient,
            value_weight_gradient,
            value_bias_gradient,
        ) = clipped_gradients
        self._w_hidden -= self.learning_rate * hidden_weight_gradient
        self._b_hidden -= self.learning_rate * hidden_bias_gradient
        self._w_actor -= self.learning_rate * actor_weight_gradient
        self._b_actor -= self.learning_rate * actor_bias_gradient
        self._w_value -= self.learning_rate * value_weight_gradient
        self._b_value -= self.learning_rate * value_bias_gradient

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "grad_norm": min(raw_gradient_norm, self.max_grad_norm),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        configuration = {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantages": self.normalize_advantages,
        }
        with target.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.array(_CHECKPOINT_VERSION, dtype=np.int64),
                configuration=np.array(json.dumps(configuration, sort_keys=True)),
                rng_state=np.array(
                    json.dumps(self._rng.bit_generator.state, sort_keys=True)
                ),
                w_hidden=self._w_hidden,
                b_hidden=self._b_hidden,
                w_actor=self._w_actor,
                b_actor=self._b_actor,
                w_value=self._w_value,
                b_value=self._b_value,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ActorCriticPolicy":
        source = Path(path)
        with np.load(source, allow_pickle=False) as checkpoint:
            required = {
                "version",
                "configuration",
                "rng_state",
                "w_hidden",
                "b_hidden",
                "w_actor",
                "b_actor",
                "w_value",
                "b_value",
            }
            missing = required - set(checkpoint.files)
            if missing:
                raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
            version = int(checkpoint["version"])
            if version != _CHECKPOINT_VERSION:
                raise ValueError(f"unsupported checkpoint version: {version}")
            configuration = json.loads(str(checkpoint["configuration"]))
            if not isinstance(configuration, dict):
                raise ValueError("checkpoint configuration must be an object")
            try:
                input_dim = _positive_integer("input_dim", configuration["input_dim"])
                action_dim = _positive_integer(
                    "action_dim", configuration["action_dim"]
                )
                hidden_dim = _positive_integer(
                    "hidden_dim", configuration["hidden_dim"]
                )
            except KeyError as exc:
                raise ValueError(
                    f"checkpoint configuration is missing {exc.args[0]}"
                ) from exc
            rng_state = json.loads(str(checkpoint["rng_state"]))
            arrays = {
                "_w_hidden": np.array(checkpoint["w_hidden"], dtype=float, copy=True),
                "_b_hidden": np.array(checkpoint["b_hidden"], dtype=float, copy=True),
                "_w_actor": np.array(checkpoint["w_actor"], dtype=float, copy=True),
                "_b_actor": np.array(checkpoint["b_actor"], dtype=float, copy=True),
                "_w_value": np.array(checkpoint["w_value"], dtype=float, copy=True),
                "_b_value": np.array(checkpoint["b_value"], dtype=float, copy=True),
            }
        expected_shapes = {
            "_w_hidden": (input_dim, hidden_dim),
            "_b_hidden": (hidden_dim,),
            "_w_actor": (hidden_dim, action_dim),
            "_b_actor": (action_dim,),
            "_w_value": (hidden_dim,),
            "_b_value": (),
        }
        for name, array in arrays.items():
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"checkpoint field {name} has shape {array.shape}, "
                    f"expected {expected_shapes[name]}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"checkpoint field {name} contains non-finite values")
        policy = cls(**configuration)
        for name, array in arrays.items():
            setattr(policy, name, array)
        policy._rng.bit_generator.state = rng_state
        return policy

    def _validate_state(self, state: Any) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        if state_array.shape != (self.input_dim,):
            raise ValueError(
                f"state must have shape ({self.input_dim},), got {state_array.shape}"
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("state must contain only finite values")
        return state_array

    def _validate_mask(self, mask: Any) -> np.ndarray:
        mask_array = np.asarray(mask)
        if mask_array.shape != (self.action_dim,):
            raise ValueError(
                f"mask must have shape ({self.action_dim},), got {mask_array.shape}"
            )
        if not np.issubdtype(mask_array.dtype, np.bool_):
            raise TypeError("mask must contain boolean values")
        if not np.any(mask_array):
            raise ValueError("mask must allow at least one action")
        return mask_array.astype(bool, copy=False)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _masked_distribution(logits, mask)[0]


def _masked_distribution(
    logits: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_logits = logits[mask]
    if not np.all(np.isfinite(valid_logits)):
        raise FloatingPointError("policy logits are not finite")
    shifted = valid_logits - np.max(valid_logits)
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("masked softmax denominator is invalid")
    probabilities = np.zeros(logits.shape, dtype=float)
    probabilities[mask] = exponentials / denominator
    log_probabilities = np.full(logits.shape, -np.inf, dtype=float)
    log_probabilities[mask] = shifted - np.log(denominator)
    return probabilities, log_probabilities


def _integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _positive_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_integer(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _integer(name, value)


def _finite(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_in_range(
    name: str,
    value: Any,
    *,
    lower: float,
    upper: float,
) -> float:
    result = _finite(name, value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


__all__ = ["ActorCriticPolicy"]

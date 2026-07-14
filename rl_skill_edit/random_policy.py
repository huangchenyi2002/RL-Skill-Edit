from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import PolicyDecision, Transition


class RandomEditPolicy:
    """Uniform valid-action policy for the random-search sanity baseline."""

    def __init__(self, *, action_dim: int, seed: int) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def select(
        self, state: Any, mask: Any, deterministic: bool = False
    ) -> PolicyDecision:
        del state
        valid = np.flatnonzero(np.asarray(mask, dtype=bool))
        if not len(valid):
            raise ValueError("mask must allow at least one action")
        action = int(valid[0] if deterministic else self._rng.choice(valid))
        probabilities = np.zeros(self.action_dim, dtype=float)
        probabilities[valid] = 1.0 / len(valid)
        return PolicyDecision(
            action_index=action,
            probability=float(probabilities[action]),
            probabilities=probabilities,
            entropy=float(np.log(len(valid))),
            value=0.0,
        )

    def update(self, transitions: Iterable[Transition]) -> dict[str, float]:
        if not tuple(transitions):
            raise ValueError("transitions must not be empty")
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "grad_norm": 0.0,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "policy": "uniform_valid_random",
                    "action_dim": self.action_dim,
                    "seed": self.seed,
                    "rng_state": self._rng.bit_generator.state,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


__all__ = ["RandomEditPolicy"]

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .types import ModuleDiagnostics, SkillModule


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def _token_levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


class StateEncoder:
    _GLOBAL_FEATURES = 6
    _MODULE_FEATURES = 9

    def __init__(self, max_modules: int, action_space_size: int) -> None:
        if (
            isinstance(max_modules, bool)
            or not isinstance(max_modules, int)
            or max_modules < 1
        ):
            raise ValueError("max_modules must be a positive integer")
        if (
            isinstance(action_space_size, bool)
            or not isinstance(action_space_size, int)
            or action_space_size < 1
        ):
            raise ValueError("action_space_size must be a positive integer")
        self.max_modules = max_modules
        self.action_space_size = action_space_size
        self.state_dim = (
            self._GLOBAL_FEATURES
            + self.action_space_size
            + 1
            + self.max_modules * self._MODULE_FEATURES
        )

    def encode(
        self,
        *,
        current_text: str,
        initial_text: str,
        modules: Sequence[SkillModule],
        diagnostics: Mapping[str, ModuleDiagnostics],
        accepted_edit_counts: Mapping[str, int],
        round_index: int,
        horizon: int,
        remaining_rollout_fraction: float,
        last_action_index: int | None,
        last_reward: float,
    ) -> np.ndarray:
        if not isinstance(current_text, str) or not isinstance(initial_text, str):
            raise TypeError("current_text and initial_text must be strings")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or not 0 <= round_index <= horizon
        ):
            raise ValueError("round_index must be between zero and horizon")
        remaining = float(remaining_rollout_fraction)
        reward = float(last_reward)
        if not math.isfinite(remaining) or not 0.0 <= remaining <= 1.0:
            raise ValueError("remaining_rollout_fraction must be finite and in [0, 1]")
        if not math.isfinite(reward):
            raise ValueError("last_reward must be finite")

        module_by_slot: dict[int, SkillModule] = {}
        module_ids: set[str] = set()
        for module in modules:
            if not 0 <= module.slot < self.max_modules:
                raise ValueError(f"module slot {module.slot} exceeds max_modules")
            if module.slot in module_by_slot:
                raise ValueError(f"duplicate module slot: {module.slot}")
            if module.module_id in module_ids:
                raise ValueError(f"duplicate module ID: {module.module_id}")
            module_by_slot[module.slot] = module
            module_ids.add(module.module_id)
        if not module_by_slot or 0 not in module_by_slot:
            raise ValueError("modules must contain slot zero")

        action_one_hot = np.zeros(self.action_space_size + 1, dtype=np.float64)
        if last_action_index is None:
            action_one_hot[0] = 1.0
        else:
            if isinstance(last_action_index, bool) or not isinstance(
                last_action_index, (int, np.integer)
            ):
                raise TypeError("last_action_index must be an integer or None")
            numeric_action = int(last_action_index)
            if numeric_action == -1:
                action_one_hot[0] = 1.0
            elif 0 <= numeric_action < self.action_space_size:
                action_one_hot[numeric_action + 1] = 1.0
            else:
                raise ValueError("last_action_index is outside the action space")

        current_tokens = _tokens(current_text)
        initial_tokens = _tokens(initial_text)
        edit_distance = _token_levenshtein(initial_tokens, current_tokens)
        vector = np.zeros(self.state_dim, dtype=np.float64)
        vector[: self._GLOBAL_FEATURES] = (
            round_index / horizon,
            math.log1p(len(current_tokens)),
            remaining,
            reward,
            edit_distance / max(1, len(initial_tokens), len(current_tokens)),
            len(module_by_slot) / self.max_modules,
        )
        action_start = self._GLOBAL_FEATURES
        action_end = action_start + len(action_one_hot)
        vector[action_start:action_end] = action_one_hot

        for slot, module in module_by_slot.items():
            diagnostics_for_module = diagnostics.get(module.module_id)
            if diagnostics_for_module is not None:
                if diagnostics_for_module.module_id != module.module_id:
                    raise ValueError("diagnostic key and module_id do not match")
                if (
                    diagnostics_for_module.failure_count < 0
                    or diagnostics_for_module.sample_count < 0
                ):
                    raise ValueError("diagnostic counts must be non-negative")
                mean_reward = float(diagnostics_for_module.mean_reward)
                success_rate = float(diagnostics_for_module.success_rate)
                if not math.isfinite(mean_reward) or not math.isfinite(success_rate):
                    raise ValueError("diagnostic rewards must be finite")
                if not 0.0 <= success_rate <= 1.0:
                    raise ValueError("diagnostic success_rate must be in [0, 1]")
                failure_count = diagnostics_for_module.failure_count
                task_count = len(diagnostics_for_module.task_ids)
                sample_count = diagnostics_for_module.sample_count
            else:
                failure_count = 0
                task_count = 0
                mean_reward = 0.0
                success_rate = 0.0
                sample_count = 0

            accepted_count = accepted_edit_counts.get(module.module_id, 0)
            if (
                isinstance(accepted_count, bool)
                or not isinstance(accepted_count, (int, np.integer))
                or int(accepted_count) < 0
            ):
                raise ValueError("accepted edit counts must be non-negative integers")

            start = action_end + slot * self._MODULE_FEATURES
            vector[start : start + self._MODULE_FEATURES] = (
                1.0,
                module.level / 6.0,
                math.log1p(len(_tokens(module.text))),
                math.log1p(failure_count),
                math.log1p(task_count),
                mean_reward,
                success_rate,
                math.log1p(sample_count),
                math.log1p(int(accepted_count)),
            )

        if not np.isfinite(vector).all():
            raise ValueError("encoded state contains a non-finite value")
        return vector


__all__ = ["StateEncoder"]

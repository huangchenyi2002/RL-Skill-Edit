from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .types import EvaluationBatch, RewardBreakdown


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


def _penalty_weight(config: Mapping[str, float], key: str) -> float:
    if key not in config:
        raise ValueError(f"reward config is missing {key}")
    try:
        value = float(config[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reward config {key} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"reward config {key} must be finite and non-negative")
    return value


def compute_incremental_reward(
    before: EvaluationBatch,
    after: EvaluationBatch,
    initial_text: str,
    current_text: str,
    candidate_text: str,
    config: Mapping[str, float],
    invalid: bool,
) -> RewardBreakdown:
    """Compute paired score improvement minus exact structural costs."""

    if not all(
        isinstance(text, str) for text in (initial_text, current_text, candidate_text)
    ):
        raise TypeError("skill texts must be strings")
    if not isinstance(invalid, bool):
        raise TypeError("invalid must be a bool")
    if before.split != after.split:
        raise ValueError("paired evaluation batches must use the same split")
    before_ids = before.ordered_task_ids
    after_ids = after.ordered_task_ids
    if not before_ids or before_ids != after_ids:
        raise ValueError("paired task results must contain the same ordered task IDs")
    if len(before_ids) != len(set(before_ids)):
        raise ValueError("paired task IDs must be unique")

    deltas: list[float] = []
    for incumbent, candidate in zip(before.results, after.results, strict=True):
        incumbent_reward = float(incumbent.reward)
        candidate_reward = float(candidate.reward)
        if not math.isfinite(incumbent_reward) or not math.isfinite(candidate_reward):
            raise ValueError("paired task rewards must be finite")
        deltas.append(candidate_reward - incumbent_reward)
    paired_delta = sum(deltas) / len(deltas)

    initial_tokens = _tokens(initial_text)
    current_tokens = _tokens(current_text)
    candidate_tokens = _tokens(candidate_text)
    length_cost = max(0, len(candidate_tokens) - len(initial_tokens))
    edit_cost = _token_levenshtein(current_tokens, candidate_tokens)
    invalid_cost = int(invalid)

    length_penalty = _penalty_weight(config, "beta_len") * length_cost
    edit_penalty = _penalty_weight(config, "beta_edit") * edit_cost
    invalid_penalty = _penalty_weight(config, "beta_invalid") * invalid_cost
    total = paired_delta - length_penalty - edit_penalty - invalid_penalty
    return RewardBreakdown(
        paired_delta=paired_delta,
        length_cost=length_cost,
        edit_cost=edit_cost,
        invalid_cost=invalid_cost,
        length_penalty=length_penalty,
        edit_penalty=edit_penalty,
        invalid_penalty=invalid_penalty,
        total=total,
    )


__all__ = ["compute_incremental_reward"]

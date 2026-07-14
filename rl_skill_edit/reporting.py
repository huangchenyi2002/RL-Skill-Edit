from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .types import EvaluationBatch, SkillArtifact, Split


_METHOD_NAMES = ("initial_skill", "rl_skill_edit")
_RESOURCE_INTEGER_FIELDS = frozenset(
    {
        "student_rollouts",
        "editor_calls",
        "evaluator_calls",
        "input_tokens",
        "output_tokens",
        "cache_hits",
        "cached_student_rollouts",
        "cached_editor_calls",
        "cached_evaluator_calls",
        "total_tokens",
        "edit_count",
    }
)
_RESOURCE_NUMBER_FIELDS = frozenset(
    {
        "wall_time_seconds",
        "cost_usd",
        "rollout_cost_usd",
        "editor_cost_usd",
        "wall_time_s",
    }
)
_REPORTING_INTEGER_FIELDS = frozenset(
    {
        "student_rollouts",
        "evaluator_calls",
        "cached_student_rollouts",
        "cached_evaluator_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
)
_REPORTING_NUMBER_FIELDS = frozenset({"cost_usd", "elapsed_s"})
_BATCH_USAGE_FIELDS = frozenset(
    {
        "student_rollouts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "trajectory_total_tokens",
        "cost_usd",
        "elapsed_s",
    }
)


@dataclass(frozen=True)
class ResourceUsage:
    student_rollouts: int = 0
    editor_calls: int = 0
    evaluator_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0
    cache_hits: int = 0
    cached_student_rollouts: int = 0
    cached_editor_calls: int = 0
    cached_evaluator_calls: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    rollout_cost_usd: float = 0.0
    editor_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    edit_count: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResourceUsage:
        if not isinstance(value, Mapping):
            raise TypeError("optimization_usage must be a mapping")
        allowed = _RESOURCE_INTEGER_FIELDS | _RESOURCE_NUMBER_FIELDS
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"optimization_usage has unknown fields: {sorted(unknown)}"
            )
        parsed: dict[str, int | float] = {}
        for name in _RESOURCE_INTEGER_FIELDS:
            if name in value:
                parsed[name] = _nonnegative_integer(
                    f"optimization_usage.{name}", value[name]
                )
        for name in _RESOURCE_NUMBER_FIELDS:
            if name in value:
                parsed[name] = _nonnegative_number(
                    f"optimization_usage.{name}", value[name]
                )
        return cls(**parsed)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PairedStats:
    mean_reward: float
    mean_delta: float
    standard_error: float
    success_rate: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True)
class MethodResult:
    method: str
    skill_digest: str
    stats: PairedStats
    usage: ResourceUsage
    skill_length_tokens: int


@dataclass(frozen=True)
class ComparisonResult:
    methods: tuple[MethodResult, ...]
    batches: dict[str, EvaluationBatch]


def paired_bootstrap_ci(
    initial: Iterable[float],
    candidate: Iterable[float],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    sample_count = _positive_integer("samples", samples)
    random_seed = _integer("seed", seed)
    confidence_alpha = _number("alpha", alpha)
    initial_array = np.asarray(tuple(initial), dtype=float)
    candidate_array = np.asarray(tuple(candidate), dtype=float)
    if initial_array.shape != candidate_array.shape or initial_array.ndim != 1:
        raise ValueError(
            "paired bootstrap inputs must be aligned one-dimensional arrays"
        )
    if initial_array.size == 0:
        raise ValueError("paired bootstrap requires at least one task")
    if not 0.0 < confidence_alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    differences = candidate_array - initial_array
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, differences.size, size=(sample_count, differences.size))
    means = differences[indices].mean(axis=1)
    return (
        float(np.quantile(means, confidence_alpha / 2.0)),
        float(np.quantile(means, 1.0 - confidence_alpha / 2.0)),
    )


def paired_statistics(
    initial: Iterable[float],
    candidate: Iterable[float],
    *,
    successes: Iterable[bool],
    samples: int,
    seed: int,
    tie_tolerance: float = 1e-12,
) -> PairedStats:
    tolerance = _nonnegative_number("tie_tolerance", tie_tolerance)
    initial_array = np.asarray(tuple(initial), dtype=float)
    candidate_array = np.asarray(tuple(candidate), dtype=float)
    success_values = tuple(successes)
    if any(type(value) is not bool for value in success_values):
        raise TypeError("successes must contain only booleans")
    success_array = np.asarray(success_values, dtype=bool)
    if (
        initial_array.shape != candidate_array.shape
        or candidate_array.shape != success_array.shape
    ):
        raise ValueError("paired statistics require aligned task-level values")
    if candidate_array.size == 0:
        raise ValueError("paired statistics require at least one task")
    differences = candidate_array - initial_array
    standard_error = (
        float(np.std(differences, ddof=1) / math.sqrt(differences.size))
        if differences.size > 1
        else 0.0
    )
    ci_low, ci_high = paired_bootstrap_ci(
        initial_array, candidate_array, samples=samples, seed=seed
    )
    wins = int(np.sum(differences > tolerance))
    losses = int(np.sum(differences < -tolerance))
    ties = int(differences.size - wins - losses)
    return PairedStats(
        mean_reward=float(np.mean(candidate_array)),
        mean_delta=float(np.mean(differences)),
        standard_error=standard_error,
        success_rate=float(np.mean(success_array)),
        ci_low=ci_low,
        ci_high=ci_high,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def run_frozen_report(
    *,
    initial_skill: SkillArtifact,
    rl_skill: SkillArtifact,
    evaluator: Any,
    test_tasks: tuple[Any, ...],
    output_dir: Path,
    seed: int,
    repetitions: int,
    bootstrap_samples: int,
    optimization_usage: Mapping[str, Any],
    reporting_usage: Mapping[str, Mapping[str, Any]],
) -> ComparisonResult:
    if not isinstance(initial_skill, SkillArtifact):
        raise TypeError("initial_skill must be a SkillArtifact")
    if not isinstance(rl_skill, SkillArtifact):
        raise TypeError("rl_skill must be a SkillArtifact")
    if not isinstance(test_tasks, tuple):
        raise TypeError("test_tasks must be a tuple")
    if not test_tasks:
        raise ValueError("frozen reporting requires at least one test task")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    report_seed = _integer("seed", seed)
    report_repetitions = _positive_integer("repetitions", repetitions)
    sample_count = _positive_integer("bootstrap_samples", bootstrap_samples)
    if not isinstance(reporting_usage, Mapping):
        raise TypeError("reporting_usage must be a mapping")
    unknown_methods = set(reporting_usage) - set(_METHOD_NAMES)
    if unknown_methods:
        raise ValueError(
            f"reporting_usage has unknown methods: {sorted(unknown_methods)}"
        )
    parsed_reporting_usage = {
        name: _parse_reporting_usage(reporting_usage.get(name, {}), method=name)
        for name in _METHOD_NAMES
    }
    optimization = ResourceUsage.from_mapping(optimization_usage)
    expected_ids = tuple(_task_id(task) for task in test_tasks)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("formal test task IDs must be unique")
    freeze = getattr(evaluator, "freeze", None)
    evaluate = getattr(evaluator, "evaluate", None)
    if not callable(freeze) or not callable(evaluate):
        raise TypeError("evaluator must expose callable freeze() and evaluate()")

    methods = (
        ("initial_skill", initial_skill),
        ("rl_skill_edit", rl_skill),
    )
    freeze()
    batches: dict[str, EvaluationBatch] = {}
    for name, skill in methods:
        batch = evaluate(
            skill,
            test_tasks,
            split=Split.TEST,
            seed=report_seed,
            repetitions=report_repetitions,
            use_cache=False,
            blind=True,
        )
        if not isinstance(batch, EvaluationBatch):
            raise TypeError(
                f"formal evaluator returned a non-EvaluationBatch for {name}"
            )
        if batch.split is not Split.TEST:
            raise RuntimeError(f"formal evaluator returned a non-test batch for {name}")
        if batch.cache_hit:
            raise RuntimeError(f"formal evaluator returned cached results for {name}")
        if batch.ordered_task_ids != expected_ids:
            raise RuntimeError(f"formal test task order changed for {name}")
        _validate_batch_usage(
            batch.usage,
            method=name,
            expected_rollouts=len(test_tasks) * report_repetitions,
        )
        batches[name] = batch

    initial_rewards = tuple(
        result.reward for result in batches["initial_skill"].results
    )
    usage_by_method = {
        "initial_skill": ResourceUsage(),
        "rl_skill_edit": optimization,
    }
    method_results: list[MethodResult] = []
    method_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for method_index, (name, skill) in enumerate(methods):
        batch = batches[name]
        rewards = tuple(result.reward for result in batch.results)
        successes = tuple(result.success for result in batch.results)
        stats = paired_statistics(
            initial_rewards,
            rewards,
            successes=successes,
            samples=sample_count,
            seed=report_seed + method_index,
        )
        usage = usage_by_method[name]
        formal_usage = _formal_reporting_usage(
            parsed_reporting_usage[name],
            batch,
            test_rollouts=len(test_tasks) * report_repetitions,
        )
        length = _token_count(skill.body)
        method_results.append(MethodResult(name, skill.digest, stats, usage, length))
        optimization_row = usage.to_dict()
        method_row = {
            "method": name,
            "skill_digest": skill.digest,
            "test_reward": stats.mean_reward,
            "test_improvement_over_initial": stats.mean_delta,
            "mean_reward": stats.mean_reward,
            "mean_delta": stats.mean_delta,
            "success_rate": stats.success_rate,
            "standard_error": stats.standard_error,
            "ci_low": stats.ci_low,
            "ci_high": stats.ci_high,
            "wins": stats.wins,
            "ties": stats.ties,
            "losses": stats.losses,
            "skill_length_tokens": length,
            **optimization_row,
            **{f"optimization_{key}": value for key, value in optimization_row.items()},
            **{f"reporting_{key}": value for key, value in formal_usage.items()},
            "reporting_executed_student_rollouts": (
                formal_usage["student_rollouts"]
                - formal_usage["cached_student_rollouts"]
            ),
            "reporting_executed_evaluator_calls": (
                formal_usage["evaluator_calls"] - formal_usage["cached_evaluator_calls"]
            ),
            "total_student_rollouts": (
                usage.student_rollouts + formal_usage["student_rollouts"]
            ),
            "total_editor_calls": usage.editor_calls,
            "total_evaluator_calls": (
                usage.evaluator_calls + formal_usage["evaluator_calls"]
            ),
            "total_observed_tokens": (
                usage.total_tokens + formal_usage["total_tokens"]
            ),
            "total_observed_cost_usd": (usage.cost_usd + formal_usage["cost_usd"]),
            "total_observed_wall_time_s": (
                usage.wall_time_s + formal_usage["elapsed_s"]
            ),
            "total_cached_student_rollouts": (
                usage.cached_student_rollouts + formal_usage["cached_student_rollouts"]
            ),
            "total_cached_evaluator_calls": (
                usage.cached_evaluator_calls + formal_usage["cached_evaluator_calls"]
            ),
        }
        method_row["student_rollouts"] = method_row["total_student_rollouts"]
        method_row["evaluator_calls"] = method_row["total_evaluator_calls"]
        method_rows.append(method_row)
        for result, initial_reward in zip(batch.results, initial_rewards, strict=True):
            task_rows.append(
                {
                    "method": name,
                    "split": Split.TEST.value,
                    "task_id": result.task_id,
                    "reward": result.reward,
                    "success": result.success,
                    "initial_reward": initial_reward,
                    "paired_delta": result.reward - initial_reward,
                    "skill_digest": skill.digest,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_record = {
        name: {
            "skill_digest": skill.digest,
            "skill_id": skill.skill_id,
            "length_tokens": _token_count(skill.body),
        }
        for name, skill in methods
    }
    (output_dir / "frozen_method_artifacts.json").write_text(
        json.dumps(frozen_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "method_comparison.csv", method_rows)
    _write_csv(output_dir / "task_level_scores.csv", task_rows)
    _write_csv(output_dir / "test_task_level_results.csv", task_rows)
    report = {
        "seed": report_seed,
        "test_repetitions": report_repetitions,
        "ordered_test_task_ids": list(expected_ids),
        "methods": method_rows,
        "evaluation": {"blind": True, "use_cache": False},
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ComparisonResult(tuple(method_results), batches)


def _parse_reporting_usage(
    value: Mapping[str, Any], *, method: str
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"reporting_usage.{method} must be a mapping")
    allowed = _REPORTING_INTEGER_FIELDS | _REPORTING_NUMBER_FIELDS
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"reporting_usage.{method} has unknown fields: {sorted(unknown)}"
        )
    parsed: dict[str, int | float] = {name: 0 for name in _REPORTING_INTEGER_FIELDS}
    parsed.update({name: 0.0 for name in _REPORTING_NUMBER_FIELDS})
    for name in _REPORTING_INTEGER_FIELDS:
        if name in value:
            parsed[name] = _nonnegative_integer_like(
                f"reporting_usage.{method}.{name}", value[name]
            )
    for name in _REPORTING_NUMBER_FIELDS:
        if name in value:
            parsed[name] = _nonnegative_number(
                f"reporting_usage.{method}.{name}", value[name]
            )
    return parsed


def _validate_batch_usage(
    value: Mapping[str, Any], *, method: str, expected_rollouts: int
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"formal batch usage for {method} must be a mapping")
    unknown = set(value) - _BATCH_USAGE_FIELDS
    if unknown:
        raise ValueError(
            f"formal batch usage for {method} has unknown fields: {sorted(unknown)}"
        )
    if "student_rollouts" in value:
        actual_rollouts = _nonnegative_integer(
            f"formal batch usage for {method}.student_rollouts",
            value["student_rollouts"],
        )
        if actual_rollouts != expected_rollouts:
            raise ValueError(
                f"formal batch usage for {method} reported "
                f"student_rollouts={actual_rollouts}, expected {expected_rollouts}"
            )
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if name in value:
            _nonnegative_integer(f"formal batch usage for {method}.{name}", value[name])
    if "trajectory_total_tokens" in value:
        _nonnegative_integer(
            f"formal batch usage for {method}.trajectory_total_tokens",
            value["trajectory_total_tokens"],
        )
    for name in ("cost_usd", "elapsed_s"):
        if name in value:
            _nonnegative_number(f"formal batch usage for {method}.{name}", value[name])


def _formal_reporting_usage(
    prior: Mapping[str, int | float],
    batch: EvaluationBatch,
    *,
    test_rollouts: int,
) -> dict[str, int | float]:
    usage = dict(prior)
    usage["student_rollouts"] += test_rollouts
    usage["evaluator_calls"] += 1
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        usage[name] += int(batch.usage.get(name, 0))
    for name in ("cost_usd", "elapsed_s"):
        usage[name] += float(batch.usage.get(name, 0.0))
    return usage


def _task_id(task: Any) -> str:
    value = (
        task.get("task_id")
        if isinstance(task, Mapping)
        else getattr(task, "task_id", None)
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every formal test task must have a non-empty task_id")
    return value


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _integer(name: str, value: Any) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _nonnegative_integer_like(name: str, value: Any) -> int:
    if type(value) is int:
        return _nonnegative_integer(name, value)
    if type(value) is float and math.isfinite(value) and value.is_integer():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return int(value)
    raise TypeError(f"{name} must be a nonnegative integer")


def _number(name: str, value: Any) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_number(name: str, value: Any) -> float:
    result = _number(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


__all__ = [
    "ComparisonResult",
    "MethodResult",
    "PairedStats",
    "ResourceUsage",
    "paired_bootstrap_ci",
    "paired_statistics",
    "run_frozen_report",
]

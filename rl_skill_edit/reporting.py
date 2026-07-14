from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .types import EvaluationBatch, SkillArtifact, Split, TaskResult


_METHOD_NAMES = ("initial_skill", "rl_skill_edit")
_REPORT_FILE_NAMES = (
    "frozen_method_artifacts.json",
    "method_comparison.csv",
    "task_level_scores.csv",
    "test_task_level_results.csv",
    "comparison_report.json",
)
_RESOURCE_INTEGER_FIELDS = (
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
)
_RESOURCE_NUMBER_FIELDS = (
    "wall_time_seconds",
    "cost_usd",
    "rollout_cost_usd",
    "editor_cost_usd",
    "wall_time_s",
)
_REPORTING_INTEGER_FIELDS = (
    "student_rollouts",
    "evaluator_calls",
    "cached_student_rollouts",
    "cached_evaluator_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
_REPORTING_NUMBER_FIELDS = ("cost_usd", "elapsed_s")
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
_BATCH_REQUIRED_USAGE_FIELDS = _BATCH_USAGE_FIELDS - {"trajectory_total_tokens"}


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
        payload = dict(value)
        if len(payload) == 0:
            return cls()
        allowed = set(_RESOURCE_INTEGER_FIELDS) | set(_RESOURCE_NUMBER_FIELDS)
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"optimization_usage is missing fields: {sorted(missing)}")
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"optimization_usage has unknown fields: {sorted(unknown)}"
            )
        parsed: dict[str, int | float] = {}
        for name in _RESOURCE_INTEGER_FIELDS:
            parsed[name] = _nonnegative_integer(
                f"optimization_usage.{name}", payload[name]
            )
        for name in _RESOURCE_NUMBER_FIELDS:
            parsed[name] = _nonnegative_number(
                f"optimization_usage.{name}", payload[name]
            )
        _validate_usage_totals("optimization_usage", parsed)
        _validate_cached_count(
            "optimization_usage",
            parsed,
            cached_name="cached_student_rollouts",
            total_name="student_rollouts",
        )
        _validate_cached_count(
            "optimization_usage",
            parsed,
            cached_name="cached_editor_calls",
            total_name="editor_calls",
        )
        _validate_cached_count(
            "optimization_usage",
            parsed,
            cached_name="cached_evaluator_calls",
            total_name="evaluator_calls",
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
    initial_array = _reward_array("initial rewards", initial)
    candidate_array = _reward_array("candidate rewards", candidate)
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
    initial_array = _reward_array("initial rewards", initial)
    candidate_array = _reward_array("candidate rewards", candidate)
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
        tuple(float(value) for value in initial_array),
        tuple(float(value) for value in candidate_array),
        samples=samples,
        seed=seed,
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
    reporting_payload = dict(reporting_usage)
    if len(reporting_payload) == 0:
        parsed_reporting_usage = {
            name: _zero_reporting_usage() for name in _METHOD_NAMES
        }
    else:
        missing_methods = set(_METHOD_NAMES) - set(reporting_payload)
        if missing_methods:
            raise ValueError(
                f"reporting_usage is missing methods: {sorted(missing_methods)}"
            )
        unknown_methods = set(reporting_payload) - set(_METHOD_NAMES)
        if unknown_methods:
            raise ValueError(
                f"reporting_usage has unknown methods: {sorted(unknown_methods)}"
            )
        parsed_reporting_usage = {
            name: _parse_reporting_usage(reporting_payload[name], method=name)
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
        if type(batch) is not EvaluationBatch:
            raise TypeError(
                f"formal evaluator returned a non-EvaluationBatch for {name}"
            )
        _validate_blind_batch(
            batch,
            method=name,
            expected_ids=expected_ids,
            repetitions=report_repetitions,
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

    frozen_record = {
        name: {
            "skill_digest": skill.digest,
            "skill_id": skill.skill_id,
            "length_tokens": _token_count(skill.body),
        }
        for name, skill in methods
    }
    report = {
        "seed": report_seed,
        "test_repetitions": report_repetitions,
        "ordered_test_task_ids": list(expected_ids),
        "methods": method_rows,
        "evaluation": {"blind": True, "use_cache": False},
    }
    _write_report_bundle(
        output_dir,
        frozen_record=frozen_record,
        method_rows=method_rows,
        task_rows=task_rows,
        report=report,
    )
    return ComparisonResult(tuple(method_results), batches)


def _parse_reporting_usage(
    value: Mapping[str, Any], *, method: str
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"reporting_usage.{method} must be a mapping")
    payload = dict(value)
    allowed = set(_REPORTING_INTEGER_FIELDS) | set(_REPORTING_NUMBER_FIELDS)
    missing = allowed - set(payload)
    if missing:
        raise ValueError(
            f"reporting_usage.{method} is missing fields: {sorted(missing)}"
        )
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"reporting_usage.{method} has unknown fields: {sorted(unknown)}"
        )
    parsed: dict[str, int | float] = {}
    for name in _REPORTING_INTEGER_FIELDS:
        parsed[name] = _nonnegative_integer(
            f"reporting_usage.{method}.{name}", payload[name]
        )
    for name in _REPORTING_NUMBER_FIELDS:
        parsed[name] = _nonnegative_number(
            f"reporting_usage.{method}.{name}", payload[name]
        )
    prefix = f"reporting_usage.{method}"
    _validate_usage_totals(prefix, parsed)
    _validate_cached_count(
        prefix,
        parsed,
        cached_name="cached_student_rollouts",
        total_name="student_rollouts",
    )
    _validate_cached_count(
        prefix,
        parsed,
        cached_name="cached_evaluator_calls",
        total_name="evaluator_calls",
    )
    return parsed


def _zero_reporting_usage() -> dict[str, int | float]:
    return {
        **{name: 0 for name in _REPORTING_INTEGER_FIELDS},
        **{name: 0.0 for name in _REPORTING_NUMBER_FIELDS},
    }


def _validate_blind_batch(
    batch: EvaluationBatch,
    *,
    method: str,
    expected_ids: tuple[str, ...],
    repetitions: int,
    expected_rollouts: int,
) -> None:
    if batch.split is not Split.TEST:
        raise RuntimeError(f"formal evaluator returned a non-test batch for {method}")
    if type(batch.cache_hit) is not bool:
        raise TypeError(f"formal batch for {method}.cache_hit must be a boolean")
    if batch.cache_hit:
        raise RuntimeError(f"formal evaluator returned cached results for {method}")
    if type(batch.results) is not tuple:
        raise TypeError(f"formal batch for {method}.results must be a tuple")
    if len(batch.results) != len(expected_ids):
        raise ValueError(
            f"formal batch for {method}.results must contain {len(expected_ids)} items"
        )
    for index, (result, expected_id) in enumerate(
        zip(batch.results, expected_ids, strict=True)
    ):
        prefix = f"formal batch for {method}.results[{index}]"
        if type(result) is not TaskResult:
            raise TypeError(f"{prefix} must be a TaskResult")
        if type(result.task_id) is not str:
            raise TypeError(f"{prefix}.task_id must be text")
        if not result.task_id.strip():
            raise ValueError(f"{prefix}.task_id must be non-empty")
        if result.task_id != expected_id:
            raise RuntimeError(f"formal test task order changed for {method}")
        reward = _unit_number(f"{prefix}.reward", result.reward)
        if type(result.success) is not bool:
            raise TypeError(f"{prefix}.success must be a boolean")
        for field_name in ("feedback", "evaluator_output", "final_answer"):
            field_value = getattr(result, field_name)
            if type(field_value) is not str:
                raise TypeError(f"{prefix}.{field_name} must be text")
            if field_value:
                raise ValueError(
                    f"{prefix}.{field_name} must be empty in blind reporting"
                )
        if type(result.visible_logs) is not tuple:
            raise TypeError(f"{prefix}.visible_logs must be a tuple")
        for log_index, log in enumerate(result.visible_logs):
            if type(log) is not str:
                raise TypeError(f"{prefix}.visible_logs[{log_index}] must be text")
        if result.visible_logs:
            raise ValueError(f"{prefix}.visible_logs must be empty in blind reporting")
        if type(result.raw_rewards) is not tuple:
            raise TypeError(f"{prefix}.raw_rewards must be a tuple")
        if len(result.raw_rewards) != repetitions:
            raise ValueError(f"{prefix}.raw_rewards must contain {repetitions} values")
        raw_rewards = tuple(
            _unit_number(f"{prefix}.raw_rewards[{raw_index}]", raw_reward)
            for raw_index, raw_reward in enumerate(result.raw_rewards)
        )
        raw_mean = math.fsum(raw_rewards) / repetitions
        if not math.isclose(raw_mean, reward, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{prefix}.raw_rewards mean {raw_mean} does not equal reward {reward}"
            )

    value = batch.usage
    if type(value) is not dict:
        raise TypeError(f"formal batch usage for {method} must be a dict")
    missing = _BATCH_REQUIRED_USAGE_FIELDS - set(value)
    if missing:
        raise ValueError(
            f"formal batch usage is missing fields for {method}: {sorted(missing)}"
        )
    unknown = set(value) - _BATCH_USAGE_FIELDS
    if unknown:
        raise ValueError(
            f"formal batch usage has unknown fields for {method}: {sorted(unknown)}"
        )
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
        _nonnegative_integer(f"formal batch usage for {method}.{name}", value[name])
    if value["total_tokens"] != value["input_tokens"] + value["output_tokens"]:
        raise ValueError(
            f"formal batch usage for {method}.total_tokens must equal "
            "input_tokens plus output_tokens"
        )
    if "trajectory_total_tokens" in value:
        _nonnegative_integer(
            f"formal batch usage for {method}.trajectory_total_tokens",
            value["trajectory_total_tokens"],
        )
    for name in ("cost_usd", "elapsed_s"):
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
        usage[name] += batch.usage[name]
    for name in ("cost_usd", "elapsed_s"):
        usage[name] += batch.usage[name]
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_report_bundle(
    output_dir: Path,
    *,
    frozen_record: Mapping[str, Any],
    method_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in _REPORT_FILE_NAMES:
        target = output_dir / name
        if target.is_symlink() or (os.path.lexists(target) and not target.is_file()):
            raise ValueError(f"report target must be a file: {target}")

    with tempfile.TemporaryDirectory(
        prefix=".report-staging-", dir=output_dir
    ) as staging_name:
        staging_dir = Path(staging_name)
        _write_json(staging_dir / "frozen_method_artifacts.json", frozen_record)
        _write_csv(staging_dir / "method_comparison.csv", method_rows)
        _write_csv(staging_dir / "task_level_scores.csv", task_rows)
        _write_csv(staging_dir / "test_task_level_results.csv", task_rows)
        _write_json(staging_dir / "comparison_report.json", report)

        with tempfile.TemporaryDirectory(
            prefix=".report-backup-", dir=output_dir
        ) as backup_name:
            _commit_staged_report(output_dir, staging_dir, Path(backup_name))


def _commit_staged_report(
    output_dir: Path, staging_dir: Path, backup_dir: Path
) -> None:
    originally_present = {
        name: (output_dir / name).exists() for name in _REPORT_FILE_NAMES
    }
    backed_up: set[str] = set()
    try:
        for name in _REPORT_FILE_NAMES:
            if originally_present[name]:
                os.replace(output_dir / name, backup_dir / name)
                backed_up.add(name)
        for name in _REPORT_FILE_NAMES:
            os.replace(staging_dir / name, output_dir / name)
    except Exception as commit_error:
        rollback_errors: list[Exception] = []
        for name in reversed(_REPORT_FILE_NAMES):
            target = output_dir / name
            backup = backup_dir / name
            try:
                if name in backed_up:
                    os.replace(backup, target)
                elif not originally_present[name] and target.exists():
                    target.unlink()
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            details = "; ".join(
                f"{type(error).__name__}: {error}" for error in rollback_errors
            )
            raise RuntimeError(
                f"report commit failed and rollback was incomplete: {details}"
            ) from commit_error
        raise


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


def _validate_usage_totals(prefix: str, usage: Mapping[str, int | float]) -> None:
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise ValueError(
            f"{prefix}.total_tokens must equal input_tokens plus output_tokens"
        )


def _validate_cached_count(
    prefix: str,
    usage: Mapping[str, int | float],
    *,
    cached_name: str,
    total_name: str,
) -> None:
    if usage[cached_name] > usage[total_name]:
        raise ValueError(
            f"{prefix}.{cached_name} must not exceed {prefix}.{total_name}"
        )


def _unit_number(name: str, value: Any) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _reward_array(name: str, values: Iterable[float]) -> np.ndarray:
    return np.asarray(
        tuple(
            _unit_number(f"{name}[{index}]", value)
            for index, value in enumerate(values)
        ),
        dtype=float,
    )


__all__ = [
    "ComparisonResult",
    "MethodResult",
    "PairedStats",
    "ResourceUsage",
    "paired_bootstrap_ci",
    "paired_statistics",
    "run_frozen_report",
]

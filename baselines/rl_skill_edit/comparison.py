from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .types import EvaluationBatch, SkillArtifact, Split


@dataclass(frozen=True)
class ResourceUsage:
    student_rollouts: int = 0
    teacher_rollouts: int = 0
    reference_rollouts: int = 0
    editor_calls: int = 0
    evaluator_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    rollout_cost_usd: float = 0.0
    editor_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    edit_count: int = 0
    archived_reporting_student_rollouts: int = 0
    cost_scope: str = "optimization"
    cached_student_rollouts: int = 0
    cached_teacher_rollouts: int = 0
    cached_reference_rollouts: int = 0
    cached_editor_calls: int = 0
    cached_evaluator_calls: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ResourceUsage":
        value = value or {}
        fields = cls.__dataclass_fields__
        return cls(**{name: value[name] for name in fields if name in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportedMethod:
    skill: SkillArtifact
    usage: ResourceUsage
    provenance: dict[str, Any]


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
    initial_array = np.asarray(tuple(initial), dtype=float)
    candidate_array = np.asarray(tuple(candidate), dtype=float)
    if initial_array.shape != candidate_array.shape or initial_array.ndim != 1:
        raise ValueError(
            "paired bootstrap inputs must be aligned one-dimensional arrays"
        )
    if initial_array.size == 0:
        raise ValueError("paired bootstrap requires at least one task")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    differences = candidate_array - initial_array
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, differences.size, size=(int(samples), differences.size))
    means = differences[indices].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
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
    initial_array = np.asarray(tuple(initial), dtype=float)
    candidate_array = np.asarray(tuple(candidate), dtype=float)
    success_array = np.asarray(tuple(successes), dtype=bool)
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
    wins = int(np.sum(differences > tie_tolerance))
    losses = int(np.sum(differences < -tie_tolerance))
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_run_id(value: Any) -> str:
    normalized = str(value or "").strip()
    return normalized.removeprefix("study1_")


def load_current_method_artifact(
    skill_path: str | Path,
    history_path: str | Path,
    jsonl_path: str | Path,
) -> ImportedMethod:
    skill_source = Path(skill_path).resolve()
    history_source = Path(history_path).resolve()
    jsonl_source = Path(jsonl_path).resolve()
    skill = SkillArtifact.from_file(skill_source, skill_id="current_method")
    history = json.loads(history_source.read_text(encoding="utf-8"))
    events = _read_jsonl(jsonl_source)
    session_start = next(
        (event for event in events if event.get("event") == "session_start"), {}
    )
    history_run_id = _normalized_run_id(
        history.get("run_id") or history.get("timestamp")
    )
    jsonl_run_id = _normalized_run_id(
        session_start.get("run_id") or session_start.get("timestamp")
    )
    if history_run_id and jsonl_run_id and history_run_id != jsonl_run_id:
        raise ValueError(
            "current-method history and JSONL run IDs do not match: "
            f"{history_run_id!r} != {jsonl_run_id!r}"
        )
    provenance = {
        "skill_path": str(skill_source),
        "skill_sha256": _sha256_path(skill_source),
        "history_path": str(history_source),
        "history_sha256": _sha256_path(history_source),
        "jsonl_path": str(jsonl_source),
        "jsonl_sha256": _sha256_path(jsonl_source),
        "history_run_id": history_run_id or None,
        "jsonl_run_id": jsonl_run_id or None,
        "history_jsonl_binding": (
            "verified" if history_run_id and jsonl_run_id else "unavailable"
        ),
        "skill_history_binding": "unavailable_in_legacy_archive",
    }
    cost = dict(history.get("cost_summary") or {})
    breakdown = dict(cost.get("breakdown_by_type") or {})

    counters = {
        "student": 0,
        "teacher": 0,
        "reference": 0,
        "editor": 0,
        "evaluator": 0,
    }
    explicit_roles: set[str] = set()
    for event in events:
        if event.get("event") == "usage":
            role = str(event.get("role", "")).lower()
            explicit_roles.add(role)
            if role == "editor":
                counters[role] += int(event.get("calls", 0))
            elif role == "evaluator":
                counters[role] += int(event.get("calls", 0))
            elif role in {"student", "teacher", "reference"}:
                counters[role] += int(event.get("rollouts", 0))

    repetitions = int((history.get("hyperparams") or {}).get("B_W", 1))
    archived_reporting_rollouts = 0
    formal_event_index = len(events)
    for index, event in enumerate(events):
        event_name = event.get("event")
        is_formal = event_name == "witness_eval" and str(
            event.get("tag", "")
        ).startswith("final_holdout")
        if is_formal:
            formal_event_index = min(formal_event_index, index)
            archived_reporting_rollouts += int(event.get("n_tasks", 0)) * repetitions
            continue
        if event_name == "rollout_summary":
            mode = event.get("activation_mode")
            role = (
                "teacher"
                if mode == "teacher_no_skill"
                else "reference"
                if mode == "reference_no_skill"
                else "student"
            )
            if role not in explicit_roles:
                counters[role] += int(event.get("n", 0))
            if "evaluator" not in explicit_roles:
                counters["evaluator"] += 1
        elif event_name == "witness_eval":
            if "student" not in explicit_roles:
                counters["student"] += int(event.get("n_tasks", 0)) * repetitions
            if "evaluator" not in explicit_roles:
                counters["evaluator"] += 1
    if "editor" not in explicit_roles:
        counters["editor"] = int(
            (breakdown.get("schema_proposal") or {}).get("calls", 0)
        )

    optimization_events = events[:formal_event_index]
    wall_time = max(
        (float(event.get("t_elapsed", 0.0)) for event in optimization_events),
        default=0.0,
    )
    usage = ResourceUsage(
        student_rollouts=counters["student"],
        teacher_rollouts=counters["teacher"],
        reference_rollouts=counters["reference"],
        editor_calls=counters["editor"],
        evaluator_calls=counters["evaluator"],
        input_tokens=int(cost.get("total_input_tokens", 0)),
        output_tokens=int(cost.get("total_output_tokens", 0)),
        total_tokens=int(cost.get("total_tokens", 0)),
        cost_usd=float(cost.get("total_cost_usd", 0.0)),
        rollout_cost_usd=sum(
            float((breakdown.get(name) or {}).get("cost_usd", 0.0))
            for name in ("student_rollout", "teacher_rollout")
        ),
        editor_cost_usd=float(
            (breakdown.get("schema_proposal") or {}).get("cost_usd", 0.0)
        ),
        wall_time_s=wall_time,
        edit_count=int(history.get("total_accepted", 0)),
        archived_reporting_student_rollouts=archived_reporting_rollouts,
        cost_scope=(
            "archived_run_total_including_original_final"
            if archived_reporting_rollouts
            else "optimization"
        ),
    )
    return ImportedMethod(skill=skill, usage=usage, provenance=provenance)


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    config: Mapping[str, Any], method_names: Iterable[str], *, seed: int
) -> ComparisonResult:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = config["evaluator"]
    tasks = tuple(config["test_tasks"])
    methods: Mapping[str, SkillArtifact] = config["frozen_methods"]
    usage_by_method = config.get("method_usage") or {}
    reporting_usage_by_method = config.get("reporting_usage") or {}
    names = tuple(method_names)
    if not names or names[0] != "initial_skill":
        raise ValueError("initial_skill must be the first comparison method")
    missing = [name for name in names if name not in methods]
    if missing:
        raise ValueError(f"missing frozen method artifacts: {missing}")
    if not tasks:
        raise ValueError("formal comparison requires test tasks")

    evaluation = dict(config.get("evaluation") or {})
    repetitions = int(
        evaluation.get("test_repetitions", config.get("test_repetitions", 1))
    )
    if not bool(evaluation.get("blind", True)):
        raise ValueError("formal test comparison must be blind")
    bootstrap_samples = int(config.get("bootstrap_samples", 2000))
    method_metrics = config.get("method_metrics") or {}

    frozen_record = {
        name: {
            "skill_digest": methods[name].digest,
            "skill_id": methods[name].skill_id,
            "length_tokens": _token_count(methods[name].body),
        }
        for name in names
    }
    (output_dir / "frozen_method_artifacts.json").write_text(
        json.dumps(frozen_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluator.freeze()
    batches: dict[str, EvaluationBatch] = {}
    expected_ids = tuple(_task_id(task) for task in tasks)
    for name in names:
        batch = evaluator.evaluate(
            methods[name],
            tasks,
            split=Split.TEST,
            seed=int(seed),
            repetitions=repetitions,
            use_cache=False,
            blind=True,
        )
        if batch.ordered_task_ids != expected_ids:
            raise RuntimeError(f"formal test task order changed for {name}")
        batches[name] = batch

    initial_rewards = tuple(
        result.reward for result in batches["initial_skill"].results
    )
    method_results: list[MethodResult] = []
    method_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for method_index, name in enumerate(names):
        batch = batches[name]
        rewards = tuple(result.reward for result in batch.results)
        successes = tuple(result.success for result in batch.results)
        stats = paired_statistics(
            initial_rewards,
            rewards,
            successes=successes,
            samples=bootstrap_samples,
            seed=int(seed) + method_index,
        )
        usage = ResourceUsage.from_mapping(usage_by_method.get(name))
        reporting_usage = dict(reporting_usage_by_method.get(name) or {})
        for key, value in batch.usage.items():
            if key in {
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "elapsed_s",
            }:
                reporting_usage[key] = reporting_usage.get(key, 0) + value
        test_rollouts = len(tasks) * repetitions
        reporting_usage["student_rollouts"] = (
            int(reporting_usage.get("student_rollouts", 0)) + test_rollouts
        )
        reporting_usage["evaluator_calls"] = (
            int(reporting_usage.get("evaluator_calls", 0)) + 1
        )
        if batch.cache_hit:
            reporting_usage["cached_student_rollouts"] = (
                int(reporting_usage.get("cached_student_rollouts", 0)) + test_rollouts
            )
            reporting_usage["cached_evaluator_calls"] = (
                int(reporting_usage.get("cached_evaluator_calls", 0)) + 1
            )
        length = _token_count(methods[name].body)
        item = MethodResult(name, methods[name].digest, stats, usage, length)
        method_results.append(item)
        optimization_usage = usage.to_dict()
        method_row = {
            "method": name,
            "skill_digest": methods[name].digest,
            "train_reward": (method_metrics.get(name) or {}).get("train_reward"),
            "validation_reward": (method_metrics.get(name) or {}).get(
                "validation_reward"
            ),
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
            **optimization_usage,
            "optimization_student_rollouts": usage.student_rollouts,
            "optimization_evaluator_calls": usage.evaluator_calls,
            "optimization_cached_student_rollouts": usage.cached_student_rollouts,
            "optimization_executed_student_rollouts": (
                usage.student_rollouts - usage.cached_student_rollouts
            ),
            "optimization_cached_editor_calls": usage.cached_editor_calls,
            "optimization_executed_editor_calls": (
                usage.editor_calls - usage.cached_editor_calls
            ),
            "optimization_cached_evaluator_calls": usage.cached_evaluator_calls,
            "optimization_executed_evaluator_calls": (
                usage.evaluator_calls - usage.cached_evaluator_calls
            ),
            "reporting_student_rollouts": int(
                reporting_usage.get("student_rollouts", 0)
            ),
            "reporting_evaluator_calls": int(reporting_usage.get("evaluator_calls", 0)),
            "reporting_input_tokens": int(reporting_usage.get("input_tokens", 0)),
            "reporting_output_tokens": int(reporting_usage.get("output_tokens", 0)),
            "reporting_total_tokens": int(reporting_usage.get("total_tokens", 0)),
            "reporting_cost_usd": float(reporting_usage.get("cost_usd", 0.0)),
            "reporting_wall_time_s": float(reporting_usage.get("elapsed_s", 0.0)),
            "reporting_cached_student_rollouts": int(
                reporting_usage.get("cached_student_rollouts", 0)
            ),
            "reporting_executed_student_rollouts": int(
                reporting_usage.get("student_rollouts", 0)
            )
            - int(reporting_usage.get("cached_student_rollouts", 0)),
            "reporting_cached_evaluator_calls": int(
                reporting_usage.get("cached_evaluator_calls", 0)
            ),
            "reporting_executed_evaluator_calls": int(
                reporting_usage.get("evaluator_calls", 0)
            )
            - int(reporting_usage.get("cached_evaluator_calls", 0)),
            "total_student_rollouts": usage.student_rollouts
            + int(reporting_usage.get("student_rollouts", 0)),
            "total_evaluator_calls": usage.evaluator_calls
            + int(reporting_usage.get("evaluator_calls", 0)),
            "total_observed_tokens": usage.total_tokens
            + int(reporting_usage.get("total_tokens", 0)),
            "total_observed_cost_usd": usage.cost_usd
            + float(reporting_usage.get("cost_usd", 0.0)),
            "total_observed_wall_time_s": usage.wall_time_s
            + float(reporting_usage.get("elapsed_s", 0.0)),
            "total_cached_student_rollouts": usage.cached_student_rollouts
            + int(reporting_usage.get("cached_student_rollouts", 0)),
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
                    "skill_digest": methods[name].digest,
                }
            )

    _write_csv(output_dir / "method_comparison.csv", list(method_rows[0]), method_rows)
    task_fields = list(task_rows[0])
    _write_csv(output_dir / "task_level_scores.csv", task_fields, task_rows)
    _write_csv(output_dir / "test_task_level_results.csv", task_fields, task_rows)
    report = {
        "seed": int(seed),
        "test_repetitions": repetitions,
        "ordered_test_task_ids": list(expected_ids),
        "methods": method_rows,
        "evaluation": evaluation,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ComparisonResult(tuple(method_results), batches)


def _task_id(task: Any) -> str:
    value = (
        task.get("task_id")
        if isinstance(task, dict)
        else getattr(task, "task_id", None)
    )
    if not isinstance(value, str) or not value:
        raise ValueError("every formal test task must have a non-empty task_id")
    return value


__all__ = [
    "ComparisonResult",
    "ImportedMethod",
    "MethodResult",
    "PairedStats",
    "ResourceUsage",
    "load_current_method_artifact",
    "paired_bootstrap_ci",
    "paired_statistics",
    "run_comparison",
]

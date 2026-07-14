from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .cache import JsonFileCache
from .types import EvaluationBatch, SkillArtifact, Split, TaskResult


def _split(value: Split | str) -> Split:
    return value if isinstance(value, Split) else Split(str(value).lower())


def _task_id(task: Any) -> str:
    if isinstance(task, Mapping):
        value = task.get("task_id")
    else:
        value = getattr(task, "task_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every evaluation task must have a non-empty task_id")
    return value


def _cache_key(
    skill: SkillArtifact,
    tasks: Iterable[Any],
    split: Split,
    seed: int,
    repetitions: int,
    blind: bool,
    protocol: str,
    evaluator_signature: Mapping[str, Any],
) -> str:
    payload = {
        "protocol": protocol,
        "skill_digest": skill.digest,
        "ordered_tasks": [_task_cache_identity(task) for task in tasks],
        "split": split.value,
        "seed": int(seed),
        "repetitions": int(repetitions),
        "blind": bool(blind),
        "evaluator": dict(evaluator_signature),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(value: Any) -> dict[str, Any]:
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _task_cache_identity(task: Any) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        return {"task_id": _task_id(task), "type": type(task).__qualname__}
    if task.get("_entity_fingerprint") and task.get("_content_fingerprint"):
        return {
            "task_id": _task_id(task),
            "entity_fingerprint": str(task["_entity_fingerprint"]),
            "content_fingerprint": str(task["_content_fingerprint"]),
        }
    normalized = json.loads(json.dumps(dict(task), ensure_ascii=False, sort_keys=True))
    spreadsheet = normalized.get("spreadsheet")
    if isinstance(spreadsheet, dict):
        for key in ("init_file", "golden_file"):
            if key in spreadsheet:
                spreadsheet[key] = _file_identity(spreadsheet[key])
    return normalized


def _batch_to_dict(batch: EvaluationBatch) -> dict[str, Any]:
    return {
        "split": batch.split.value,
        "results": [result.to_dict() for result in batch.results],
        "usage": dict(batch.usage),
    }


def _strict_unit_number(value: Any, name: str) -> int | float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a JSON number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _strict_usage(payload: Any) -> dict[str, int | float]:
    if type(payload) is not dict:
        raise TypeError("cached usage must be a JSON object")
    integer_fields = {
        "student_rollouts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    optional_integer_fields = {"trajectory_total_tokens"}
    number_fields = {"cost_usd", "elapsed_s"}
    required_fields = integer_fields | number_fields
    missing = required_fields - payload.keys()
    if missing:
        raise ValueError(f"cached usage is missing fields: {sorted(missing)}")
    unknown = payload.keys() - required_fields - optional_integer_fields
    if unknown:
        raise ValueError(f"cached usage has unknown fields: {sorted(unknown)}")
    for field_name in integer_fields | optional_integer_fields:
        if field_name not in payload:
            continue
        value = payload[field_name]
        if type(value) is not int or value < 0:
            raise TypeError(f"cached usage.{field_name} must be a non-negative integer")
    for field_name in number_fields:
        value = payload[field_name]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0.0:
            raise TypeError(
                f"cached usage.{field_name} must be finite and non-negative"
            )
    return dict(payload)


def _batch_from_dict(payload: dict[str, Any], *, cache_hit: bool) -> EvaluationBatch:
    if type(payload) is not dict:
        raise TypeError("cached evaluation must be a JSON object")
    split_value = payload.get("split")
    if type(split_value) is not str:
        raise TypeError("cached split must be a string")
    result_payloads = payload.get("results")
    if type(result_payloads) is not list:
        raise TypeError("cached results must be a JSON array")
    results = []
    for index, item in enumerate(result_payloads):
        if type(item) is not dict:
            raise TypeError(f"cached results[{index}] must be a JSON object")
        task_id = item.get("task_id")
        if type(task_id) is not str or not task_id.strip():
            raise ValueError(f"cached results[{index}].task_id must be non-empty text")
        success = item.get("success")
        if type(success) is not bool:
            raise TypeError(f"cached results[{index}].success must be a boolean")
        text_fields = {}
        for field_name in ("feedback", "evaluator_output", "final_answer"):
            value = item.get(field_name)
            if type(value) is not str:
                raise TypeError(f"cached results[{index}].{field_name} must be text")
            text_fields[field_name] = value
        visible_logs = item.get("visible_logs")
        if type(visible_logs) is not list or not all(
            type(value) is str for value in visible_logs
        ):
            raise TypeError(
                f"cached results[{index}].visible_logs must be a text array"
            )
        raw_reward_values = item.get("raw_rewards")
        if type(raw_reward_values) is not list:
            raise TypeError(
                f"cached results[{index}].raw_rewards must be a number array"
            )
        raw_rewards = tuple(
            _strict_unit_number(
                value,
                f"cached results[{index}].raw_rewards[{reward_index}]",
            )
            for reward_index, value in enumerate(raw_reward_values)
        )
        results.append(
            TaskResult(
                task_id=task_id,
                reward=_strict_unit_number(
                    item.get("reward"),
                    f"cached results[{index}].reward",
                ),
                success=success,
                feedback=text_fields["feedback"],
                evaluator_output=text_fields["evaluator_output"],
                final_answer=text_fields["final_answer"],
                visible_logs=tuple(visible_logs),
                raw_rewards=raw_rewards,
            )
        )
    usage = _strict_usage(payload.get("usage"))
    cache_hit_usage = {
        "student_rollouts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "elapsed_s": 0.0,
    }
    if "trajectory_total_tokens" in usage:
        cache_hit_usage["trajectory_total_tokens"] = 0
    return EvaluationBatch(
        split=Split(split_value),
        results=tuple(results),
        cache_hit=cache_hit,
        usage=cache_hit_usage if cache_hit else usage,
    )


class MockSkillEvaluator:
    """Deterministic evaluator used by tests and the zero-cost smoke run."""

    def __init__(
        self,
        score_fn: Callable[[SkillArtifact, Any, int, int], TaskResult],
        cache: JsonFileCache | None = None,
        cache_signature: Mapping[str, Any] | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.cache = cache
        self.cache_signature = dict(
            cache_signature
            or {
                "score_fn_module": getattr(score_fn, "__module__", ""),
                "score_fn_name": getattr(
                    score_fn, "__qualname__", type(score_fn).__qualname__
                ),
            }
        )
        json.dumps(self.cache_signature, sort_keys=True)
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def cache_will_hit(
        self,
        skill: SkillArtifact,
        tasks: Iterable[Any],
        split: Split | str,
        seed: int,
        repetitions: int,
        blind: bool,
    ) -> bool:
        if self.cache is None:
            return False
        split = _split(split)
        tasks = tuple(tasks)
        key = _cache_key(
            skill,
            tasks,
            split,
            seed,
            repetitions,
            blind,
            "rl-skill-edit-mock-v2",
            self.cache_signature,
        )
        return self.cache.get("rollout", key) is not None

    def evaluate(
        self,
        skill: SkillArtifact,
        tasks: Iterable[Any],
        split: Split | str,
        seed: int,
        repetitions: int,
        use_cache: bool,
        blind: bool,
    ) -> EvaluationBatch:
        split = _split(split)
        tasks = tuple(tasks)
        if split is Split.TEST and not self._frozen:
            raise RuntimeError("formal test evaluation requires freeze")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")

        key = _cache_key(
            skill,
            tasks,
            split,
            seed,
            repetitions,
            blind,
            "rl-skill-edit-mock-v2",
            self.cache_signature,
        )
        if use_cache and self.cache is not None:
            cached = self.cache.get("rollout", key)
            if cached is not None:
                return _batch_from_dict(cached, cache_hit=True)

        aggregated: list[TaskResult] = []
        for task in tasks:
            repeated = tuple(
                self.score_fn(skill, task, repetition, int(seed))
                for repetition in range(repetitions)
            )
            expected_id = _task_id(task)
            if any(item.task_id != expected_id for item in repeated):
                raise ValueError(
                    "score_fn returned a task_id that does not match its task"
                )
            rewards = tuple(float(item.reward) for item in repeated)
            representative = repeated[-1]
            aggregated.append(
                replace(
                    representative,
                    reward=sum(rewards) / len(rewards),
                    success=(
                        sum(bool(item.success) for item in repeated) / len(repeated)
                    )
                    >= 0.5,
                    raw_rewards=rewards,
                )
            )
        batch = EvaluationBatch(
            split=split,
            results=tuple(aggregated),
            cache_hit=False,
            usage={
                "student_rollouts": len(tasks) * repetitions,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            },
        )
        if use_cache and self.cache is not None:
            self.cache.set("rollout", key, _batch_to_dict(batch))
        return batch


def select_score(
    hard: float,
    soft: float,
    metric: str,
    mixed_weight: float,
) -> float:
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        weight = min(1.0, max(0.0, float(mixed_weight)))
        return (1.0 - weight) * float(hard) + weight * float(soft)
    raise ValueError(f"unknown metric: {metric}")


class SpreadsheetSkillEvaluator:
    def __init__(
        self,
        student: Any,
        *,
        cache: JsonFileCache | None,
        gate_metric: str,
        gate_mixed_weight: float,
        success_threshold: float = 0.8,
    ) -> None:
        if gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError("gate_metric must be hard, soft, or mixed")
        self.student = student
        self.cache = cache
        self.gate_metric = gate_metric
        self.gate_mixed_weight = float(gate_mixed_weight)
        self.success_threshold = float(success_threshold)
        self.cache_signature = {
            "adapter": "SpreadsheetSkillEvaluator-v1",
            "student_model": str(getattr(student, "model", "")),
            "student_temperature": float(getattr(student, "temperature", 0.0)),
            "student_max_tokens": int(getattr(student, "max_tokens", 0)),
            "student_max_steps": int(getattr(student, "max_steps", 0)),
            "activation_mode": "forced-skill-only",
            "gate_metric": self.gate_metric,
            "gate_mixed_weight": self.gate_mixed_weight,
            "success_threshold": self.success_threshold,
            "blind_protocol": "single-call-hide-answer-metadata-v1",
        }
        json.dumps(self.cache_signature, sort_keys=True, allow_nan=False)
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def cache_will_hit(
        self,
        skill: SkillArtifact,
        tasks: Iterable[Mapping[str, Any]],
        split: Split | str,
        seed: int,
        repetitions: int,
        blind: bool,
    ) -> bool:
        split = _split(split)
        if split is Split.TEST and not self._frozen:
            raise RuntimeError("formal test evaluation requires freeze")
        if self.cache is None:
            return False
        tasks = tuple(tasks)
        key = _cache_key(
            skill,
            tasks,
            split,
            seed,
            repetitions,
            blind,
            "rl-skill-edit-spreadsheet-v1",
            self.cache_signature,
        )
        cached = self.cache.get("rollout", key)
        if cached is None:
            return False
        _validated_cached_batch(
            cached,
            split=split,
            tasks=tasks,
            repetitions=repetitions,
            success_threshold=self.success_threshold,
        )
        return True

    def evaluate(
        self,
        skill: SkillArtifact,
        tasks: Iterable[Mapping[str, Any]],
        split: Split | str,
        seed: int,
        repetitions: int,
        use_cache: bool,
        blind: bool,
    ) -> EvaluationBatch:
        split = _split(split)
        if split is Split.TEST and not self._frozen:
            raise RuntimeError("formal test evaluation requires freeze")
        tasks = tuple(tasks)
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not tasks:
            raise ValueError("evaluation task bundle must not be empty")

        key = _cache_key(
            skill,
            tasks,
            split,
            seed,
            repetitions,
            blind,
            "rl-skill-edit-spreadsheet-v1",
            self.cache_signature,
        )
        if use_cache and self.cache is not None:
            cached = self.cache.get("rollout", key)
            if cached is not None:
                return _validated_cached_batch(
                    cached,
                    split=split,
                    tasks=tasks,
                    repetitions=repetitions,
                    success_threshold=self.success_threshold,
                )

        client = getattr(self.student, "client", None)
        has_client_usage = client is not None and all(
            hasattr(client, field)
            for field in (
                "total_input_tokens",
                "total_output_tokens",
                "total_cost_usd",
            )
        )
        input_before = int(client.total_input_tokens) if has_client_usage else 0
        output_before = int(client.total_output_tokens) if has_client_usage else 0
        cost_before = float(client.total_cost_usd) if has_client_usage else 0.0
        started = time.monotonic()
        results: list[TaskResult] = []
        total_tokens = 0
        total_cost_usd = 0.0
        for task_index, task in enumerate(tasks):
            expected_task_id = _task_id(task)
            trajectory_list = []
            for repetition in range(repetitions):
                trajectory = self.student.run_task(
                    task,
                    skill,
                    blind=blind,
                    seed=int(seed) + task_index * repetitions + repetition,
                )
                error = _trajectory_bundle_error(trajectory, expected_task_id)
                if error:
                    raise RuntimeError(
                        "incomplete evaluation bundle: "
                        f"task={expected_task_id} repetition={repetition}: {error}"
                    )
                trajectory_list.append(trajectory)
            trajectories = tuple(trajectory_list)
            raw_rewards = tuple(
                select_score(
                    trajectory.hard_reward,
                    trajectory.soft_reward,
                    self.gate_metric,
                    self.gate_mixed_weight,
                )
                for trajectory in trajectories
            )
            reward = sum(raw_rewards) / len(raw_rewards)
            representative = trajectories[-1]
            total_tokens += sum(
                int(trajectory.total_tokens) for trajectory in trajectories
            )
            total_cost_usd += sum(
                float(trajectory.total_cost_usd) for trajectory in trajectories
            )
            results.append(
                TaskResult(
                    task_id=_task_id(task),
                    reward=reward,
                    success=reward >= self.success_threshold,
                    feedback=(
                        ""
                        if blind
                        else f"reward={reward:.6f}; "
                        f"success={reward >= self.success_threshold}"
                    ),
                    final_answer="" if blind else representative.final_answer,
                    visible_logs=() if blind else representative.visible_logs,
                    raw_rewards=raw_rewards,
                )
            )
        if has_client_usage:
            input_tokens = int(client.total_input_tokens) - input_before
            output_tokens = int(client.total_output_tokens) - output_before
            client_cost_usd = float(client.total_cost_usd) - cost_before
            if (
                input_tokens < 0
                or output_tokens < 0
                or not math.isfinite(client_cost_usd)
                or client_cost_usd < 0.0
            ):
                raise RuntimeError("incomplete evaluation bundle: invalid client usage")
        else:
            input_tokens = 0
            output_tokens = 0
            client_cost_usd = total_cost_usd
        batch = EvaluationBatch(
            split=split,
            results=tuple(results),
            cache_hit=False,
            usage={
                "student_rollouts": len(tasks) * repetitions,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens if has_client_usage else total_tokens
                ),
                "trajectory_total_tokens": total_tokens,
                "cost_usd": client_cost_usd,
                "elapsed_s": time.monotonic() - started,
            },
        )
        if use_cache and self.cache is not None:
            self.cache.set("rollout", key, _batch_to_dict(batch))
        return batch


def _trajectory_bundle_error(trajectory: Any, expected_task_id: str) -> str:
    if getattr(trajectory, "task_id", None) != expected_task_id:
        return "trajectory task_id does not match its task"
    if getattr(trajectory, "evaluation_valid", None) is not True:
        reason = str(getattr(trajectory, "invalid_reason", "") or "invalid rollout")
        return reason
    for field_name in ("hard_reward", "soft_reward"):
        value = getattr(trajectory, field_name, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            return f"invalid {field_name}"
    total_tokens = getattr(trajectory, "total_tokens", None)
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < 0
    ):
        return "invalid total_tokens"
    total_cost = getattr(trajectory, "total_cost_usd", None)
    if (
        isinstance(total_cost, bool)
        or not isinstance(total_cost, (int, float))
        or not math.isfinite(float(total_cost))
        or float(total_cost) < 0.0
    ):
        return "invalid total_cost_usd"
    if not isinstance(getattr(trajectory, "final_answer", None), str):
        return "invalid final_answer"
    if not isinstance(getattr(trajectory, "invalid_reason", None), str):
        return "invalid invalid_reason"
    visible_logs = getattr(trajectory, "visible_logs", None)
    if not isinstance(visible_logs, tuple) or not all(
        isinstance(log, str) for log in visible_logs
    ):
        return "invalid visible_logs"
    return ""


def _cached_bundle_error(
    batch: EvaluationBatch,
    *,
    split: Split,
    tasks: tuple[Mapping[str, Any], ...],
    repetitions: int,
    success_threshold: float,
) -> str:
    if batch.split is not split:
        return "cached split does not match request"
    expected_ids = tuple(_task_id(task) for task in tasks)
    if batch.ordered_task_ids != expected_ids:
        return "cached task order does not match request"
    for result in batch.results:
        if len(result.raw_rewards) != repetitions:
            return f"cached repetition count is incomplete for {result.task_id}"
        if not 0.0 <= result.reward <= 1.0:
            return f"cached reward is outside [0, 1] for {result.task_id}"
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in result.raw_rewards
        ):
            return f"cached reward is non-finite for {result.task_id}"
        expected_reward = sum(result.raw_rewards) / repetitions
        if result.reward != expected_reward:
            return f"cached mean reward is inconsistent for {result.task_id}"
        if result.success is not (result.reward >= success_threshold):
            return f"cached success flag is inconsistent for {result.task_id}"
    return ""


def _validated_cached_batch(
    payload: Any,
    *,
    split: Split,
    tasks: tuple[Mapping[str, Any], ...],
    repetitions: int,
    success_threshold: float,
) -> EvaluationBatch:
    try:
        batch = _batch_from_dict(payload, cache_hit=True)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(f"incomplete cached evaluation bundle: {exc}") from exc
    cache_error = _cached_bundle_error(
        batch,
        split=split,
        tasks=tasks,
        repetitions=repetitions,
        success_threshold=success_threshold,
    )
    if cache_error:
        raise RuntimeError(f"incomplete cached evaluation bundle: {cache_error}")
    return batch


__all__ = [
    "MockSkillEvaluator",
    "SpreadsheetSkillEvaluator",
    "select_score",
]

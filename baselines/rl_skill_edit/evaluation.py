from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.evaluator import select_gate_score

from .cache import JsonFileCache
from .types import EvaluationBatch, SkillArtifact, Split, TaskResult


def _split(value: Split | str) -> Split:
    return value if isinstance(value, Split) else Split(str(value).lower())


def _task_id(task: Any) -> str:
    if isinstance(task, dict):
        value = task.get("task_id")
    else:
        value = getattr(task, "task_id", None)
    if not isinstance(value, str) or not value:
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
    if not isinstance(task, dict):
        return {"task_id": _task_id(task), "type": type(task).__qualname__}
    if task.get("_entity_fingerprint") and task.get("_content_fingerprint"):
        return {
            "task_id": _task_id(task),
            "entity_fingerprint": str(task["_entity_fingerprint"]),
            "content_fingerprint": str(task["_content_fingerprint"]),
        }
    normalized = json.loads(json.dumps(task, ensure_ascii=False, sort_keys=True))
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


def _batch_from_dict(payload: dict[str, Any], *, cache_hit: bool) -> EvaluationBatch:
    results = tuple(
        TaskResult(
            task_id=str(item["task_id"]),
            reward=float(item["reward"]),
            success=bool(item["success"]),
            feedback=str(item.get("feedback", "")),
            evaluator_output=str(item.get("evaluator_output", "")),
            final_answer=str(item.get("final_answer", "")),
            visible_logs=tuple(str(value) for value in item.get("visible_logs", ())),
            raw_rewards=tuple(float(value) for value in item.get("raw_rewards", ())),
        )
        for item in payload["results"]
    )
    return EvaluationBatch(
        split=Split(payload["split"]),
        results=results,
        cache_hit=cache_hit,
        usage=(
            {
                "student_rollouts": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
            }
            if cache_hit
            else dict(payload.get("usage") or {})
        ),
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


class RepositorySkillEvaluator:
    """Adapter from the repository's frozen Student/Evaluator to RL task batches."""

    def __init__(
        self,
        evaluator,
        *,
        cache: JsonFileCache | None,
        gate_metric: str,
        gate_mixed_weight: float,
        success_threshold: float = 0.8,
    ) -> None:
        if gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError("gate_metric must be hard, soft, or mixed")
        self.evaluator = evaluator
        self.cache = cache
        self.gate_metric = gate_metric
        self.gate_mixed_weight = float(gate_mixed_weight)
        self.success_threshold = float(success_threshold)
        agent = evaluator.agent
        self.cache_signature = {
            "adapter": "RepositorySkillEvaluator-v2",
            "student_model": str(getattr(agent, "model", "")),
            "student_temperature": float(getattr(agent, "temp", 0.0)),
            "student_max_tokens": int(getattr(agent, "max_tok", 0)),
            "student_max_steps": int(getattr(agent, "max_steps", 0)),
            "activation_mode": "harness",
            "gate_metric": self.gate_metric,
            "gate_mixed_weight": self.gate_mixed_weight,
            "success_threshold": self.success_threshold,
            "blind_protocol": "hide-answer-metadata-and-verifier-retries-v1",
        }
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
            "rl-skill-edit-repo-v2",
            self.cache_signature,
        )
        return self.cache.get("rollout", key) is not None

    def evaluate(
        self,
        skill: SkillArtifact,
        tasks: Iterable[dict[str, Any]],
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
        if not tasks:
            raise ValueError("evaluation task bundle must not be empty")

        key = _cache_key(
            skill,
            tasks,
            split,
            seed,
            repetitions,
            blind,
            "rl-skill-edit-repo-v2",
            self.cache_signature,
        )
        if use_cache and self.cache is not None:
            cached = self.cache.get("rollout", key)
            if cached is not None:
                return _batch_from_dict(cached, cache_hit=True)

        library = skill.to_library()
        agent = self.evaluator.agent
        client = getattr(agent, "client", None)
        input_before = int(getattr(client, "total_input_tokens", 0))
        output_before = int(getattr(client, "total_output_tokens", 0))
        cost_before = float(getattr(client, "total_cost_usd", 0.0))
        started = time.monotonic()
        if hasattr(agent, "clear_cache"):
            agent.clear_cache()
        estimate = self.evaluator.witness_estimate(
            library,
            list(tasks),
            B_W=repetitions,
            desc=f"RL-Skill-Edit {split.value}",
            activation_mode="harness",
            forced_skill_id=skill.skill_id,
            return_trajectories=True,
            require_complete=True,
            verifier_feedback=not blind,
            expose_answer_metadata=not blind,
            seed=int(seed),
        )
        elapsed = time.monotonic() - started
        input_tokens = int(getattr(client, "total_input_tokens", 0)) - input_before
        output_tokens = int(getattr(client, "total_output_tokens", 0)) - output_before
        cost_usd = float(getattr(client, "total_cost_usd", 0.0)) - cost_before
        hard_means = estimate["per_task_means"]
        soft_means = estimate["per_task_soft_means"]
        hard_raw = estimate["per_task_rewards"]
        soft_raw = estimate["per_task_soft_rewards"]
        trajectories = estimate["per_task_trajectories"]
        if not (
            len(tasks)
            == len(hard_means)
            == len(soft_means)
            == len(hard_raw)
            == len(soft_raw)
            == len(trajectories)
        ):
            raise RuntimeError("repository evaluator returned an unaligned task bundle")

        results: list[TaskResult] = []
        total_tokens = 0
        total_cost = 0.0
        for index, task in enumerate(tasks):
            raw_rewards = tuple(
                select_gate_score(hard, soft, self.gate_metric, self.gate_mixed_weight)
                for hard, soft in zip(hard_raw[index], soft_raw[index], strict=True)
            )
            reward = select_gate_score(
                hard_means[index],
                soft_means[index],
                self.gate_metric,
                self.gate_mixed_weight,
            )
            task_trajectories = tuple(trajectories[index])
            for trajectory in task_trajectories:
                total_tokens += int(getattr(trajectory, "total_tokens", 0))
                total_cost += float(getattr(trajectory, "total_cost_usd", 0.0))

            feedback = ""
            evaluator_output = ""
            final_answer = ""
            visible_logs: tuple[str, ...] = ()
            if not blind and task_trajectories:
                trajectory = task_trajectories[-1]
                final_answer = (
                    str(trajectory.steps[-1].action) if trajectory.steps else ""
                )
                signals = tuple(
                    json.dumps(signal, ensure_ascii=False, sort_keys=True)
                    for signal in getattr(trajectory, "step_execution_signals", ())
                )
                visible_logs = signals
                detail = dict(getattr(trajectory, "score_detail", {}) or {})
                detail.pop("answer_pos", None)
                detail.pop("answer_sheet", None)
                evaluator_output = json.dumps(
                    detail, ensure_ascii=False, sort_keys=True
                )
                feedback = (
                    f"reward={reward:.6f}; success={reward >= self.success_threshold}"
                )
            results.append(
                TaskResult(
                    task_id=_task_id(task),
                    reward=float(reward),
                    success=bool(reward >= self.success_threshold),
                    feedback=feedback,
                    evaluator_output=evaluator_output,
                    final_answer=final_answer,
                    visible_logs=visible_logs,
                    raw_rewards=raw_rewards,
                )
            )

        batch = EvaluationBatch(
            split=split,
            results=tuple(results),
            cache_hit=False,
            usage={
                "student_rollouts": len(tasks) * repetitions,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "trajectory_total_tokens": total_tokens,
                "cost_usd": cost_usd if client is not None else total_cost,
                "elapsed_s": elapsed,
            },
        )
        if use_cache and self.cache is not None:
            self.cache.set("rollout", key, _batch_to_dict(batch))
        return batch


__all__ = ["MockSkillEvaluator", "RepositorySkillEvaluator"]

from __future__ import annotations

import csv
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.rl_skill_edit.action_space import EditOperator
from baselines.rl_skill_edit.cache import JsonFileCache
from baselines.rl_skill_edit.evaluation import MockSkillEvaluator
from baselines.rl_skill_edit.optimizer import RLSkillEditOptimizer
from baselines.rl_skill_edit.types import (
    EditPatch,
    EvaluationBatch,
    GeneratedPatch,
    SkillArtifact,
    Split,
    TaskResult,
)


REWRITE_RULE_INDEX = len(EditOperator) + list(EditOperator).index(
    EditOperator.REWRITE_RULE
)
STOP_INDEX = list(EditOperator).index(EditOperator.STOP)


@dataclass(frozen=True)
class FakeTask:
    task_id: str


def _task_id(task) -> str:
    if hasattr(task, "task_id"):
        return str(task.task_id)
    if isinstance(task, dict):
        return str(task["task_id"])
    raise TypeError(f"unsupported task: {task!r}")


def _split_name(split) -> str:
    return str(getattr(split, "value", split)).lower()


class RecordingEvaluator:
    """Deterministic evaluator double; optimizer orchestration remains real."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def evaluate(
        self,
        skill,
        tasks,
        split,
        seed,
        repetitions,
        use_cache,
        blind,
    ) -> EvaluationBatch:
        split_name = _split_name(split)
        if split_name == "test" and not self.frozen:
            raise RuntimeError("test evaluation requires freeze")

        ordered_ids = tuple(_task_id(task) for task in tasks)
        self.calls.append(
            {
                "skill_body": skill.body,
                "task_ids": ordered_ids,
                "split": split_name,
                "seed": seed,
                "repetitions": repetitions,
                "use_cache": use_cache,
                "blind": blind,
            }
        )

        match = re.search(r"RULE_(\d+)", skill.body)
        version = int(match.group(1)) if match else 0
        if split_name == "validation":
            score = {0: 0.20, 1: 0.90, 2: 0.30}.get(version, 0.10)
        else:
            score = {0: 0.20, 1: 0.70, 2: 0.80}.get(version, 0.90)
        return EvaluationBatch(
            split=split,
            results=tuple(
                TaskResult(
                    task_id=task_id,
                    reward=score,
                    success=score >= 0.5,
                    feedback=f"visible feedback for {task_id}",
                )
                for task_id in ordered_ids
            ),
        )


class ScriptedPolicy:
    def __init__(self, action_indices) -> None:
        self.action_indices = list(action_indices)
        self.updates = []

    def select(self, state, mask, deterministic=False):
        index = self.action_indices.pop(0) if self.action_indices else STOP_INDEX
        assert bool(np.asarray(mask, dtype=bool)[index]), (
            f"scripted action {index} is masked"
        )
        return SimpleNamespace(
            action_index=index,
            log_probability=0.0,
            value=0.0,
            probabilities=np.asarray(mask, dtype=float) / np.asarray(mask).sum(),
        )

    def update(self, transitions):
        self.updates.append(tuple(transitions))
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    def save(self, path) -> None:
        Path(path).write_bytes(b"deterministic fake policy checkpoint")


class LocalRewritePatchGenerator:
    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid
        self.calls: list[dict] = []

    def generate(self, skill, module, operator, train_batch, edit_history) -> EditPatch:
        ids = tuple(result.task_id for result in train_batch.results)
        self.calls.append(
            {"skill_body": skill.body, "task_ids": ids, "operator": operator}
        )
        match = re.search(r"RULE_(\d+)", skill.body)
        assert match is not None
        old_text = match.group(0)
        new_text = f"RULE_{int(match.group(1)) + 1}"
        return EditPatch(
            target_module="wrong-module" if self.invalid else module.module_id,
            operator=operator,
            rationale="Replace one local rule for a deterministic test edit.",
            old_text=old_text,
            new_text=new_text,
            expected_effect="raise the deterministic mock score",
        )


class InvalidStructuredPatchGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, skill, module, operator, train_batch, edit_history):
        del skill, train_batch, edit_history
        self.calls.append({"module_id": module.module_id, "operator": operator})
        return GeneratedPatch(
            patch=None,
            cache_hit=False,
            request_hash="invalid-structured-output",
            usage={"input_tokens": 7, "output_tokens": 3},
            error="Editor response is not strict JSON",
        )


class RecordingBudget:
    def __init__(self) -> None:
        self.roles: list[str] = []
        self.counters = {
            "student_rollouts": 0,
            "teacher_rollouts": 0,
            "reference_rollouts": 0,
            "editor_calls": 0,
            "evaluator_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": 0.0,
        }

    def reserve_evaluation(self, *args, **kwargs) -> None:
        role = str(
            kwargs.get(
                "role", args[0] if args and isinstance(args[0], str) else "student"
            )
        ).lower()
        self.roles.append(role)

    def record_evaluation(self, *args, **kwargs) -> None:
        role = str(
            kwargs.get(
                "role", args[0] if args and isinstance(args[0], str) else "student"
            )
        ).lower()
        task_count = int(kwargs.get("task_count", kwargs.get("rollouts", 0)))
        repetitions = int(kwargs.get("repetitions", 1))
        key = f"{role}_rollouts"
        if key in self.counters:
            self.counters[key] += task_count * repetitions
        self.counters["evaluator_calls"] += 1

    def reserve_editor(self, *args, **kwargs) -> None:
        self.roles.append("editor")

    def record_editor(self, *args, **kwargs) -> None:
        self.counters["editor_calls"] += 1

    def remaining_fraction(self) -> float:
        return 1.0

    def snapshot(self) -> dict:
        return dict(self.counters)


def _skill() -> SkillArtifact:
    return SkillArtifact(
        skill_id="mock-skill",
        name="Mock Skill",
        description="A local deterministic optimizer fixture.",
        body="# Mock Skill\n\n## Rules\n\nRULE_0\n",
    )


def _tasks(prefix: str, count: int = 3) -> tuple[FakeTask, ...]:
    return tuple(FakeTask(f"{prefix}-{index}") for index in range(count))


def _config(*, episodes=1, horizon=1, minibatch_size=2) -> dict:
    # Top-level aliases keep this focused on behavior while the final YAML schema is built in Task 7.
    return {
        "episodes": episodes,
        "horizon": horizon,
        "minibatch_size": minibatch_size,
        "validation_interval": 1,
        "train_repetitions": 1,
        "validation_repetitions": 1,
        "max_modules": 2,
        "optimizer": {
            "episodes": episodes,
            "horizon": horizon,
            "minibatch_size": minibatch_size,
            "validation_interval": 1,
        },
        "evaluation": {"train_repetitions": 1, "validation_repetitions": 1},
        "action_space": {"max_modules": 2},
        "reward": {"beta_len": 0.0, "beta_edit": 0.0, "beta_invalid": 1.0},
    }


def _run_optimizer(
    tmp_path,
    *,
    actions,
    episodes=1,
    horizon=1,
    patch_generator=None,
    evaluator=None,
    budget=None,
):
    evaluator = evaluator or RecordingEvaluator()
    patch_generator = patch_generator or LocalRewritePatchGenerator()
    budget = budget or RecordingBudget()
    optimizer = RLSkillEditOptimizer(
        config=_config(episodes=episodes, horizon=horizon),
        evaluator=evaluator,
        patch_generator=patch_generator,
        policy=ScriptedPolicy(actions),
        output_dir=tmp_path,
    )
    result = optimizer.optimize(
        initial_skill=_skill(),
        train_tasks=_tasks("train"),
        validation_tasks=_tasks("validation"),
        budget=budget,
        seed=17,
    )
    return result, evaluator, patch_generator, budget


def _step_logs(tmp_path) -> list[dict]:
    path = tmp_path / "rl_training_log.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_optimizer_has_no_test_input_and_mock_evaluator_blocks_test_until_freeze():
    constructor_parameters = inspect.signature(RLSkillEditOptimizer).parameters
    optimize_parameters = inspect.signature(RLSkillEditOptimizer.optimize).parameters
    assert not any("test" in name.lower() for name in constructor_parameters)
    assert not any("test" in name.lower() for name in optimize_parameters)

    evaluator = MockSkillEvaluator(
        score_fn=lambda skill, task, repetition, seed: TaskResult(
            task_id=_task_id(task), reward=1.0, success=True
        )
    )
    kwargs = dict(
        skill=_skill(),
        tasks=_tasks("test", 1),
        split=Split.TEST,
        seed=3,
        repetitions=1,
        use_cache=False,
        blind=True,
    )
    with pytest.raises(RuntimeError, match="freeze"):
        evaluator.evaluate(**kwargs)
    evaluator.freeze()
    assert tuple(result.task_id for result in evaluator.evaluate(**kwargs).results) == (
        "test-0",
    )


def test_rollout_cache_binds_task_content_and_evaluator_signature(tmp_path):
    calls = []

    def score_fn(skill, task, repetition, seed):
        del skill, repetition, seed
        calls.append(task["description"])
        return TaskResult(task_id=task["task_id"], reward=0.5, success=True)

    cache = JsonFileCache(tmp_path / "rollouts.json")
    evaluator = MockSkillEvaluator(
        score_fn=score_fn,
        cache=cache,
        cache_signature={"model": "student-a", "temperature": 0.0},
    )
    kwargs = dict(
        skill=_skill(),
        split=Split.TRAIN,
        seed=3,
        repetitions=1,
        use_cache=True,
        blind=False,
    )
    first_task = ({"task_id": "same-id", "description": "first content"},)
    changed_task = ({"task_id": "same-id", "description": "changed content"},)

    assert evaluator.evaluate(tasks=first_task, **kwargs).cache_hit is False
    assert evaluator.evaluate(tasks=first_task, **kwargs).cache_hit is True
    assert evaluator.evaluate(tasks=changed_task, **kwargs).cache_hit is False

    changed_evaluator = MockSkillEvaluator(
        score_fn=score_fn,
        cache=cache,
        cache_signature={"model": "student-b", "temperature": 0.0},
    )
    assert changed_evaluator.evaluate(tasks=first_task, **kwargs).cache_hit is False
    assert calls == ["first content", "changed content", "first content"]


def test_current_and_candidate_use_the_identical_ordered_train_minibatch(tmp_path):
    _, evaluator, _, _ = _run_optimizer(tmp_path, actions=[REWRITE_RULE_INDEX])
    train_calls = [call for call in evaluator.calls if call["split"] == "train"]

    assert len(train_calls) >= 2
    current_call, candidate_call = train_calls[-2:]
    assert current_call["task_ids"] == candidate_call["task_ids"]
    assert current_call["seed"] == candidate_call["seed"]
    assert current_call["repetitions"] == candidate_call["repetitions"]


def test_checkpoint_validation_uses_the_existing_witness_protocol(tmp_path):
    _, evaluator, _, _ = _run_optimizer(
        tmp_path,
        actions=[REWRITE_RULE_INDEX],
    )
    validation_calls = [
        call for call in evaluator.calls if call["split"] == "validation"
    ]

    assert validation_calls
    assert all(call["blind"] is False for call in validation_calls)


def test_invalid_patch_is_logged_as_invalid_and_does_not_change_skill(tmp_path):
    result, _, _, _ = _run_optimizer(
        tmp_path,
        actions=[REWRITE_RULE_INDEX],
        patch_generator=LocalRewritePatchGenerator(invalid=True),
    )
    log = _step_logs(tmp_path)[0]

    assert log["status"] == "invalid"
    assert log["current_skill_digest"] == log["candidate_skill_digest"]
    assert log["reward_components"]["invalid_cost"] > 0
    assert result.final_skill.digest == _skill().digest


def test_malformed_editor_output_is_an_invalid_transition_not_a_crash(tmp_path):
    result, _, _, budget = _run_optimizer(
        tmp_path,
        actions=[REWRITE_RULE_INDEX],
        patch_generator=InvalidStructuredPatchGenerator(),
    )
    log = _step_logs(tmp_path)[0]

    assert log["status"] == "invalid"
    assert log["patch"] is None
    assert "strict JSON" in log["patch_reason"]
    assert log["reward_components"]["invalid_cost"] == 1
    assert result.final_skill.digest == _skill().digest
    assert budget.snapshot()["editor_calls"] == 1


def test_stop_ends_episode_without_editor_or_candidate_train_call(tmp_path):
    result, evaluator, generator, _ = _run_optimizer(tmp_path, actions=[STOP_INDEX])

    assert generator.calls == []
    train_calls = [call for call in evaluator.calls if call["split"] == "train"]
    assert len(train_calls) <= 1  # one incumbent diagnostic batch is allowed
    assert all(call["skill_body"] == _skill().body for call in train_calls)
    assert result.final_skill.digest == _skill().digest
    rows = list(
        csv.DictReader((tmp_path / "rl_episode_summary.csv").open(encoding="utf-8"))
    )
    assert rows[0]["termination"] == "stop"


def test_horizon_ends_episode_after_exactly_h_non_stop_actions(tmp_path):
    _run_optimizer(
        tmp_path,
        actions=[REWRITE_RULE_INDEX, REWRITE_RULE_INDEX, REWRITE_RULE_INDEX],
        horizon=2,
    )

    assert len(_step_logs(tmp_path)) == 2
    rows = list(
        csv.DictReader((tmp_path / "rl_episode_summary.csv").open(encoding="utf-8"))
    )
    assert rows[0]["termination"] == "horizon"


def test_each_episode_resets_to_the_identical_initial_skill(tmp_path):
    _, _, generator, _ = _run_optimizer(
        tmp_path,
        actions=[REWRITE_RULE_INDEX, REWRITE_RULE_INDEX],
        episodes=2,
        horizon=1,
    )

    assert [call["skill_body"] for call in generator.calls] == [
        _skill().body,
        _skill().body,
    ]


def test_validation_selects_best_checkpoint_not_last_episode_state(tmp_path):
    result, _, _, _ = _run_optimizer(
        tmp_path,
        actions=[REWRITE_RULE_INDEX, REWRITE_RULE_INDEX],
        horizon=2,
    )

    assert "RULE_1" in result.best_skill.body
    assert "RULE_2" not in result.best_skill.body


def test_pure_rl_never_reserves_teacher_or_reference_work(tmp_path):
    budget = RecordingBudget()
    _run_optimizer(tmp_path, actions=[REWRITE_RULE_INDEX], budget=budget)

    assert "teacher" not in budget.roles
    assert "reference" not in budget.roles
    assert budget.snapshot()["teacher_rollouts"] == 0
    assert budget.snapshot()["reference_rollouts"] == 0


def test_optimizer_writes_required_auditable_artifacts_and_step_fields(tmp_path):
    _run_optimizer(tmp_path, actions=[REWRITE_RULE_INDEX])

    for filename in (
        "best_rl_skill.md",
        "final_rl_policy.pt",
        "rl_training_log.jsonl",
        "rl_episode_summary.csv",
    ):
        artifact = tmp_path / filename
        assert artifact.is_file(), filename
        assert artifact.stat().st_size > 0, filename

    log = _step_logs(tmp_path)[0]
    assert {
        "episode",
        "step",
        "action_index",
        "module_id",
        "operator",
        "status",
        "patch",
        "train_task_ids",
        "current_task_scores",
        "candidate_task_scores",
        "reward",
        "reward_components",
        "current_skill_digest",
        "candidate_skill_digest",
        "budget",
    } <= log.keys()
    assert log["train_task_ids"] == list(log["current_task_scores"])
    assert log["train_task_ids"] == list(log["candidate_task_scores"])
    assert {
        "paired_delta",
        "length_cost",
        "edit_cost",
        "invalid_cost",
        "total",
    } <= log["reward_components"].keys()

    summary_columns = set(
        next(
            csv.DictReader((tmp_path / "rl_episode_summary.csv").open(encoding="utf-8"))
        ).keys()
    )
    assert {
        "episode",
        "steps",
        "episode_return",
        "termination",
        "best_validation_score",
        "best_skill_digest",
    } <= summary_columns

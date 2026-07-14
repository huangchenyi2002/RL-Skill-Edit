from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass

import pytest

from rl_skill_edit.comparison import (
    load_current_method_artifact,
    paired_bootstrap_ci,
    paired_statistics,
    run_comparison,
)
from rl_skill_edit.types import EvaluationBatch, SkillArtifact, TaskResult


@dataclass(frozen=True)
class FakeTask:
    task_id: str


def _skill(skill_id: str, marker: str) -> SkillArtifact:
    return SkillArtifact(
        skill_id=skill_id,
        name=skill_id,
        description="Frozen mock comparison artifact.",
        body=f"# {skill_id}\n\n## Rules\n\n{marker}\n",
    )


class FrozenComparisonEvaluator:
    def __init__(self) -> None:
        self.frozen = False
        self.events: list[tuple[str, object]] = []

    def freeze(self) -> None:
        self.frozen = True
        self.events.append(("freeze", None))

    def evaluate(self, skill, tasks, split, seed, repetitions, use_cache, blind):
        split_name = str(getattr(split, "value", split)).lower()
        assert split_name == "test"
        assert self.frozen, "formal test evaluation happened before freeze"
        ordered_ids = tuple(task.task_id for task in tasks)
        self.events.append(
            (
                "evaluate",
                {
                    "skill_id": skill.skill_id,
                    "task_ids": ordered_ids,
                    "seed": seed,
                    "repetitions": repetitions,
                    "use_cache": use_cache,
                    "blind": blind,
                },
            )
        )
        base = {"initial_skill": 0.25, "current_method": 0.50, "rl_skill_edit": 0.75}[
            skill.skill_id
        ]
        return EvaluationBatch(
            split=split,
            results=tuple(
                TaskResult(
                    task_id=task_id,
                    reward=min(1.0, base + 0.05 * index),
                    success=(base + 0.05 * index) >= 0.5,
                )
                for index, task_id in enumerate(ordered_ids)
            ),
        )


def _comparison_config(tmp_path, evaluator) -> dict:
    methods = {
        "initial_skill": _skill("initial_skill", "INITIAL"),
        "current_method": _skill("current_method", "CURRENT"),
        "rl_skill_edit": _skill("rl_skill_edit", "RL"),
    }
    usage = {
        name: {
            "student_rollouts": 0,
            "teacher_rollouts": 0,
            "reference_rollouts": 0,
            "editor_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "wall_time_s": 0.0,
        }
        for name in methods
    }
    return {
        "output_dir": tmp_path,
        "evaluator": evaluator,
        "test_tasks": tuple(FakeTask(f"test-{index}") for index in range(4)),
        "frozen_methods": methods,
        "method_usage": usage,
        "test_repetitions": 2,
        "bootstrap_samples": 200,
        "evaluation": {
            "test_repetitions": 2,
            "blind": True,
            "model": "mock-student",
            "temperature": 0.0,
            "max_steps": 3,
        },
    }


def test_paired_statistics_cover_mean_se_ci_success_and_win_tie_loss():
    initial = [0.0, 0.0, 1.0, 1.0]
    candidate = [1.0, 0.0, 1.0, 0.0]
    stats = paired_statistics(
        initial,
        candidate,
        successes=[True, False, True, False],
        samples=500,
        seed=19,
    )

    assert stats.mean_reward == pytest.approx(0.5)
    assert stats.mean_delta == pytest.approx(0.0)
    assert stats.standard_error == pytest.approx(math.sqrt(2 / 3) / 2)
    assert stats.success_rate == pytest.approx(0.5)
    assert (stats.wins, stats.ties, stats.losses) == (1, 2, 1)
    assert stats.ci_low <= stats.mean_delta <= stats.ci_high

    ci_low, ci_high = paired_bootstrap_ci(
        [0.0, 0.2, 0.4], [0.25, 0.45, 0.65], samples=100, seed=3
    )
    assert ci_low == pytest.approx(0.25)
    assert ci_high == pytest.approx(0.25)


def test_current_method_import_recovers_usage_and_wall_time(tmp_path):
    skill_path = tmp_path / "current_skill.md"
    skill_path.write_text("# Current\n\n## Rules\n\nCURRENT\n", encoding="utf-8")
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "cost_summary": {
                    "total_input_tokens": 120,
                    "total_output_tokens": 30,
                    "total_tokens": 150,
                    "total_cost_usd": 1.75,
                }
            }
        ),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "events.jsonl"
    events = [
        {"event": "usage", "role": "student", "rollouts": 11},
        {"event": "usage", "role": "teacher", "rollouts": 7},
        {"event": "usage", "role": "reference", "rollouts": 5},
        {"event": "usage", "role": "editor", "calls": 3},
        {"event": "session_end", "t_elapsed": 42.5},
    ]
    jsonl_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    method = load_current_method_artifact(skill_path, history_path, jsonl_path)

    assert "CURRENT" in method.skill.body
    assert method.usage.student_rollouts == 11
    assert method.usage.teacher_rollouts == 7
    assert method.usage.reference_rollouts == 5
    assert method.usage.editor_calls == 3
    assert method.usage.input_tokens == 120
    assert method.usage.output_tokens == 30
    assert method.usage.total_tokens == 150
    assert method.usage.cost_usd == pytest.approx(1.75)
    assert method.usage.wall_time_s == pytest.approx(42.5)
    assert (
        method.provenance["skill_sha256"]
        == hashlib.sha256(skill_path.read_bytes()).hexdigest()
    )
    assert (
        method.provenance["history_sha256"]
        == hashlib.sha256(history_path.read_bytes()).hexdigest()
    )
    assert (
        method.provenance["jsonl_sha256"]
        == hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    )
    assert method.provenance["skill_history_binding"] == (
        "unavailable_in_legacy_archive"
    )


def test_current_method_import_rejects_history_jsonl_run_mismatch(tmp_path):
    skill_path = tmp_path / "current_skill.md"
    skill_path.write_text("# Current\n", encoding="utf-8")
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps({"timestamp": "20260101_010101"}), encoding="utf-8"
    )
    jsonl_path = tmp_path / "events.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "event": "session_start",
                "run_id": "study1_20260202_020202",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run IDs do not match"):
        load_current_method_artifact(skill_path, history_path, jsonl_path)


def test_all_methods_get_one_identical_fresh_blind_test_call_after_freeze(tmp_path):
    evaluator = FrozenComparisonEvaluator()
    method_names = ["initial_skill", "current_method", "rl_skill_edit"]
    run_comparison(_comparison_config(tmp_path, evaluator), method_names, seed=23)

    assert evaluator.events[0][0] == "freeze"
    calls = [payload for event, payload in evaluator.events if event == "evaluate"]
    assert len(calls) == len(method_names)
    assert [call["skill_id"] for call in calls] == method_names
    assert all(
        call["task_ids"] == ("test-0", "test-1", "test-2", "test-3") for call in calls
    )
    assert all(call["seed"] == 23 for call in calls)
    assert all(call["repetitions"] == 2 for call in calls)
    assert all(call["blind"] is True for call in calls)
    assert all(call["use_cache"] is False for call in calls)


def test_mock_end_to_end_comparison_writes_method_and_raw_task_csvs(tmp_path):
    evaluator = FrozenComparisonEvaluator()
    method_names = ["initial_skill", "current_method", "rl_skill_edit"]
    result = run_comparison(
        _comparison_config(tmp_path, evaluator), method_names, seed=29
    )

    assert tuple(item.method for item in result.methods) == tuple(method_names)

    method_csv = tmp_path / "method_comparison.csv"
    task_csv = tmp_path / "task_level_scores.csv"
    assert method_csv.is_file()
    assert task_csv.is_file()

    with method_csv.open(encoding="utf-8", newline="") as handle:
        method_rows = list(csv.DictReader(handle))
    assert len(method_rows) == len(method_names)
    assert {
        "method",
        "skill_digest",
        "mean_reward",
        "success_rate",
        "standard_error",
        "ci_low",
        "ci_high",
        "wins",
        "ties",
        "losses",
        "student_rollouts",
        "teacher_rollouts",
        "reference_rollouts",
        "editor_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "wall_time_s",
        "optimization_cached_student_rollouts",
        "optimization_executed_student_rollouts",
        "reporting_cached_student_rollouts",
        "reporting_executed_student_rollouts",
    } <= set(method_rows[0])
    assert all(int(row["reporting_student_rollouts"]) == 8 for row in method_rows)
    assert all(
        int(row["reporting_executed_student_rollouts"]) == 8 for row in method_rows
    )

    with task_csv.open(encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
    assert len(task_rows) == len(method_names) * 4
    assert {
        "method",
        "split",
        "task_id",
        "reward",
        "success",
        "initial_reward",
        "paired_delta",
        "skill_digest",
    } <= set(task_rows[0])
    assert {(row["method"], row["task_id"]) for row in task_rows} == {
        (method, f"test-{index}") for method in method_names for index in range(4)
    }

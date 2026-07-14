from __future__ import annotations

import csv
import importlib.util
import json
import math
from dataclasses import dataclass

import pytest

import rl_skill_edit.reporting as reporting
from rl_skill_edit.types import EvaluationBatch, SkillArtifact, TaskResult


def test_reporting_module_exists():
    assert importlib.util.find_spec("rl_skill_edit.reporting") is not None


@dataclass(frozen=True)
class FakeTask:
    task_id: str


def _skill(skill_id: str, marker: str) -> SkillArtifact:
    return SkillArtifact(
        skill_id=skill_id,
        name=skill_id,
        description="Frozen mock reporting artifact.",
        body=f"# {skill_id}\n\n## Rules\n\n{marker}\n",
    )


class FrozenEvaluator:
    def __init__(self) -> None:
        self.frozen = False
        self.events: list[tuple[str, object]] = []

    def freeze(self) -> None:
        self.frozen = True
        self.events.append(("freeze", None))

    def evaluate(self, skill, tasks, split, seed, repetitions, use_cache, blind):
        assert self.frozen
        assert blind is True
        assert use_cache is False
        ordered_ids = tuple(task.task_id for task in tasks)
        self.events.append(
            (
                "evaluate",
                {
                    "skill_id": skill.skill_id,
                    "task_ids": ordered_ids,
                    "seed": seed,
                    "repetitions": repetitions,
                },
            )
        )
        base = 0.25 if skill.skill_id == "initial_skill" else 0.75
        return EvaluationBatch(
            split=split,
            results=tuple(
                TaskResult(task_id, base, base >= 0.5) for task_id in ordered_ids
            ),
            usage={
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
                "cost_usd": 0.1,
                "elapsed_s": 0.2,
            },
        )


def _run_report(
    tmp_path, *, evaluator=None, optimization_usage=None, reporting_usage=None
):
    evaluator = evaluator if evaluator is not None else FrozenEvaluator()
    result = reporting.run_frozen_report(
        initial_skill=_skill("initial_skill", "INITIAL"),
        rl_skill=_skill("rl_skill_edit", "RL"),
        evaluator=evaluator,
        test_tasks=tuple(FakeTask(f"test-{index}") for index in range(4)),
        output_dir=tmp_path,
        seed=23,
        repetitions=2,
        bootstrap_samples=200,
        optimization_usage=(
            optimization_usage if optimization_usage is not None else {}
        ),
        reporting_usage=reporting_usage if reporting_usage is not None else {},
    )
    return result, evaluator


def test_paired_statistics_cover_mean_se_ci_success_and_win_tie_loss():
    initial = [0.0, 0.0, 1.0, 1.0]
    candidate = [1.0, 0.0, 1.0, 0.0]
    stats = reporting.paired_statistics(
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

    ci_low, ci_high = reporting.paired_bootstrap_ci(
        [0.0, 0.2, 0.4], [0.25, 0.45, 0.65], samples=100, seed=3
    )
    assert ci_low == pytest.approx(0.25)
    assert ci_high == pytest.approx(0.25)


def test_frozen_report_evaluates_only_initial_and_rl(tmp_path):
    evaluator = FrozenEvaluator()
    result, _ = _run_report(tmp_path, evaluator=evaluator)

    assert [row.method for row in result.methods] == [
        "initial_skill",
        "rl_skill_edit",
    ]
    assert evaluator.events[0] == ("freeze", None)
    calls = [payload for event, payload in evaluator.events if event == "evaluate"]
    assert [call["skill_id"] for call in calls] == [
        "initial_skill",
        "rl_skill_edit",
    ]
    assert all(
        call["task_ids"] == ("test-0", "test-1", "test-2", "test-3") for call in calls
    )
    assert all(call["seed"] == 23 for call in calls)
    assert all(call["repetitions"] == 2 for call in calls)


def test_frozen_report_writes_exactly_two_method_rows_and_two_rows_per_task(tmp_path):
    optimization_usage = {
        "student_rollouts": 6,
        "editor_calls": 1,
        "evaluator_calls": 3,
        "input_tokens": 20,
        "output_tokens": 10,
        "wall_time_seconds": 1.5,
        "cache_hits": 1,
        "cached_student_rollouts": 2,
        "cached_editor_calls": 0,
        "cached_evaluator_calls": 1,
        "total_tokens": 30,
        "rollout_cost_usd": 0.2,
        "editor_cost_usd": 0.3,
        "cost_usd": 0.5,
        "wall_time_s": 2.0,
        "edit_count": 1,
    }
    prior_reporting = {
        "initial_skill": {
            "student_rollouts": 4,
            "evaluator_calls": 2,
            "cached_student_rollouts": 0,
            "cached_evaluator_calls": 0,
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "cost_usd": 0.05,
            "elapsed_s": 0.4,
        },
        "rl_skill_edit": {
            "student_rollouts": 4,
            "evaluator_calls": 2,
            "cached_student_rollouts": 1,
            "cached_evaluator_calls": 1,
            "input_tokens": 6,
            "output_tokens": 3,
            "total_tokens": 9,
            "cost_usd": 0.06,
            "elapsed_s": 0.5,
        },
    }
    _run_report(
        tmp_path,
        optimization_usage=optimization_usage,
        reporting_usage=prior_reporting,
    )

    with (tmp_path / "method_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        method_rows = list(csv.DictReader(handle))
    assert [row["method"] for row in method_rows] == [
        "initial_skill",
        "rl_skill_edit",
    ]
    assert all(
        "teacher" not in column.lower() and "reference" not in column.lower()
        for column in method_rows[0]
    )
    assert int(method_rows[0]["optimization_student_rollouts"]) == 0
    assert int(method_rows[1]["optimization_student_rollouts"]) == 6
    assert all(int(row["reporting_student_rollouts"]) == 12 for row in method_rows)

    with (tmp_path / "task_level_scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    assert len(task_rows) == 8
    assert {(row["method"], row["task_id"]) for row in task_rows} == {
        (method, f"test-{index}")
        for method in ("initial_skill", "rl_skill_edit")
        for index in range(4)
    }

    report = json.loads(
        (tmp_path / "comparison_report.json").read_text(encoding="utf-8")
    )
    assert [row["method"] for row in report["methods"]] == [
        "initial_skill",
        "rl_skill_edit",
    ]
    frozen = json.loads(
        (tmp_path / "frozen_method_artifacts.json").read_text(encoding="utf-8")
    )
    assert list(frozen) == ["initial_skill", "rl_skill_edit"]


def test_frozen_report_rejects_empty_test_bundle_before_freeze(tmp_path):
    evaluator = FrozenEvaluator()
    with pytest.raises(ValueError, match="at least one test task"):
        reporting.run_frozen_report(
            initial_skill=_skill("initial_skill", "INITIAL"),
            rl_skill=_skill("rl_skill_edit", "RL"),
            evaluator=evaluator,
            test_tasks=(),
            output_dir=tmp_path,
            seed=23,
            repetitions=2,
            bootstrap_samples=200,
            optimization_usage={},
            reporting_usage={},
        )
    assert evaluator.events == []


@pytest.mark.parametrize(
    ("name", "value"),
    (("seed", True), ("repetitions", 1.5), ("bootstrap_samples", "200")),
)
def test_frozen_report_rejects_invalid_control_types(tmp_path, name, value):
    kwargs = {
        "initial_skill": _skill("initial_skill", "INITIAL"),
        "rl_skill": _skill("rl_skill_edit", "RL"),
        "evaluator": FrozenEvaluator(),
        "test_tasks": (FakeTask("test-0"),),
        "output_dir": tmp_path,
        "seed": 23,
        "repetitions": 2,
        "bootstrap_samples": 200,
        "optimization_usage": {},
        "reporting_usage": {},
    }
    kwargs[name] = value

    with pytest.raises(TypeError, match=rf"{name} must be an integer"):
        reporting.run_frozen_report(**kwargs)


def test_reporting_has_no_generic_or_current_method_paths():
    assert not hasattr(reporting, "ImportedMethod")
    assert not hasattr(reporting, "load_current_method_artifact")
    assert not hasattr(reporting, "run_comparison")

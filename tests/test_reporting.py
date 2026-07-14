from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from dataclasses import dataclass, replace

import pytest

import rl_skill_edit.reporting as reporting
from rl_skill_edit.types import EvaluationBatch, SkillArtifact, TaskResult


OUTPUT_FILES = (
    "frozen_method_artifacts.json",
    "method_comparison.csv",
    "task_level_scores.csv",
    "test_task_level_results.csv",
    "comparison_report.json",
)


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
                TaskResult(
                    task_id,
                    base,
                    base >= 0.5,
                    raw_rewards=(base,) * repetitions,
                )
                for task_id in ordered_ids
            ),
            usage={
                "student_rollouts": len(tasks) * repetitions,
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
                "cost_usd": 0.1,
                "elapsed_s": 0.2,
            },
        )


class BatchOverrideEvaluator(FrozenEvaluator):
    def __init__(self, override) -> None:
        super().__init__()
        self.override = override

    def evaluate(self, *args, **kwargs):
        return self.override(super().evaluate(*args, **kwargs))


def _replace_first_result(batch: EvaluationBatch, **changes) -> EvaluationBatch:
    first, *rest = batch.results
    return replace(batch, results=(replace(first, **changes), *rest))


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


def _full_optimization_usage() -> dict:
    return {
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


def _full_reporting_usage() -> dict:
    return {
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


@pytest.mark.parametrize(
    "invalid",
    ("0.25", True, float("nan"), float("inf"), -0.01, 1.01),
)
def test_paired_bootstrap_rejects_coerced_nonfinite_or_out_of_range_rewards(invalid):
    with pytest.raises((TypeError, ValueError), match="initial rewards"):
        reporting.paired_bootstrap_ci([invalid], [0.5], samples=10, seed=3)


@pytest.mark.parametrize("invalid", ("0.25", True, float("nan"), float("inf")))
def test_paired_statistics_rejects_invalid_rewards_before_numpy_conversion(invalid):
    with pytest.raises((TypeError, ValueError), match="candidate rewards"):
        reporting.paired_statistics(
            [0.25],
            [invalid],
            successes=[True],
            samples=10,
            seed=3,
        )


@pytest.mark.parametrize(
    ("override", "error", "message"),
    (
        (
            lambda batch: replace(batch, cache_hit=0),
            TypeError,
            "cache_hit must be a boolean",
        ),
        (
            lambda batch: replace(batch, results=list(batch.results)),
            TypeError,
            "results must be a tuple",
        ),
        (
            lambda batch: replace(batch, results=(object(), *batch.results[1:])),
            TypeError,
            "results\\[0\\] must be a TaskResult",
        ),
        (
            lambda batch: replace(batch, results=batch.results[:-1]),
            ValueError,
            "results must contain 4 items",
        ),
        (
            lambda batch: _replace_first_result(batch, task_id=7),
            TypeError,
            "results\\[0\\].task_id must be text",
        ),
        (
            lambda batch: _replace_first_result(batch, reward=True),
            TypeError,
            "results\\[0\\].reward must be numeric",
        ),
        (
            lambda batch: _replace_first_result(batch, reward=float("nan")),
            ValueError,
            "results\\[0\\].reward must be finite",
        ),
        (
            lambda batch: _replace_first_result(batch, raw_rewards=(0.25,)),
            ValueError,
            "raw_rewards must contain 2 values",
        ),
        (
            lambda batch: _replace_first_result(batch, raw_rewards=[0.25, 0.25]),
            TypeError,
            "raw_rewards must be a tuple",
        ),
        (
            lambda batch: _replace_first_result(batch, raw_rewards=("0.25", 0.25)),
            TypeError,
            "raw_rewards\\[0\\] must be numeric",
        ),
        (
            lambda batch: _replace_first_result(batch, raw_rewards=(0.0, 0.0)),
            ValueError,
            "raw_rewards mean",
        ),
        (
            lambda batch: _replace_first_result(batch, success=1),
            TypeError,
            "success must be a boolean",
        ),
        (
            lambda batch: _replace_first_result(batch, feedback=7),
            TypeError,
            "feedback must be text",
        ),
        (
            lambda batch: _replace_first_result(batch, feedback="leaked feedback"),
            ValueError,
            "feedback must be empty in blind reporting",
        ),
        (
            lambda batch: _replace_first_result(batch, evaluator_output="leaked"),
            ValueError,
            "evaluator_output must be empty in blind reporting",
        ),
        (
            lambda batch: _replace_first_result(batch, final_answer="leaked"),
            ValueError,
            "final_answer must be empty in blind reporting",
        ),
        (
            lambda batch: _replace_first_result(batch, visible_logs=("leaked",)),
            ValueError,
            "visible_logs must be empty in blind reporting",
        ),
        (
            lambda batch: _replace_first_result(batch, visible_logs=[]),
            TypeError,
            "visible_logs must be a tuple",
        ),
        (
            lambda batch: _replace_first_result(batch, visible_logs=(7,)),
            TypeError,
            "visible_logs\\[0\\] must be text",
        ),
        (
            lambda batch: replace(batch, usage={}),
            ValueError,
            "usage is missing fields",
        ),
        (
            lambda batch: replace(batch, usage={**batch.usage, "unknown": 1}),
            ValueError,
            "usage has unknown fields",
        ),
        (
            lambda batch: replace(batch, usage={**batch.usage, "total_tokens": 999}),
            ValueError,
            "total_tokens must equal input_tokens plus output_tokens",
        ),
    ),
)
def test_frozen_report_rejects_invalid_blind_batch_schema(
    tmp_path, override, error, message
):
    evaluator = BatchOverrideEvaluator(override)

    with pytest.raises(error, match=message):
        _run_report(tmp_path, evaluator=evaluator)


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
    _run_report(
        tmp_path,
        optimization_usage=_full_optimization_usage(),
        reporting_usage=_full_reporting_usage(),
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


def test_nonempty_optimization_usage_requires_every_field(tmp_path):
    usage = _full_optimization_usage()
    usage.pop("edit_count")

    with pytest.raises(ValueError, match="optimization_usage is missing fields"):
        _run_report(tmp_path, optimization_usage=usage)


def test_optimization_usage_requires_exact_token_total(tmp_path):
    usage = _full_optimization_usage()
    usage["total_tokens"] = 31

    with pytest.raises(
        ValueError,
        match="optimization_usage.total_tokens must equal input_tokens plus output_tokens",
    ):
        _run_report(tmp_path, optimization_usage=usage)


@pytest.mark.parametrize(
    ("cached_field", "total_field"),
    (
        ("cached_student_rollouts", "student_rollouts"),
        ("cached_editor_calls", "editor_calls"),
        ("cached_evaluator_calls", "evaluator_calls"),
    ),
)
def test_optimization_usage_rejects_cached_work_above_logical_work(
    tmp_path, cached_field, total_field
):
    usage = _full_optimization_usage()
    usage[cached_field] = usage[total_field] + 1

    with pytest.raises(ValueError, match=rf"{cached_field} must not exceed"):
        _run_report(tmp_path, optimization_usage=usage)


def test_nonempty_reporting_usage_requires_both_methods(tmp_path):
    usage = _full_reporting_usage()
    usage.pop("rl_skill_edit")

    with pytest.raises(ValueError, match="reporting_usage is missing methods"):
        _run_report(tmp_path, reporting_usage=usage)


def test_nonempty_reporting_usage_requires_every_field(tmp_path):
    usage = _full_reporting_usage()
    usage["initial_skill"].pop("elapsed_s")

    with pytest.raises(
        ValueError, match="reporting_usage.initial_skill is missing fields"
    ):
        _run_report(tmp_path, reporting_usage=usage)


def test_reporting_usage_requires_exact_token_total(tmp_path):
    usage = _full_reporting_usage()
    usage["initial_skill"]["total_tokens"] = 8

    with pytest.raises(
        ValueError,
        match=(
            "reporting_usage.initial_skill.total_tokens must equal "
            "input_tokens plus output_tokens"
        ),
    ):
        _run_report(tmp_path, reporting_usage=usage)


@pytest.mark.parametrize(
    ("cached_field", "total_field"),
    (
        ("cached_student_rollouts", "student_rollouts"),
        ("cached_evaluator_calls", "evaluator_calls"),
    ),
)
def test_reporting_usage_rejects_cached_work_above_logical_work(
    tmp_path, cached_field, total_field
):
    usage = _full_reporting_usage()
    usage["initial_skill"][cached_field] = usage["initial_skill"][total_field] + 1

    with pytest.raises(ValueError, match=rf"{cached_field} must not exceed"):
        _run_report(tmp_path, reporting_usage=usage)


def _seed_old_outputs(tmp_path, names=OUTPUT_FILES) -> dict[str, bytes]:
    previous = {
        name: f"old report bytes for {name}\n".encode("utf-8") for name in names
    }
    for name, content in previous.items():
        (tmp_path / name).write_bytes(content)
    return previous


def test_report_staging_failure_leaves_all_existing_outputs_unchanged(
    tmp_path, monkeypatch
):
    previous = _seed_old_outputs(tmp_path)
    original_write_csv = reporting._write_csv
    calls = 0

    def fail_second_csv(path, rows):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging write failure")
        original_write_csv(path, rows)

    monkeypatch.setattr(reporting, "_write_csv", fail_second_csv)

    with pytest.raises(OSError, match="injected staging write failure"):
        _run_report(tmp_path)

    assert {name: (tmp_path / name).read_bytes() for name in OUTPUT_FILES} == previous
    assert {path.name for path in tmp_path.iterdir()} == set(OUTPUT_FILES)


@pytest.mark.parametrize("failure_call", range(1, len(OUTPUT_FILES) + 3))
def test_every_report_commit_replace_failure_rolls_back_existing_and_new_outputs(
    tmp_path, monkeypatch, failure_call
):
    existing_names = OUTPUT_FILES[:2]
    previous = _seed_old_outputs(tmp_path, existing_names)
    real_replace = os.replace
    calls = 0
    injected = False

    def fail_one_replace(source, target):
        nonlocal calls, injected
        calls += 1
        if not injected and calls == failure_call:
            injected = True
            raise OSError("injected commit replace failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_one_replace)

    with pytest.raises(OSError, match="injected commit replace failure"):
        _run_report(tmp_path)

    assert {name: (tmp_path / name).read_bytes() for name in existing_names} == previous
    assert all(not (tmp_path / name).exists() for name in OUTPUT_FILES[2:])
    assert {path.name for path in tmp_path.iterdir()} == set(existing_names)


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

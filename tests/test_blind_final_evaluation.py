from __future__ import annotations

import copy
import importlib.util
import os
import signal
import subprocess
import time
from pathlib import Path
from types import MappingProxyType

import pytest
from openpyxl import Workbook, load_workbook

import rl_skill_edit.evaluation as evaluation
from rl_skill_edit.adapters.spreadsheet import (
    ExecutionResult,
    SpreadsheetExecutor,
    SpreadsheetStudent,
    StudentTrajectory,
)
from rl_skill_edit.cache import JsonFileCache
from rl_skill_edit.types import SkillArtifact, Split


STUDENT_CONFIG = {
    "student": {
        "model": "fake-student",
        "temperature": 0.0,
        "max_tokens": 256,
        "max_steps": 3,
    }
}


class RecordingClient:
    def __init__(
        self,
        response: str | None = None,
        usage: dict | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.response = response
        self.usage = usage

    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = (
            self.response
            if self.response is not None
            else (
                "```python\n"
                "from openpyxl import load_workbook\n"
                "wb = load_workbook(wb_path)\n"
                "wb.active['A1'] = 'candidate'\n"
                "wb.save(wb_path)\n"
                "```"
            )
        )
        usage = (
            self.usage
            if self.usage is not None
            else {"total_tokens": 1, "cost_usd": 0.0, "ok": True}
        )
        return response, usage


def _write_workbook(path: Path, value: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "SECRET_FINAL_SHEET"
    worksheet["A1"] = value
    workbook.save(path)


def _process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_exit(process_id: int, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_exists(process_id):
            return True
        time.sleep(0.01)
    return not _process_exists(process_id)


def _final_task(tmp_path: Path) -> dict:
    init_file = tmp_path / "final_input.xlsx"
    golden_file = tmp_path / "final_golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    return {
        "task_id": "final-1",
        "description": "Update the workbook.",
        "spreadsheet": {
            "init_file": str(init_file),
            "golden_file": str(golden_file),
            "answer_sheet": "SECRET_FINAL_SHEET",
            "answer_position": "B17:C19",
        },
    }


def _evaluation_task(tmp_path: Path, task_id: str) -> dict:
    init_file = tmp_path / f"{task_id}_input.xlsx"
    golden_file = tmp_path / f"{task_id}_golden.xlsx"
    _write_workbook(init_file, f"input-{task_id}")
    _write_workbook(golden_file, f"golden-{task_id}")
    return {
        "task_id": task_id,
        "description": f"Update workbook {task_id}.",
        "spreadsheet": {
            "init_file": str(init_file),
            "golden_file": str(golden_file),
            "answer_sheet": "SECRET_FINAL_SHEET",
            "answer_position": "A1",
        },
    }


class RecordingStudent:
    model = "frozen-student"
    temperature = 0.0
    max_tokens = 256
    max_steps = 3

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_task(self, task, skill, *, blind, seed):
        self.calls.append(
            {
                "task_id": task["task_id"],
                "skill_digest": skill.digest,
                "blind": blind,
                "seed": seed,
            }
        )
        hard = seed / 1000.0
        return StudentTrajectory(
            task_id=task["task_id"],
            hard_reward=hard,
            soft_reward=hard + 0.4,
            final_answer=f"answer-{seed}",
            visible_logs=(f"log-{seed}",),
            total_tokens=3,
            total_cost_usd=0.01,
        )


class IncompleteStudent(RecordingStudent):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def run_task(self, task, skill, *, blind, seed):
        trajectory = super().run_task(task, skill, blind=blind, seed=seed)
        if len(self.calls) != 2:
            return trajectory
        if self.failure == "invalid":
            return StudentTrajectory(
                task_id=trajectory.task_id,
                hard_reward=0.0,
                soft_reward=0.0,
                final_answer=trajectory.final_answer,
                visible_logs=trajectory.visible_logs,
                total_tokens=trajectory.total_tokens,
                total_cost_usd=trajectory.total_cost_usd,
                evaluation_valid=False,
                invalid_reason="missing_executable_code",
            )
        return StudentTrajectory(
            task_id="wrong-task-id",
            hard_reward=trajectory.hard_reward,
            soft_reward=trajectory.soft_reward,
            final_answer=trajectory.final_answer,
            visible_logs=trajectory.visible_logs,
            total_tokens=trajectory.total_tokens,
            total_cost_usd=trajectory.total_cost_usd,
        )


class UsageCountingStudent(RecordingStudent):
    def __init__(self) -> None:
        super().__init__()
        self.client = type(
            "UsageClient",
            (),
            {
                "total_input_tokens": 10,
                "total_output_tokens": 20,
                "total_cost_usd": 1.0,
            },
        )()

    def run_task(self, task, skill, *, blind, seed):
        trajectory = super().run_task(task, skill, blind=blind, seed=seed)
        self.client.total_input_tokens += 2
        self.client.total_output_tokens += 3
        self.client.total_cost_usd += 0.05
        return StudentTrajectory(
            task_id=trajectory.task_id,
            hard_reward=trajectory.hard_reward,
            soft_reward=trajectory.soft_reward,
            final_answer=trajectory.final_answer,
            visible_logs=trajectory.visible_logs,
            total_tokens=5,
            total_cost_usd=0.05,
        )


def test_spreadsheet_adapter_module_exists() -> None:
    assert importlib.util.find_spec("rl_skill_edit.adapters.spreadsheet") is not None


def test_spreadsheet_skill_evaluator_exists() -> None:
    assert callable(getattr(evaluation, "SpreadsheetSkillEvaluator", None))


def test_runtime_has_no_legacy_repository_evaluator_or_src_dependency() -> None:
    assert not hasattr(evaluation, "RepositorySkillEvaluator")
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "src.evaluator" not in source


@pytest.mark.parametrize(
    ("metric", "weight", "expected"),
    [
        ("hard", 0.5, 0.2),
        ("soft", 0.5, 0.8),
        ("mixed", 0.25, 0.35),
        ("mixed", -3.0, 0.2),
        ("mixed", 3.0, 0.8),
    ],
)
def test_select_score_is_local_and_clamps_mixed_weight(
    metric: str,
    weight: float,
    expected: float,
) -> None:
    assert evaluation.select_score(0.2, 0.8, metric, weight) == pytest.approx(
        expected
    )


def test_select_score_rejects_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        evaluation.select_score(0.2, 0.8, "unknown", 0.5)


def test_spreadsheet_evaluator_preserves_order_and_derives_each_seed(
    tmp_path: Path,
) -> None:
    student = RecordingStudent()
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=None,
        gate_metric="mixed",
        gate_mixed_weight=0.25,
        success_threshold=0.8,
    )
    tasks = (
        _evaluation_task(tmp_path, "task-b"),
        _evaluation_task(tmp_path, "task-a"),
    )

    batch = evaluator.evaluate(
        SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        tasks,
        split=Split.TRAIN,
        seed=100,
        repetitions=2,
        use_cache=False,
        blind=False,
    )

    assert batch.ordered_task_ids == ("task-b", "task-a")
    assert [call["seed"] for call in student.calls] == [100, 101, 102, 103]
    assert [call["task_id"] for call in student.calls] == [
        "task-b",
        "task-b",
        "task-a",
        "task-a",
    ]
    assert batch.results[0].raw_rewards == pytest.approx((0.2, 0.201))
    assert batch.results[1].raw_rewards == pytest.approx((0.202, 0.203))
    assert batch.usage["student_rollouts"] == 4


def test_spreadsheet_evaluator_records_client_input_and_output_usage(
    tmp_path: Path,
) -> None:
    student = UsageCountingStudent()
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=None,
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )

    batch = evaluator.evaluate(
        SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        (_evaluation_task(tmp_path, "task-a"),),
        split=Split.TRAIN,
        seed=10,
        repetitions=2,
        use_cache=False,
        blind=False,
    )

    assert batch.usage["input_tokens"] == 4
    assert batch.usage["output_tokens"] == 6
    assert batch.usage["total_tokens"] == 10
    assert batch.usage["cost_usd"] == pytest.approx(0.1)


@pytest.mark.parametrize("failure", ["invalid", "wrong_task_id"])
def test_spreadsheet_evaluator_rejects_an_incomplete_bundle_immediately(
    tmp_path: Path,
    failure: str,
) -> None:
    student = IncompleteStudent(failure)
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=None,
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )

    with pytest.raises(RuntimeError, match="incomplete evaluation bundle"):
        evaluator.evaluate(
            SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
            (_evaluation_task(tmp_path, "task-a"),),
            split=Split.TRAIN,
            seed=10,
            repetitions=3,
            use_cache=False,
            blind=False,
        )

    assert len(student.calls) == 2


def test_spreadsheet_evaluator_requires_freeze_before_test(
    tmp_path: Path,
) -> None:
    student = RecordingStudent()
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=JsonFileCache(tmp_path / "test-rollouts.json"),
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )
    kwargs = {
        "skill": SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        "tasks": (_evaluation_task(tmp_path, "test-a"),),
        "split": Split.TEST,
        "seed": 10,
        "repetitions": 1,
        "use_cache": False,
        "blind": True,
    }

    with pytest.raises(RuntimeError, match="freeze"):
        evaluator.evaluate(**kwargs)
    with pytest.raises(RuntimeError, match="freeze"):
        evaluator.cache_will_hit(
            **{key: value for key, value in kwargs.items() if key != "use_cache"}
        )
    assert student.calls == []

    evaluator.freeze()
    batch = evaluator.evaluate(**kwargs)
    assert batch.ordered_task_ids == ("test-a",)
    assert student.calls[0]["blind"] is True


def test_spreadsheet_evaluator_cache_binds_task_files_and_student_signature(
    tmp_path: Path,
) -> None:
    cache = JsonFileCache(tmp_path / "rollouts.json")
    student = RecordingStudent()
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=cache,
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )
    skill = SkillArtifact("main", "main", "", "## Rule\nUpdate cells.")
    task = _evaluation_task(tmp_path, "train-a")
    kwargs = {
        "skill": skill,
        "tasks": (task,),
        "split": Split.TRAIN,
        "seed": 10,
        "repetitions": 1,
        "blind": False,
    }

    assert evaluator.cache_will_hit(**kwargs) is False
    assert evaluator.evaluate(**kwargs, use_cache=True).cache_hit is False
    assert evaluator.cache_will_hit(**kwargs) is True
    assert evaluator.evaluate(**kwargs, use_cache=True).cache_hit is True
    assert len(student.calls) == 1

    changed_task = copy.deepcopy(task)
    changed_task["description"] = "Changed task content with the same task_id."
    assert (
        evaluator.evaluate(
            **{**kwargs, "tasks": (changed_task,)},
            use_cache=True,
        ).cache_hit
        is False
    )

    _write_workbook(Path(task["spreadsheet"]["init_file"]), "changed-input")
    assert evaluator.evaluate(**kwargs, use_cache=True).cache_hit is False

    changed_student = RecordingStudent()
    changed_student.model = "different-frozen-student"
    changed_evaluator = evaluation.SpreadsheetSkillEvaluator(
        changed_student,
        cache=cache,
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )
    assert changed_evaluator.evaluate(**kwargs, use_cache=True).cache_hit is False


@pytest.mark.parametrize("tamper", ["task_order", "repetitions"])
def test_spreadsheet_evaluator_rejects_an_incomplete_cached_bundle(
    tmp_path: Path,
    tamper: str,
) -> None:
    cache_path = tmp_path / "rollouts.json"
    cache = JsonFileCache(cache_path)
    student = RecordingStudent()
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=cache,
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )
    kwargs = {
        "skill": SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        "tasks": (
            _evaluation_task(tmp_path, "task-a"),
            _evaluation_task(tmp_path, "task-b"),
        ),
        "split": Split.TRAIN,
        "seed": 10,
        "repetitions": 2,
        "use_cache": True,
        "blind": False,
    }
    evaluator.evaluate(**kwargs)
    payload = __import__("json").loads(cache_path.read_text(encoding="utf-8"))
    cached_batch = next(iter(payload["rollout"].values()))
    if tamper == "task_order":
        cached_batch["results"].reverse()
    else:
        cached_batch["results"][0]["raw_rewards"].pop()
    cache_path.write_text(
        __import__("json").dumps(payload),
        encoding="utf-8",
    )
    student.calls.clear()

    with pytest.raises(RuntimeError, match="incomplete cached evaluation bundle"):
        evaluator.evaluate(**kwargs)
    assert student.calls == []


def test_generic_task_mapping_keeps_strong_cache_identity(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path / "rollouts.json")
    student = RecordingStudent()
    evaluator = evaluation.SpreadsheetSkillEvaluator(
        student,
        cache=cache,
        gate_metric="hard",
        gate_mixed_weight=0.5,
    )
    task = _evaluation_task(tmp_path, "task-a")
    changed_task = copy.deepcopy(task)
    changed_task["description"] = "Different content, same task id and files."
    kwargs = {
        "skill": SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        "split": Split.TRAIN,
        "seed": 10,
        "repetitions": 1,
        "use_cache": True,
        "blind": False,
    }

    first = evaluator.evaluate(
        tasks=(MappingProxyType(task),),
        **kwargs,
    )
    changed = evaluator.evaluate(
        tasks=(MappingProxyType(changed_task),),
        **kwargs,
    )

    assert first.cache_hit is False
    assert changed.cache_hit is False
    assert len(student.calls) == 2


def test_blind_test_hides_answer_metadata_and_does_not_retry_from_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = RecordingClient()
    student = SpreadsheetStudent(STUDENT_CONFIG, client)
    monkeypatch.setattr(
        student.executor,
        "execute_and_score",
        lambda **kwargs: ExecutionResult(0.25, 1, 4, ""),
    )
    skill = SkillArtifact("main", "main", "", "## Rule\nUpdate cells.")

    trajectory = student.run_task(_final_task(tmp_path), skill, blind=True, seed=11)

    assert trajectory.hard_reward == 0.25
    assert len(client.calls) == 1
    prompt = repr(client.calls[0])
    assert "SECRET_FINAL_SHEET" not in prompt
    assert "B17:C19" not in prompt
    assert "score=" not in prompt
    assert client.calls[0]["seed"] == 11


def test_student_has_no_implicit_or_no_skill_entrypoints() -> None:
    assert not hasattr(SpreadsheetStudent, "run_task_no_skill")
    assert not hasattr(SpreadsheetStudent, "run_task_with_model")


def test_executor_copies_input_executes_code_and_scores_configured_range(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    code = (
        "from openpyxl import load_workbook\n"
        "wb = load_workbook(wb_path)\n"
        "wb['SECRET_FINAL_SHEET']['A1'] = 'golden'\n"
        "wb.save(wb_path)\n"
    )

    result = SpreadsheetExecutor().execute_and_score(
        code=code,
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result == ExecutionResult(score=1.0, matched=1, total=1)
    assert load_workbook(init_file)["SECRET_FINAL_SHEET"]["A1"].value == "input"


def test_executor_rejects_empty_code_without_running_a_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "same")
    _write_workbook(golden_file, "same")
    monkeypatch.setattr(
        "rl_skill_edit.adapters.spreadsheet.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("empty code must not be executed"),
    )

    result = SpreadsheetExecutor().execute_and_score(
        code="   \n",
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result.score == 0.0
    assert result.total == 0
    assert result.error == "missing_executable_code"


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("init_file", "missing-input.xlsx", "missing_input_workbook"),
        ("golden_file", "missing-golden.xlsx", "missing_golden_workbook"),
        ("answer_position", "", "missing_answer_position"),
        ("answer_sheet", "", "missing_answer_sheet"),
    ],
)
def test_executor_preflight_failures_return_explicit_invalid_results(
    tmp_path: Path,
    monkeypatch,
    field: str,
    replacement: str,
    expected_error: str,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    kwargs = {
        "code": "raise AssertionError('must not run')",
        "init_file": str(init_file),
        "golden_file": str(golden_file),
        "answer_position": "A1",
        "answer_sheet": "SECRET_FINAL_SHEET",
    }
    kwargs[field] = (
        str(tmp_path / replacement)
        if field in {"init_file", "golden_file"}
        else replacement
    )
    monkeypatch.setattr(
        "rl_skill_edit.adapters.spreadsheet.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("invalid inputs must not be executed"),
    )

    result = SpreadsheetExecutor().execute_and_score(**kwargs)

    assert result == ExecutionResult(0.0, 0, 0, expected_error)


def test_executor_timeout_is_invalid_and_uses_only_allowed_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    captured: dict = {"timeouts": []}

    class TimedOutProcess:
        pid = 12345
        returncode = None

        def communicate(self, *, timeout):
            captured["timeouts"].append(timeout)
            if len(captured["timeouts"]) == 1:
                raise subprocess.TimeoutExpired(captured["command"], timeout)
            self.returncode = -9
            return "", ""

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return TimedOutProcess()

    monkeypatch.setattr(
        "rl_skill_edit.adapters.spreadsheet.subprocess.Popen",
        fake_popen,
    )
    executor = SpreadsheetExecutor()
    killed: list[int] = []
    monkeypatch.setattr(
        executor,
        "_kill_process_tree",
        lambda process: killed.append(process.pid),
    )

    result = executor.execute_and_score(
        code="while True:\n    pass",
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result == ExecutionResult(0.0, 0, 0, "timeout_8s")
    assert captured["timeouts"] == [8, 1]
    assert killed == [12345]
    assert set(captured["env"]) == {
        "PATH",
        "PYTHONIOENCODING",
        "PYTHONDONTWRITEBYTECODE",
    }
    assert captured["start_new_session"] is (os.name != "nt")


def test_executor_timeout_terminates_descendant_processes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    child_pid_file = tmp_path / "child.pid"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    executor = SpreadsheetExecutor()
    monkeypatch.setattr(executor, "_TIMEOUT_SECONDS", 0.2, raising=False)
    code = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )

    result = executor.execute_and_score(
        code=code,
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    try:
        assert result == ExecutionResult(0.0, 0, 0, "timeout_8s")
        assert _wait_for_process_exit(child_pid)
    finally:
        if _process_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_executor_does_not_insert_an_automatic_save(tmp_path: Path) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    code_without_save = (
        "from openpyxl import load_workbook\n"
        "wb = load_workbook(wb_path)\n"
        "wb['SECRET_FINAL_SHEET']['A1'] = 'golden'\n"
    )

    result = SpreadsheetExecutor().execute_and_score(
        code=code_without_save,
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result == ExecutionResult(score=0.0, matched=0, total=1)


@pytest.mark.parametrize(
    "code",
    [
        "import module_that_does_not_exist_rl_skill_edit",
        "raise RuntimeError('explicit execution failure')",
    ],
)
def test_executor_import_or_runtime_failure_is_explicitly_invalid(
    tmp_path: Path,
    code: str,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")

    result = SpreadsheetExecutor().execute_and_score(
        code=code,
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result.score == 0.0
    assert result.total == 0
    assert result.error


@pytest.mark.parametrize(
    ("answer_sheet", "answer_position", "expected_error"),
    [
        ("MISSING_SHEET", "A1", "missing_golden_sheet"),
        ("SECRET_FINAL_SHEET", "not-a-range", "invalid_answer_position"),
    ],
)
def test_executor_rejects_misaligned_golden_metadata_before_execution(
    tmp_path: Path,
    monkeypatch,
    answer_sheet: str,
    answer_position: str,
    expected_error: str,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")
    monkeypatch.setattr(
        "rl_skill_edit.adapters.spreadsheet.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("invalid metadata must not be executed"),
    )

    result = SpreadsheetExecutor().execute_and_score(
        code="raise AssertionError('must not run')",
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position=answer_position,
        answer_sheet=answer_sheet,
    )

    assert result == ExecutionResult(0.0, 0, 0, expected_error)


def test_executor_rejects_an_invalid_input_workbook_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    init_file.write_bytes(b"not an xlsx workbook")
    _write_workbook(golden_file, "golden")
    monkeypatch.setattr(
        "rl_skill_edit.adapters.spreadsheet.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("invalid input must not be executed"),
    )

    result = SpreadsheetExecutor().execute_and_score(
        code="raise AssertionError('must not run')",
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result == ExecutionResult(0.0, 0, 0, "invalid_input_workbook")


@pytest.mark.parametrize(
    ("code", "expected_error"),
    [
        (
            "from pathlib import Path\nPath(wb_path).unlink()\n",
            "missing_result_workbook",
        ),
        (
            "from pathlib import Path\nPath(wb_path).write_bytes(b'not xlsx')\n",
            "invalid_result_workbook",
        ),
        (
            "from openpyxl import load_workbook\n"
            "wb = load_workbook(wb_path)\n"
            "wb.create_sheet('OTHER')\n"
            "del wb['SECRET_FINAL_SHEET']\n"
            "wb.save(wb_path)\n",
            "missing_result_sheet",
        ),
    ],
)
def test_executor_rejects_missing_or_misaligned_execution_artifacts(
    tmp_path: Path,
    code: str,
    expected_error: str,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")

    result = SpreadsheetExecutor().execute_and_score(
        code=code,
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result == ExecutionResult(0.0, 0, 0, expected_error)


def test_executor_process_launch_failure_is_explicitly_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_file = tmp_path / "input.xlsx"
    golden_file = tmp_path / "golden.xlsx"
    _write_workbook(init_file, "input")
    _write_workbook(golden_file, "golden")

    def fail_launch(*args, **kwargs):
        raise OSError("interpreter missing")

    monkeypatch.setattr(
        "rl_skill_edit.adapters.spreadsheet.subprocess.Popen",
        fail_launch,
    )

    result = SpreadsheetExecutor().execute_and_score(
        code="print('attempt')",
        init_file=str(init_file),
        golden_file=str(golden_file),
        answer_position="A1",
        answer_sheet="SECRET_FINAL_SHEET",
    )

    assert result == ExecutionResult(
        0.0,
        0,
        0,
        "execution_error:interpreter missing",
    )


@pytest.mark.parametrize(
    "usage",
    [
        {"ok": True, "total_tokens": 1},
        {"ok": False, "total_tokens": 1, "cost_usd": 0.0},
    ],
)
def test_student_rejects_incomplete_or_failed_api_bundles_before_execution(
    tmp_path: Path,
    monkeypatch,
    usage: dict,
) -> None:
    client = RecordingClient(usage=usage)
    student = SpreadsheetStudent(STUDENT_CONFIG, client)
    monkeypatch.setattr(
        student.executor,
        "execute_and_score",
        lambda **kwargs: pytest.fail("an invalid API bundle must not execute"),
    )

    trajectory = student.run_task(
        _final_task(tmp_path),
        SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        blind=True,
        seed=11,
    )

    assert trajectory.evaluation_valid is False
    assert trajectory.hard_reward == 0.0
    assert trajectory.invalid_reason.startswith("api:")


def test_student_prompt_has_exactly_one_forced_skill_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = RecordingClient()
    student = SpreadsheetStudent(STUDENT_CONFIG, client)
    monkeypatch.setattr(
        student.executor,
        "execute_and_score",
        lambda **kwargs: ExecutionResult(1.0, 1, 1),
    )
    skill = SkillArtifact(
        "main",
        "Only Active Skill",
        "",
        "## Unique Rule\nWrite the requested value.",
    )

    trajectory = student.run_task(_final_task(tmp_path), skill, blind=False, seed=19)

    assert trajectory.evaluation_valid is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["system"].count("[ACTIVE SKILL:") == 1
    assert "[ACTIVE SKILL: Only Active Skill]" in call["system"]
    assert skill.body in call["system"]
    assert "SECRET_FINAL_SHEET" in call["system"]
    assert "B17:C19" in call["system"]
    assert call["messages"] == [
        {"role": "user", "content": "Update the workbook."}
    ]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "I did not provide executable Python code.",
        "```python\nthis is not valid Python !!!\n```",
    ],
)
def test_student_empty_missing_or_invalid_code_is_invalid(
    tmp_path: Path,
    response: str,
) -> None:
    client = RecordingClient(response=response)
    student = SpreadsheetStudent(STUDENT_CONFIG, client)

    trajectory = student.run_task(
        _final_task(tmp_path),
        SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
        blind=True,
        seed=7,
    )

    assert len(client.calls) == 1
    assert trajectory.evaluation_valid is False
    assert trajectory.hard_reward == 0.0
    assert trajectory.invalid_reason


@pytest.mark.parametrize(
    "invalid_case",
    ["missing_description", "missing_answer_position", "missing_input_file"],
)
def test_student_rejects_an_incomplete_task_before_the_api_call(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    task = _final_task(tmp_path)
    if invalid_case == "missing_description":
        del task["description"]
    elif invalid_case == "missing_answer_position":
        del task["spreadsheet"]["answer_position"]
    else:
        task["spreadsheet"]["init_file"] = str(tmp_path / "missing.xlsx")
    client = RecordingClient()
    student = SpreadsheetStudent(STUDENT_CONFIG, client)

    with pytest.raises(ValueError, match="task"):
        student.run_task(
            task,
            SkillArtifact("main", "main", "", "## Rule\nUpdate cells."),
            blind=True,
            seed=7,
        )

    assert client.calls == []

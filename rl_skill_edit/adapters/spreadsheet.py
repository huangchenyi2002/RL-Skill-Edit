"""Forced-skill SpreadsheetBench Student runtime."""

from __future__ import annotations

import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from openpyxl.utils.exceptions import InvalidFileException

from ..types import SkillArtifact


@dataclass(frozen=True)
class ExecutionResult:
    score: float
    matched: int
    total: int
    error: str = ""


@dataclass(frozen=True)
class StudentTrajectory:
    task_id: str
    hard_reward: float
    soft_reward: float
    final_answer: str
    visible_logs: tuple[str, ...]
    total_tokens: int
    total_cost_usd: float
    evaluation_valid: bool = True
    invalid_reason: str = ""


class SpreadsheetExecutor:
    _TIMEOUT_SECONDS = 8

    def execute_and_score(
        self,
        *,
        code: str,
        init_file: str,
        golden_file: str,
        answer_position: str,
        answer_sheet: str,
    ) -> ExecutionResult:
        if not isinstance(code, str) or not code.strip():
            return ExecutionResult(0.0, 0, 0, "missing_executable_code")
        input_path = Path(init_file)
        if not input_path.is_file():
            return ExecutionResult(0.0, 0, 0, "missing_input_workbook")
        try:
            input_workbook = load_workbook(
                input_path,
                data_only=True,
                read_only=True,
            )
        except (BadZipFile, InvalidFileException, OSError, ValueError):
            return ExecutionResult(0.0, 0, 0, "invalid_input_workbook")
        input_workbook.close()
        golden_path = Path(golden_file)
        if not golden_path.is_file():
            return ExecutionResult(0.0, 0, 0, "missing_golden_workbook")
        if not isinstance(answer_position, str) or not answer_position.strip():
            return ExecutionResult(0.0, 0, 0, "missing_answer_position")
        if not isinstance(answer_sheet, str) or not answer_sheet.strip():
            return ExecutionResult(0.0, 0, 0, "missing_answer_sheet")
        try:
            min_column, min_row, max_column, max_row = range_boundaries(
                answer_position
            )
            if min(min_column, min_row, max_column, max_row) < 1:
                raise ValueError("range coordinates must be positive")
        except (TypeError, ValueError):
            return ExecutionResult(0.0, 0, 0, "invalid_answer_position")
        try:
            golden_workbook = load_workbook(
                golden_path,
                data_only=True,
                read_only=True,
            )
        except (BadZipFile, InvalidFileException, OSError, ValueError):
            return ExecutionResult(0.0, 0, 0, "invalid_golden_workbook")
        try:
            if answer_sheet not in golden_workbook.sheetnames:
                return ExecutionResult(0.0, 0, 0, "missing_golden_sheet")
        finally:
            golden_workbook.close()
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            workbook_path = temporary_path / "workbook.xlsx"
            shutil.copy2(input_path, workbook_path)
            script_path = temporary_path / "solution.py"
            script_path.write_text(
                f"wb_path = {str(workbook_path)!r}\n{code}",
                encoding="utf-8",
            )
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=temporary_path,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=os.name != "nt",
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=self._TIMEOUT_SECONDS
                    )
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(process)
                    try:
                        process.communicate(timeout=1)
                    except (OSError, subprocess.SubprocessError):
                        pass
                    return ExecutionResult(0.0, 0, 0, "timeout_8s")
            except OSError as exc:
                if process is not None:
                    self._kill_process_tree(process)
                return ExecutionResult(0.0, 0, 0, f"execution_error:{exc}")
            if process.returncode != 0:
                return ExecutionResult(
                    0.0,
                    0,
                    0,
                    (stderr or stdout or "execution_failed")[:500],
                )
            return self._compare_workbooks(
                workbook_path,
                golden_path,
                answer_position=answer_position,
                answer_sheet=answer_sheet,
            )

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    [
                        "taskkill",
                        "/F",
                        "/T",
                        "/PID",
                        str(process.pid),
                    ],
                    capture_output=True,
                    timeout=3,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass

    @staticmethod
    def _compare_workbooks(
        result_file: Path,
        golden_file: Path,
        *,
        answer_position: str,
        answer_sheet: str,
    ) -> ExecutionResult:
        if not result_file.is_file():
            return ExecutionResult(0.0, 0, 0, "missing_result_workbook")
        try:
            result_workbook = load_workbook(result_file, data_only=True)
        except (BadZipFile, InvalidFileException, OSError, ValueError):
            return ExecutionResult(0.0, 0, 0, "invalid_result_workbook")
        try:
            golden_workbook = load_workbook(golden_file, data_only=True)
        except (BadZipFile, InvalidFileException, OSError, ValueError):
            result_workbook.close()
            return ExecutionResult(0.0, 0, 0, "invalid_golden_workbook")
        try:
            if answer_sheet not in result_workbook.sheetnames:
                return ExecutionResult(0.0, 0, 0, "missing_result_sheet")
            if answer_sheet not in golden_workbook.sheetnames:
                return ExecutionResult(0.0, 0, 0, "missing_golden_sheet")
            try:
                min_column, min_row, max_column, max_row = range_boundaries(
                    answer_position
                )
            except (TypeError, ValueError):
                return ExecutionResult(0.0, 0, 0, "invalid_answer_position")
            result_sheet = result_workbook[answer_sheet]
            golden_sheet = golden_workbook[answer_sheet]
            matched = 0
            total = 0
            for row in range(min_row, max_row + 1):
                for column in range(min_column, max_column + 1):
                    total += 1
                    if (
                        result_sheet.cell(row=row, column=column).value
                        == golden_sheet.cell(row=row, column=column).value
                    ):
                        matched += 1
            return ExecutionResult(matched / total, matched, total)
        finally:
            result_workbook.close()
            golden_workbook.close()


def build_student_system(
    task: Mapping[str, Any], *, expose_answer_metadata: bool
) -> str:
    metadata = ""
    if expose_answer_metadata:
        spreadsheet = task.get("spreadsheet")
        if isinstance(spreadsheet, Mapping):
            answer_sheet = str(spreadsheet.get("answer_sheet", ""))
            answer_position = str(spreadsheet.get("answer_position", ""))
            if answer_sheet or answer_position:
                metadata = (
                    f"\nTarget sheet: {answer_sheet}"
                    f"\nTarget range: {answer_position}"
                )
    return (
        "You are an AI agent solving an Excel task. Write Python code that "
        "uses the provided wb_path, modifies the workbook, and explicitly "
        "saves it back to wb_path. Return executable Python code."
        f"{metadata}"
    )


def _extract_python(response: str) -> str:
    for match in re.finditer(
        r"```(?:python|py)?\s*\n?(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        code = match.group(1).strip()
        if code:
            return code
    return ""


def _validate_task(
    task: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any]]:
    if not isinstance(task, Mapping):
        raise ValueError("task must be a mapping")
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task.task_id must be a non-empty string")
    description = task.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("task.description must be a non-empty string")
    spreadsheet = task.get("spreadsheet")
    if not isinstance(spreadsheet, Mapping):
        raise ValueError("task.spreadsheet must be a mapping")
    for key in ("init_file", "golden_file"):
        value = spreadsheet.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"task.spreadsheet.{key} must be a file path")
        if not Path(value).is_file():
            raise ValueError(f"task.spreadsheet.{key} does not exist")
    for key in ("answer_sheet", "answer_position"):
        value = spreadsheet.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"task.spreadsheet.{key} must be a non-empty string"
            )
    return task_id, description, spreadsheet


class SpreadsheetStudent:
    def __init__(self, config: Mapping[str, Any], client: Any) -> None:
        student_config = config["student"]
        self.client = client
        self.model = str(student_config["model"])
        self.temperature = float(student_config["temperature"])
        self.max_tokens = int(student_config["max_tokens"])
        self.max_steps = int(student_config["max_steps"])
        self.executor = SpreadsheetExecutor()

    def run_task(
        self,
        task: Mapping[str, Any],
        skill: SkillArtifact,
        *,
        blind: bool,
        seed: int,
    ) -> StudentTrajectory:
        task_id, description, spreadsheet = _validate_task(task)
        system = build_student_system(task, expose_answer_metadata=not blind)
        system += f"\n\n[ACTIVE SKILL: {skill.name}]\n{skill.body}"
        response, usage = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": description}],
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            call_type="student_rollout",
            seed=seed,
        )
        if not isinstance(usage, Mapping) or not {
            "ok",
            "total_tokens",
            "cost_usd",
        } <= usage.keys():
            return StudentTrajectory(
                task_id=task_id,
                hard_reward=0.0,
                soft_reward=0.0,
                final_answer="",
                visible_logs=(),
                total_tokens=0,
                total_cost_usd=0.0,
                evaluation_valid=False,
                invalid_reason="api:incomplete_usage",
            )
        total_tokens_value = usage["total_tokens"]
        total_cost_value = usage["cost_usd"]
        if (
            not isinstance(usage["ok"], bool)
            or isinstance(total_tokens_value, bool)
            or not isinstance(total_tokens_value, int)
            or total_tokens_value < 0
            or isinstance(total_cost_value, bool)
            or not isinstance(total_cost_value, (int, float))
            or not math.isfinite(float(total_cost_value))
            or float(total_cost_value) < 0.0
        ):
            return StudentTrajectory(
                task_id=task_id,
                hard_reward=0.0,
                soft_reward=0.0,
                final_answer="",
                visible_logs=(),
                total_tokens=0,
                total_cost_usd=0.0,
                evaluation_valid=False,
                invalid_reason="api:invalid_usage",
            )
        total_tokens = total_tokens_value
        total_cost_usd = float(total_cost_value)
        if not usage["ok"]:
            return StudentTrajectory(
                task_id=task_id,
                hard_reward=0.0,
                soft_reward=0.0,
                final_answer="",
                visible_logs=(),
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                evaluation_valid=False,
                invalid_reason=f"api:{usage.get('error_kind', 'request_failed')}",
            )
        if not response:
            return StudentTrajectory(
                task_id=task_id,
                hard_reward=0.0,
                soft_reward=0.0,
                final_answer="",
                visible_logs=(),
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                evaluation_valid=False,
                invalid_reason=(
                    f"api:{usage.get('error_kind', 'empty_response')}"
                    if not usage.get("ok", True)
                    else "empty_response"
                ),
            )

        code = _extract_python(response)
        if not code:
            return StudentTrajectory(
                task_id=task_id,
                hard_reward=0.0,
                soft_reward=0.0,
                final_answer=response,
                visible_logs=(),
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                evaluation_valid=False,
                invalid_reason="missing_executable_code",
            )

        result = self.executor.execute_and_score(
            code=code,
            init_file=str(spreadsheet["init_file"]),
            golden_file=str(spreadsheet["golden_file"]),
            answer_position=str(spreadsheet["answer_position"]),
            answer_sheet=str(spreadsheet["answer_sheet"]),
        )
        return StudentTrajectory(
            task_id=task_id,
            hard_reward=float(result.score),
            soft_reward=float(result.score),
            final_answer=response,
            visible_logs=((result.error,) if result.error else ()),
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            evaluation_valid=not bool(result.error),
            invalid_reason=result.error,
        )


__all__ = [
    "ExecutionResult",
    "SpreadsheetExecutor",
    "SpreadsheetStudent",
    "StudentTrajectory",
    "build_student_system",
]

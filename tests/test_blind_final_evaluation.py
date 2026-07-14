import copy
from pathlib import Path
from types import SimpleNamespace

from openpyxl import Workbook

from src.agent import StudentAgent
from src.evaluator import Evaluator
from src.skill_library import SkillCore, SkillLibrary


STUDENT_CONFIG = {
    "student": {
        "model": "fake-student",
        "temperature": 0.0,
        "max_tokens": 256,
        "max_steps": 3,
    }
}


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = (
            "```python\n"
            "from openpyxl import load_workbook\n"
            "wb = load_workbook(wb_path)\n"
            "wb.active['A1'] = 'candidate'\n"
            "wb.save(wb_path)\n"
            "```"
        )
        return response, {
            "total_tokens": 1,
            "cost_usd": 0.0,
            "ok": True,
        }


def _write_workbook(path: Path, value: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "SECRET_FINAL_SHEET"
    worksheet["A1"] = value
    workbook.save(path)


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


def _library() -> SkillLibrary:
    return SkillLibrary(
        [
            SkillCore(
                skill_id="sheet-skill",
                name="Sheet Skill",
                execution_body="## Rule\nUpdate the requested cells.",
            )
        ]
    )


def test_student_blind_final_hides_answer_metadata_and_never_retries_from_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = RecordingClient()
    agent = StudentAgent(STUDENT_CONFIG, client)
    score_calls: list[dict] = []

    def score_once(**kwargs):
        score_calls.append(kwargs)
        return 0.25, {"score": 0.25, "matched": 1, "total": 4}

    monkeypatch.setattr(agent, "_execute_and_score", score_once)
    monkeypatch.setattr(
        agent,
        "_compute_reward",
        lambda **kwargs: (0.25, {"score": 0.25}),
    )

    agent.run_task(
        _final_task(tmp_path),
        _library(),
        activation_mode="harness",
        forced_skill_id="sheet-skill",
        verifier_feedback=False,
        expose_answer_metadata=False,
        seed=11,
    )

    assert len(score_calls) == 1
    assert len(client.calls) == 1
    prompt = repr(client.calls[0])
    assert "SECRET_FINAL_SHEET" not in prompt
    assert "B17:C19" not in prompt
    assert "score=" not in prompt
    assert client.calls[0]["seed"] == 11


class RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_task(self, task, library, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            task_id=task["task_id"],
            evaluation_valid=True,
            hard_reward=0.0,
            soft_reward=0.25,
        )


def test_evaluator_forwards_blind_flags_to_every_student_rollout() -> None:
    agent = RecordingAgent()
    evaluator = Evaluator(
        {
            "study1": {
                "B_W": 2,
                "parallel_witness": False,
                "gate_metric": "soft",
            }
        },
        agent,
    )

    evaluator.witness_estimate(
        object(),
        [{"task_id": "final-1"}],
        B_W=2,
        activation_mode="harness",
        forced_skill_id="sheet-skill",
        verifier_feedback=False,
        expose_answer_metadata=False,
        seed=101,
    )

    assert len(agent.calls) == 2
    assert all(call["verifier_feedback"] is False for call in agent.calls)
    assert all(call["expose_answer_metadata"] is False for call in agent.calls)
    assert [call["seed"] for call in agent.calls] == [101, 102]

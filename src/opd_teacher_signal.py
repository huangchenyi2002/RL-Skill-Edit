from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def clamp01(value: Any, default: float = 0.0) -> float:
    """Return a finite value in [0, 1], using a sanitized default on bad input."""

    def _coerce_finite(candidate: Any) -> float | None:
        try:
            numeric = float(candidate)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    sanitized_default = _coerce_finite(default)
    if sanitized_default is None:
        sanitized_default = 0.0
    sanitized_default = max(0.0, min(1.0, sanitized_default))

    numeric = _coerce_finite(value)
    if numeric is None:
        return sanitized_default
    return max(0.0, min(1.0, numeric))


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class TeacherStepGrade:
    """Teacher feedback for one student-visited trajectory step."""

    step: int
    student_action_score: float = 0.0
    teacher_preferred_action: str = ""
    teacher_step_score: float = 0.0
    confidence: float = 0.0
    error_type: str = ""
    failure_type: str = ""
    implicated_skill_id: str = ""
    skill_edit_hint: str = ""
    rationale: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherStepGrade":
        if not isinstance(payload, dict):
            payload = {}
        return cls(
            step=_safe_int(payload.get("step", 0), default=0) or 0,
            student_action_score=clamp01(
                payload.get("student_action_score", 0.0)
            ),
            teacher_preferred_action=str(
                payload.get("teacher_preferred_action", "")
            )[:500],
            teacher_step_score=clamp01(
                payload.get("teacher_step_score", 0.0)
            ),
            confidence=clamp01(payload.get("confidence", 0.0)),
            error_type=str(payload.get("error_type", ""))[:500],
            failure_type=str(payload.get("failure_type", ""))[:500],
            implicated_skill_id=str(
                payload.get("implicated_skill_id", "")
            )[:500],
            skill_edit_hint=str(payload.get("skill_edit_hint", ""))[:500],
            rationale=str(payload.get("rationale", ""))[:500],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_step_signal_patch(self) -> dict[str, Any]:
        return {
            "has_teacher_grade": True,
            "teacher_score": self.teacher_step_score,
            "teacher_confidence": self.confidence,
            "teacher_preferred_action": self.teacher_preferred_action,
            "teacher_error_type": self.error_type,
            "b4_error_type": self.error_type,
            "b4_failure_type": self.failure_type,
            "b4_implicated_skill_id": self.implicated_skill_id,
            "b4_skill_edit_hint": self.skill_edit_hint,
            "b4_rationale": self.rationale,
            "student_action_score": self.student_action_score,
        }


@dataclass
class TrajectoryTeacherGrade:
    """Teacher grading of a complete student trajectory."""

    task_id: str = ""
    trajectory_teacher_score: float = 0.0
    overall_confidence: float = 0.0
    step_grades: list[TeacherStepGrade] = field(default_factory=list)
    summary: str = ""
    trajectory_teacher_score_present: bool = False
    appendix_notes: list[str] = field(default_factory=list)
    parse_failed: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        task_id: str = "",
    ) -> "TrajectoryTeacherGrade":
        if not isinstance(data, dict):
            return cls(task_id=task_id, parse_failed=True)

        raw_step_grades = data.get("step_grades", [])
        if not isinstance(raw_step_grades, list):
            raw_step_grades = []
        step_grades = [
            TeacherStepGrade.from_dict(item)
            for item in raw_step_grades
            if isinstance(item, dict)
        ]

        raw_notes = data.get("appendix_notes", [])
        if not isinstance(raw_notes, list):
            raw_notes = []
        appendix_notes = []
        for note in raw_notes:
            normalized = str(note).strip()
            if normalized:
                appendix_notes.append(normalized[:300])

        # Presence is semantic. Explicit zero, null, or invalid input is an
        # aggregate verdict of zero; only an absent key triggers step fallback.
        aggregate_present = "trajectory_teacher_score" in data
        return cls(
            task_id=str(data.get("task_id") or task_id),
            trajectory_teacher_score=clamp01(
                data.get("trajectory_teacher_score", 0.0)
            ),
            overall_confidence=clamp01(data.get("overall_confidence", 0.0)),
            step_grades=step_grades,
            summary=str(data.get("summary", ""))[:500],
            trajectory_teacher_score_present=aggregate_present,
            appendix_notes=appendix_notes,
            parse_failed=bool(data.get("parse_failed", False)),
        )

    def teacher_score(self) -> float:
        if self.trajectory_teacher_score_present:
            return clamp01(self.trajectory_teacher_score)
        if not self.step_grades:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        for step_grade in self.step_grades:
            weight = max(clamp01(step_grade.confidence), 0.05)
            weighted_sum += clamp01(step_grade.teacher_step_score) * weight
            total_weight += weight
        if total_weight <= 0.0:
            return 0.0
        return clamp01(weighted_sum / total_weight)

    def mean_confidence(self) -> float:
        if self.step_grades:
            return clamp01(
                sum(step.confidence for step in self.step_grades)
                / len(self.step_grades),
                default=self.overall_confidence,
            )
        return clamp01(self.overall_confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_step_grade(value: Any) -> TeacherStepGrade | None:
    if isinstance(value, TeacherStepGrade):
        return value
    if isinstance(value, dict):
        return TeacherStepGrade.from_dict(value)
    return None


def attach_teacher_grades_to_trajectories(
    trajectories: list,
    grades: dict,
) -> None:
    """Attach parsed teacher grades and inject step-level execution signals."""
    if not grades:
        return

    for trajectory in trajectories:
        task_id = getattr(trajectory, "task_id", "")
        grade = grades.get(task_id)
        if grade is None or getattr(grade, "parse_failed", False):
            continue

        step_grade_by_step: dict[int, TeacherStepGrade] = {}
        for raw_step_grade in getattr(grade, "step_grades", []) or []:
            step_grade = _coerce_step_grade(raw_step_grade)
            if step_grade is not None:
                step_grade_by_step[step_grade.step] = step_grade

        for signal in getattr(trajectory, "step_execution_signals", []) or []:
            if not isinstance(signal, dict):
                continue
            signal_step = _safe_int(signal.get("step"), default=None)
            step_grade = step_grade_by_step.get(signal_step)
            if step_grade is not None:
                signal.update(step_grade.to_step_signal_patch())

        # Both attribute names have existed in repository callers.
        setattr(trajectory, "teacher_grade", grade)
        setattr(trajectory, "b4_teacher_grade", grade)


def teacher_scores_from_grades(
    trajectories: list,
    grades: dict,
) -> dict[str, float]:
    """Return teacher scores, falling back to each student's observed reward."""
    scores = {}
    grades = grades or {}
    for trajectory in trajectories:
        task_id = getattr(trajectory, "task_id", "")
        grade = grades.get(task_id)
        if grade is not None and not getattr(grade, "parse_failed", False):
            teacher_score = getattr(grade, "teacher_score", None)
            if callable(teacher_score):
                scores[task_id] = teacher_score()
                continue
        scores[task_id] = clamp01(getattr(trajectory, "final_reward", 0.0))
    return scores


def summarize_teacher_grade_for_prompt(
    grade: TrajectoryTeacherGrade | None,
    max_len: int = 600,
) -> str:
    if grade is None or getattr(grade, "parse_failed", False):
        return ""

    lines = [
        f"Teacher grade for task {grade.task_id}.",
        (
            "student-visited trajectory "
            f"teacher_score={grade.teacher_score():.2f} "
            f"confidence={grade.mean_confidence():.2f}."
        ),
    ]
    if grade.summary:
        lines.append(grade.summary)

    for raw_step_grade in grade.step_grades:
        step_grade = _coerce_step_grade(raw_step_grade)
        if step_grade is None:
            continue
        lines.append(
            " ".join(
                [
                    f"Step {step_grade.step}",
                    (
                        "student-visited "
                        f"score={clamp01(step_grade.student_action_score):.2f};"
                    ),
                    f"teacher prefers: {step_grade.teacher_preferred_action}",
                    f"implicated_skill_id={step_grade.implicated_skill_id};",
                    f"edit_hint={step_grade.skill_edit_hint}",
                    f"error_type={step_grade.error_type}.",
                    f"Rationale: {step_grade.rationale}",
                ]
            )
        )
    return "\n".join(lines)[:max_len]

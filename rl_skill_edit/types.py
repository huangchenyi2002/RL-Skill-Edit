from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SkillArtifact:
    skill_id: str
    name: str
    description: str
    body: str

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        skill_id: str = "main",
    ) -> "SkillArtifact":
        raw = Path(path).read_text(encoding="utf-8-sig")
        metadata: dict[str, Any] = {}
        body = raw.strip()
        if raw.lstrip().startswith("---"):
            stripped = raw.lstrip("\ufeff")
            parts = stripped.split("---", 2)
            if len(parts) == 3:
                metadata = yaml.safe_load(parts[1]) or {}
                if not isinstance(metadata, dict):
                    raise ValueError("skill frontmatter must be a mapping")
                body = parts[2].strip()
        name = str(metadata.get("name") or skill_id)
        description = str(metadata.get("description") or "")
        return cls(
            skill_id=str(skill_id),
            name=name,
            description=description,
            body=body,
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "skill_id": self.skill_id,
                "name": self.name,
                "description": self.description,
                "body": self.body,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = yaml.safe_dump(
            {"name": self.name, "description": self.description},
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        target.write_text(
            f"---\n{metadata}\n---\n\n{self.body.rstrip()}\n",
            encoding="utf-8",
        )

    def with_body(self, body: str) -> "SkillArtifact":
        return SkillArtifact(
            self.skill_id,
            self.name,
            self.description,
            body,
        )


@dataclass(frozen=True)
class SkillModule:
    module_id: str
    title: str
    level: int
    start: int
    end: int
    text: str
    slot: int


@dataclass(frozen=True)
class ModuleDiagnostics:
    module_id: str
    failure_count: int = 0
    task_ids: tuple[str, ...] = ()
    mean_reward: float = 0.0
    success_rate: float = 0.0
    sample_count: int = 0


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    reward: float
    success: bool
    feedback: str = ""
    evaluator_output: str = ""
    final_answer: str = ""
    visible_logs: tuple[str, ...] = ()
    raw_rewards: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reward": float(self.reward),
            "success": bool(self.success),
            "feedback": self.feedback,
            "evaluator_output": self.evaluator_output,
            "final_answer": self.final_answer,
            "visible_logs": list(self.visible_logs),
            "raw_rewards": list(self.raw_rewards),
        }


@dataclass(frozen=True)
class EvaluationBatch:
    split: Split
    results: tuple[TaskResult, ...]
    cache_hit: bool = False
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ordered_task_ids(self) -> tuple[str, ...]:
        return tuple(result.task_id for result in self.results)

    @property
    def mean_reward(self) -> float:
        if not self.results:
            raise ValueError("evaluation batch must not be empty")
        return sum(result.reward for result in self.results) / len(self.results)

    @property
    def success_rate(self) -> float:
        if not self.results:
            raise ValueError("evaluation batch must not be empty")
        return sum(bool(result.success) for result in self.results) / len(self.results)


@dataclass(frozen=True)
class Action:
    module_id: str
    operator: Any
    index: int


@dataclass(frozen=True)
class EditPatch:
    target_module: str
    operator: Any
    rationale: str
    old_text: str
    new_text: str
    expected_effect: str

    def to_dict(self) -> dict[str, Any]:
        operator = getattr(self.operator, "value", self.operator)
        return {
            "target_module": self.target_module,
            "operator": operator,
            "rationale": self.rationale,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "expected_effect": self.expected_effect,
        }


@dataclass(frozen=True)
class PatchApplication:
    accepted: bool
    skill: SkillArtifact
    reason: str
    changed_tokens: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    probability: float
    probabilities: Any
    entropy: float
    value: float


@dataclass(frozen=True)
class Transition:
    state: Any
    action_index: int
    reward: float
    done: bool
    mask: Any


@dataclass(frozen=True)
class RewardBreakdown:
    paired_delta: float
    length_cost: int
    edit_cost: int
    invalid_cost: int
    length_penalty: float
    edit_penalty: float
    invalid_penalty: float
    total: float


@dataclass(frozen=True)
class GeneratedPatch:
    patch: EditPatch | None
    cache_hit: bool
    request_hash: str
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

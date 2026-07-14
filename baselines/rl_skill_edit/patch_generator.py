from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from .action_space import EditOperator
from .cache import JsonFileCache
from .types import (
    EditPatch,
    EvaluationBatch,
    GeneratedPatch,
    SkillArtifact,
    SkillModule,
    Split,
)


_PATCH_FIELDS = {
    "target_module",
    "operator",
    "rationale",
    "old_text",
    "new_text",
    "expected_effect",
}


def _parse_patch(payload: Any) -> EditPatch:
    if not isinstance(payload, dict):
        raise ValueError("Editor response must be one JSON object")
    if set(payload) != _PATCH_FIELDS:
        missing = sorted(_PATCH_FIELDS - set(payload))
        extra = sorted(set(payload) - _PATCH_FIELDS)
        raise ValueError(
            f"Editor patch fields mismatch: missing={missing}, extra={extra}"
        )
    for key in _PATCH_FIELDS:
        if not isinstance(payload[key], str):
            raise ValueError(f"Editor patch field {key} must be a string")
    if not payload["target_module"].strip():
        raise ValueError("target_module must not be empty")
    try:
        operator = EditOperator(payload["operator"])
    except ValueError as exc:
        raise ValueError(f"unknown edit operator: {payload['operator']}") from exc
    if not payload["rationale"].strip():
        raise ValueError("rationale must not be empty")
    if not payload["expected_effect"].strip():
        raise ValueError("expected_effect must not be empty")
    return EditPatch(
        target_module=payload["target_module"],
        operator=operator,
        rationale=payload["rationale"],
        old_text=payload["old_text"],
        new_text=payload["new_text"],
        expected_effect=payload["expected_effect"],
    )


def _history_to_json(history: Iterable[Any]) -> list[Any]:
    result = []
    for item in history:
        if hasattr(item, "to_dict"):
            result.append(item.to_dict())
        elif is_dataclass(item):
            result.append(asdict(item))
        elif isinstance(item, (dict, str, int, float, bool)) or item is None:
            result.append(item)
        else:
            raise TypeError(f"unsupported edit history item: {type(item).__name__}")
    return result


def _train_evidence(batch: EvaluationBatch) -> list[dict[str, Any]]:
    if batch.split is not Split.TRAIN:
        raise ValueError("Editor context must contain train results only")
    return [result.to_dict() for result in batch.results]


class OpenRouterPatchGenerator:
    def __init__(
        self,
        *,
        client,
        cache: JsonFileCache,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int = 0,
    ) -> None:
        self.client = client
        self.cache = cache
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)

    def _request(
        self,
        skill: SkillArtifact,
        module: SkillModule,
        operator: EditOperator,
        train_batch: EvaluationBatch,
        edit_history: Iterable[Any],
    ) -> tuple[str, str]:
        evidence = _train_evidence(train_batch)
        request = {
            "protocol": "rl-skill-edit-v1",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "skill": {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description,
                "body": skill.body,
            },
            "selected_action": {
                "target_module": module.module_id,
                "module_title": module.title,
                "module_text": module.text,
                "operator": operator.value,
            },
            "train_evidence": evidence,
            "edit_history": _history_to_json(edit_history),
            "required_output_fields": sorted(_PATCH_FIELDS),
        }
        request_json = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        return request_json, request_hash

    def cache_will_hit(
        self,
        skill: SkillArtifact,
        module: SkillModule,
        operator: EditOperator,
        train_batch: EvaluationBatch,
        edit_history: Iterable[Any],
    ) -> bool:
        _, request_hash = self._request(
            skill, module, operator, train_batch, edit_history
        )
        return self.cache.get("editor", request_hash) is not None

    def generate(
        self,
        skill: SkillArtifact,
        module: SkillModule,
        operator: EditOperator,
        train_batch: EvaluationBatch,
        edit_history: Iterable[Any],
    ) -> GeneratedPatch:
        request_json, request_hash = self._request(
            skill, module, operator, train_batch, edit_history
        )
        cached = self.cache.get("editor", request_hash)
        if cached is not None:
            cached_patch = cached.get("patch")
            patch = None if cached_patch is None else _parse_patch(cached_patch)
            return GeneratedPatch(
                patch=patch,
                cache_hit=True,
                request_hash=request_hash,
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "elapsed_s": 0.0,
                },
                error=str(cached.get("error", "")),
            )

        prompt = (
            "You edit one local module of an external skill for a frozen agent. "
            "Apply exactly the selected operator to exactly the selected module. "
            "Return only one JSON object with the required fields; do not rewrite "
            "the whole skill.\n\n" + request_json
        )
        response, usage = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            call_type="editor",
            seed=self.seed,
        )
        if not usage.get("ok", True):
            raise RuntimeError(
                f"Editor call failed: {usage.get('error_kind', 'unknown')}"
            )
        stored_usage = dict(usage)
        try:
            parsed = json.loads(response)
            patch = _parse_patch(parsed)
        except json.JSONDecodeError:
            error = "Editor response is not strict JSON"
            self.cache.set(
                "editor",
                request_hash,
                {
                    "patch": None,
                    "error": error,
                    "response_sha256": hashlib.sha256(
                        response.encode("utf-8")
                    ).hexdigest(),
                    "usage": stored_usage,
                },
            )
            return GeneratedPatch(
                patch=None,
                cache_hit=False,
                request_hash=request_hash,
                usage=stored_usage,
                error=error,
            )
        except ValueError as exc:
            error = str(exc)
            self.cache.set(
                "editor",
                request_hash,
                {
                    "patch": None,
                    "error": error,
                    "response_sha256": hashlib.sha256(
                        response.encode("utf-8")
                    ).hexdigest(),
                    "usage": stored_usage,
                },
            )
            return GeneratedPatch(
                patch=None,
                cache_hit=False,
                request_hash=request_hash,
                usage=stored_usage,
                error=error,
            )
        self.cache.set(
            "editor",
            request_hash,
            {"patch": patch.to_dict(), "error": "", "usage": stored_usage},
        )
        return GeneratedPatch(
            patch=patch,
            cache_hit=False,
            request_hash=request_hash,
            usage=stored_usage,
        )


class MockPatchGenerator:
    def __init__(self, patches: Iterable[EditPatch]) -> None:
        self._patches = list(patches)
        self._index = 0

    def generate(
        self,
        skill: SkillArtifact,
        module: SkillModule,
        operator: EditOperator,
        train_batch: EvaluationBatch,
        edit_history: Iterable[Any],
    ) -> GeneratedPatch:
        del skill, edit_history
        _train_evidence(train_batch)
        if self._index >= len(self._patches):
            raise RuntimeError("mock patch sequence exhausted")
        patch = self._patches[self._index]
        self._index += 1
        if patch.target_module != module.module_id or patch.operator is not operator:
            raise ValueError("mock patch does not match selected action")
        payload = json.dumps(patch.to_dict(), sort_keys=True).encode("utf-8")
        return GeneratedPatch(
            patch=patch,
            cache_hit=False,
            request_hash=hashlib.sha256(payload).hexdigest(),
            usage={},
        )


class SyntheticPatchGenerator:
    """API-free local patch generator used only by the reproducible smoke run."""

    def generate(
        self,
        skill: SkillArtifact,
        module: SkillModule,
        operator: EditOperator,
        train_batch: EvaluationBatch,
        edit_history: Iterable[Any],
    ) -> GeneratedPatch:
        del skill, edit_history
        _train_evidence(train_batch)
        lines = [line for line in module.text.splitlines(keepends=True) if line.strip()]
        if not lines:
            raise ValueError("synthetic smoke module must contain editable text")
        plain = [line.rstrip("\r\n") for line in lines]
        rule_lines = [
            line
            for line in plain
            if "example" not in line.casefold()
            and re.fullmatch(r" {0,3}#{1,6}(?:\s+.*)?", line) is None
        ]
        first = rule_lines[0] if rule_lines else plain[0]
        if operator is EditOperator.ADD_RULE:
            old, new = first, first + "\n- IMPROVED smoke rule."
        elif operator is EditOperator.REWRITE_RULE:
            old, new = first, first + " IMPROVED"
        elif operator is EditOperator.DELETE_RULE:
            old = next(
                (line for line in lines if line.rstrip("\r\n") == first),
                first,
            )
            new = ""
        elif operator is EditOperator.ADD_EXAMPLE:
            old, new = first, first + "\nExample: IMPROVED input -> output."
        elif operator is EditOperator.REWRITE_EXAMPLE:
            examples = [line for line in plain if "example" in line.casefold()]
            old = examples[0] if examples else first
            new = old + " IMPROVED"
        elif operator is EditOperator.MERGE_REDUNDANT_RULES:
            selected = rule_lines[:2]
            old = "\n".join(selected)
            new = selected[0] + " IMPROVED merged."
        elif operator is EditOperator.REORDER_CONTENT:
            selected = rule_lines[:2]
            old = "\n".join(selected)
            new = "\n".join(reversed(selected))
        else:
            raise ValueError("STOP must not call the synthetic patch generator")
        patch = EditPatch(
            target_module=module.module_id,
            operator=operator,
            rationale="Deterministic API-free smoke edit.",
            old_text=old,
            new_text=new,
            expected_effect="Change the mock environment reward.",
        )
        payload = json.dumps(patch.to_dict(), sort_keys=True).encode("utf-8")
        return GeneratedPatch(
            patch=patch,
            cache_hit=False,
            request_hash=hashlib.sha256(payload).hexdigest(),
            usage={},
        )


__all__ = [
    "MockPatchGenerator",
    "OpenRouterPatchGenerator",
    "SyntheticPatchGenerator",
]

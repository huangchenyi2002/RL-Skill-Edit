from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .action_space import EditOperator
from .types import Action, EditPatch, PatchApplication, SkillArtifact, SkillModule


_PROTECTED_MARKERS = (
    ("<!-- SLOW_UPDATE_START -->", "<!-- SLOW_UPDATE_END -->"),
    ("<!-- APPENDIX_START -->", "<!-- APPENDIX_END -->"),
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, 1):
        current = [row]
        for column, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _intersects_protected_region(body: str, start_index: int, end_index: int) -> bool:
    for start_marker, end_marker in _PROTECTED_MARKERS:
        cursor = 0
        while True:
            protected_start = body.find(start_marker, cursor)
            if protected_start < 0:
                break
            marker_end = body.find(end_marker, protected_start + len(start_marker))
            protected_end = (
                len(body) if marker_end < 0 else marker_end + len(end_marker)
            )
            if start_index < protected_end and end_index > protected_start:
                return True
            cursor = protected_end
    return False


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_example(text: str) -> bool:
    lowered = text.casefold()
    return "example" in lowered or "```" in text


def _is_heading_line(line: str) -> bool:
    return re.fullmatch(r" {0,3}#{1,6}(?:\s+.*)?", line.strip("\r\n")) is not None


def _heading_signature(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip("\r\n") for line in text.splitlines() if _is_heading_line(line)
    )


def _is_rule_text(text: str) -> bool:
    lines = _nonempty_lines(text)
    return (
        bool(lines)
        and not _is_example(text)
        and not any(_is_heading_line(line) for line in lines)
    )


def _operator_semantics(operator: EditOperator, old: str, new: str) -> bool:
    if operator is EditOperator.ADD_RULE:
        return bool(
            _is_rule_text(old)
            and old in new
            and len(_nonempty_lines(new)) > len(_nonempty_lines(old))
        )
    if operator is EditOperator.REWRITE_RULE:
        return bool(_is_rule_text(old) and _is_rule_text(new) and old != new)
    if operator is EditOperator.DELETE_RULE:
        return bool(_is_rule_text(old) and new == "")
    if operator is EditOperator.ADD_EXAMPLE:
        return bool(old and new and old in new and new != old and _is_example(new))
    if operator is EditOperator.REWRITE_EXAMPLE:
        return bool(
            old and new and old != new and _is_example(old) and _is_example(new)
        )
    if operator is EditOperator.MERGE_REDUNDANT_RULES:
        return bool(
            old
            and new
            and _is_rule_text(old)
            and _is_rule_text(new)
            and len(_nonempty_lines(old)) >= 2
            and len(_nonempty_lines(new)) < len(_nonempty_lines(old))
        )
    if operator is EditOperator.REORDER_CONTENT:
        old_lines = _nonempty_lines(old)
        new_lines = _nonempty_lines(new)
        return bool(
            old_lines
            and old_lines != new_lines
            and Counter(old_lines) == Counter(new_lines)
        )
    return False


def _reject(skill: SkillArtifact, reason: str) -> PatchApplication:
    return PatchApplication(False, skill, reason, 0)


def validate_and_apply_patch(
    skill: SkillArtifact,
    modules: Iterable[SkillModule],
    selected_action: Action,
    patch: EditPatch,
    limits: dict,
) -> PatchApplication:
    module_by_id = {module.module_id: module for module in modules}
    module = module_by_id.get(selected_action.module_id)
    if module is None:
        return _reject(skill, "selected module is not present")
    if patch.target_module != selected_action.module_id:
        return _reject(skill, "patch target does not match selected module")
    if patch.operator is not selected_action.operator:
        return _reject(skill, "patch operator does not match selected operator")
    if patch.operator is EditOperator.STOP:
        return _reject(skill, "STOP does not accept an Editor patch")
    if not _operator_semantics(patch.operator, patch.old_text, patch.new_text):
        return _reject(skill, "patch does not satisfy operator semantics")
    if not patch.old_text or skill.body.count(patch.old_text) != 1:
        return _reject(skill, "old_text must occur exactly once in the skill")
    if module.text.count(patch.old_text) != 1:
        return _reject(skill, "old_text must occur exactly once in the selected module")

    local_index = module.text.find(patch.old_text)
    absolute_index = module.start + local_index
    absolute_end = absolute_index + len(patch.old_text)
    if absolute_index < module.start or absolute_end > module.end:
        return _reject(skill, "patch crosses the selected module boundary")
    if absolute_index == 0 and absolute_end == len(skill.body):
        return _reject(skill, "patch may not replace the entire skill")
    if module.level > 0 and local_index == 0:
        return _reject(skill, "patch may not modify a module heading")
    if _intersects_protected_region(skill.body, absolute_index, absolute_end):
        return _reject(skill, "patch targets a protected region")
    if any(marker in patch.new_text for pair in _PROTECTED_MARKERS for marker in pair):
        return _reject(skill, "patch introduces a protected marker")

    old_tokens = _tokens(patch.old_text)
    new_tokens = _tokens(patch.new_text)
    changed_tokens = _levenshtein(old_tokens, new_tokens)
    growth = max(0, len(new_tokens) - len(old_tokens))
    if changed_tokens > int(limits["max_changed_tokens"]):
        return _reject(skill, "patch exceeds max_changed_tokens")
    if growth > int(limits["max_length_growth_tokens"]):
        return _reject(skill, "patch exceeds max_length_growth_tokens")

    updated_body = (
        skill.body[:absolute_index]
        + patch.new_text
        + skill.body[absolute_index + len(patch.old_text) :]
    )
    if _heading_signature(updated_body) != _heading_signature(skill.body):
        return _reject(skill, "patch may not modify Markdown heading topology")
    if len(_tokens(updated_body)) > int(limits["max_skill_tokens"]):
        return _reject(skill, "candidate exceeds max_skill_tokens")
    return PatchApplication(
        accepted=True,
        skill=skill.with_body(updated_body),
        reason="applied",
        changed_tokens=changed_tokens,
    )


__all__ = ["validate_and_apply_patch"]

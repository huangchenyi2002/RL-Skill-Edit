from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from .types import EvaluationBatch, ModuleDiagnostics, SkillModule, TaskResult


_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)[ \t]*|[ \t]*)$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "module",
    "of",
    "on",
    "or",
    "rule",
    "that",
    "the",
    "this",
    "to",
    "use",
    "was",
    "were",
    "with",
}


def _heading_title(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING.fullmatch(line.rstrip("\r\n"))
    if match is None:
        return None
    raw_title = (match.group(2) or "").strip()
    title = re.sub(r"[ \t]+#+[ \t]*$", "", raw_title).strip()
    return len(match.group(1)), title


def _heading_positions(body: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    offset = 0
    fence_character: str | None = None
    fence_length = 0

    for line in body.splitlines(keepends=True):
        line_without_newline = line.rstrip("\r\n")
        if fence_character is not None:
            stripped = line_without_newline.lstrip(" ")
            leading_spaces = len(line_without_newline) - len(stripped)
            if leading_spaces <= 3:
                closing = re.fullmatch(
                    rf"{re.escape(fence_character)}{{{fence_length},}}[ \t]*",
                    stripped,
                )
                if closing is not None:
                    fence_character = None
                    fence_length = 0
            offset += len(line)
            continue

        fence = _FENCE_OPEN.match(line_without_newline)
        if fence is not None:
            marker = fence.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            offset += len(line)
            continue

        heading = _heading_title(line)
        if heading is not None:
            level, title = heading
            headings.append((offset, level, title))
        offset += len(line)

    return headings


def _slug(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    slug = re.sub(r"[^\w]+", "-", normalized, flags=re.UNICODE).strip("-_")
    return slug or "untitled"


def parse_modules(body: str, max_modules: int) -> tuple[SkillModule, ...]:
    """Split Markdown into a global prefix and non-overlapping ATX sections."""

    if not isinstance(body, str):
        raise TypeError("body must be a string")
    if (
        isinstance(max_modules, bool)
        or not isinstance(max_modules, int)
        or max_modules < 1
    ):
        raise ValueError("max_modules must be a positive integer")

    headings = _heading_positions(body)
    module_count = 1 + len(headings)
    if module_count > max_modules:
        raise ValueError(
            f"parsed {module_count} modules, which exceeds max_modules={max_modules}"
        )

    first_heading = headings[0][0] if headings else len(body)
    modules = [
        SkillModule(
            module_id="global",
            title="global",
            level=0,
            start=0,
            end=first_heading,
            text=body[:first_heading],
            slot=0,
        )
    ]

    occurrences: dict[tuple[int, str], int] = {}
    for index, (start, level, title) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(body)
        slug = _slug(title)
        identity = (level, slug)
        occurrence = occurrences.get(identity, 0) + 1
        occurrences[identity] = occurrence
        modules.append(
            SkillModule(
                module_id=f"h{level}-{slug}-{occurrence}",
                title=title,
                level=level,
                start=start,
                end=end,
                text=body[start:end],
                slot=index + 1,
            )
        )

    return tuple(modules)


def _normalize_word(word: str) -> str:
    normalized = unicodedata.normalize("NFKC", word).casefold()
    if (
        normalized.isascii()
        and len(normalized) > 3
        and normalized.endswith("s")
        and not normalized.endswith("ss")
    ):
        normalized = normalized[:-1]
    return normalized


def _keywords(text: str) -> set[str]:
    words = {_normalize_word(match.group(0)) for match in _WORD.finditer(text)}
    return {word for word in words if len(word) >= 2 and word not in _STOPWORDS}


def _visible_failure_text(result: TaskResult) -> str:
    fields = (
        result.feedback,
        result.evaluator_output,
        result.final_answer,
        *result.visible_logs,
    )
    return "\n".join(field for field in fields if field)


def _attributed_module_id(
    modules: Sequence[SkillModule],
    result: TaskResult,
) -> str:
    evidence = _keywords(_visible_failure_text(result))
    if not evidence:
        return "global"

    scored: list[tuple[tuple[int, int, int], str]] = []
    for module in modules:
        title_overlap = evidence & _keywords(module.title)
        body_overlap = evidence & _keywords(module.text)
        union_overlap = title_overlap | body_overlap
        if union_overlap:
            score = (
                2 * len(title_overlap) + len(body_overlap),
                len(title_overlap),
                len(union_overlap),
            )
            scored.append((score, module.module_id))

    if not scored:
        return "global"
    best_score = max(score for score, _ in scored)
    winners = [module_id for score, module_id in scored if score == best_score]
    return winners[0] if len(winners) == 1 else "global"


def attribute_failures(
    modules: Sequence[SkillModule],
    batch: EvaluationBatch,
) -> dict[str, ModuleDiagnostics]:
    """Attribute failed tasks using only agent-visible textual evidence."""

    if not modules:
        raise ValueError("modules must not be empty")
    module_ids = [module.module_id for module in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("module IDs must be unique")
    if module_ids.count("global") != 1:
        raise ValueError("modules must contain exactly one global module")

    assigned: dict[str, list[TaskResult]] = {module_id: [] for module_id in module_ids}
    for result in batch.results:
        if result.success:
            continue
        assigned[_attributed_module_id(modules, result)].append(result)

    diagnostics: dict[str, ModuleDiagnostics] = {}
    sample_count = len(batch.results)
    mean_reward = batch.mean_reward if sample_count else 0.0
    success_rate = batch.success_rate if sample_count else 0.0
    for module in modules:
        failures = assigned[module.module_id]
        count = len(failures)
        diagnostics[module.module_id] = ModuleDiagnostics(
            module_id=module.module_id,
            failure_count=count,
            task_ids=tuple(result.task_id for result in failures),
            mean_reward=mean_reward,
            success_rate=success_rate,
            sample_count=sample_count,
        )
    return diagnostics


__all__ = ["attribute_failures", "parse_modules"]

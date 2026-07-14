from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Split


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_task_fingerprint(
    task: dict[str, Any],
    *,
    init_sha256: str,
    golden_sha256: str,
) -> str:
    spreadsheet = task["spreadsheet"]
    identity = {
        "init_sha256": init_sha256,
        "golden_sha256": golden_sha256,
        "answer_sheet": str(spreadsheet.get("answer_sheet", "")),
        "answer_position": str(spreadsheet.get("answer_position", "")),
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_content_fingerprint(task: dict[str, Any], entity_fingerprint: str) -> str:
    content = {
        key: value
        for key, value in task.items()
        if key not in {"task_id", "spreadsheet"}
    }
    content["entity_fingerprint"] = entity_fingerprint
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TaskManifest:
    split: Split
    source_path: Path
    tasks: tuple[dict[str, Any], ...]
    ordered_task_ids: tuple[str, ...]
    task_fingerprints: tuple[str, ...]
    init_workbook_fingerprints: tuple[str, ...]
    golden_workbook_fingerprints: tuple[str, ...]
    task_content_fingerprints: tuple[str, ...]
    digest: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        split: Split,
        expected_size: int,
    ) -> "TaskManifest":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"{split.value} manifest must be a non-empty list")
        if len(payload) != int(expected_size):
            raise ValueError(
                f"{split.value} manifest size {len(payload)} != {expected_size}"
            )

        tasks: list[dict[str, Any]] = []
        task_ids: list[str] = []
        fingerprints: list[str] = []
        init_workbook_fingerprints: list[str] = []
        golden_workbook_fingerprints: list[str] = []
        content_fingerprints: list[str] = []
        seen_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        for index, raw_task in enumerate(payload):
            if not isinstance(raw_task, dict):
                raise ValueError(f"task {index} must be an object")
            raw_id = raw_task.get("task_id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError(f"task {index} has an empty task_id")
            task_id = raw_id.strip()
            if task_id in seen_ids:
                raise ValueError(f"duplicate task_id in {split.value}: {task_id}")
            seen_ids.add(task_id)

            raw_spreadsheet = raw_task.get("spreadsheet")
            if not isinstance(raw_spreadsheet, dict):
                raise ValueError(f"task {task_id} has no spreadsheet object")
            spreadsheet = dict(raw_spreadsheet)
            for key in ("init_file", "golden_file"):
                file_path = Path(str(spreadsheet.get(key, ""))).expanduser()
                if not file_path.is_absolute():
                    file_path = source.parent / file_path
                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"task {task_id} {key} does not exist: {file_path}"
                    )
                spreadsheet[key] = str(file_path.resolve())

            task = dict(raw_task)
            task["task_id"] = task_id
            task["spreadsheet"] = spreadsheet
            init_workbook_fingerprint = _sha256_file(Path(spreadsheet["init_file"]))
            golden_workbook_fingerprint = _sha256_file(Path(spreadsheet["golden_file"]))
            fingerprint = _canonical_task_fingerprint(
                task,
                init_sha256=init_workbook_fingerprint,
                golden_sha256=golden_workbook_fingerprint,
            )
            if fingerprint in seen_fingerprints:
                raise ValueError(
                    f"duplicate canonical task entity in {split.value}: {task_id}"
                )
            seen_fingerprints.add(fingerprint)
            fingerprints.append(fingerprint)
            init_workbook_fingerprints.append(init_workbook_fingerprint)
            golden_workbook_fingerprints.append(golden_workbook_fingerprint)
            content_fingerprint = _task_content_fingerprint(task, fingerprint)
            task["_entity_fingerprint"] = fingerprint
            task["_content_fingerprint"] = content_fingerprint
            tasks.append(task)
            task_ids.append(task_id)
            content_fingerprints.append(content_fingerprint)

        digest_payload = json.dumps(
            {
                "split": split.value,
                "task_ids": task_ids,
                "fingerprints": fingerprints,
                "init_workbook_fingerprints": init_workbook_fingerprints,
                "golden_workbook_fingerprints": golden_workbook_fingerprints,
                "content_fingerprints": content_fingerprints,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            split=split,
            source_path=source,
            tasks=tuple(tasks),
            ordered_task_ids=tuple(task_ids),
            task_fingerprints=tuple(fingerprints),
            init_workbook_fingerprints=tuple(init_workbook_fingerprints),
            golden_workbook_fingerprints=tuple(golden_workbook_fingerprints),
            task_content_fingerprints=tuple(content_fingerprints),
            digest=hashlib.sha256(digest_payload).hexdigest(),
        )


def validate_manifests(*manifests: TaskManifest) -> None:
    if len(manifests) not in {2, 3}:
        raise ValueError(
            "validate_manifests requires train/validation, optionally test"
        )
    expected = (Split.TRAIN, Split.VALIDATION, Split.TEST)[: len(manifests)]
    if tuple(manifest.split for manifest in manifests) != expected:
        raise ValueError("manifests must be ordered train, validation, optionally test")
    for left_index, left in enumerate(manifests):
        for right in manifests[left_index + 1 :]:
            id_overlap = set(left.ordered_task_ids) & set(right.ordered_task_ids)
            if id_overlap:
                raise ValueError(
                    f"task_id overlap between {left.split.value} and "
                    f"{right.split.value}: {sorted(id_overlap)[:5]}"
                )
            fingerprint_overlap = set(left.task_fingerprints) & set(
                right.task_fingerprints
            )
            if fingerprint_overlap:
                raise ValueError(
                    f"canonical task overlap between {left.split.value} "
                    f"and {right.split.value}"
                )
            left_workbook_fingerprints = set(left.init_workbook_fingerprints) | set(
                left.golden_workbook_fingerprints
            )
            right_workbook_fingerprints = set(right.init_workbook_fingerprints) | set(
                right.golden_workbook_fingerprints
            )
            if left_workbook_fingerprints & right_workbook_fingerprints:
                raise ValueError(
                    f"workbook content overlap between {left.split.value} "
                    f"and {right.split.value}"
                )


__all__ = ["Split", "TaskManifest", "validate_manifests"]

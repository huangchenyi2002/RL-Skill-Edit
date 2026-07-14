import json
import shutil
from pathlib import Path

import pytest
from openpyxl import Workbook

from rl_skill_edit.manifest import (
    Split,
    TaskManifest,
    validate_manifests,
)


def _write_workbook(path: Path, value: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Data"
    worksheet["A1"] = value
    workbook.save(path)


def _make_task(tmp_path: Path, task_id: object, stem: str) -> dict:
    init_file = tmp_path / f"{stem}_input.xlsx"
    golden_file = tmp_path / f"{stem}_golden.xlsx"
    _write_workbook(init_file, f"input-{stem}")
    _write_workbook(golden_file, f"golden-{stem}")
    return {
        "task_id": task_id,
        "description": f"Complete spreadsheet task {stem}.",
        "spreadsheet": {
            "init_file": str(init_file),
            "golden_file": str(golden_file),
            "answer_sheet": "Data",
            "answer_position": "A1",
        },
    }


def _write_manifest(path: Path, tasks: list[dict]) -> Path:
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def _load(
    tmp_path: Path,
    split: Split,
    tasks: list[dict],
) -> TaskManifest:
    path = _write_manifest(tmp_path / f"{split.value}.json", tasks)
    return TaskManifest.load(path, split=split, expected_size=len(tasks))


def test_manifest_load_preserves_declared_task_order(tmp_path: Path) -> None:
    tasks = [
        _make_task(tmp_path, "task-b", "b"),
        _make_task(tmp_path, "task-a", "a"),
    ]

    manifest = TaskManifest.load(
        _write_manifest(tmp_path / "train.json", tasks),
        split=Split.TRAIN,
        expected_size=2,
    )

    assert [task["task_id"] for task in manifest.tasks] == ["task-b", "task-a"]


@pytest.mark.parametrize("empty_id", ["", "   ", None])
def test_manifest_rejects_empty_task_ids(tmp_path: Path, empty_id: object) -> None:
    task = _make_task(tmp_path, empty_id, "empty-id")

    with pytest.raises(ValueError):
        _load(tmp_path, Split.TRAIN, [task])


def test_manifest_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    tasks = [
        _make_task(tmp_path, "duplicate", "first"),
        _make_task(tmp_path, "duplicate", "second"),
    ]

    with pytest.raises(ValueError):
        _load(tmp_path, Split.TRAIN, tasks)


def test_manifest_requires_exact_configured_size(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "train.json",
        [_make_task(tmp_path, "only-one", "only-one")],
    )

    with pytest.raises(ValueError):
        TaskManifest.load(path, split=Split.TRAIN, expected_size=2)


@pytest.mark.parametrize("missing_key", ["init_file", "golden_file"])
def test_manifest_rejects_missing_spreadsheet_files(
    tmp_path: Path,
    missing_key: str,
) -> None:
    task = _make_task(tmp_path, "missing-file", missing_key)
    task["spreadsheet"][missing_key] = str(tmp_path / "does-not-exist.xlsx")

    with pytest.raises(FileNotFoundError):
        _load(tmp_path, Split.TRAIN, [task])


@pytest.mark.parametrize(
    ("left_split", "right_split"),
    [
        (Split.TRAIN, Split.VALIDATION),
        (Split.TRAIN, Split.TEST),
        (Split.VALIDATION, Split.TEST),
    ],
)
def test_validation_rejects_id_overlap_between_every_split_pair(
    tmp_path: Path,
    left_split: Split,
    right_split: Split,
) -> None:
    tasks = {
        Split.TRAIN: _make_task(tmp_path, "train", "train"),
        Split.VALIDATION: _make_task(tmp_path, "validation", "validation"),
        Split.TEST: _make_task(tmp_path, "test", "test"),
    }
    tasks[left_split]["task_id"] = "shared-id"
    tasks[right_split]["task_id"] = "shared-id"
    manifests = {split: _load(tmp_path, split, [task]) for split, task in tasks.items()}

    with pytest.raises(ValueError):
        validate_manifests(
            manifests[Split.TRAIN],
            manifests[Split.VALIDATION],
            manifests[Split.TEST],
        )


def test_validation_rejects_canonical_alias_with_a_different_id_and_paths(
    tmp_path: Path,
) -> None:
    train_task = _make_task(tmp_path, "train-id", "canonical-source")
    validation_task = _make_task(tmp_path, "validation-id", "validation")
    test_task = _make_task(tmp_path, "alias-id", "canonical-copy")

    test_task["description"] = train_task["description"]
    shutil.copyfile(
        train_task["spreadsheet"]["init_file"],
        test_task["spreadsheet"]["init_file"],
    )
    shutil.copyfile(
        train_task["spreadsheet"]["golden_file"],
        test_task["spreadsheet"]["golden_file"],
    )

    train = _load(tmp_path, Split.TRAIN, [train_task])
    validation = _load(tmp_path, Split.VALIDATION, [validation_task])
    test = _load(tmp_path, Split.TEST, [test_task])

    with pytest.raises(ValueError):
        validate_manifests(train, validation, test)


def test_canonical_alias_cannot_hide_behind_reworded_description(
    tmp_path: Path,
) -> None:
    train_task = _make_task(tmp_path, "train-id", "shared-entity")
    validation_task = _make_task(tmp_path, "validation-id", "validation-entity")
    test_task = _make_task(tmp_path, "test-id", "unused-test-files")
    test_task["description"] = "Completely different wording for the same task."
    test_task["spreadsheet"] = dict(train_task["spreadsheet"])

    with pytest.raises(ValueError, match="canonical"):
        validate_manifests(
            _load(tmp_path, Split.TRAIN, [train_task]),
            _load(tmp_path, Split.VALIDATION, [validation_task]),
            _load(tmp_path, Split.TEST, [test_task]),
        )


@pytest.mark.parametrize(
    ("source_field", "target_field"),
    [
        ("init_file", "init_file"),
        ("golden_file", "golden_file"),
        ("init_file", "golden_file"),
    ],
)
def test_workbook_content_overlap_cannot_hide_behind_changed_answer_range(
    tmp_path: Path,
    source_field: str,
    target_field: str,
) -> None:
    train_task = _make_task(tmp_path, "train-id", "shared-workbooks")
    validation_task = _make_task(tmp_path, "validation-id", "validation")
    test_task = _make_task(tmp_path, "test-id", "unused-test-workbooks")
    test_task["spreadsheet"][target_field] = train_task["spreadsheet"][source_field]
    test_task["spreadsheet"]["answer_position"] = "B2"

    with pytest.raises(ValueError, match="workbook content overlap"):
        validate_manifests(
            _load(tmp_path, Split.TRAIN, [train_task]),
            _load(tmp_path, Split.VALIDATION, [validation_task]),
            _load(tmp_path, Split.TEST, [test_task]),
        )


def test_manifest_rejects_duplicate_canonical_entities_inside_one_split(
    tmp_path: Path,
) -> None:
    first = _make_task(tmp_path, "first-id", "same-entity")
    second = _make_task(tmp_path, "second-id", "unused-second-files")
    second["description"] = "Reworded duplicate."
    second["spreadsheet"] = dict(first["spreadsheet"])

    with pytest.raises(ValueError, match="canonical|duplicate"):
        _load(tmp_path, Split.TRAIN, [first, second])


def test_train_and_validation_can_be_checked_before_test_is_loaded(
    tmp_path: Path,
) -> None:
    train = _load(
        tmp_path,
        Split.TRAIN,
        [_make_task(tmp_path, "train-id", "train-only")],
    )
    validation = _load(
        tmp_path,
        Split.VALIDATION,
        [_make_task(tmp_path, "validation-id", "validation-only")],
    )

    validate_manifests(train, validation)

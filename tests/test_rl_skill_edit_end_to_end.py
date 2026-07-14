from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import rl_skill_edit.cli as cli
from rl_skill_edit.cli import parse_args, run


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_FIELDS = {
    "protocol",
    "method",
    "best_skill_digest",
    "initial_skill_digest",
    "config_sha256",
    "split_digest",
    "split_digests",
    "implementation_sha256",
    "dependency_sha256",
    "summary_sha256",
    "seed",
    "skill_identity",
}
OUTPUT_MARKER_FILE = ".rl-skill-edit-output.json"
OUTPUT_MARKER_FIELDS = {"protocol", "method", "final_output_path"}


def test_cli_module_exists() -> None:
    assert importlib.util.find_spec("rl_skill_edit.cli") is not None


def _smoke_config_in(tmp_path: Path) -> Path:
    source = yaml.safe_load(
        (ROOT / "configs/rl_skill_edit_smoke.yaml").read_text(encoding="utf-8")
    )
    data_dir = tmp_path / "mock_data"
    shutil.copytree(ROOT / "data/mock_rl_skill_edit", data_dir)
    output = tmp_path / "result"
    source["paths"].update(
        {
            "initial_skill": str(data_dir / "initial_skill.md"),
            "train_manifest": str(data_dir / "train.json"),
            "validation_manifest": str(data_dir / "validation.json"),
            "test_manifest": str(data_dir / "test.json"),
            "output_dir": str(output),
            "rl_skill": str(output / "rl_skill_edit/best_rl_skill.md"),
            "rl_summary": str(output / "rl_skill_edit/rl_optimization_summary.json"),
            "rl_provenance": str(output / "rl_skill_edit/freeze_provenance.json"),
        }
    )
    target = tmp_path / "smoke.yaml"
    target.write_text(
        yaml.safe_dump(source, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_config(
    path: Path, value: dict[str, Any], *, sort_keys: bool = False
) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=sort_keys, allow_unicode=True),
        encoding="utf-8",
    )


def _provenance_path(config_path: Path) -> Path:
    return Path(_config(config_path)["paths"]["rl_provenance"])


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _tree_snapshot(path: Path) -> dict[str, tuple[Any, ...]]:
    if not os.path.lexists(path):
        return {}
    snapshot: dict[str, tuple[Any, ...]] = {".": ("directory",)}
    for item in sorted(path.rglob("*")):
        relative = str(item.relative_to(path))
        if item.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(item))
        elif item.is_dir():
            snapshot[relative] = ("directory",)
        elif item.is_file():
            snapshot[relative] = ("file", item.read_bytes())
        else:
            snapshot[relative] = ("special",)
    return snapshot


def _previous_output_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.previous"


def _set_output_paths(config: dict[str, Any], output_dir: Path) -> None:
    config["paths"].update(
        {
            "output_dir": str(output_dir),
            "rl_skill": str(output_dir / "rl_skill_edit/best_rl_skill.md"),
            "rl_summary": str(
                output_dir / "rl_skill_edit/rl_optimization_summary.json"
            ),
            "rl_provenance": str(output_dir / "rl_skill_edit/freeze_provenance.json"),
        }
    )


def _write_expected_output_marker(root: Path, output_dir: Path) -> Path:
    marker = root / OUTPUT_MARKER_FILE
    marker.write_text(
        json.dumps(
            {
                "protocol": "rl-skill-edit-output-v1",
                "method": "rl_skill_edit",
                "final_output_path": str(output_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def test_parser_has_only_the_fixed_rl_flags() -> None:
    args = parse_args(["--config", "config.yaml", "--seed", "9", "--test-only"])
    assert args.config == Path("config.yaml")
    assert args.seed == 9
    assert args.test_only is True

    with pytest.raises(SystemExit):
        parse_args(["--config", "config.yaml", "--methods", "rl_skill_edit"])


def test_api_free_training_and_test_only_report_initial_and_rl(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42, test_only=False)
    frozen = run(config_path, seed=42, test_only=True)

    assert trained["methods"] == ["initial_skill", "rl_skill_edit"]
    assert frozen["methods"] == ["initial_skill", "rl_skill_edit"]
    assert (
        trained["test_rewards"]["rl_skill_edit"]
        > trained["test_rewards"]["initial_skill"]
    )
    assert frozen["test_rewards"] == trained["test_rewards"]

    output = Path(trained["output_dir"])
    assert (output / "rl_skill_edit/best_rl_skill.md").is_file()
    assert (output / "rl_skill_edit/final_rl_policy.pt").is_file()
    assert (output / "rl_skill_edit/rl_training_log.jsonl").is_file()
    assert (output / "rl_skill_edit/rl_episode_summary.csv").is_file()
    assert (output / "rl_skill_edit/freeze_provenance.json").is_file()
    assert (output / "test_task_level_results.csv").is_file()
    assert (output / "method_comparison.csv").is_file()

    provenance = json.loads(
        (output / "rl_skill_edit/freeze_provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance) == PROVENANCE_FIELDS
    assert provenance["method"] == "rl_skill_edit"
    assert set(provenance["split_digests"]) == {"train", "validation", "test"}

    manifest = json.loads(
        (output / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["method"] == "rl_skill_edit"
    assert manifest["methods"] == ["initial_skill", "rl_skill_edit"]
    assert "current_method_artifact_provenance" not in manifest
    assert manifest["dependency_sha256"] == provenance["dependency_sha256"]

    with (output / "test_task_level_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {
        "initial_skill",
        "rl_skill_edit",
    }


def test_published_output_has_exact_stable_ownership_marker(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    config = _config(config_path)
    expected_config_sha256 = cli._normalized_config_sha256(config)

    trained = run(config_path, seed=42)

    output_dir = Path(trained["output_dir"])
    marker_path = output_dir / OUTPUT_MARKER_FILE
    marker_text = marker_path.read_text(encoding="utf-8")
    marker = json.loads(marker_text)
    manifest = json.loads(
        (output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert set(marker) == OUTPUT_MARKER_FIELDS
    assert marker == {
        "protocol": "rl-skill-edit-output-v1",
        "method": "rl_skill_edit",
        "final_output_path": str(output_dir),
    }
    assert ".rl-output-staging-" not in marker_text
    assert manifest["config_sha256"] == expected_config_sha256


def test_staged_output_rejects_marker_with_unknown_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    output_dir = Path(_config(config_path)["paths"]["output_dir"])
    original_write = cli._atomic_json_write

    def write_invalid_marker(path: Path, value: dict[str, Any]) -> None:
        if path.name == OUTPUT_MARKER_FILE:
            value = {**value, "unknown": "forbidden"}
        original_write(path, value)

    monkeypatch.setattr(cli, "_atomic_json_write", write_invalid_marker)

    with pytest.raises(ValueError, match="unknown fields"):
        run(config_path, seed=42)
    assert not os.path.lexists(output_dir)
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


def test_test_manifest_is_loaded_only_after_rl_optimization_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    test_manifest = Path(_config(config_path)["paths"]["test_manifest"])
    events: list[str] = []
    original_optimize = cli.RLSkillEditOptimizer.optimize
    original_load_test = cli._load_test_manifest
    original_read_text = Path.read_text
    original_open = Path.open

    def optimize_then_mark(self, **kwargs):
        result = original_optimize(self, **kwargs)
        events.append("optimized")
        return result

    def load_test_after_freeze(config, optimization_manifests):
        assert events == ["optimized"]
        events.append("test_loaded")
        return original_load_test(config, optimization_manifests)

    def guarded_read_text(path: Path, *args, **kwargs):
        if path == test_manifest and not events:
            raise AssertionError("Test manifest was read before optimization froze")
        return original_read_text(path, *args, **kwargs)

    def guarded_open(path: Path, *args, **kwargs):
        if path == test_manifest and not events:
            raise AssertionError("Test manifest was opened before optimization froze")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(cli.RLSkillEditOptimizer, "optimize", optimize_then_mark)
    monkeypatch.setattr(cli, "_load_test_manifest", load_test_after_freeze)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "open", guarded_open)

    run(config_path, seed=42)
    assert events == ["optimized", "test_loaded"]


def test_frozen_bundle_records_only_resolvable_relative_artifact_paths(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    method_dir = Path(trained["output_dir"]) / "rl_skill_edit"

    log_rows = tuple(
        json.loads(line)
        for line in (method_dir / "rl_training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    with (method_dir / "rl_episode_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        summary_rows = tuple(csv.DictReader(handle))
    recorded_paths = [
        row[field]
        for row in log_rows
        for field in ("current_skill_path", "candidate_skill_path")
    ]
    recorded_paths.extend(
        row[field]
        for row in summary_rows
        for field in (
            "final_skill_path",
            "best_skill_path",
            "policy_checkpoint_path",
            "trajectory_path",
        )
    )

    assert recorded_paths
    for value in recorded_paths:
        reference = Path(value)
        assert not reference.is_absolute()
        assert (method_dir / reference).is_file()
    for artifact in method_dir.rglob("*"):
        if artifact.is_file() and artifact.suffix in {".json", ".jsonl", ".csv"}:
            assert ".rl-training-" not in artifact.read_text(encoding="utf-8")


def test_normalized_config_digest_ignores_yaml_key_order(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    _write_config(config_path, _config(config_path), sort_keys=True)

    frozen = run(config_path, seed=42, test_only=True)
    assert frozen["test_rewards"] == trained["test_rewards"]


@pytest.mark.parametrize(
    ("tamper", "field"),
    (
        ("skill", "best_skill_digest"),
        ("config", "config_sha256"),
        ("split", "split_digest"),
        ("implementation", "implementation_sha256"),
        ("dependency", "dependency_sha256"),
        ("seed", "seed"),
    ),
)
def test_test_only_rejects_each_tampered_provenance_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    field: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    run(config_path, seed=42)
    test_seed = 42

    if tamper == "skill":
        skill_path = Path(_config(config_path)["paths"]["rl_skill"])
        skill_path.write_text(
            skill_path.read_text(encoding="utf-8") + "\nTampered.\n",
            encoding="utf-8",
        )
    elif tamper == "config":
        value = _config(config_path)
        value["policy"]["learning_rate"] = 0.02
        _write_config(config_path, value)
    elif tamper == "split":
        train_path = Path(_config(config_path)["paths"]["train_manifest"])
        tasks = json.loads(train_path.read_text(encoding="utf-8"))
        tasks[0]["description"] += " tampered"
        train_path.write_text(json.dumps(tasks), encoding="utf-8")
    elif tamper == "implementation":
        monkeypatch.setattr(cli, "_implementation_sha256", lambda: "0" * 64)
    elif tamper == "dependency":
        monkeypatch.setattr(cli, "_dependency_sha256", lambda: "1" * 64)
    elif tamper == "seed":
        test_seed = 43
    else:
        raise AssertionError(tamper)

    with pytest.raises(ValueError, match=rf"provenance mismatch for {field}"):
        run(config_path, seed=test_seed, test_only=True)


@pytest.mark.parametrize("mutation", ("missing", "unknown"))
def test_test_only_rejects_missing_or_unknown_provenance_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    run(config_path, seed=42)
    provenance_path = _provenance_path(config_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        provenance.pop("dependency_sha256")
    else:
        provenance["unknown"] = "forbidden"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance fields"):
        run(config_path, seed=42, test_only=True)


def test_test_only_rejects_boolean_seed_even_when_it_equals_integer_one(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    run(config_path, seed=1)
    provenance_path = _provenance_path(config_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["seed"] = True
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"provenance\.seed"):
        run(config_path, seed=1, test_only=True)


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    (
        ("protocol_type", TypeError, r"provenance\.protocol"),
        ("method_type", TypeError, r"provenance\.method"),
        ("digest_type", TypeError, r"provenance\.best_skill_digest"),
        ("digest_shape", ValueError, "64 lowercase hexadecimal"),
        ("split_extra", ValueError, r"provenance\.split_digests"),
        ("split_value_type", TypeError, r"provenance\.split_digests\.train"),
        ("identity_extra", ValueError, r"provenance\.skill_identity"),
        ("identity_value_type", TypeError, r"provenance\.skill_identity\.name"),
    ),
)
def test_test_only_validates_nested_provenance_schema_before_comparison(
    tmp_path: Path,
    mutation: str,
    error: type[Exception],
    message: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    run(config_path, seed=42)
    provenance_path = _provenance_path(config_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if mutation == "protocol_type":
        provenance["protocol"] = False
    elif mutation == "method_type":
        provenance["method"] = 7
    elif mutation == "digest_type":
        provenance["best_skill_digest"] = 7
    elif mutation == "digest_shape":
        provenance["best_skill_digest"] = "not-a-digest"
    elif mutation == "split_extra":
        provenance["split_digests"]["holdout"] = "0" * 64
    elif mutation == "split_value_type":
        provenance["split_digests"]["train"] = True
    elif mutation == "identity_extra":
        provenance["skill_identity"]["unknown"] = "forbidden"
    elif mutation == "identity_value_type":
        provenance["skill_identity"]["name"] = False
    else:
        raise AssertionError(mutation)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(error, match=message):
        run(config_path, seed=42, test_only=True)


@pytest.mark.parametrize("location", ("top", "student"))
def test_config_rejects_unknown_top_level_and_nested_fields(
    tmp_path: Path,
    location: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    if location == "top":
        value["methods"] = ["initial_skill", "rl_skill_edit"]
    else:
        value["student"]["unknown"] = True
    _write_config(config_path, value)

    with pytest.raises(ValueError, match="unknown fields"):
        run(config_path, seed=42)


def test_config_rejects_rl_artifact_path_mismatch_before_training(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    value["paths"]["rl_skill"] = str(tmp_path / "wrong.md")
    _write_config(config_path, value)

    with pytest.raises(ValueError, match="paths.rl_skill must point"):
        run(config_path, seed=42)
    assert not Path(value["paths"]["output_dir"]).exists()


def test_config_rejects_future_artifact_target_beneath_symlink(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    output_dir = Path(value["paths"]["output_dir"])
    external = tmp_path / "external-output"
    output_dir.mkdir(parents=True)
    external.mkdir()
    (output_dir / "rl_skill_edit").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        run(config_path, seed=42)
    assert tuple(external.iterdir()) == ()


def test_config_rejects_external_same_content_skill_symlink(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    original = Path(value["paths"]["initial_skill"])
    alias = tmp_path / "initial-skill-alias.md"
    alias.symlink_to(original)
    value["paths"]["initial_skill"] = str(alias)
    _write_config(config_path, value)

    with pytest.raises(ValueError, match="symbolic link"):
        run(config_path, seed=42)


def test_config_rejects_dangling_initial_skill_symlink(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    alias = tmp_path / "dangling-initial-skill.md"
    alias.symlink_to(tmp_path / "missing-initial-skill.md")
    value["paths"]["initial_skill"] = str(alias)
    _write_config(config_path, value)

    with pytest.raises(ValueError, match="symbolic link"):
        run(config_path, seed=42)


def test_manifest_rejects_symlinked_workbook_input(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    train_path = Path(value["paths"]["train_manifest"])
    tasks = json.loads(train_path.read_text(encoding="utf-8"))
    workbook = train_path.parent / tasks[0]["spreadsheet"]["init_file"]
    external = tmp_path / "external-workbook.txt"
    external.write_bytes(workbook.read_bytes())
    workbook.unlink()
    workbook.symlink_to(external)

    with pytest.raises(ValueError, match="symbolic link"):
        run(config_path, seed=42)


def test_test_only_rejects_symlink_anywhere_inside_frozen_output(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    external = tmp_path / "external-evidence.txt"
    external.write_text("outside", encoding="utf-8")
    injected = output_dir / "rl_skill_edit/injected-link.txt"
    injected.symlink_to(external)

    with pytest.raises(ValueError, match="symbolic link"):
        run(config_path, seed=42, test_only=True)
    assert injected.is_symlink()


def test_test_only_rejects_special_file_inside_frozen_output(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    injected = Path(trained["output_dir"]) / "rl_skill_edit/injected-pipe"
    os.mkfifo(injected)

    with pytest.raises(ValueError, match="special file"):
        run(config_path, seed=42, test_only=True)


def test_cli_rejects_overlapping_split_task_ids(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    train_tasks = json.loads(
        Path(value["paths"]["train_manifest"]).read_text(encoding="utf-8")
    )
    validation_path = Path(value["paths"]["validation_manifest"])
    validation_tasks = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_tasks[0]["task_id"] = train_tasks[0]["task_id"]
    validation_path.write_text(json.dumps(validation_tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="task_id overlap"):
        run(config_path, seed=42)


def test_failed_retraining_does_not_pollute_existing_frozen_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    method_dir = Path(trained["output_dir"]) / "rl_skill_edit"
    before = _tree_bytes(method_dir)
    original_write = cli._atomic_json_write

    def fail_provenance(path: Path, value: dict[str, Any]) -> None:
        if path.name == "freeze_provenance.json":
            raise OSError("injected provenance write failure")
        original_write(path, value)

    monkeypatch.setattr(cli, "_atomic_json_write", fail_provenance)

    with pytest.raises(OSError, match="injected provenance"):
        run(config_path, seed=42)
    assert _tree_bytes(method_dir) == before


def test_nonfinite_config_value_is_rejected_before_training(tmp_path: Path) -> None:
    config_path = _smoke_config_in(tmp_path)
    value = _config(config_path)
    value["budget"]["wall_time_seconds"] = float("nan")
    _write_config(config_path, value)

    with pytest.raises(ValueError, match="finite"):
        run(config_path, seed=42)


def test_training_bundle_commit_failure_rolls_back_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    before = _tree_snapshot(output_dir)
    real_replace = os.replace

    def fail_new_bundle(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if (
            source_path.name.startswith(".rl-output-staging-")
            and target_path == output_dir
        ):
            raise OSError("injected bundle commit failure")
        real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_new_bundle)

    with pytest.raises(OSError, match="injected bundle"):
        run(config_path, seed=42)
    assert _tree_snapshot(output_dir) == before
    assert not os.path.lexists(_previous_output_path(output_dir))


def test_first_run_rejects_unowned_previous_without_modifying_it(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    output_dir = Path(_config(config_path)["paths"]["output_dir"])
    previous = _previous_output_path(output_dir)
    previous.mkdir()
    (previous / "sentinel.txt").write_bytes(b"must-not-be-deleted\x00")
    before = _tree_snapshot(previous)

    with pytest.raises((FileNotFoundError, ValueError), match="previous output"):
        run(config_path, seed=42)

    assert _tree_snapshot(previous) == before
    assert not os.path.lexists(output_dir)
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


def test_pretraining_input_audit_excludes_test_workbooks_without_omitting_its_path(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    config = _config(config_path)
    inputs = cli._pretraining_input_paths(config_path, config)
    actual = set(inputs.values())
    expected = {
        config_path,
        ROOT / "requirements.txt",
        ROOT / "rl_skill_edit",
        Path(config["paths"]["initial_skill"]),
        Path(config["paths"]["train_manifest"]),
        Path(config["paths"]["validation_manifest"]),
        Path(config["paths"]["test_manifest"]),
        *(ROOT / "rl_skill_edit").rglob("*.py"),
    }
    for field in ("train_manifest", "validation_manifest"):
        manifest_path = Path(config["paths"][field])
        tasks = json.loads(manifest_path.read_text(encoding="utf-8"))
        for task in tasks:
            for workbook_field in ("init_file", "golden_file"):
                expected.add(manifest_path.parent / task["spreadsheet"][workbook_field])

    test_manifest = Path(config["paths"]["test_manifest"])
    test_tasks = json.loads(test_manifest.read_text(encoding="utf-8"))
    test_workbooks = {
        test_manifest.parent / task["spreadsheet"][field]
        for task in test_tasks
        for field in ("init_file", "golden_file")
    }

    assert actual == expected
    assert actual.isdisjoint(test_workbooks)
    assert ROOT not in actual
    cli._validate_output_input_separation(Path(config["paths"]["output_dir"]), inputs)


@pytest.mark.parametrize("input_is_ancestor", (False, True))
def test_output_input_separation_rejects_both_ancestor_directions(
    tmp_path: Path,
    input_is_ancestor: bool,
) -> None:
    if input_is_ancestor:
        input_path = tmp_path / "protected"
        output_dir = input_path / "result"
    else:
        output_dir = tmp_path / "result"
        input_path = output_dir / "protected.txt"

    with pytest.raises(ValueError, match="unsafe output path overlap"):
        cli._validate_output_input_separation(
            output_dir,
            {"protected input": input_path},
        )


def test_postload_input_audit_includes_the_actual_staging_path(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "result"
    staging = tmp_path / ".rl-output-staging-fixed"

    with pytest.raises(ValueError, match="unsafe output path overlap"):
        cli._validate_output_input_separation(
            output_dir,
            {"protected input": staging / "test-workbook.xlsx"},
            staging=staging,
        )


@pytest.mark.parametrize("mutable_path", ("output", "previous"))
def test_output_paths_must_not_overlap_the_input_tree(
    tmp_path: Path,
    mutable_path: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    config = _config(config_path)
    data_dir = Path(config["paths"]["initial_skill"]).parent
    output_dir = Path(config["paths"]["output_dir"])
    if mutable_path == "output":
        output_dir = data_dir
        _set_output_paths(config, output_dir)
        protected = data_dir
    else:
        previous = _previous_output_path(output_dir)
        data_dir.rename(previous)
        for field, filename in (
            ("initial_skill", "initial_skill.md"),
            ("train_manifest", "train.json"),
            ("validation_manifest", "validation.json"),
            ("test_manifest", "test.json"),
        ):
            config["paths"][field] = str(previous / filename)
        protected = previous
    _write_config(config_path, config)
    before = _tree_snapshot(protected)

    with pytest.raises(ValueError, match="unsafe output path overlap"):
        run(config_path, seed=42)

    assert _tree_snapshot(protected) == before
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


def test_test_workbook_overlap_is_rejected_after_optimization_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    config = _config(config_path)
    output_dir = Path(trained["output_dir"])
    test_manifest = Path(config["paths"]["test_manifest"])
    protected_workbook = output_dir / "protected-test-init.txt"
    protected_workbook.write_bytes(b"protected test input\x00")
    tasks = json.loads(test_manifest.read_text(encoding="utf-8"))
    tasks[0]["spreadsheet"]["init_file"] = str(protected_workbook)
    test_manifest.write_text(json.dumps(tasks), encoding="utf-8")
    output_before = _tree_snapshot(output_dir)
    manifest_before = test_manifest.read_bytes()
    workbook_before = protected_workbook.read_bytes()
    events: list[str] = []
    original_optimize = cli.RLSkillEditOptimizer.optimize

    def optimize_then_mark(self, **kwargs):
        result = original_optimize(self, **kwargs)
        events.append("optimized")
        return result

    monkeypatch.setattr(cli.RLSkillEditOptimizer, "optimize", optimize_then_mark)

    with pytest.raises(ValueError, match="unsafe output path overlap"):
        run(config_path, seed=43)

    assert events == ["optimized"]
    assert _tree_snapshot(output_dir) == output_before
    assert test_manifest.read_bytes() == manifest_before
    assert protected_workbook.read_bytes() == workbook_before
    assert not os.path.lexists(_previous_output_path(output_dir))
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


@pytest.mark.parametrize("test_only", (False, True))
def test_existing_output_without_marker_is_rejected_unchanged(
    tmp_path: Path,
    test_only: bool,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    marker = _write_expected_output_marker(output_dir, output_dir)
    marker.unlink()
    before = _tree_snapshot(output_dir)

    with pytest.raises(FileNotFoundError, match="ownership marker"):
        run(config_path, seed=42 if test_only else 43, test_only=test_only)

    assert _tree_snapshot(output_dir) == before
    assert not os.path.lexists(_previous_output_path(output_dir))
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "wrong_binding",
        "symlink",
        "nan",
        "unknown",
        "incomplete",
        "incomplete_episode",
    ),
)
def test_invalid_or_incomplete_previous_output_is_never_deleted(
    tmp_path: Path,
    mutation: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    _write_expected_output_marker(output_dir, output_dir)
    previous = _previous_output_path(output_dir)
    shutil.copytree(output_dir, previous)
    marker = previous / OUTPUT_MARKER_FILE
    if mutation == "missing":
        marker.unlink()
    elif mutation == "wrong_binding":
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["final_output_path"] = str(tmp_path / "somewhere-else")
        marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif mutation == "symlink":
        external = tmp_path / "external-marker.json"
        external.write_text(marker.read_text(encoding="utf-8"), encoding="utf-8")
        marker.unlink()
        marker.symlink_to(external)
    elif mutation == "nan":
        marker.write_text(
            '{"protocol":"rl-skill-edit-output-v1",'
            '"method":"rl_skill_edit","final_output_path":NaN}\n',
            encoding="utf-8",
        )
    elif mutation == "unknown":
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["unknown"] = "forbidden"
        marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif mutation == "incomplete":
        (previous / "experiment_manifest.json").unlink()
    elif mutation == "incomplete_episode":
        (
            previous
            / "rl_skill_edit/episodes/episode_0000/skills/step_000_candidate.md"
        ).unlink()
    else:
        raise AssertionError(mutation)
    output_before = _tree_snapshot(output_dir)
    previous_before = _tree_snapshot(previous)

    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        run(config_path, seed=43)

    assert _tree_snapshot(output_dir) == output_before
    assert _tree_snapshot(previous) == previous_before
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


def test_successful_retraining_keeps_complete_previous_output_snapshot(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    before = _tree_snapshot(output_dir)
    before_fingerprint = cli._tree_fingerprint(output_dir)

    run(config_path, seed=43)

    previous = _previous_output_path(output_dir)
    assert _tree_snapshot(previous) == before
    assert cli._tree_fingerprint(previous) == before_fingerprint


def test_three_publications_rotate_owned_previous_for_train_and_test_only(
    tmp_path: Path,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    first = run(config_path, seed=42)
    output_dir = Path(first["output_dir"])
    first_snapshot = _tree_snapshot(output_dir)

    run(config_path, seed=42, test_only=True)
    second_snapshot = _tree_snapshot(output_dir)
    previous = _previous_output_path(output_dir)
    assert _tree_snapshot(previous) == first_snapshot

    run(config_path, seed=43)

    assert _tree_snapshot(previous) == second_snapshot
    previous_marker = json.loads(
        (previous / OUTPUT_MARKER_FILE).read_text(encoding="utf-8")
    )
    assert previous_marker["final_output_path"] == str(output_dir)


def test_post_install_previous_validation_failure_restores_current_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    previous = _previous_output_path(output_dir)
    before = _tree_snapshot(output_dir)
    original_fingerprint = cli._tree_fingerprint
    previous_checks = 0

    def fail_second_previous_check(path: Path) -> str:
        nonlocal previous_checks
        if path == previous:
            previous_checks += 1
            if previous_checks == 2:
                raise OSError("injected post-install previous validation failure")
        return original_fingerprint(path)

    monkeypatch.setattr(cli, "_tree_fingerprint", fail_second_previous_check)

    with pytest.raises(OSError, match="post-install previous validation"):
        run(config_path, seed=43)
    assert _tree_snapshot(output_dir) == before
    assert not os.path.lexists(previous)
    assert not tuple(tmp_path.glob(".rl-output-staging-*"))


@pytest.mark.parametrize("kind", ("symlink", "special"))
def test_previous_snapshot_rejects_links_and_special_files_before_commit(
    tmp_path: Path,
    kind: str,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    previous = _previous_output_path(output_dir)
    before = _tree_snapshot(output_dir)
    if kind == "symlink":
        external = tmp_path / "external-previous"
        external.mkdir()
        previous.symlink_to(external, target_is_directory=True)
    else:
        os.mkfifo(previous)

    with pytest.raises(ValueError):
        run(config_path, seed=43)
    assert _tree_snapshot(output_dir) == before
    assert os.path.lexists(previous)


def test_manifest_write_failure_preserves_entire_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    before = _tree_snapshot(output_dir)
    original_write = cli._atomic_json_write

    def fail_manifest(path: Path, value: dict[str, Any]) -> None:
        if path.name == "experiment_manifest.json":
            raise OSError("injected manifest write failure")
        original_write(path, value)

    monkeypatch.setattr(cli, "_atomic_json_write", fail_manifest)

    with pytest.raises(OSError, match="injected manifest"):
        run(config_path, seed=42)
    assert _tree_snapshot(output_dir) == before


def test_report_failure_with_new_seed_preserves_entire_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    before = _tree_snapshot(output_dir)
    original_report = cli.run_frozen_report

    def fail_seed_43(**kwargs):
        if kwargs["seed"] == 43:
            raise RuntimeError("injected report failure for seed 43")
        return original_report(**kwargs)

    monkeypatch.setattr(cli, "run_frozen_report", fail_seed_43)

    with pytest.raises(RuntimeError, match="seed 43"):
        run(config_path, seed=43)
    assert _tree_snapshot(output_dir) == before


def test_initial_run_failure_leaves_final_output_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    output_dir = Path(_config(config_path)["paths"]["output_dir"])

    def fail_report(**kwargs):
        raise RuntimeError("injected initial report failure")

    monkeypatch.setattr(cli, "run_frozen_report", fail_report)

    with pytest.raises(RuntimeError, match="initial report"):
        run(config_path, seed=42)
    assert not os.path.lexists(output_dir)


def test_test_only_report_failure_preserves_entire_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    before = _tree_snapshot(output_dir)

    def fail_report(**kwargs):
        raise RuntimeError("injected test-only report failure")

    monkeypatch.setattr(cli, "run_frozen_report", fail_report)

    with pytest.raises(RuntimeError, match="test-only report"):
        run(config_path, seed=42, test_only=True)
    assert _tree_snapshot(output_dir) == before


@pytest.mark.parametrize("test_only", (False, True))
def test_partial_previous_cleanup_failure_preserves_current_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_only: bool,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    run(config_path, seed=42, test_only=True)
    previous = _previous_output_path(output_dir)
    assert previous.is_dir()
    before = _tree_snapshot(output_dir)
    previous_before = _tree_snapshot(previous)
    original_remove = cli._remove_tree

    def partially_remove_previous(path: Path) -> None:
        if path == previous:
            (path / "experiment_manifest.json").unlink()
            raise OSError("injected partial previous cleanup failure")
        original_remove(path)

    monkeypatch.setattr(cli, "_remove_tree", partially_remove_previous)

    with pytest.raises(OSError, match="partial previous cleanup"):
        run(config_path, seed=42 if test_only else 43, test_only=test_only)
    assert _tree_snapshot(output_dir) == before
    assert os.path.lexists(previous)
    assert _tree_snapshot(previous) != previous_before


def test_install_and_rollback_failure_retains_explicit_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42)
    output_dir = Path(trained["output_dir"])
    real_replace = os.replace

    def fail_install_and_restore(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if target_path == output_dir and (
            source_path.name.startswith(".rl-output-staging-")
            or source_path == _previous_output_path(output_dir)
        ):
            raise OSError("injected install or rollback failure")
        real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_install_and_restore)

    with pytest.raises(cli.OutputRollbackError, match="evidence retained"):
        run(config_path, seed=42)
    assert not os.path.lexists(output_dir)
    assert tuple(tmp_path.glob(".rl-output-staging-*"))
    assert _previous_output_path(output_dir).is_dir()

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


def test_test_manifest_is_loaded_only_after_rl_optimization_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _smoke_config_in(tmp_path)
    events: list[str] = []
    original_optimize = cli.RLSkillEditOptimizer.optimize
    original_load_test = cli._load_test_manifest

    def optimize_then_mark(self, **kwargs):
        result = original_optimize(self, **kwargs)
        events.append("optimized")
        return result

    def load_test_after_freeze(config, optimization_manifests):
        assert events == ["optimized"]
        events.append("test_loaded")
        return original_load_test(config, optimization_manifests)

    monkeypatch.setattr(cli.RLSkillEditOptimizer, "optimize", optimize_then_mark)
    monkeypatch.setattr(cli, "_load_test_manifest", load_test_after_freeze)

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
    method_dir = Path(trained["output_dir"]) / "rl_skill_edit"
    before = _tree_bytes(method_dir)
    real_replace = os.replace

    def fail_new_bundle(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if (
            source_path.parent.name.startswith(".rl-training-")
            and source_path.name == "rl_skill_edit"
            and target_path == method_dir
        ):
            raise OSError("injected bundle commit failure")
        real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_new_bundle)

    with pytest.raises(OSError, match="injected bundle"):
        run(config_path, seed=42)
    assert _tree_bytes(method_dir) == before

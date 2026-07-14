from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
import yaml

from experiments import run_skill_optimization_comparison as runner
from experiments.run_skill_optimization_comparison import REPOSITORY_ROOT, run


def test_api_free_training_through_frozen_final_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/rl_skill_edit_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    output_path = tmp_path / "result"
    source["paths"]["output_dir"] = str(output_path)
    source["paths"]["rl_skill"] = str(output_path / "rl_skill_edit/best_rl_skill.md")
    source["paths"]["rl_summary"] = str(
        output_path / "rl_skill_edit/rl_optimization_summary.json"
    )
    source["paths"]["rl_provenance"] = str(
        output_path / "rl_skill_edit/freeze_provenance.json"
    )
    source["optimizer"]["episodes"] = 1
    source["optimizer"]["horizon"] = 1
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    original_test_loader = runner._load_test_manifest

    def load_test_only_after_freeze(config, optimization_manifests):
        assert (output_path / "rl_skill_edit/best_rl_skill.md").is_file()
        assert (output_path / "rl_skill_edit/rl_optimization_summary.json").is_file()
        return original_test_loader(config, optimization_manifests)

    monkeypatch.setattr(runner, "_load_test_manifest", load_test_only_after_freeze)

    result = run(
        argparse.Namespace(
            config=config_path,
            methods=["initial_skill", "current_method", "rl_skill_edit"],
            seed=42,
            test_only=False,
        )
    )

    output = Path(result["output_dir"])
    assert (
        result["test_rewards"]["rl_skill_edit"]
        > result["test_rewards"]["initial_skill"]
    )
    assert (output / "rl_skill_edit/best_rl_skill.md").is_file()
    assert (output / "rl_skill_edit/final_rl_policy.pt").is_file()
    assert (output / "rl_skill_edit/rl_training_log.jsonl").is_file()
    assert (output / "rl_skill_edit/rl_episode_summary.csv").is_file()
    assert (output / "rl_skill_edit/freeze_provenance.json").is_file()
    freeze_provenance = json.loads(
        (output / "rl_skill_edit/freeze_provenance.json").read_text(encoding="utf-8")
    )
    assert freeze_provenance["implementation_sha256"]
    assert freeze_provenance["requirements_sha256"]
    experiment_manifest = json.loads(
        (output / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        experiment_manifest["implementation_sha256"]
        == freeze_provenance["implementation_sha256"]
    )
    assert experiment_manifest["current_method_artifact_provenance"]["history_sha256"]
    assert (output / "test_task_level_results.csv").is_file()
    assert (output / "method_comparison.csv").is_file()
    with (output / "test_task_level_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2
    assert {row["method"] for row in rows} == {
        "initial_skill",
        "current_method",
        "rl_skill_edit",
    }
    with (output / "method_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        trained_method_rows = {row["method"]: row for row in csv.DictReader(handle)}

    frozen = run(
        argparse.Namespace(
            config=config_path,
            methods=["initial_skill", "current_method", "rl_skill_edit"],
            seed=42,
            test_only=True,
        )
    )
    assert frozen["test_rewards"] == result["test_rewards"]
    with (output / "method_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        frozen_method_rows = {row["method"]: row for row in csv.DictReader(handle)}
    for method in trained_method_rows:
        assert (
            frozen_method_rows[method]["reporting_student_rollouts"]
            == (trained_method_rows[method]["reporting_student_rollouts"])
        )
        assert (
            frozen_method_rows[method]["reporting_evaluator_calls"]
            == (trained_method_rows[method]["reporting_evaluator_calls"])
        )

    provenance_path = output / "rl_skill_edit/freeze_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["seed"] = 43
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance mismatch for seed"):
        run(
            argparse.Namespace(
                config=config_path,
                methods=["initial_skill", "current_method", "rl_skill_edit"],
                seed=42,
                test_only=True,
            )
        )

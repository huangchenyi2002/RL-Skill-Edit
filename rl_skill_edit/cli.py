from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .action_space import ActionSpace
from .adapters.openrouter import OpenRouterClient
from .adapters.spreadsheet import SpreadsheetStudent
from .budget import BudgetLedger
from .cache import JsonFileCache
from .evaluation import MockSkillEvaluator, SpreadsheetSkillEvaluator
from .manifest import TaskManifest, validate_manifests
from .optimizer import OptimizationResult, RLSkillEditOptimizer
from .patch_generator import OpenRouterPatchGenerator, SyntheticPatchGenerator
from .policy import ActorCriticPolicy
from .reporting import ResourceUsage, run_frozen_report
from .state_encoder import StateEncoder
from .types import EvaluationBatch, SkillArtifact, Split, TaskResult


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPOSITORY_ROOT / "requirements.txt"
_METHODS = ("initial_skill", "rl_skill_edit")
_PROVENANCE_FIELDS = (
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
)
_TOP_LEVEL_FIELDS = {
    "method",
    "runtime",
    "seed",
    "openrouter",
    "cost_tracking",
    "student",
    "editor",
    "skill_identity",
    "paths",
    "split_sizes",
    "optimizer",
    "action_space",
    "policy",
    "reward",
    "patch_limits",
    "evaluation",
    "budget",
}
_PATH_FIELDS = {
    "initial_skill",
    "train_manifest",
    "validation_manifest",
    "test_manifest",
    "output_dir",
    "rl_skill",
    "rl_summary",
    "rl_provenance",
}
_BATCH_USAGE_FIELDS = {
    "student_rollouts",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "elapsed_s",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL-Skill-Edit.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args(argv)


def _exact_mapping(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    payload = dict(value)
    optional = optional or set()
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    unknown = set(payload) - required - optional
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return payload


def _text(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be text")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _integer(name: str, value: Any) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _number(name: str, value: Any) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_number(name: str, value: Any) -> float:
    result = _number(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _unit_number(name: str, value: Any) -> float:
    result = _number(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    _exact_mapping(config, name="config", required=_TOP_LEVEL_FIELDS)
    if _text("method", config["method"]) != "rl_skill_edit":
        raise ValueError("method must be rl_skill_edit")
    runtime = _text("runtime", config["runtime"])
    if runtime not in {"mock", "spreadsheet"}:
        raise ValueError("runtime must be mock or spreadsheet")
    _integer("seed", config["seed"])

    openrouter = _exact_mapping(
        config["openrouter"],
        name="openrouter",
        required={"base_url"},
        optional={"proxy"},
    )
    _text("openrouter.base_url", openrouter["base_url"])
    if "proxy" in openrouter and openrouter["proxy"] is not None:
        _text("openrouter.proxy", openrouter["proxy"])

    cost_tracking = _exact_mapping(
        config["cost_tracking"],
        name="cost_tracking",
        required={"enabled", "cost_per_1k_tokens"},
    )
    enabled = _boolean("cost_tracking.enabled", cost_tracking["enabled"])
    prices = cost_tracking["cost_per_1k_tokens"]
    if type(prices) is not dict:
        raise TypeError("cost_tracking.cost_per_1k_tokens must be a mapping")
    for model, price in prices.items():
        _text("cost_tracking model", model)
        _nonnegative_number(f"cost_tracking price for {model}", price)

    student = _exact_mapping(
        config["student"],
        name="student",
        required={"model", "temperature", "max_tokens", "max_steps"},
    )
    editor = _exact_mapping(
        config["editor"],
        name="editor",
        required={"model", "temperature", "max_tokens"},
    )
    for section_name, section in (("student", student), ("editor", editor)):
        model = _text(f"{section_name}.model", section["model"])
        _number(f"{section_name}.temperature", section["temperature"])
        _positive_integer(f"{section_name}.max_tokens", section["max_tokens"])
        if enabled and model not in prices:
            raise ValueError(f"missing token price for {section_name}.model")
    _positive_integer("student.max_steps", student["max_steps"])

    identity = _exact_mapping(
        config["skill_identity"],
        name="skill_identity",
        required={"skill_id", "name", "description"},
    )
    _text("skill_identity.skill_id", identity["skill_id"])
    _text("skill_identity.name", identity["name"])
    _text("skill_identity.description", identity["description"], allow_empty=True)

    paths = _exact_mapping(config["paths"], name="paths", required=_PATH_FIELDS)
    for field, value in paths.items():
        _text(f"paths.{field}", value)

    sizes = _exact_mapping(
        config["split_sizes"],
        name="split_sizes",
        required={"train", "validation", "test"},
    )
    for field, value in sizes.items():
        _positive_integer(f"split_sizes.{field}", value)

    optimizer = _exact_mapping(
        config["optimizer"],
        name="optimizer",
        required={"episodes", "horizon", "minibatch_size", "validation_interval"},
    )
    for field, value in optimizer.items():
        _positive_integer(f"optimizer.{field}", value)

    action_space = _exact_mapping(
        config["action_space"],
        name="action_space",
        required={"max_modules"},
    )
    _positive_integer("action_space.max_modules", action_space["max_modules"])

    policy = _exact_mapping(
        config["policy"],
        name="policy",
        required={
            "hidden_dim",
            "learning_rate",
            "gamma",
            "entropy_coef",
            "value_coef",
            "max_grad_norm",
            "normalize_advantages",
        },
    )
    _positive_integer("policy.hidden_dim", policy["hidden_dim"])
    if _number("policy.learning_rate", policy["learning_rate"]) <= 0.0:
        raise ValueError("policy.learning_rate must be positive")
    _unit_number("policy.gamma", policy["gamma"])
    _nonnegative_number("policy.entropy_coef", policy["entropy_coef"])
    _nonnegative_number("policy.value_coef", policy["value_coef"])
    if _number("policy.max_grad_norm", policy["max_grad_norm"]) <= 0.0:
        raise ValueError("policy.max_grad_norm must be positive")
    _boolean("policy.normalize_advantages", policy["normalize_advantages"])

    reward = _exact_mapping(
        config["reward"],
        name="reward",
        required={"beta_len", "beta_edit", "beta_invalid"},
    )
    for field, value in reward.items():
        _nonnegative_number(f"reward.{field}", value)

    patch_limits = _exact_mapping(
        config["patch_limits"],
        name="patch_limits",
        required={
            "max_changed_tokens",
            "max_length_growth_tokens",
            "max_skill_tokens",
        },
    )
    for field, value in patch_limits.items():
        _positive_integer(f"patch_limits.{field}", value)

    evaluation = _exact_mapping(
        config["evaluation"],
        name="evaluation",
        required={
            "metric",
            "mixed_weight",
            "success_threshold",
            "train_repetitions",
            "validation_repetitions",
            "report_train_repetitions",
            "test_repetitions",
            "bootstrap_samples",
            "use_cache",
            "paired_tasks",
            "save_task_level_results",
            "blind",
        },
    )
    if _text("evaluation.metric", evaluation["metric"]) not in {
        "hard",
        "soft",
        "mixed",
    }:
        raise ValueError("evaluation.metric must be hard, soft, or mixed")
    _unit_number("evaluation.mixed_weight", evaluation["mixed_weight"])
    _unit_number("evaluation.success_threshold", evaluation["success_threshold"])
    for field in (
        "train_repetitions",
        "validation_repetitions",
        "report_train_repetitions",
        "test_repetitions",
        "bootstrap_samples",
    ):
        _positive_integer(f"evaluation.{field}", evaluation[field])
    for field in ("use_cache", "paired_tasks", "save_task_level_results", "blind"):
        _boolean(f"evaluation.{field}", evaluation[field])
    for field in ("paired_tasks", "save_task_level_results", "blind"):
        if not evaluation[field]:
            raise ValueError(f"evaluation.{field} must be true")

    budget = _exact_mapping(
        config["budget"],
        name="budget",
        required={
            "student_rollouts",
            "editor_calls",
            "evaluator_calls",
            "input_tokens",
            "output_tokens",
            "wall_time_seconds",
        },
    )
    for field in (
        "student_rollouts",
        "editor_calls",
        "evaluator_calls",
        "input_tokens",
        "output_tokens",
    ):
        _nonnegative_integer(f"budget.{field}", budget[field])
    _nonnegative_number("budget.wall_time_seconds", budget["wall_time_seconds"])


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"config must be a YAML mapping: {path}")
    config = dict(payload)
    _validate_config(config)
    return config


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve(strict=False)


def _path(config: Mapping[str, Any], name: str) -> Path:
    return _resolve(config["paths"][name])


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    output_dir = _path(config, "output_dir")
    method_dir = output_dir / "rl_skill_edit"
    expected = {
        "rl_skill": method_dir / "best_rl_skill.md",
        "rl_summary": method_dir / "rl_optimization_summary.json",
        "rl_provenance": method_dir / "freeze_provenance.json",
    }
    for name, expected_path in expected.items():
        configured = _path(config, name)
        if configured != expected_path.resolve(strict=False):
            raise ValueError(
                f"paths.{name} must point to the generated artifact: {expected_path}"
            )
    return {"output_dir": output_dir, "method_dir": method_dir, **expected}


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(item.resolve() for item in paths):
        relative = path.relative_to(REPOSITORY_ROOT)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _implementation_sha256() -> str:
    files = tuple((REPOSITORY_ROOT / "rl_skill_edit").rglob("*.py"))
    if not files:
        raise RuntimeError("rl_skill_edit implementation files are missing")
    return _sha256_files(files)


def _dependency_sha256() -> str:
    return _sha256_file(REQUIREMENTS_PATH)


def _normalized_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    normalized["paths"] = {
        name: str(_path(config, name)) for name in sorted(_PATH_FIELDS)
    }
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_skill(
    skill: SkillArtifact,
    *,
    identity: Mapping[str, Any],
) -> SkillArtifact:
    return SkillArtifact(
        skill_id=str(identity["skill_id"]),
        name=str(identity["name"]),
        description=str(identity["description"]),
        body=skill.body,
    )


def _load_initial_skill(
    config: Mapping[str, Any], identity: Mapping[str, Any]
) -> SkillArtifact:
    loaded = SkillArtifact.from_file(
        _path(config, "initial_skill"), skill_id=str(identity["skill_id"])
    )
    return _canonical_skill(loaded, identity=identity)


def _load_optimization_manifests(
    config: Mapping[str, Any],
) -> dict[Split, TaskManifest]:
    sizes = config["split_sizes"]
    manifests = {
        Split.TRAIN: TaskManifest.load(
            _path(config, "train_manifest"),
            split=Split.TRAIN,
            expected_size=sizes["train"],
        ),
        Split.VALIDATION: TaskManifest.load(
            _path(config, "validation_manifest"),
            split=Split.VALIDATION,
            expected_size=sizes["validation"],
        ),
    }
    validate_manifests(manifests[Split.TRAIN], manifests[Split.VALIDATION])
    return manifests


def _load_test_manifest(
    config: Mapping[str, Any],
    optimization_manifests: Mapping[Split, TaskManifest],
) -> dict[Split, TaskManifest]:
    manifests = dict(optimization_manifests)
    manifests[Split.TEST] = TaskManifest.load(
        _path(config, "test_manifest"),
        split=Split.TEST,
        expected_size=config["split_sizes"]["test"],
    )
    validate_manifests(
        manifests[Split.TRAIN],
        manifests[Split.VALIDATION],
        manifests[Split.TEST],
    )
    return manifests


def _split_provenance(
    manifests: Mapping[Split, TaskManifest],
) -> tuple[dict[str, str], str]:
    split_digests = {split.value: manifests[split].digest for split in Split}
    encoded = json.dumps(
        split_digests,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return split_digests, hashlib.sha256(encoded).hexdigest()


def _mock_score(initial_digest: str) -> Callable[..., TaskResult]:
    def score(
        skill: SkillArtifact, task: Any, repetition: int, seed: int
    ) -> TaskResult:
        del repetition, seed
        task_id = str(task["task_id"] if isinstance(task, dict) else task.task_id)
        offset = (sum(task_id.encode("utf-8")) % 3) * 0.025
        reward = 0.20 + offset if skill.digest == initial_digest else 0.80 + offset
        reward = min(1.0, reward)
        return TaskResult(task_id=task_id, reward=reward, success=reward >= 0.5)

    return score


def _make_evaluator(
    config: Mapping[str, Any],
    *,
    initial_skill: SkillArtifact,
    cache_path: Path,
) -> Any:
    cache = JsonFileCache(cache_path)
    if config["runtime"] == "mock":
        return MockSkillEvaluator(
            _mock_score(initial_skill.digest),
            cache=cache,
            cache_signature={
                "adapter": "rl-skill-edit-smoke-oracle-v2",
                "initial_skill_digest": initial_skill.digest,
            },
        )
    client = OpenRouterClient(config)
    student = SpreadsheetStudent(config, client)
    evaluation = config["evaluation"]
    return SpreadsheetSkillEvaluator(
        student,
        cache=cache,
        gate_metric=evaluation["metric"],
        gate_mixed_weight=evaluation["mixed_weight"],
        success_threshold=evaluation["success_threshold"],
    )


def _make_training_components(
    config: Mapping[str, Any],
    *,
    initial_skill: SkillArtifact,
    method_dir: Path,
    seed: int,
) -> tuple[Any, Any]:
    if config["runtime"] == "mock":
        evaluator = MockSkillEvaluator(
            _mock_score(initial_skill.digest),
            cache=JsonFileCache(method_dir / "rollout_cache.json"),
            cache_signature={
                "adapter": "rl-skill-edit-smoke-oracle-v2",
                "initial_skill_digest": initial_skill.digest,
            },
        )
        return evaluator, SyntheticPatchGenerator()
    client = OpenRouterClient(config)
    student = SpreadsheetStudent(config, client)
    evaluation = config["evaluation"]
    evaluator = SpreadsheetSkillEvaluator(
        student,
        cache=JsonFileCache(method_dir / "rollout_cache.json"),
        gate_metric=evaluation["metric"],
        gate_mixed_weight=evaluation["mixed_weight"],
        success_threshold=evaluation["success_threshold"],
    )
    editor = config["editor"]
    generator = OpenRouterPatchGenerator(
        client=client,
        cache=JsonFileCache(method_dir / "editor_cache.json"),
        model=editor["model"],
        temperature=editor["temperature"],
        max_tokens=editor["max_tokens"],
        seed=seed,
    )
    return evaluator, generator


def _policy(config: Mapping[str, Any], *, seed: int) -> ActorCriticPolicy:
    max_modules = config["action_space"]["max_modules"]
    action_space = ActionSpace(max_modules)
    encoder = StateEncoder(max_modules, action_space.size)
    policy = config["policy"]
    return ActorCriticPolicy(
        input_dim=encoder.state_dim,
        action_dim=action_space.size,
        hidden_dim=policy["hidden_dim"],
        seed=seed,
        learning_rate=policy["learning_rate"],
        gamma=policy["gamma"],
        entropy_coef=policy["entropy_coef"],
        value_coef=policy["value_coef"],
        max_grad_norm=policy["max_grad_norm"],
        normalize_advantages=policy["normalize_advantages"],
    )


def _optimizer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: copy.deepcopy(config[name])
        for name in (
            "optimizer",
            "evaluation",
            "action_space",
            "reward",
            "patch_limits",
        )
    }


def _strict_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if type(payload) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    return dict(payload)


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (os.path.lexists(path) and not path.is_file()):
        raise ValueError(f"JSON target must be a regular file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_summary(
    summary: Mapping[str, Any], *, skill_digest: str
) -> dict[str, int | float]:
    payload = _exact_mapping(
        dict(summary),
        name="RL optimization summary",
        required={
            "best_skill_digest",
            "final_skill_digest",
            "best_validation_score",
            "accepted_edits",
            "total_applied_edits",
            "budget",
        },
    )
    if payload["best_skill_digest"] != skill_digest:
        raise ValueError("RL summary and frozen Skill digest do not match")
    _text("summary.final_skill_digest", payload["final_skill_digest"])
    _unit_number("summary.best_validation_score", payload["best_validation_score"])
    _nonnegative_integer("summary.accepted_edits", payload["accepted_edits"])
    _nonnegative_integer("summary.total_applied_edits", payload["total_applied_edits"])
    return ResourceUsage.from_mapping(payload["budget"]).to_dict()


def _provenance(
    *,
    skill: SkillArtifact,
    initial_skill: SkillArtifact,
    config_sha256: str,
    manifests: Mapping[Split, TaskManifest],
    implementation_sha256: str,
    dependency_sha256: str,
    summary_sha256: str,
    seed: int,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    split_digests, split_digest = _split_provenance(manifests)
    return {
        "protocol": "rl-skill-edit-freeze-v1",
        "method": "rl_skill_edit",
        "best_skill_digest": skill.digest,
        "initial_skill_digest": initial_skill.digest,
        "config_sha256": config_sha256,
        "split_digest": split_digest,
        "split_digests": split_digests,
        "implementation_sha256": implementation_sha256,
        "dependency_sha256": dependency_sha256,
        "summary_sha256": summary_sha256,
        "seed": seed,
        "skill_identity": dict(identity),
    }


def _commit_training_bundle(staged: Path, target: Path) -> None:
    if staged.is_symlink() or not staged.is_dir():
        raise ValueError("staged RL bundle must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (os.path.lexists(target) and not target.is_dir()):
        raise ValueError(f"RL bundle target must be a directory: {target}")
    backup = target.parent / f".rl-skill-edit-backup-{os.urandom(12).hex()}"
    had_target = os.path.lexists(target)
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except BaseException:
        if os.path.lexists(target):
            shutil.rmtree(target)
        if had_target:
            os.replace(backup, target)
        raise
    if had_target:
        shutil.rmtree(backup)


def _train_rl(
    *,
    config: Mapping[str, Any],
    initial_skill: SkillArtifact,
    optimization_manifests: Mapping[Split, TaskManifest],
    artifact_paths: Mapping[str, Path],
    seed: int,
    config_sha256: str,
    implementation_sha256: str,
    dependency_sha256: str,
    identity: Mapping[str, Any],
) -> tuple[SkillArtifact, dict[str, int | float], dict[Split, TaskManifest]]:
    output_dir = artifact_paths["output_dir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".rl-training-", dir=output_dir.parent))
    staged_method_dir = staging_root / "rl_skill_edit"
    try:
        evaluator, patch_generator = _make_training_components(
            config,
            initial_skill=initial_skill,
            method_dir=staged_method_dir,
            seed=seed,
        )
        optimizer = RLSkillEditOptimizer(
            config=_optimizer_config(config),
            evaluator=evaluator,
            patch_generator=patch_generator,
            policy=_policy(config, seed=seed),
            output_dir=staged_method_dir,
        )
        result: OptimizationResult = optimizer.optimize(
            initial_skill=initial_skill,
            train_tasks=optimization_manifests[Split.TRAIN].tasks,
            validation_tasks=optimization_manifests[Split.VALIDATION].tasks,
            budget=BudgetLedger(config["budget"]),
            seed=seed,
        )

        staged_skill_path = staged_method_dir / "best_rl_skill.md"
        staged_summary_path = staged_method_dir / "rl_optimization_summary.json"
        staged_skill = _canonical_skill(
            SkillArtifact.from_file(staged_skill_path, skill_id=identity["skill_id"]),
            identity=identity,
        )
        if staged_skill.digest != result.best_skill.digest:
            raise ValueError("saved RL Skill does not match optimizer result")
        summary = _strict_json_mapping(
            staged_summary_path, name="RL optimization summary"
        )
        usage = _validate_summary(summary, skill_digest=staged_skill.digest)

        manifests = _load_test_manifest(config, optimization_manifests)
        provenance = _provenance(
            skill=staged_skill,
            initial_skill=initial_skill,
            config_sha256=config_sha256,
            manifests=manifests,
            implementation_sha256=implementation_sha256,
            dependency_sha256=dependency_sha256,
            summary_sha256=_sha256_file(staged_summary_path),
            seed=seed,
            identity=identity,
        )
        _atomic_json_write(staged_method_dir / "freeze_provenance.json", provenance)
        _commit_training_bundle(staged_method_dir, artifact_paths["method_dir"])
        return staged_skill, usage, manifests
    finally:
        shutil.rmtree(staging_root, ignore_errors=False)


def _load_frozen_rl(
    *,
    config: Mapping[str, Any],
    initial_skill: SkillArtifact,
    optimization_manifests: Mapping[Split, TaskManifest],
    artifact_paths: Mapping[str, Path],
    seed: int,
    config_sha256: str,
    implementation_sha256: str,
    dependency_sha256: str,
    identity: Mapping[str, Any],
) -> tuple[SkillArtifact, dict[str, int | float], dict[Split, TaskManifest]]:
    skill = _canonical_skill(
        SkillArtifact.from_file(
            artifact_paths["rl_skill"], skill_id=identity["skill_id"]
        ),
        identity=identity,
    )
    summary = _strict_json_mapping(
        artifact_paths["rl_summary"], name="RL optimization summary"
    )
    provenance = _strict_json_mapping(
        artifact_paths["rl_provenance"], name="RL freeze provenance"
    )
    actual_fields = set(provenance)
    expected_fields = set(_PROVENANCE_FIELDS)
    if actual_fields != expected_fields:
        raise ValueError(
            "RL freeze provenance fields do not match the required schema: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"unknown={sorted(actual_fields - expected_fields)}"
        )

    manifests = _load_test_manifest(config, optimization_manifests)
    expected = _provenance(
        skill=skill,
        initial_skill=initial_skill,
        config_sha256=config_sha256,
        manifests=manifests,
        implementation_sha256=implementation_sha256,
        dependency_sha256=dependency_sha256,
        summary_sha256=_sha256_file(artifact_paths["rl_summary"]),
        seed=seed,
        identity=identity,
    )
    for field in _PROVENANCE_FIELDS:
        if provenance[field] != expected[field]:
            raise ValueError(
                f"RL test-only provenance mismatch for {field}: "
                f"{provenance[field]!r} != {expected[field]!r}"
            )
    usage = _validate_summary(summary, skill_digest=skill.digest)
    return skill, usage, manifests


def _validated_common_batch_usage(
    batch: EvaluationBatch,
    *,
    expected_rollouts: int,
) -> dict[str, int | float]:
    if type(batch) is not EvaluationBatch:
        raise TypeError("common reporting evaluator must return EvaluationBatch")
    if type(batch.cache_hit) is not bool:
        raise TypeError("common reporting cache_hit must be a boolean")
    usage = batch.usage
    if type(usage) is not dict:
        raise TypeError("common reporting usage must be a dict")
    allowed = _BATCH_USAGE_FIELDS | {"trajectory_total_tokens"}
    missing = _BATCH_USAGE_FIELDS - set(usage)
    unknown = set(usage) - allowed
    if missing:
        raise ValueError(f"common reporting usage is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(
            f"common reporting usage has unknown fields: {sorted(unknown)}"
        )
    parsed = {
        "student_rollouts": _nonnegative_integer(
            "common usage.student_rollouts", usage["student_rollouts"]
        ),
        "input_tokens": _nonnegative_integer(
            "common usage.input_tokens", usage["input_tokens"]
        ),
        "output_tokens": _nonnegative_integer(
            "common usage.output_tokens", usage["output_tokens"]
        ),
        "total_tokens": _nonnegative_integer(
            "common usage.total_tokens", usage["total_tokens"]
        ),
        "cost_usd": _nonnegative_number("common usage.cost_usd", usage["cost_usd"]),
        "elapsed_s": _nonnegative_number("common usage.elapsed_s", usage["elapsed_s"]),
    }
    expected_reported = 0 if batch.cache_hit else expected_rollouts
    if parsed["student_rollouts"] != expected_reported:
        raise ValueError("common reporting usage has inconsistent student_rollouts")
    if parsed["total_tokens"] != parsed["input_tokens"] + parsed["output_tokens"]:
        raise ValueError("common reporting total_tokens is inconsistent")
    if "trajectory_total_tokens" in usage:
        _nonnegative_integer(
            "common usage.trajectory_total_tokens", usage["trajectory_total_tokens"]
        )
    return parsed


def _common_reporting(
    *,
    evaluator: Any,
    initial_skill: SkillArtifact,
    rl_skill: SkillArtifact,
    manifests: Mapping[Split, TaskManifest],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int | float]]]:
    evaluation = config["evaluation"]
    repetitions = {
        Split.TRAIN: evaluation["report_train_repetitions"],
        Split.VALIDATION: evaluation["validation_repetitions"],
    }
    metrics: dict[str, dict[str, float]] = {}
    usage_by_method: dict[str, dict[str, int | float]] = {}
    for method, skill in zip(_METHODS, (initial_skill, rl_skill), strict=True):
        method_metrics: dict[str, float] = {}
        usage: dict[str, int | float] = {
            "student_rollouts": 0,
            "evaluator_calls": 0,
            "cached_student_rollouts": 0,
            "cached_evaluator_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "elapsed_s": 0.0,
        }
        for split in (Split.TRAIN, Split.VALIDATION):
            tasks = manifests[split].tasks
            repeat_count = repetitions[split]
            logical_rollouts = len(tasks) * repeat_count
            batch = evaluator.evaluate(
                skill,
                tasks,
                split=split,
                seed=seed,
                repetitions=repeat_count,
                use_cache=evaluation["use_cache"],
                blind=True,
            )
            if batch.split is not split:
                raise RuntimeError("common reporting evaluator changed the split")
            if batch.ordered_task_ids != manifests[split].ordered_task_ids:
                raise RuntimeError("common reporting evaluator changed task order")
            batch_usage = _validated_common_batch_usage(
                batch, expected_rollouts=logical_rollouts
            )
            method_metrics[f"{split.value}_reward"] = _unit_number(
                f"{method}.{split.value}_reward", batch.mean_reward
            )
            usage["student_rollouts"] += logical_rollouts
            usage["evaluator_calls"] += 1
            if batch.cache_hit:
                usage["cached_student_rollouts"] += logical_rollouts
                usage["cached_evaluator_calls"] += 1
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                usage[field] += int(batch_usage[field])
            for field in ("cost_usd", "elapsed_s"):
                usage[field] += float(batch_usage[field])
        metrics[method] = method_metrics
        usage_by_method[method] = usage
    return metrics, usage_by_method


def run(
    config_path: Path,
    seed: int | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be a Path")
    if type(test_only) is not bool:
        raise TypeError("test_only must be a boolean")
    resolved_config_path = config_path.expanduser().resolve()
    config = _load_yaml(resolved_config_path)
    artifact_paths = _artifact_paths(config)
    run_seed = config["seed"] if seed is None else _integer("seed", seed)
    config_sha256 = _normalized_config_sha256(config)
    implementation_sha256 = _implementation_sha256()
    dependency_sha256 = _dependency_sha256()
    identity = dict(config["skill_identity"])
    optimization_manifests = _load_optimization_manifests(config)
    initial_skill = _load_initial_skill(config, identity)

    if test_only:
        rl_skill, optimization_usage, manifests = _load_frozen_rl(
            config=config,
            initial_skill=initial_skill,
            optimization_manifests=optimization_manifests,
            artifact_paths=artifact_paths,
            seed=run_seed,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            dependency_sha256=dependency_sha256,
            identity=identity,
        )
    else:
        rl_skill, optimization_usage, manifests = _train_rl(
            config=config,
            initial_skill=initial_skill,
            optimization_manifests=optimization_manifests,
            artifact_paths=artifact_paths,
            seed=run_seed,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            dependency_sha256=dependency_sha256,
            identity=identity,
        )

    output_dir = artifact_paths["output_dir"]
    reporting_evaluator = _make_evaluator(
        config,
        initial_skill=initial_skill,
        cache_path=output_dir / "comparison_rollout_cache.json",
    )
    common_metrics, reporting_usage = _common_reporting(
        evaluator=reporting_evaluator,
        initial_skill=initial_skill,
        rl_skill=rl_skill,
        manifests=manifests,
        config=config,
        seed=run_seed,
    )
    comparison = run_frozen_report(
        initial_skill=initial_skill,
        rl_skill=rl_skill,
        evaluator=reporting_evaluator,
        test_tasks=manifests[Split.TEST].tasks,
        output_dir=output_dir,
        seed=run_seed,
        repetitions=config["evaluation"]["test_repetitions"],
        bootstrap_samples=config["evaluation"]["bootstrap_samples"],
        optimization_usage=optimization_usage,
        reporting_usage=reporting_usage,
    )

    split_digests, split_digest = _split_provenance(manifests)
    experiment_manifest = {
        "protocol": "rl-skill-edit-v2",
        "method": "rl_skill_edit",
        "config_path": str(resolved_config_path),
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "dependency_sha256": dependency_sha256,
        "python_version": sys.version,
        "seed": run_seed,
        "methods": list(_METHODS),
        "runtime": config["runtime"],
        "split_digest": split_digest,
        "split_digests": split_digests,
        "ordered_task_ids": {
            split.value: list(manifests[split].ordered_task_ids) for split in Split
        },
        "skill_digests": {
            "initial_skill": initial_skill.digest,
            "rl_skill_edit": rl_skill.digest,
        },
        "common_metrics": common_metrics,
    }
    _atomic_json_write(output_dir / "experiment_manifest.json", experiment_manifest)
    return {
        "output_dir": str(output_dir),
        "methods": list(_METHODS),
        "test_rewards": {
            item.method: item.stats.mean_reward for item in comparison.methods
        },
    }


__all__ = ["parse_args", "run"]

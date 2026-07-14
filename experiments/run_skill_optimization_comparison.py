#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from baselines.rl_skill_edit.action_space import ActionSpace  # noqa: E402
from baselines.rl_skill_edit.budget import BudgetLedger  # noqa: E402
from baselines.rl_skill_edit.cache import JsonFileCache  # noqa: E402
from baselines.rl_skill_edit.comparison import (  # noqa: E402
    ResourceUsage,
    load_current_method_artifact,
    run_comparison,
)
from baselines.rl_skill_edit.evaluation import (  # noqa: E402
    MockSkillEvaluator,
    RepositorySkillEvaluator,
)
from baselines.rl_skill_edit.manifest import (  # noqa: E402
    TaskManifest,
    validate_manifests,
)
from baselines.rl_skill_edit.optimizer import RLSkillEditOptimizer  # noqa: E402
from baselines.rl_skill_edit.patch_generator import (  # noqa: E402
    OpenRouterPatchGenerator,
    SyntheticPatchGenerator,
)
from baselines.rl_skill_edit.policy import ActorCriticPolicy  # noqa: E402
from baselines.rl_skill_edit.random_policy import RandomEditPolicy  # noqa: E402
from baselines.rl_skill_edit.state_encoder import StateEncoder  # noqa: E402
from baselines.rl_skill_edit.types import (  # noqa: E402
    SkillArtifact,
    Split,
    TaskResult,
)


SUPPORTED_METHODS = (
    "initial_skill",
    "current_method",
    "rl_skill_edit",
    "random_edit_search",
)


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _implementation_sha256() -> str:
    files = list((REPOSITORY_ROOT / "baselines/rl_skill_edit").glob("*.py"))
    files.extend(
        REPOSITORY_ROOT / relative
        for relative in (
            "experiments/run_skill_optimization_comparison.py",
            "src/agent.py",
            "src/client.py",
            "src/evaluator.py",
            "study1_main.py",
        )
    )
    return _sha256_files(files)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare frozen-agent external-skill optimizers."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load a previously frozen RL skill instead of training it.",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _path(config: Mapping[str, Any], name: str) -> Path:
    paths = config.get("paths")
    if not isinstance(paths, Mapping) or name not in paths:
        raise ValueError(f"config is missing paths.{name}")
    return _resolve(paths[name])


def _canonical_skill(
    skill: SkillArtifact,
    *,
    identity: Mapping[str, Any],
) -> SkillArtifact:
    return SkillArtifact(
        skill_id=str(identity["skill_id"]),
        name=str(identity["name"]),
        description=str(identity.get("description", "")),
        body=skill.body,
    )


def _skill_identity(config: Mapping[str, Any]) -> dict[str, str]:
    value = config.get("skill_identity")
    if not isinstance(value, Mapping):
        raise ValueError("config is missing skill_identity")
    if (
        not str(value.get("skill_id", "")).strip()
        or not str(value.get("name", "")).strip()
    ):
        raise ValueError("skill_identity requires non-empty skill_id and name")
    return {
        "skill_id": str(value["skill_id"]),
        "name": str(value["name"]),
        "description": str(value.get("description", "")),
    }


def _load_optimization_manifests(
    config: Mapping[str, Any],
) -> dict[Split, TaskManifest]:
    sizes = config.get("split_sizes")
    if not isinstance(sizes, Mapping):
        raise ValueError("config is missing split_sizes")
    manifests = {
        Split.TRAIN: TaskManifest.load(
            _path(config, "train_manifest"),
            split=Split.TRAIN,
            expected_size=int(sizes["train"]),
        ),
        Split.VALIDATION: TaskManifest.load(
            _path(config, "validation_manifest"),
            split=Split.VALIDATION,
            expected_size=int(sizes["validation"]),
        ),
    }
    validate_manifests(
        manifests[Split.TRAIN],
        manifests[Split.VALIDATION],
    )
    return manifests


def _load_test_manifest(
    config: Mapping[str, Any],
    optimization_manifests: Mapping[Split, TaskManifest],
) -> dict[Split, TaskManifest]:
    sizes = config.get("split_sizes")
    if not isinstance(sizes, Mapping):
        raise ValueError("config is missing split_sizes")
    manifests = dict(optimization_manifests)
    manifests[Split.TEST] = TaskManifest.load(
        _path(config, "test_manifest"),
        split=Split.TEST,
        expected_size=int(sizes["test"]),
    )
    validate_manifests(
        manifests[Split.TRAIN],
        manifests[Split.VALIDATION],
        manifests[Split.TEST],
    )
    return manifests


def _mock_score(initial_digest: str):
    def score(
        skill: SkillArtifact, task: Any, repetition: int, seed: int
    ) -> TaskResult:
        del repetition, seed
        task_id = str(task["task_id"] if isinstance(task, dict) else task.task_id)
        offset = (sum(task_id.encode("utf-8")) % 3) * 0.025
        if skill.digest == initial_digest:
            reward = 0.20 + offset
        elif "CURRENT_METHOD" in skill.body:
            reward = 0.55 + offset
        else:
            reward = 0.80 + offset
        return TaskResult(
            task_id=task_id,
            reward=min(1.0, reward),
            success=reward >= 0.5,
            feedback=f"mock visible reward={reward:.3f}",
            final_answer="mock final answer",
            visible_logs=("mock visible tool log",),
        )

    return score


def _repository_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = _load_yaml(_path(config, "repository_config"))
    models = config.get("models") or {}
    for role in ("student", "editor"):
        if role in models:
            if not isinstance(models[role], Mapping):
                raise ValueError(f"models.{role} must be a mapping")
            base.setdefault(role, {}).update(dict(models[role]))
    evaluation = config.get("evaluation") or {}
    base.setdefault("study1", {}).update(
        {
            "gate_metric": evaluation.get("metric", "mixed"),
            "gate_mixed_weight": evaluation.get("mixed_weight", 0.5),
            "parallel_witness": evaluation.get("parallel", True),
            "witness_workers": evaluation.get("workers", 4),
        }
    )
    return base


def _make_evaluator(
    config: Mapping[str, Any],
    *,
    initial_skill: SkillArtifact,
    cache_path: Path,
):
    runtime = str(config.get("runtime", "repository"))
    cache = JsonFileCache(cache_path)
    if runtime == "mock":
        return MockSkillEvaluator(
            _mock_score(initial_skill.digest),
            cache=cache,
            cache_signature={
                "adapter": "rl-skill-edit-smoke-oracle-v1",
                "initial_skill_digest": initial_skill.digest,
            },
        )
    if runtime != "repository":
        raise ValueError("runtime must be repository or mock")

    from src.agent import StudentAgent
    from src.client import OpenRouterClient
    from src.evaluator import Evaluator

    repository_config = _repository_config(config)
    client = OpenRouterClient(repository_config)
    agent = StudentAgent(repository_config, client)
    core_evaluator = Evaluator(repository_config, agent)
    evaluation = config.get("evaluation") or {}
    return RepositorySkillEvaluator(
        core_evaluator,
        cache=cache,
        gate_metric=str(evaluation.get("metric", "mixed")),
        gate_mixed_weight=float(evaluation.get("mixed_weight", 0.5)),
        success_threshold=float(evaluation.get("success_threshold", 0.8)),
    )


def _make_patch_generator(config: Mapping[str, Any], *, cache_path: Path, seed: int):
    runtime = str(config.get("runtime", "repository"))
    if runtime == "mock":
        return SyntheticPatchGenerator()

    from src.client import OpenRouterClient

    repository_config = _repository_config(config)
    client = OpenRouterClient(repository_config)
    editor = repository_config["editor"]
    return OpenRouterPatchGenerator(
        client=client,
        cache=JsonFileCache(cache_path),
        model=str(editor["model"]),
        temperature=float(editor["temperature"]),
        max_tokens=int(editor["max_tokens"]),
        seed=seed,
    )


def _policy(config: Mapping[str, Any], *, seed: int, random: bool):
    max_modules = int((config.get("action_space") or {})["max_modules"])
    action_space = ActionSpace(max_modules)
    if random:
        return RandomEditPolicy(action_dim=action_space.size, seed=seed)
    encoder = StateEncoder(max_modules, action_space.size)
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("config is missing policy")
    return ActorCriticPolicy(
        input_dim=encoder.state_dim,
        action_dim=action_space.size,
        hidden_dim=int(policy["hidden_dim"]),
        seed=seed,
        learning_rate=float(policy["learning_rate"]),
        gamma=float(policy["gamma"]),
        entropy_coef=float(policy["entropy_coef"]),
        value_coef=float(policy["value_coef"]),
        max_grad_norm=float(policy["max_grad_norm"]),
        normalize_advantages=bool(policy["normalize_advantages"]),
    )


def _optimizer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "optimizer",
        "evaluation",
        "action_space",
        "reward",
        "patch_limits",
    )
    missing = [name for name in required if not isinstance(config.get(name), Mapping)]
    if missing:
        raise ValueError(f"config is missing mappings: {missing}")
    return {name: copy.deepcopy(config[name]) for name in required}


def _train_method(
    name: str,
    config: Mapping[str, Any],
    train_manifest: TaskManifest,
    validation_manifest: TaskManifest,
    initial_skill: SkillArtifact,
    output_dir: Path,
    seed: int,
):
    method_dir = output_dir / name
    evaluator = _make_evaluator(
        config,
        initial_skill=initial_skill,
        cache_path=method_dir / "rollout_cache.json",
    )
    patch_generator = _make_patch_generator(
        config, cache_path=method_dir / "editor_cache.json", seed=seed
    )
    policy = _policy(config, seed=seed, random=name == "random_edit_search")
    budget_limits = config.get("budget")
    if not isinstance(budget_limits, Mapping):
        raise ValueError("config is missing budget")
    budget = BudgetLedger(budget_limits)
    optimizer = RLSkillEditOptimizer(
        config=_optimizer_config(config),
        evaluator=evaluator,
        patch_generator=patch_generator,
        policy=policy,
        output_dir=method_dir,
    )
    result = optimizer.optimize(
        initial_skill=initial_skill,
        train_tasks=train_manifest.tasks,
        validation_tasks=validation_manifest.tasks,
        budget=budget,
        seed=seed,
    )
    if name == "rl_skill_edit":
        if result.budget.get("teacher_rollouts", 0) != 0:
            raise RuntimeError("pure RL used teacher rollouts")
        if result.budget.get("reference_rollouts", 0) != 0:
            raise RuntimeError("pure RL used reference rollouts")
    else:
        result.best_skill.save(method_dir / "best_random_skill.md")
    return result


def _load_rl_test_only(
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    manifests: Mapping[Split, TaskManifest],
    *,
    initial_skill_digest: str,
    seed: int,
    config_sha256: str,
    implementation_sha256: str,
    requirements_sha256: str,
) -> tuple[SkillArtifact, dict[str, Any]]:
    loaded = SkillArtifact.from_file(
        _path(config, "rl_skill"), skill_id=str(identity["skill_id"])
    )
    skill = _canonical_skill(loaded, identity=identity)
    summary_path = _path(config, "rl_summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(_path(config, "rl_provenance").read_text(encoding="utf-8"))
    expected = {
        "method": "rl_skill_edit",
        "best_skill_digest": skill.digest,
        "initial_skill_digest": initial_skill_digest,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "requirements_sha256": requirements_sha256,
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "seed": int(seed),
        "train_manifest_digest": manifests[Split.TRAIN].digest,
        "validation_manifest_digest": manifests[Split.VALIDATION].digest,
        "test_manifest_digest": manifests[Split.TEST].digest,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"RL test-only provenance mismatch for {key}: "
                f"{provenance.get(key)!r} != {value!r}"
            )
    if summary.get("best_skill_digest") != skill.digest:
        raise ValueError("RL summary and frozen Skill digest do not match")
    return skill, dict(summary.get("budget") or {})


def _common_metrics(
    evaluator: Any,
    methods: Mapping[str, SkillArtifact],
    manifests: Mapping[Split, TaskManifest],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    evaluation = config.get("evaluation") or {}
    train_repetitions = int(evaluation.get("report_train_repetitions", 1))
    validation_repetitions = int(evaluation.get("validation_repetitions", 1))
    metrics: dict[str, dict[str, float]] = {}
    reporting_usage: dict[str, dict[str, float]] = {}
    for name, skill in methods.items():
        train = evaluator.evaluate(
            skill,
            manifests[Split.TRAIN].tasks,
            split=Split.TRAIN,
            seed=seed,
            repetitions=train_repetitions,
            use_cache=True,
            blind=True,
        )
        validation = evaluator.evaluate(
            skill,
            manifests[Split.VALIDATION].tasks,
            split=Split.VALIDATION,
            seed=seed,
            repetitions=validation_repetitions,
            use_cache=True,
            blind=True,
        )
        metrics[name] = {
            "train_reward": train.mean_reward,
            "validation_reward": validation.mean_reward,
        }
        train_rollouts = len(manifests[Split.TRAIN].tasks) * train_repetitions
        validation_rollouts = (
            len(manifests[Split.VALIDATION].tasks) * validation_repetitions
        )
        usage = {
            "student_rollouts": float(train_rollouts + validation_rollouts),
            "evaluator_calls": 2.0,
            "cached_student_rollouts": float(
                (train_rollouts if train.cache_hit else 0)
                + (validation_rollouts if validation.cache_hit else 0)
            ),
            "cached_evaluator_calls": float(train.cache_hit)
            + float(validation.cache_hit),
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "cost_usd": 0.0,
            "elapsed_s": 0.0,
        }
        for batch in (train, validation):
            for key in usage:
                if key not in {
                    "student_rollouts",
                    "evaluator_calls",
                    "cached_student_rollouts",
                    "cached_evaluator_calls",
                }:
                    usage[key] += float(batch.usage.get(key, 0.0))
        reporting_usage[name] = usage
    return metrics, reporting_usage


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    implementation_sha256 = _implementation_sha256()
    requirements_sha256 = hashlib.sha256(
        (REPOSITORY_ROOT / "requirements-rl.txt").read_bytes()
    ).hexdigest()
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    methods = tuple(args.methods or config.get("methods") or SUPPORTED_METHODS[:3])
    if len(methods) != len(set(methods)):
        raise ValueError("methods must not contain duplicates")
    output_dir = _path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = _load_optimization_manifests(config)
    identity = _skill_identity(config)
    initial_skill = _canonical_skill(
        SkillArtifact.from_file(
            _path(config, "initial_skill"), skill_id=identity["skill_id"]
        ),
        identity=identity,
    )
    frozen_methods: dict[str, SkillArtifact] = {}
    method_usage: dict[str, dict[str, Any]] = {}
    current_method_provenance: dict[str, Any] | None = None
    trained_rl_result = None
    load_frozen_rl_after_test_manifest = False

    if "initial_skill" in methods:
        frozen_methods["initial_skill"] = initial_skill
        method_usage["initial_skill"] = ResourceUsage().to_dict()

    if "current_method" in methods:
        current = load_current_method_artifact(
            _path(config, "current_skill"),
            _path(config, "current_history"),
            _path(config, "current_jsonl"),
        )
        frozen_methods["current_method"] = _canonical_skill(
            current.skill, identity=identity
        )
        method_usage["current_method"] = current.usage.to_dict()
        current_method_provenance = current.provenance

    if "rl_skill_edit" in methods:
        if args.test_only:
            load_frozen_rl_after_test_manifest = True
        else:
            rl_result = _train_method(
                "rl_skill_edit",
                config,
                manifests[Split.TRAIN],
                manifests[Split.VALIDATION],
                initial_skill,
                output_dir,
                seed,
            )
            trained_rl_result = rl_result
            rl_skill, rl_usage = rl_result.best_skill, rl_result.budget
            frozen_methods["rl_skill_edit"] = rl_skill
            method_usage["rl_skill_edit"] = rl_usage

    if "random_edit_search" in methods:
        if args.test_only:
            raise ValueError("--test-only does not load random_edit_search")
        random_result = _train_method(
            "random_edit_search",
            config,
            manifests[Split.TRAIN],
            manifests[Split.VALIDATION],
            initial_skill,
            output_dir,
            seed,
        )
        frozen_methods["random_edit_search"] = random_result.best_skill
        method_usage["random_edit_search"] = random_result.budget

    manifests = _load_test_manifest(config, manifests)

    if load_frozen_rl_after_test_manifest:
        rl_skill, rl_usage = _load_rl_test_only(
            config,
            identity,
            manifests,
            initial_skill_digest=initial_skill.digest,
            seed=seed,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            requirements_sha256=requirements_sha256,
        )
        frozen_methods["rl_skill_edit"] = rl_skill
        method_usage["rl_skill_edit"] = rl_usage
    elif trained_rl_result is not None:
        generated_skill_path = (output_dir / "rl_skill_edit/best_rl_skill.md").resolve()
        generated_summary_path = (
            output_dir / "rl_skill_edit/rl_optimization_summary.json"
        ).resolve()
        generated_provenance_path = (
            output_dir / "rl_skill_edit/freeze_provenance.json"
        ).resolve()
        configured_artifacts = {
            "paths.rl_skill": (_path(config, "rl_skill"), generated_skill_path),
            "paths.rl_summary": (_path(config, "rl_summary"), generated_summary_path),
            "paths.rl_provenance": (
                _path(config, "rl_provenance"),
                generated_provenance_path,
            ),
        }
        for label, (configured, generated) in configured_artifacts.items():
            if configured.resolve() != generated:
                raise ValueError(
                    f"{label} must point to the generated artifact: {generated}"
                )
        provenance = {
            "method": "rl_skill_edit",
            "best_skill_digest": trained_rl_result.best_skill.digest,
            "initial_skill_digest": initial_skill.digest,
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "requirements_sha256": requirements_sha256,
            "summary_sha256": hashlib.sha256(
                generated_summary_path.read_bytes()
            ).hexdigest(),
            "seed": seed,
            "train_manifest_digest": manifests[Split.TRAIN].digest,
            "validation_manifest_digest": manifests[Split.VALIDATION].digest,
            "test_manifest_digest": manifests[Split.TEST].digest,
            "skill_identity": identity,
        }
        generated_provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if "initial_skill" not in frozen_methods:
        raise ValueError("initial_skill is required for paired comparison")
    ordered_methods = tuple(name for name in methods if name in frozen_methods)
    if ordered_methods[0] != "initial_skill":
        raise ValueError("initial_skill must be the first requested method")

    comparison_evaluator = _make_evaluator(
        config,
        initial_skill=initial_skill,
        cache_path=output_dir / "comparison_rollout_cache.json",
    )
    method_metrics, reporting_usage = _common_metrics(
        comparison_evaluator, frozen_methods, manifests, config, seed
    )
    evaluation = config.get("evaluation") or {}
    comparison = run_comparison(
        {
            "output_dir": output_dir,
            "evaluator": comparison_evaluator,
            "test_tasks": manifests[Split.TEST].tasks,
            "frozen_methods": frozen_methods,
            "method_usage": method_usage,
            "method_metrics": method_metrics,
            "reporting_usage": reporting_usage,
            "bootstrap_samples": int(evaluation.get("bootstrap_samples", 2000)),
            "evaluation": {
                **dict(evaluation),
                "blind": True,
                "test_repetitions": int(evaluation.get("test_repetitions", 1)),
            },
        },
        ordered_methods,
        seed=seed,
    )

    experiment_manifest = {
        "protocol": "rl-skill-edit-v1",
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "requirements_sha256": requirements_sha256,
        "python_version": sys.version,
        "seed": seed,
        "methods": list(ordered_methods),
        "runtime": config.get("runtime", "repository"),
        "split_digests": {split.value: manifests[split].digest for split in Split},
        "ordered_task_ids": {
            split.value: list(manifests[split].ordered_task_ids) for split in Split
        },
        "skill_digests": {name: skill.digest for name, skill in frozen_methods.items()},
        "current_method_artifact_provenance": current_method_provenance,
        "resource_scope": (
            "optimization and common reporting usage are separate; imported "
            "current-method token/cost totals may include its archived Final pass "
            "when the source log lacks phase-level token accounting"
        ),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "methods": [item.method for item in comparison.methods],
        "test_rewards": {
            item.method: item.stats.mean_reward for item in comparison.methods
        },
    }


def main() -> int:
    result = run(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

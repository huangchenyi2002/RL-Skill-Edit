from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .action_space import ActionSpace, EditOperator
from .modules import attribute_failures, parse_modules
from .patch_validator import validate_and_apply_patch
from .reward import compute_incremental_reward
from .state_encoder import StateEncoder
from .types import (
    EditPatch,
    EvaluationBatch,
    GeneratedPatch,
    PatchApplication,
    RewardBreakdown,
    SkillArtifact,
    SkillModule,
    Split,
    Transition,
)


@dataclass(frozen=True)
class OptimizationResult:
    best_skill: SkillArtifact
    final_skill: SkillArtifact
    best_validation_score: float
    budget: dict[str, Any]
    accepted_edits: int
    total_applied_edits: int


def _setting(
    config: Mapping[str, Any],
    name: str,
    section: str,
    *,
    default: Any = None,
    required: bool = False,
) -> Any:
    if name in config:
        return config[name]
    nested = config.get(section)
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    if required:
        raise ValueError(f"RL config is missing {section}.{name}")
    return default


def _optimizer_modules(body: str, max_modules: int) -> tuple[SkillModule, ...]:
    """Treat a content-free leading H1 as the document title, not an edit module."""

    parsed = parse_modules(body, max_modules=max_modules + 1)
    if (
        len(parsed) >= 3
        and not parsed[0].text.strip()
        and parsed[1].level == 1
        and not "\n".join(parsed[1].text.splitlines()[1:]).strip()
    ):
        global_module = SkillModule(
            module_id="global",
            title="global",
            level=0,
            start=0,
            end=parsed[2].start,
            text=body[: parsed[2].start],
            slot=0,
        )
        remaining = tuple(
            SkillModule(
                module_id=module.module_id,
                title=module.title,
                level=module.level,
                start=module.start,
                end=module.end,
                text=module.text,
                slot=index,
            )
            for index, module in enumerate(parsed[2:], start=1)
        )
        modules = (global_module, *remaining)
    else:
        modules = parsed
    if len(modules) > max_modules:
        raise ValueError(
            f"parsed {len(modules)} optimizer modules, which exceeds max_modules={max_modules}"
        )
    return modules


def _scores(batch: EvaluationBatch) -> dict[str, float]:
    return {result.task_id: float(result.reward) for result in batch.results}


def _snapshot(budget: Any) -> dict[str, Any]:
    snapshot = budget.snapshot()
    if hasattr(snapshot, "to_dict"):
        return dict(snapshot.to_dict())
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    raise TypeError("budget snapshot must be a mapping or expose to_dict()")


def _usage(batch: EvaluationBatch) -> tuple[int, int, float]:
    usage = batch.usage
    return (
        int(usage.get("input_tokens", 0)),
        int(usage.get("output_tokens", 0)),
        float(usage.get("elapsed_s", 0.0)),
    )


class RLSkillEditOptimizer:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        evaluator: Any,
        patch_generator: Any,
        policy: Any,
        output_dir: str | Path,
    ) -> None:
        self.config = dict(config)
        self.evaluator = evaluator
        self.patch_generator = patch_generator
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.episodes = int(_setting(config, "episodes", "optimizer", required=True))
        self.horizon = int(_setting(config, "horizon", "optimizer", required=True))
        self.minibatch_size = int(
            _setting(config, "minibatch_size", "optimizer", required=True)
        )
        self.validation_interval = int(
            _setting(config, "validation_interval", "optimizer", default=1)
        )
        self.train_repetitions = int(
            _setting(config, "train_repetitions", "evaluation", default=1)
        )
        self.validation_repetitions = int(
            _setting(config, "validation_repetitions", "evaluation", default=1)
        )
        self.max_modules = int(
            _setting(config, "max_modules", "action_space", required=True)
        )
        for name, value in (
            ("episodes", self.episodes),
            ("horizon", self.horizon),
            ("minibatch_size", self.minibatch_size),
            ("validation_interval", self.validation_interval),
            ("train_repetitions", self.train_repetitions),
            ("validation_repetitions", self.validation_repetitions),
            ("max_modules", self.max_modules),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self.reward_config = dict(config.get("reward") or {})
        self.patch_limits = dict(
            config.get("patch_limits")
            or {
                "max_changed_tokens": 256,
                "max_length_growth_tokens": 128,
                "max_skill_tokens": 20000,
            }
        )
        self.use_cache = bool(_setting(config, "use_cache", "evaluation", default=True))
        self.action_space = ActionSpace(self.max_modules)
        self.state_encoder = StateEncoder(self.max_modules, self.action_space.size)

    def optimize(
        self,
        *,
        initial_skill: SkillArtifact,
        train_tasks: Iterable[Any],
        validation_tasks: Iterable[Any],
        budget: Any,
        seed: int,
    ) -> OptimizationResult:
        train_tasks = tuple(train_tasks)
        validation_tasks = tuple(validation_tasks)
        if len(train_tasks) < self.minibatch_size:
            raise ValueError("train task count is smaller than minibatch_size")
        if not validation_tasks:
            raise ValueError("validation task bundle must not be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "rl_training_log.jsonl"
        summary_path = self.output_dir / "rl_episode_summary.csv"
        log_path.write_text("", encoding="utf-8")
        run_started = time.monotonic()
        self._rollout_cost_usd = 0.0
        self._editor_cost_usd = 0.0

        rng = np.random.default_rng(int(seed))
        initial_validation = self._evaluate(
            initial_skill,
            validation_tasks,
            Split.VALIDATION,
            int(seed),
            self.validation_repetitions,
            blind=False,
            budget=budget,
        )
        best_skill = initial_skill
        best_validation_score = initial_validation.mean_reward
        final_skill = initial_skill
        best_edit_count = 0
        total_applied_edits = 0
        global_step = 0
        episode_rows: list[dict[str, Any]] = []

        for episode in range(self.episodes):
            episode_dir = self.output_dir / "episodes" / f"episode_{episode:04d}"
            skill_dir = episode_dir / "skills"
            skill_dir.mkdir(parents=True, exist_ok=True)
            current_skill = initial_skill
            accepted_counts_by_slot: dict[int, int] = {}
            last_action_index: int | None = None
            last_reward = 0.0
            edit_history: list[dict[str, Any]] = []
            transitions: list[Transition] = []
            episode_log_rows: list[dict[str, Any]] = []
            episode_return = 0.0
            termination = "horizon"
            non_stop_steps = 0

            for step in range(self.horizon):
                modules = _optimizer_modules(current_skill.body, self.max_modules)
                task_indices = rng.choice(
                    len(train_tasks), size=self.minibatch_size, replace=False
                )
                minibatch = tuple(train_tasks[int(index)] for index in task_indices)
                pair_seed = int(seed) + 100000 * episode + step
                current_batch = self._evaluate(
                    current_skill,
                    minibatch,
                    Split.TRAIN,
                    pair_seed,
                    self.train_repetitions,
                    blind=False,
                    budget=budget,
                )
                diagnostics = attribute_failures(modules, current_batch)
                state = self.state_encoder.encode(
                    current_text=current_skill.body,
                    initial_text=initial_skill.body,
                    modules=modules,
                    diagnostics=diagnostics,
                    accepted_edit_counts={
                        module.module_id: accepted_counts_by_slot.get(module.slot, 0)
                        for module in modules
                    },
                    round_index=step,
                    horizon=self.horizon,
                    remaining_rollout_fraction=float(budget.remaining_fraction()),
                    last_action_index=last_action_index,
                    last_reward=last_reward,
                )
                mask = self.action_space.mask(modules)
                decision = self.policy.select(state, mask, deterministic=False)
                action = self.action_space.decode(decision.action_index, modules)
                current_skill_path = skill_dir / f"step_{step:03d}_current.md"
                current_skill.save(current_skill_path)
                probabilities = np.asarray(decision.probabilities, dtype=float)
                selected_probability = float(probabilities[action.index])
                positive = probabilities > 0.0
                policy_entropy = float(
                    -np.sum(probabilities[positive] * np.log(probabilities[positive]))
                )

                if action.operator is EditOperator.STOP:
                    transition = Transition(
                        state=state,
                        action_index=action.index,
                        reward=0.0,
                        done=True,
                        mask=mask,
                    )
                    transitions.append(transition)
                    log_row = {
                        "episode": episode,
                        "step": step,
                        "current_skill_path": self._artifact_reference(
                            current_skill_path
                        ),
                        "candidate_skill_path": self._artifact_reference(
                            current_skill_path
                        ),
                        "action_index": action.index,
                        "module_id": action.module_id,
                        "operator": action.operator.value,
                        "policy_probability": selected_probability,
                        "policy_entropy": policy_entropy,
                        "value_prediction": float(decision.value),
                        "status": "stop",
                        "patch": None,
                        "train_task_ids": list(current_batch.ordered_task_ids),
                        "current_task_scores": _scores(current_batch),
                        "candidate_task_scores": _scores(current_batch),
                        "reward": 0.0,
                        "train_score_before": current_batch.mean_reward,
                        "train_score_after": current_batch.mean_reward,
                        "validation_score": best_validation_score,
                        "reward_components": asdict(
                            RewardBreakdown(0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
                        ),
                        "current_skill_digest": current_skill.digest,
                        "candidate_skill_digest": current_skill.digest,
                        "budget": _snapshot(budget),
                    }
                    self._append_log(log_path, log_row)
                    episode_log_rows.append(log_row)
                    termination = "stop"
                    break

                module = next(
                    module for module in modules if module.module_id == action.module_id
                )
                history_snapshot = tuple(edit_history)
                editor_cache_hit = False
                cache_probe = getattr(self.patch_generator, "cache_will_hit", None)
                if callable(cache_probe):
                    editor_cache_hit = bool(
                        cache_probe(
                            current_skill,
                            module,
                            action.operator,
                            current_batch,
                            history_snapshot,
                        )
                    )
                editor_started = time.monotonic()
                reservation = budget.reserve_editor(cache_hit=editor_cache_hit)
                generated_value = self.patch_generator.generate(
                    current_skill,
                    module,
                    action.operator,
                    current_batch,
                    history_snapshot,
                )
                if isinstance(generated_value, EditPatch):
                    generated = GeneratedPatch(
                        patch=generated_value,
                        cache_hit=False,
                        request_hash="test-double",
                        usage={},
                    )
                elif isinstance(generated_value, GeneratedPatch):
                    generated = generated_value
                else:
                    raise TypeError(
                        "patch generator must return EditPatch or GeneratedPatch"
                    )
                if generated.cache_hit != editor_cache_hit:
                    raise RuntimeError(
                        "Editor cache state changed after budget reservation"
                    )
                editor_elapsed = time.monotonic() - editor_started
                self._record_editor(
                    budget, reservation, generated, editor_elapsed=editor_elapsed
                )

                if generated.patch is None:
                    application = PatchApplication(
                        accepted=False,
                        skill=current_skill,
                        reason=generated.error or "Editor returned no structured patch",
                    )
                else:
                    application = validate_and_apply_patch(
                        current_skill,
                        modules,
                        action,
                        generated.patch,
                        self.patch_limits,
                    )
                if application.accepted:
                    candidate_skill = application.skill
                    candidate_batch = self._evaluate(
                        candidate_skill,
                        minibatch,
                        Split.TRAIN,
                        pair_seed,
                        self.train_repetitions,
                        blind=False,
                        budget=budget,
                    )
                    status = "applied"
                    invalid = False
                else:
                    candidate_skill = current_skill
                    candidate_batch = current_batch
                    status = "invalid"
                    invalid = True

                reward = compute_incremental_reward(
                    before=current_batch,
                    after=candidate_batch,
                    initial_text=initial_skill.body,
                    current_text=current_skill.body,
                    candidate_text=candidate_skill.body,
                    config=self.reward_config,
                    invalid=invalid,
                )
                is_done = step == self.horizon - 1
                transitions.append(
                    Transition(
                        state=state,
                        action_index=action.index,
                        reward=reward.total,
                        done=is_done,
                        mask=mask,
                    )
                )
                episode_return += reward.total
                non_stop_steps += 1
                global_step += 1
                validation_score = None
                if global_step % self.validation_interval == 0:
                    validation = self._evaluate(
                        candidate_skill,
                        validation_tasks,
                        Split.VALIDATION,
                        int(seed),
                        self.validation_repetitions,
                        blind=False,
                        budget=budget,
                    )
                    validation_score = validation.mean_reward
                    if validation.mean_reward > best_validation_score:
                        best_validation_score = validation.mean_reward
                        best_skill = candidate_skill
                        best_edit_count = sum(accepted_counts_by_slot.values()) + int(
                            application.accepted
                        )
                candidate_skill_path = skill_dir / f"step_{step:03d}_candidate.md"
                candidate_skill.save(candidate_skill_path)
                log_row = {
                    "episode": episode,
                    "step": step,
                    "current_skill_path": self._artifact_reference(current_skill_path),
                    "candidate_skill_path": self._artifact_reference(
                        candidate_skill_path
                    ),
                    "action_index": action.index,
                    "module_id": action.module_id,
                    "operator": action.operator.value,
                    "policy_probability": selected_probability,
                    "policy_entropy": policy_entropy,
                    "value_prediction": float(decision.value),
                    "status": status,
                    "patch": (
                        generated.patch.to_dict()
                        if generated.patch is not None
                        else None
                    ),
                    "patch_reason": application.reason,
                    "request_hash": generated.request_hash,
                    "cache_hit": generated.cache_hit,
                    "train_task_ids": list(current_batch.ordered_task_ids),
                    "current_task_scores": _scores(current_batch),
                    "candidate_task_scores": _scores(candidate_batch),
                    "reward": reward.total,
                    "train_score_before": current_batch.mean_reward,
                    "train_score_after": candidate_batch.mean_reward,
                    "validation_score": validation_score,
                    "reward_components": asdict(reward),
                    "current_skill_digest": current_skill.digest,
                    "candidate_skill_digest": candidate_skill.digest,
                    "budget": _snapshot(budget),
                }
                self._append_log(log_path, log_row)
                episode_log_rows.append(log_row)
                edit_history.append(
                    {
                        "episode": episode,
                        "step": step,
                        "status": status,
                        "patch": (
                            generated.patch.to_dict()
                            if generated.patch is not None
                            else None
                        ),
                        "reward": reward.total,
                    }
                )
                if application.accepted:
                    current_skill = candidate_skill
                    accepted_counts_by_slot[module.slot] = (
                        accepted_counts_by_slot.get(module.slot, 0) + 1
                    )
                    total_applied_edits += 1
                last_action_index = action.index
                last_reward = reward.total

            if transitions:
                update_metrics = self.policy.update(tuple(transitions))
            else:
                update_metrics = {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "grad_norm": 0.0,
                }
            final_skill = current_skill
            final_skill.save(episode_dir / "final_skill.md")
            best_skill.save(episode_dir / "best_validation_skill.md")
            self.policy.save(episode_dir / "policy_checkpoint.pt")
            (episode_dir / "edit_trajectory.json").write_text(
                json.dumps(episode_log_rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            episode_rows.append(
                {
                    "episode": episode,
                    "steps": non_stop_steps,
                    "episode_return": episode_return,
                    "termination": termination,
                    "best_validation_score": best_validation_score,
                    "best_skill_digest": best_skill.digest,
                    "final_skill_path": self._artifact_reference(
                        episode_dir / "final_skill.md"
                    ),
                    "best_skill_path": self._artifact_reference(
                        episode_dir / "best_validation_skill.md"
                    ),
                    "policy_checkpoint_path": self._artifact_reference(
                        episode_dir / "policy_checkpoint.pt"
                    ),
                    "trajectory_path": self._artifact_reference(
                        episode_dir / "edit_trajectory.json"
                    ),
                    **update_metrics,
                }
            )

        best_skill.save(self.output_dir / "best_rl_skill.md")
        self.policy.save(self.output_dir / "final_rl_policy.pt")
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
        usage_summary = _snapshot(budget)
        usage_summary.update(
            {
                "total_tokens": int(usage_summary.get("input_tokens", 0))
                + int(usage_summary.get("output_tokens", 0)),
                "rollout_cost_usd": self._rollout_cost_usd,
                "editor_cost_usd": self._editor_cost_usd,
                "cost_usd": self._rollout_cost_usd + self._editor_cost_usd,
                "wall_time_s": time.monotonic() - run_started,
                "edit_count": best_edit_count,
            }
        )
        result = OptimizationResult(
            best_skill=best_skill,
            final_skill=final_skill,
            best_validation_score=best_validation_score,
            budget=usage_summary,
            accepted_edits=best_edit_count,
            total_applied_edits=total_applied_edits,
        )
        (self.output_dir / "rl_optimization_summary.json").write_text(
            json.dumps(
                {
                    "best_skill_digest": result.best_skill.digest,
                    "final_skill_digest": result.final_skill.digest,
                    "best_validation_score": result.best_validation_score,
                    "accepted_edits": result.accepted_edits,
                    "total_applied_edits": result.total_applied_edits,
                    "budget": result.budget,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def _evaluate(
        self,
        skill: SkillArtifact,
        tasks: tuple[Any, ...],
        split: Split,
        seed: int,
        repetitions: int,
        *,
        blind: bool,
        budget: Any,
    ) -> EvaluationBatch:
        started = time.monotonic()
        rollout_cache_hit = False
        cache_probe = getattr(self.evaluator, "cache_will_hit", None)
        if self.use_cache and callable(cache_probe):
            rollout_cache_hit = bool(
                cache_probe(skill, tasks, split, seed, repetitions, blind)
            )
        reservation = budget.reserve_evaluation(
            task_count=len(tasks),
            repetitions=repetitions,
            cache_hit=rollout_cache_hit,
        )
        batch = self.evaluator.evaluate(
            skill,
            tasks,
            split=split,
            seed=seed,
            repetitions=repetitions,
            use_cache=self.use_cache,
            blind=blind,
        )
        if batch.cache_hit != rollout_cache_hit:
            raise RuntimeError("rollout cache state changed after budget reservation")
        elapsed = time.monotonic() - started
        input_tokens, output_tokens, recorded_elapsed = _usage(batch)
        self._rollout_cost_usd += float(batch.usage.get("cost_usd", 0.0))
        self._record_evaluation(
            budget,
            reservation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=(
                0.0 if rollout_cache_hit else max(elapsed, recorded_elapsed)
            ),
        )
        return batch

    @staticmethod
    def _record_evaluation(
        budget: Any,
        reservation: Any,
        *,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
    ) -> None:
        budget.record_evaluation(
            reservation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed_seconds,
        )

    def _record_editor(
        self,
        budget: Any,
        reservation: Any,
        generated: GeneratedPatch,
        *,
        editor_elapsed: float,
    ) -> None:
        usage = generated.usage
        self._editor_cost_usd += float(usage.get("cost_usd", 0.0))
        kwargs = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "elapsed_seconds": float(usage.get("elapsed_s", editor_elapsed)),
        }
        budget.record_editor(reservation, **kwargs)

    def _artifact_reference(self, path: Path) -> str:
        return path.relative_to(self.output_dir).as_posix()

    @staticmethod
    def _append_log(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["OptimizationResult", "RLSkillEditOptimizer"]

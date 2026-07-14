from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_skill_edit.action_space import ActionSpace, EditOperator
from rl_skill_edit.modules import attribute_failures, parse_modules
from rl_skill_edit.reward import compute_incremental_reward
from rl_skill_edit.state_encoder import StateEncoder
from rl_skill_edit.types import (
    EvaluationBatch,
    ModuleDiagnostics,
    SkillArtifact,
    Split,
    TaskResult,
)


def _batch(split: Split, scores: dict[str, float]) -> EvaluationBatch:
    return EvaluationBatch(
        split=split,
        results=tuple(
            TaskResult(task_id=task_id, reward=score, success=score >= 0.5)
            for task_id, score in scores.items()
        ),
    )


def test_skill_artifact_round_trip_digest_without_legacy_library_adapter(
    tmp_path: Path,
):
    source = tmp_path / "SKILL.md"
    source.write_text(
        "---\nname: Lookup Skill\ndescription: Safe lookup\n---\n\n# Rules\nUse exact keys.\n",
        encoding="utf-8",
    )

    artifact = SkillArtifact.from_file(source, skill_id="spreadsheet__lookup")
    saved = tmp_path / "saved.md"
    artifact.save(saved)

    assert artifact == SkillArtifact.from_file(saved, skill_id="spreadsheet__lookup")
    assert (
        artifact.digest
        == SkillArtifact.from_file(source, skill_id="spreadsheet__lookup").digest
    )
    assert (
        artifact.digest
        != SkillArtifact(
            artifact.skill_id,
            artifact.name,
            artifact.description,
            artifact.body + "Changed",
        ).digest
    )
    assert not hasattr(SkillArtifact, "to_library")


def test_parser_is_stable_non_overlapping_and_distinguishes_duplicate_titles():
    body = (
        "Opening guidance.\n\n"
        "# Rules\n- Match exact keys.\n\n"
        "## Details\n- Normalize whitespace.\n\n"
        "# Rules\n- Verify the output.\n"
    )

    modules = parse_modules(body, max_modules=6)
    modules_again = parse_modules(
        body.replace("exact keys", "exact workbook keys"), max_modules=6
    )

    assert [module.title for module in modules] == [
        "global",
        "Rules",
        "Details",
        "Rules",
    ]
    assert [module.level for module in modules] == [0, 1, 2, 1]
    assert [module.slot for module in modules] == list(range(4))
    assert modules[0].module_id == "global"
    assert len({module.module_id for module in modules}) == len(modules)
    assert [module.module_id for module in modules] == [
        module.module_id for module in modules_again
    ]
    assert all(left.end <= right.start for left, right in zip(modules, modules[1:]))
    assert all(module.text == body[module.start : module.end] for module in modules)


def test_parser_keeps_frontmatter_free_unheaded_text_in_global_and_fails_on_capacity():
    assert parse_modules("Use exact keys before saving.\n", max_modules=1)[0].text == (
        "Use exact keys before saving.\n"
    )

    with pytest.raises(ValueError, match="max_modules"):
        parse_modules("intro\n# One\na\n# Two\nb\n", max_modules=2)


def test_failure_attribution_uses_visible_text_and_falls_back_to_global():
    modules = parse_modules(
        "# Lookup Rules\nUse customer lookup keys.\n\n# Formatting\nApply number formats.\n",
        max_modules=4,
    )
    batch = EvaluationBatch(
        split=Split.TRAIN,
        results=(
            TaskResult(
                task_id="lookup-failure",
                reward=0.0,
                success=False,
                feedback="The customer lookup key was wrong.",
                final_answer="Lookup returned no row.",
                evaluator_output="Formatting module secretly named here.",
                raw_rewards=(1.0,),
            ),
            TaskResult(
                task_id="unresolved",
                reward=0.0,
                success=False,
                feedback="The output is incorrect.",
            ),
            TaskResult(
                task_id="success-is-not-a-failure",
                reward=1.0,
                success=True,
                feedback="lookup",
            ),
        ),
    )

    diagnostics = attribute_failures(modules, batch)
    lookup = next(module for module in modules if module.title == "Lookup Rules")

    assert diagnostics[lookup.module_id].task_ids == ("lookup-failure",)
    assert diagnostics["global"].task_ids == ("unresolved",)
    assert sum(item.failure_count for item in diagnostics.values()) == 2


def test_action_space_masks_padding_stop_and_all_non_stop_operators():
    modules = parse_modules(
        "# Rules\nA rule.\n# Examples\nAn example.\n", max_modules=4
    )
    space = ActionSpace(max_modules=4)
    mask = space.mask(modules)

    assert mask.dtype == np.bool_
    assert mask.shape == (4 * len(EditOperator),)
    stop_indices = [
        space.encode(slot, EditOperator.STOP)
        for slot in range(space.max_modules)
        if mask[space.encode(slot, EditOperator.STOP)]
    ]
    assert stop_indices == [space.encode(0, EditOperator.STOP)]
    for module in modules[1:]:
        for operator in EditOperator:
            assert bool(mask[space.encode(module.slot, operator)]) is (
                operator is not EditOperator.STOP
            )
    for slot in range(len(modules), space.max_modules):
        assert not mask[slot * len(EditOperator) : (slot + 1) * len(EditOperator)].any()

    index = space.encode(modules[1].slot, EditOperator.REWRITE_RULE)
    assert space.decode(index, modules).module_id == modules[1].module_id
    assert space.decode(index, modules).operator is EditOperator.REWRITE_RULE
    assert space.decode(index, modules).index == index


def test_state_encoder_has_fixed_finite_shape_and_every_required_signal_changes_it():
    modules = parse_modules("# Rules\nUse exact keys.\n", max_modules=3)
    diagnostics = {
        module.module_id: ModuleDiagnostics(
            module_id=module.module_id,
            failure_count=1 if module.title == "Rules" else 0,
            task_ids=("t1",) if module.title == "Rules" else (),
            mean_reward=0.25,
        )
        for module in modules
    }
    encoder = StateEncoder(max_modules=3, action_space_size=3 * len(EditOperator))
    base = dict(
        current_text="one two three",
        initial_text="one two three",
        modules=modules,
        diagnostics=diagnostics,
        accepted_edit_counts={module.module_id: 0 for module in modules},
        round_index=1,
        horizon=5,
        remaining_rollout_fraction=0.8,
        last_action_index=2,
        last_reward=0.1,
    )
    encoded = encoder.encode(**base)

    assert encoded.shape == (encoder.state_dim,)
    assert np.isfinite(encoded).all()
    one_module = parse_modules("plain text\n", max_modules=3)
    fewer = dict(
        base, modules=one_module, diagnostics={"global": diagnostics["global"]}
    )
    assert encoder.encode(**fewer).shape == encoded.shape

    variants = (
        dict(round_index=2),
        dict(current_text="one two three four"),
        dict(remaining_rollout_fraction=0.2),
        dict(last_action_index=3),
        dict(last_reward=-0.4),
        dict(current_text="one X three"),
        dict(accepted_edit_counts={modules[1].module_id: 2}),
        dict(
            diagnostics={
                **diagnostics,
                modules[1].module_id: ModuleDiagnostics(
                    module_id=modules[1].module_id,
                    failure_count=3,
                    task_ids=("t1", "t2", "t3"),
                    mean_reward=0.0,
                ),
            }
        ),
    )
    for changed in variants:
        assert not np.array_equal(encoded, encoder.encode(**(base | changed)))


def test_reward_is_paired_and_uses_exact_token_levenshtein_costs():
    breakdown = compute_incremental_reward(
        before=_batch(Split.TRAIN, {"a": 0.2, "b": 0.5}),
        after=_batch(Split.TRAIN, {"a": 0.5, "b": 0.6}),
        initial_text="a b c",
        current_text="a b c",
        candidate_text="a x c d",
        config={"beta_len": 0.1, "beta_edit": 0.05, "beta_invalid": 0.3},
        invalid=True,
    )

    assert breakdown.paired_delta == pytest.approx(0.2)
    assert breakdown.length_cost == 1
    assert breakdown.edit_cost == 2
    assert breakdown.invalid_cost == 1
    assert breakdown.total == pytest.approx(-0.3)


def test_reward_rejects_unpaired_or_reordered_task_results():
    with pytest.raises(ValueError, match="paired|task"):
        compute_incremental_reward(
            before=_batch(Split.TRAIN, {"a": 0.0, "b": 0.0}),
            after=_batch(Split.TRAIN, {"b": 1.0, "a": 1.0}),
            initial_text="a",
            current_text="a",
            candidate_text="b",
            config={"beta_len": 0.0, "beta_edit": 0.0, "beta_invalid": 0.0},
            invalid=False,
        )

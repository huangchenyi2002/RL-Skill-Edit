from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.rl_skill_edit.action_space import EditOperator
from baselines.rl_skill_edit.cache import JsonFileCache
from baselines.rl_skill_edit.modules import parse_modules
from baselines.rl_skill_edit.patch_generator import (
    MockPatchGenerator,
    OpenRouterPatchGenerator,
)
from baselines.rl_skill_edit.patch_validator import validate_and_apply_patch
from baselines.rl_skill_edit.types import (
    Action,
    EditPatch,
    EvaluationBatch,
    SkillArtifact,
    Split,
    TaskResult,
)


BODY = """# Rules
- Rule alpha.
- Rule beta.
- Redundant rule A.
- Redundant rule A restated.
Example: old input -> old output.
Example: second input -> second output.

# Formatting
- Keep dates formatted.
"""

LIMITS = {
    "max_changed_tokens": 40,
    "max_length_growth_tokens": 20,
    "max_skill_tokens": 200,
}


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected Editor request")
        return self.responses.pop(0), {
            "ok": True,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }


def _skill(body: str = BODY) -> SkillArtifact:
    return SkillArtifact("spreadsheet", "Spreadsheet", "Safe spreadsheet work", body)


def _rules_module(skill: SkillArtifact):
    return next(
        module
        for module in parse_modules(skill.body, max_modules=5)
        if module.title == "Rules"
    )


def _patch(module_id: str, operator: EditOperator, old: str, new: str) -> EditPatch:
    return EditPatch(
        target_module=module_id,
        operator=operator,
        rationale="Make one local improvement.",
        old_text=old,
        new_text=new,
        expected_effect="Improve task reward.",
    )


@pytest.mark.parametrize(
    ("operator", "old", "new"),
    (
        (EditOperator.ADD_RULE, "- Rule alpha.", "- Rule alpha.\n- Rule gamma."),
        (EditOperator.REWRITE_RULE, "- Rule alpha.", "- Rule alpha with exact keys."),
        (EditOperator.DELETE_RULE, "- Rule beta.\n", ""),
        (
            EditOperator.ADD_EXAMPLE,
            "Example: old input -> old output.",
            "Example: old input -> old output.\nExample: new input -> new output.",
        ),
        (
            EditOperator.REWRITE_EXAMPLE,
            "Example: old input -> old output.",
            "Example: exact key -> matched row.",
        ),
        (
            EditOperator.MERGE_REDUNDANT_RULES,
            "- Redundant rule A.\n- Redundant rule A restated.",
            "- Combined rule A.",
        ),
        (
            EditOperator.REORDER_CONTENT,
            "- Rule alpha.\n- Rule beta.",
            "- Rule beta.\n- Rule alpha.",
        ),
    ),
)
def test_each_non_stop_operator_applies_exactly_one_local_edit(operator, old, new):
    skill = _skill()
    modules = parse_modules(skill.body, max_modules=5)
    module = _rules_module(skill)
    action = Action(module_id=module.module_id, operator=operator, index=1)

    applied = validate_and_apply_patch(
        skill, modules, action, _patch(module.module_id, operator, old, new), LIMITS
    )

    assert applied.accepted is True
    assert applied.skill.body == skill.body.replace(old, new, 1)
    assert applied.skill.skill_id == skill.skill_id
    assert skill.body == BODY


@pytest.mark.parametrize("mismatch", ["target", "operator"])
def test_target_and_operator_mismatch_fail_closed(mismatch: str):
    skill = _skill()
    modules = parse_modules(skill.body, max_modules=5)
    module = _rules_module(skill)
    action = Action(module.module_id, EditOperator.REWRITE_RULE, 1)
    patch = _patch(
        module.module_id, EditOperator.REWRITE_RULE, "- Rule alpha.", "- Better rule."
    )
    if mismatch == "target":
        patch = _patch(
            "some-other-module", patch.operator, patch.old_text, patch.new_text
        )
    else:
        patch = _patch(module.module_id, EditOperator.DELETE_RULE, patch.old_text, "")

    result = validate_and_apply_patch(skill, modules, action, patch, LIMITS)

    assert result.accepted is False
    assert result.skill == skill
    assert result.reason


@pytest.mark.parametrize(
    ("body", "operator", "old", "new", "limits"),
    (
        (
            "# Rules\n- Duplicate.\n- Duplicate.\n",
            EditOperator.REWRITE_RULE,
            "- Duplicate.",
            "- Unique.",
            LIMITS,
        ),
        (
            "# Rules\n<!-- SLOW_UPDATE_START -->\nsecret\n<!-- SLOW_UPDATE_END -->\n",
            EditOperator.REWRITE_RULE,
            "secret",
            "changed",
            LIMITS,
        ),
        (
            BODY,
            EditOperator.REWRITE_RULE,
            "- Rule alpha.",
            "- Rule alpha.",
            LIMITS,
        ),
        (
            BODY,
            EditOperator.REWRITE_RULE,
            "- Rule alpha.",
            "- This replacement contains far too many changed tokens for the configured local limit.",
            {**LIMITS, "max_changed_tokens": 2},
        ),
    ),
)
def test_ambiguous_protected_noop_and_oversized_edits_are_rejected(
    body, operator, old, new, limits
):
    skill = _skill(body)
    modules = parse_modules(body, max_modules=5)
    module = next(module for module in modules if module.title == "Rules")
    result = validate_and_apply_patch(
        skill,
        modules,
        Action(module.module_id, operator, 1),
        _patch(module.module_id, operator, old, new),
        limits,
    )

    assert result.accepted is False
    assert result.skill == skill


def test_whole_skill_cross_module_and_semantically_wrong_patch_are_rejected():
    skill = _skill()
    modules = parse_modules(skill.body, max_modules=5)
    module = _rules_module(skill)
    cases = (
        _patch(
            module.module_id,
            EditOperator.REWRITE_RULE,
            skill.body,
            "# Replacement\nEverything changed.",
        ),
        _patch(
            module.module_id,
            EditOperator.REWRITE_RULE,
            "# Formatting\n- Keep dates formatted.",
            "# Formatting\n- Remove formats.",
        ),
        _patch(module.module_id, EditOperator.ADD_RULE, "- Rule alpha.", ""),
    )

    for patch in cases:
        action = Action(module.module_id, patch.operator, 1)
        result = validate_and_apply_patch(
            skill, modules, action, patch, {**LIMITS, "max_changed_tokens": 500}
        )
        assert result.accepted is False
        assert result.skill == skill


def test_patch_spanning_into_protected_region_and_whole_global_skill_are_rejected():
    protected_body = "prefix\n<!-- APPENDIX_START -->\nsecret\n<!-- APPENDIX_END -->\n"
    protected_skill = _skill(protected_body)
    protected_modules = parse_modules(protected_body, max_modules=1)
    protected_patch = _patch(
        "global",
        EditOperator.REWRITE_RULE,
        "prefix\n<!-- APPENDIX_START -->\nsecret",
        "replacement",
    )
    protected_result = validate_and_apply_patch(
        protected_skill,
        protected_modules,
        Action("global", EditOperator.REWRITE_RULE, 1),
        protected_patch,
        {**LIMITS, "max_changed_tokens": 500},
    )
    assert protected_result.accepted is False

    whole_skill = _skill("one global rule only")
    whole_modules = parse_modules(whole_skill.body, max_modules=1)
    whole_patch = _patch(
        "global",
        EditOperator.REWRITE_RULE,
        whole_skill.body,
        "replacement global rule",
    )
    whole_result = validate_and_apply_patch(
        whole_skill,
        whole_modules,
        Action("global", EditOperator.REWRITE_RULE, 1),
        whole_patch,
        {**LIMITS, "max_changed_tokens": 500},
    )
    assert whole_result.accepted is False


@pytest.mark.parametrize(
    ("operator", "old", "new"),
    (
        (EditOperator.REWRITE_RULE, "Rules", "Policies"),
        (
            EditOperator.ADD_RULE,
            "- Rule alpha.",
            "- Rule alpha.\n### Injected module\n- Hidden rule.",
        ),
    ),
)
def test_patch_cannot_rename_or_inject_markdown_headings(operator, old, new):
    skill = _skill()
    modules = parse_modules(skill.body, max_modules=5)
    module = _rules_module(skill)

    result = validate_and_apply_patch(
        skill,
        modules,
        Action(module.module_id, operator, 1),
        _patch(module.module_id, operator, old, new),
        LIMITS,
    )

    assert result.accepted is False
    assert result.skill == skill
    assert "heading" in result.reason or "topology" in result.reason


def _train_batch(**overrides) -> EvaluationBatch:
    result = TaskResult(
        task_id="train-1",
        reward=0.0,
        success=False,
        feedback="VISIBLE_FEEDBACK: lookup key failed.",
        final_answer="VISIBLE_FINAL_ANSWER",
        evaluator_output="VISIBLE_EVALUATOR_OUTPUT",
        visible_logs=("VISIBLE_TOOL_LOG",),
        raw_rewards=(0.1, 0.2),
    )
    return EvaluationBatch(split=overrides.get("split", Split.TRAIN), results=(result,))


def _response(module_id: str, **changes) -> str:
    payload = {
        "target_module": module_id,
        "operator": EditOperator.REWRITE_RULE.value,
        "rationale": "Clarify the key rule.",
        "old_text": "- Rule alpha.",
        "new_text": "- Rule alpha with exact keys.",
        "expected_effect": "Fewer lookup failures.",
    }
    payload.update(changes)
    return json.dumps(payload)


def _openrouter_generator(
    tmp_path: Path, client: FakeClient
) -> OpenRouterPatchGenerator:
    return OpenRouterPatchGenerator(
        client=client,
        cache=JsonFileCache(tmp_path / "editor-cache.json"),
        model="fake-editor",
        temperature=0.0,
        max_tokens=512,
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.pop("rationale"),
        lambda payload: payload.update(extra_field="not allowed"),
        lambda payload: payload.update(operator="UNKNOWN_OPERATOR"),
        lambda payload: payload.update(target_module=""),
    ),
)
def test_editor_json_schema_requires_exact_fields_and_operator_enum(
    tmp_path: Path, mutate
):
    skill = _skill()
    module = _rules_module(skill)
    payload = json.loads(_response(module.module_id))
    mutate(payload)
    generator = _openrouter_generator(tmp_path, FakeClient([json.dumps(payload)]))

    generated = generator.generate(
        skill, module, EditOperator.REWRITE_RULE, _train_batch(), ()
    )

    assert generated.patch is None
    assert generated.error


def test_non_json_editor_response_is_an_auditable_invalid_patch(tmp_path: Path):
    skill = _skill()
    module = _rules_module(skill)
    generator = _openrouter_generator(tmp_path, FakeClient(["not-json"]))

    generated = generator.generate(
        skill, module, EditOperator.REWRITE_RULE, _train_batch(), ()
    )

    assert generated.patch is None
    assert generated.cache_hit is False
    assert "strict JSON" in generated.error


def test_editor_prompt_uses_train_visible_evidence_only_and_request_is_cached(
    tmp_path: Path,
):
    skill = _skill()
    module = _rules_module(skill)
    client = FakeClient([_response(module.module_id)])
    generator = _openrouter_generator(tmp_path, client)

    first = generator.generate(
        skill, module, EditOperator.REWRITE_RULE, _train_batch(), ()
    )
    second = generator.generate(
        skill, module, EditOperator.REWRITE_RULE, _train_batch(), ()
    )

    prompt = json.dumps(client.calls[0], sort_keys=True)
    assert "VISIBLE_FEEDBACK" in prompt
    assert "VISIBLE_FINAL_ANSWER" in prompt
    assert "VISIBLE_EVALUATOR_OUTPUT" in prompt
    assert "VISIBLE_TOOL_LOG" in prompt
    assert "teacher" not in prompt.casefold()
    assert "reference" not in prompt.casefold()
    assert "lambda" not in prompt.casefold()
    assert "projection" not in prompt.casefold()
    assert client.calls[0]["call_type"] == "editor"
    assert len(client.calls) == 1
    assert first.patch == second.patch
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.request_hash == second.request_hash


@pytest.mark.parametrize("split", [Split.VALIDATION, Split.TEST])
def test_editor_rejects_validation_and_test_batches_before_any_request(
    tmp_path: Path, split: Split
):
    skill = _skill()
    module = _rules_module(skill)
    client = FakeClient([])
    generator = _openrouter_generator(tmp_path, client)

    with pytest.raises(ValueError, match="train"):
        generator.generate(
            skill, module, EditOperator.REWRITE_RULE, _train_batch(split=split), ()
        )

    assert client.calls == []


def test_mock_generator_has_the_same_train_only_boundary():
    skill = _skill()
    module = _rules_module(skill)
    patch = _patch(
        module.module_id, EditOperator.REWRITE_RULE, "- Rule alpha.", "- Better rule."
    )
    generator = MockPatchGenerator((patch,))

    assert (
        generator.generate(skill, module, patch.operator, _train_batch(), ()).patch
        == patch
    )
    with pytest.raises(ValueError, match="train"):
        generator.generate(
            skill, module, patch.operator, _train_batch(split=Split.TEST), ()
        )

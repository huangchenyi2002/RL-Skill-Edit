from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rl_skill_edit.budget import BudgetExceeded, BudgetLedger
from rl_skill_edit.cache import JsonFileCache


LIMITS = {
    "student_rollouts": 6,
    "editor_calls": 2,
    "evaluator_calls": 4,
    "input_tokens": 100,
    "output_tokens": 50,
    "wall_time_seconds": 20.0,
}


def test_budget_accepts_only_rl_resources():
    ledger = BudgetLedger(
        {
            "student_rollouts": 4,
            "editor_calls": 1,
            "evaluator_calls": 2,
            "input_tokens": 100,
            "output_tokens": 100,
            "wall_time_seconds": 10.0,
        }
    )

    snapshot = ledger.snapshot().to_dict()

    assert set(snapshot) == {
        "student_rollouts",
        "editor_calls",
        "evaluator_calls",
        "input_tokens",
        "output_tokens",
        "wall_time_seconds",
        "cache_hits",
        "cached_student_rollouts",
        "cached_editor_calls",
        "cached_evaluator_calls",
    }


def test_budget_rejects_non_rl_limits():
    with pytest.raises(ValueError, match="unknown budget limits"):
        BudgetLedger(
            {
                **LIMITS,
                "teacher_rollouts": 0,
                "reference_rollouts": 0,
            }
        )


def test_evaluation_bundle_reservation_fails_atomically_before_partial_work():
    ledger = BudgetLedger(LIMITS)
    before = ledger.snapshot()

    with pytest.raises(BudgetExceeded):
        ledger.reserve_evaluation(task_count=4, repetitions=2, cache_hit=False)

    assert ledger.snapshot() == before


def test_budget_counts_logical_work_and_marks_cache_reuse_separately():
    ledger = BudgetLedger(LIMITS)
    evaluated = ledger.reserve_evaluation(task_count=2, repetitions=2, cache_hit=False)
    ledger.record_evaluation(
        evaluated, input_tokens=20, output_tokens=8, elapsed_seconds=1.5
    )
    cached = ledger.reserve_evaluation(task_count=1, repetitions=1, cache_hit=True)
    ledger.record_evaluation(
        cached, input_tokens=0, output_tokens=0, elapsed_seconds=0.0
    )
    editor = ledger.reserve_editor(cache_hit=False)
    ledger.record_editor(editor, input_tokens=9, output_tokens=7, elapsed_seconds=0.3)
    cached_editor = ledger.reserve_editor(cache_hit=True)
    ledger.record_editor(
        cached_editor, input_tokens=0, output_tokens=0, elapsed_seconds=0.0
    )

    snapshot = ledger.snapshot()
    assert snapshot.student_rollouts == 5
    assert snapshot.editor_calls == 2
    assert snapshot.evaluator_calls == 2
    assert snapshot.input_tokens == 29
    assert snapshot.output_tokens == 15
    assert snapshot.wall_time_seconds == pytest.approx(1.8)
    assert snapshot.cache_hits == 2
    assert snapshot.cached_student_rollouts == 1
    assert snapshot.cached_editor_calls == 1
    assert snapshot.cached_evaluator_calls == 1


def test_recording_actual_usage_over_reserved_limit_fails_closed():
    ledger = BudgetLedger(LIMITS)
    reservation = ledger.reserve_editor(cache_hit=False)
    before = ledger.snapshot()
    with pytest.raises(BudgetExceeded):
        ledger.record_editor(
            reservation, input_tokens=101, output_tokens=1, elapsed_seconds=1.0
        )
    assert ledger.snapshot() == before


def test_json_cache_persists_namespaces_and_concurrent_atomic_writes(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache = JsonFileCache(path)
    cache.set("rollout", "same", {"value": 1})
    cache.set("editor", "same", {"value": 2})

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda index: cache.set("rollout", f"k{index}", index), range(20)
            )
        )

    reopened = JsonFileCache(path)
    assert reopened.get("rollout", "same") == {"value": 1}
    assert reopened.get("editor", "same") == {"value": 2}
    assert reopened.get("missing", "key") is None
    assert [reopened.get("rollout", f"k{i}") for i in range(20)] == list(range(20))
    json.loads(path.read_text(encoding="utf-8"))


def test_failed_cache_serialization_does_not_corrupt_existing_file(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache = JsonFileCache(path)
    cache.set("safe", "key", {"kept": True})
    original = path.read_bytes()

    with pytest.raises(TypeError):
        cache.set("safe", "bad", object())

    assert path.read_bytes() == original
    assert JsonFileCache(path).get("safe", "key") == {"kept": True}

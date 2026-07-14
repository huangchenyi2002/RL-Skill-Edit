# Standalone RL-Skill-Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the published repository into a self-contained implementation of RL-Skill-Edit with no original OSD, current-method, or random-search method code.

**Architecture:** Promote the RL package to `rl_skill_edit/`, replace its old `src` dependencies with a small forced-skill OpenRouter/Spreadsheet runtime, and expose one fixed `initial skill -> RL optimization -> frozen paired test` CLI. Initial Skill remains input and reporting baseline only; the only optimizer is RL-Skill-Edit.

**Tech Stack:** Python 3.12+, NumPy, OpenAI-compatible OpenRouter client, httpx, PyYAML, openpyxl, pandas, pytest, Ruff.

## Global Constraints

- Work directly on the existing `kaggle_data` branch; do not create a branch or worktree.
- Do not modify the sibling `../ooosd` checkout.
- The repository contains exactly one optimization method: RL-Skill-Edit.
- `initial_skill` is required input and a paired reporting baseline, not another optimizer.
- Delete original OSD, current-method, random-search, Teacher, Reference, Parser, label-space, lambda, and Study 1 runtime paths.
- Real evaluation supports one forced active Skill only; no implicit activation and no no-skill endpoint.
- Missing data, incomplete API bundles, invalid code, and invalid Editor patches fail closed; do not add repair or heuristic fallbacks.
- Test remains inaccessible until the Validation-selected RL Skill and provenance are frozen.
- Do not track `.env`, `.venv`, results, caches, private workbooks, or absolute local paths.

---

### Task 1: Promote the RL Core to the Repository Package

**Files:**
- Create: `tests/test_repository_boundary.py`
- Move: `baselines/rl_skill_edit/*.py` to `rl_skill_edit/*.py`
- Modify: all retained tests importing `baselines.rl_skill_edit`
- Delete: `baselines/__init__.py`
- Delete after move: `baselines/`

**Interfaces:**
- Consumes: existing RL classes and functions without behavior changes.
- Produces: importable package `rl_skill_edit`; `SkillArtifact`, `Split`, `RLSkillEditOptimizer`, `ActorCriticPolicy`, and patch/reward APIs retain their current signatures.

- [ ] **Step 1: Write the package-boundary test**

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rl_is_the_top_level_package():
    assert importlib.util.find_spec("rl_skill_edit") is not None
    assert not (ROOT / "baselines").exists()
```

- [ ] **Step 2: Run the boundary test and verify the old namespace fails it**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_repository_boundary.py -v`

Expected: FAIL by assertion because `rl_skill_edit` is absent and `baselines/` exists; test collection succeeds.

- [ ] **Step 3: Move the package and rewrite retained imports mechanically**

Run:

```bash
git mv baselines/rl_skill_edit rl_skill_edit
git rm baselines/__init__.py
```

Change imports from:

```python
from baselines.rl_skill_edit.types import SkillArtifact
```

to:

```python
from rl_skill_edit.types import SkillArtifact
```

Do not change algorithm logic in this task.

- [ ] **Step 4: Run core tests**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_repository_boundary.py tests/test_rl_skill_edit_core.py tests/test_rl_skill_edit_policy.py tests/test_rl_skill_edit_patch.py tests/test_rl_skill_edit_optimizer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A baselines rl_skill_edit tests
git commit -m "refactor: promote RL package"
```

### Task 2: Add the Standalone OpenRouter Client

**Files:**
- Create: `rl_skill_edit/adapters/__init__.py`
- Create: `rl_skill_edit/adapters/openrouter.py`
- Modify: `rl_skill_edit/patch_generator.py`
- Modify: `tests/test_client_usage_accounting.py`

**Interfaces:**
- Consumes: config mappings `openrouter`, `cost_tracking`, and role-specific model settings.
- Produces: `OpenRouterClient.chat(model, messages, system=None, temperature=0.0, max_tokens=2048, call_type="unknown", seed=None) -> tuple[str, dict[str, Any]]` and thread-safe usage properties.

- [ ] **Step 1: Point the accounting test at the future adapter**

```python
import importlib.util


def test_openrouter_adapter_module_exists():
    assert importlib.util.find_spec(
        "rl_skill_edit.adapters.openrouter"
    ) is not None
```

- [ ] **Step 2: Run the test and verify the adapter is missing**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_client_usage_accounting.py -v`

Expected: FAIL by assertion because the adapter module is absent; test collection succeeds.

- [ ] **Step 3: Implement the minimal client**

The new class must expose this request behavior:

```python
request = {
    "model": model,
    "messages": full_messages,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "extra_headers": self.extra_headers,
}
if seed is not None:
    request["seed"] = int(seed)
response = self.client.chat.completions.create(**request)
```

It must read only `OPENROUTER_API_KEY`, optional proxy variables, and RL config; record input/output tokens and cost under a lock; make one provider request; and return an explicit `ok=False` usage object on provider failure. Set the default app title to `RL-Skill-Edit`. Do not copy Teacher-specific comments, retry loops, or Study configuration.

Then point the existing accounting test at the adapter:

```python
from rl_skill_edit.adapters.openrouter import OpenRouterClient
```

- [ ] **Step 4: Type the Editor against the shared chat surface**

Add a small protocol in `patch_generator.py`:

```python
class ChatClient(Protocol):
    def chat(self, **kwargs: Any) -> tuple[str, dict[str, Any]]: ...
```

Use `ChatClient` in `OpenRouterPatchGenerator.__init__`; do not import the old client.

- [ ] **Step 5: Run adapter and patch tests**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_client_usage_accounting.py tests/test_rl_skill_edit_patch.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_skill_edit/adapters rl_skill_edit/patch_generator.py tests/test_client_usage_accounting.py
git commit -m "refactor: add standalone OpenRouter client"
```

### Task 3: Replace the OSD Student/Evaluator with a Forced-Skill Runtime

**Files:**
- Create: `rl_skill_edit/adapters/spreadsheet.py`
- Modify: `rl_skill_edit/evaluation.py`
- Modify: `rl_skill_edit/types.py`
- Rewrite: `tests/test_blind_final_evaluation.py`
- Add tests in: `tests/test_blind_final_evaluation.py`

**Interfaces:**
- Consumes: `SkillArtifact`, a SpreadsheetBench task mapping, `OpenRouterClient`, and frozen Student configuration.
- Produces: `StudentTrajectory`, `SpreadsheetExecutor`, `SpreadsheetStudent.run_task(task, skill, blind, seed)`, and `SpreadsheetSkillEvaluator.evaluate(...) -> EvaluationBatch`.

- [ ] **Step 1: Write a failing adapter-boundary test**

```python
import importlib.util


def test_spreadsheet_adapter_module_exists():
    assert importlib.util.find_spec(
        "rl_skill_edit.adapters.spreadsheet"
    ) is not None
```

- [ ] **Step 2: Run the boundary test and verify a clean assertion failure**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_blind_final_evaluation.py::test_spreadsheet_adapter_module_exists -v`

Expected: FAIL by assertion because the adapter module is absent; test collection succeeds.

- [ ] **Step 3: Create only the runtime API skeleton**

Create the records below and `SpreadsheetExecutor`/`SpreadsheetStudent` method signatures. Their behavior methods must raise `NotImplementedError`; this is the minimum code needed to satisfy the module-boundary test and make the behavior test collect.

```python
@dataclass(frozen=True)
class ExecutionResult:
    score: float
    matched: int
    total: int
    error: str = ""


@dataclass(frozen=True)
class StudentTrajectory:
    task_id: str
    hard_reward: float
    soft_reward: float
    final_answer: str
    visible_logs: tuple[str, ...]
    total_tokens: int
    total_cost_usd: float
    evaluation_valid: bool = True
    invalid_reason: str = ""


class SpreadsheetStudent:
    def __init__(self, config: Mapping[str, Any], client: Any) -> None:
        self.executor = SpreadsheetExecutor()

    def run_task(
        self, task: Mapping[str, Any], skill: SkillArtifact, *, blind: bool, seed: int
    ) -> StudentTrajectory:
        raise NotImplementedError
```

- [ ] **Step 4: Replace old blind tests with failing forced-skill behavior tests**

```python
from rl_skill_edit.adapters.spreadsheet import ExecutionResult, SpreadsheetStudent
from rl_skill_edit.types import SkillArtifact


def test_blind_test_hides_answer_metadata_and_does_not_retry_from_score(
    tmp_path, monkeypatch
):
    client = RecordingClient()
    student = SpreadsheetStudent(STUDENT_CONFIG, client)
    monkeypatch.setattr(
        student.executor,
        "execute_and_score",
        lambda **kwargs: ExecutionResult(0.25, 1, 4, ""),
    )
    skill = SkillArtifact("main", "main", "", "## Rule\nUpdate cells.")

    trajectory = student.run_task(_final_task(tmp_path), skill, blind=True, seed=11)

    assert trajectory.hard_reward == 0.25
    assert len(client.calls) == 1
    prompt = repr(client.calls[0])
    assert "SECRET_FINAL_SHEET" not in prompt
    assert "B17:C19" not in prompt
    assert "score=" not in prompt
    assert client.calls[0]["seed"] == 11


def test_student_has_no_implicit_or_no_skill_entrypoints():
    assert not hasattr(SpreadsheetStudent, "run_task_no_skill")
    assert not hasattr(SpreadsheetStudent, "run_task_with_model")
```

- [ ] **Step 5: Run the blind behavior test and verify the skeleton fails**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_blind_final_evaluation.py -v`

Expected: FAIL at `SpreadsheetStudent.run_task` with `NotImplementedError`.

- [ ] **Step 6: Implement focused runtime records and executor**

Use these records:

```python
@dataclass(frozen=True)
class ExecutionResult:
    score: float
    matched: int
    total: int
    error: str = ""


@dataclass(frozen=True)
class StudentTrajectory:
    task_id: str
    hard_reward: float
    soft_reward: float
    final_answer: str
    visible_logs: tuple[str, ...]
    total_tokens: int
    total_cost_usd: float
    evaluation_valid: bool = True
    invalid_reason: str = ""
```

`SpreadsheetExecutor.execute_and_score` must copy the input workbook to a temporary directory, run extracted Python with only `PATH`, `PYTHONIOENCODING`, and `PYTHONDONTWRITEBYTECODE`, enforce the existing eight-second timeout, and compare the configured golden sheet/range. Empty code, missing files/range, timeout, import failure, or nonzero subprocess exit returns an explicit invalid result. Do not auto-insert `wb.save(...)` and do not compute heuristic reward.

- [ ] **Step 7: Implement the forced-skill Student**

The only prompt path is:

```python
system = build_student_system(task, expose_answer_metadata=not blind)
system += f"\n\n[ACTIVE SKILL: {skill.name}]\n{skill.body}"
response, usage = self.client.chat(
    model=self.model,
    messages=[{"role": "user", "content": task["description"]}],
    system=system,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
    call_type="student_rollout",
    seed=seed,
)
```

For `blind=True`, make exactly one model call and never return score/verifier feedback to the model. For Train/Validation, bounded retries may use only execution error text, not golden answer values. An empty response or missing executable code is `evaluation_valid=False`.

- [ ] **Step 8: Localize score selection and implement ordered evaluation**

Remove `from src.evaluator import select_gate_score` and define:

```python
def select_score(hard: float, soft: float, metric: str, mixed_weight: float) -> float:
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        weight = min(1.0, max(0.0, float(mixed_weight)))
        return (1.0 - weight) * float(hard) + weight * float(soft)
    raise ValueError(f"unknown metric: {metric}")
```

`SpreadsheetSkillEvaluator` must preserve task order, derive each rollout seed as `seed + task_index * repetitions + repetition`, reject an incomplete bundle, enforce freeze-before-Test, and use the existing cache identity.

- [ ] **Step 9: Remove SkillLibrary conversion**

Delete `SkillArtifact.to_library()` from `types.py`. The real evaluator receives `SkillArtifact` directly.

- [ ] **Step 10: Run runtime, isolation, and optimizer tests**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_blind_final_evaluation.py tests/test_data_split_isolation.py tests/test_rl_skill_edit_optimizer.py -v`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add rl_skill_edit/evaluation.py rl_skill_edit/types.py rl_skill_edit/adapters/spreadsheet.py tests/test_blind_final_evaluation.py
git commit -m "feat: add standalone spreadsheet runtime"
```

### Task 4: Remove Non-RL Budget and Reporting Paths

**Files:**
- Modify: `rl_skill_edit/budget.py`
- Modify: `rl_skill_edit/optimizer.py`
- Move and rewrite: `rl_skill_edit/comparison.py` to `rl_skill_edit/reporting.py`
- Delete: `rl_skill_edit/random_policy.py`
- Rewrite: `tests/test_budget_accounting.py`
- Rewrite and rename: `tests/test_skill_optimization_comparison.py` to `tests/test_reporting.py`
- Modify: `tests/test_rl_skill_edit_optimizer.py`

**Interfaces:**
- Consumes: initial and frozen RL `SkillArtifact`, one frozen evaluator, task bundle, and two usage mappings.
- Produces: `BudgetSnapshot` with RL-only counters and `run_frozen_report(initial_skill, rl_skill, evaluator, test_tasks, output_dir, seed, repetitions, bootstrap_samples)`.

- [ ] **Step 1: Write failing RL-only budget and reporting-module assertions**

Create `tests/test_reporting.py` for the reporting assertion; keep the old comparison
test untouched until Step 3 so this first RED run still collects cleanly.

```python
import importlib.util


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
    assert "teacher_rollouts" not in snapshot
    assert "reference_rollouts" not in snapshot


def test_reporting_module_exists():
    assert importlib.util.find_spec("rl_skill_edit.reporting") is not None
```

- [ ] **Step 2: Run RED for the old budget and missing reporting module**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_budget_accounting.py tests/test_reporting.py -v`

Expected: FAIL by assertions because old budget fields exist and `reporting.py` is absent; test collection succeeds.

- [ ] **Step 3: Move comparison to reporting without changing behavior**

Run: `git mv rl_skill_edit/comparison.py rl_skill_edit/reporting.py`

Update imports that still reference `rl_skill_edit.comparison`. Do not add `run_frozen_report` yet.

- [ ] **Step 4: Add the failing two-artifact reporting assertion**

```python
import rl_skill_edit.reporting as reporting


class FrozenEvaluator:
    def __init__(self) -> None:
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def evaluate(self, skill, tasks, split, seed, repetitions, use_cache, blind):
        assert self.frozen
        assert blind is True
        assert use_cache is False
        base = 0.25 if skill.skill_id == "initial_skill" else 0.75
        return EvaluationBatch(
            split=split,
            results=tuple(
                TaskResult(task.task_id, base, base >= 0.5) for task in tasks
            ),
        )


def test_frozen_report_evaluates_only_initial_and_rl(tmp_path):
    assert callable(getattr(reporting, "run_frozen_report", None))
    result = reporting.run_frozen_report(
        initial_skill=_skill("initial_skill", "INITIAL"),
        rl_skill=_skill("rl_skill_edit", "RL"),
        evaluator=FrozenEvaluator(),
        test_tasks=tuple(FakeTask(f"test-{i}") for i in range(4)),
        output_dir=tmp_path,
        seed=23,
        repetitions=2,
        bootstrap_samples=200,
        optimization_usage={},
        reporting_usage={},
    )
    assert [row.method for row in result.methods] == [
        "initial_skill",
        "rl_skill_edit",
    ]
```

- [ ] **Step 5: Run RED for the absent fixed reporting function**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_budget_accounting.py tests/test_reporting.py -v`

Expected: FAIL by assertion because `run_frozen_report` is absent and because the old ledger still exposes other method fields.

- [ ] **Step 6: Simplify the budget ledger**

Set structural limits to:

```python
_STRUCTURAL_LIMITS = ("student_rollouts", "editor_calls", "evaluator_calls")
_USAGE_LIMITS = ("input_tokens", "output_tokens", "wall_time_seconds")
```

Change `reserve_evaluation` to remove `role`; every reservation increments only `student_rollouts` and `evaluator_calls`. Keep `cache_hits`, `cached_student_rollouts`, `cached_editor_calls`, and `cached_evaluator_calls` because they describe the RL run itself.

Update `optimizer.py` call sites to use:

```python
reservation = budget.reserve_evaluation(
    task_count=len(tasks),
    repetitions=repetitions,
    cache_hit=cache_hit,
)
```

- [ ] **Step 7: Rewrite reporting for exactly two artifacts**

Retain paired bootstrap/statistics and CSV writers, but expose only:

```python
def run_frozen_report(
    *,
    initial_skill: SkillArtifact,
    rl_skill: SkillArtifact,
    evaluator: Any,
    test_tasks: tuple[Any, ...],
    output_dir: Path,
    seed: int,
    repetitions: int,
    bootstrap_samples: int,
    optimization_usage: Mapping[str, Any],
    reporting_usage: Mapping[str, Mapping[str, Any]],
) -> ComparisonResult:
    ...
```

Freeze first; evaluate initial then RL with `use_cache=False` and `blind=True`; write only two method rows and `2 * len(test_tasks)` task rows. Delete `ImportedMethod`, current archive parsing, generic method registry, and Teacher/Reference columns.

- [ ] **Step 8: Remove random search and update optimizer tests**

Delete `random_policy.py`. Replace assertions about zero Teacher/Reference budgets with assertions that the public snapshot keys are exactly the RL counters.

- [ ] **Step 9: Run budget/reporting/optimizer tests**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_budget_accounting.py tests/test_reporting.py tests/test_rl_skill_edit_optimizer.py -v`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add rl_skill_edit tests
git commit -m "refactor: keep only RL budget and reporting"
```

### Task 5: Replace the Multi-Method Runner with One RL CLI

**Files:**
- Create: `rl_skill_edit/cli.py`
- Create: `rl_skill_edit/__main__.py`
- Rewrite: `configs/rl_skill_edit.yaml`
- Rewrite: `configs/rl_skill_edit_smoke.yaml`
- Rewrite: `scripts/run_rl_skill_edit_smoke.sh` to `scripts/run_smoke.sh`
- Rewrite: `tests/test_rl_skill_edit_end_to_end.py`
- Delete: `experiments/run_skill_optimization_comparison.py`
- Delete after migration: `experiments/`

**Interfaces:**
- Consumes: a self-contained RL YAML config.
- Produces: `parse_args(argv=None)`, `run(config_path: Path, seed: int | None = None, test_only: bool = False) -> dict[str, Any]`, and `python -m rl_skill_edit`.

- [ ] **Step 1: Write a failing CLI module-boundary test**

```python
import importlib.util


def test_cli_module_exists():
    assert importlib.util.find_spec("rl_skill_edit.cli") is not None
```

- [ ] **Step 2: Run the boundary test and verify a clean assertion failure**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_rl_skill_edit_end_to_end.py::test_cli_module_exists -v`

Expected: FAIL by assertion because the CLI module is absent; test collection succeeds.

- [ ] **Step 3: Create only the CLI API skeleton**

Create `parse_args` with the final three flags, create `__main__.py`, and define:

```python
def run(
    config_path: Path,
    seed: int | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    raise NotImplementedError
```

- [ ] **Step 4: Rewrite the end-to-end test for the fixed workflow**

```python
from rl_skill_edit.cli import run


def _smoke_config_in(tmp_path: Path) -> Path:
    source = yaml.safe_load(
        (ROOT / "configs/rl_skill_edit_smoke.yaml").read_text(encoding="utf-8")
    )
    output = tmp_path / "result"
    source["paths"]["output_dir"] = str(output)
    source["paths"]["rl_skill"] = str(output / "rl_skill_edit/best_rl_skill.md")
    source["paths"]["rl_summary"] = str(
        output / "rl_skill_edit/rl_optimization_summary.json"
    )
    source["paths"]["rl_provenance"] = str(
        output / "rl_skill_edit/freeze_provenance.json"
    )
    target = tmp_path / "smoke.yaml"
    target.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return target


def test_api_free_training_and_test_only_report_initial_and_rl(tmp_path):
    config_path = _smoke_config_in(tmp_path)
    trained = run(config_path, seed=42, test_only=False)
    frozen = run(config_path, seed=42, test_only=True)

    assert trained["methods"] == ["initial_skill", "rl_skill_edit"]
    assert frozen["methods"] == ["initial_skill", "rl_skill_edit"]
    assert frozen["test_rewards"] == trained["test_rewards"]
    manifest = json.loads(
        (Path(trained["output_dir"]) / "experiment_manifest.json").read_text()
    )
    assert manifest["method"] == "rl_skill_edit"
    assert "current_method_artifact_provenance" not in manifest
```

- [ ] **Step 5: Run the end-to-end test and verify the skeleton fails**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_rl_skill_edit_end_to_end.py -v`

Expected: FAIL at `run` with `NotImplementedError`.

- [ ] **Step 6: Implement the fixed CLI**

Parser surface:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL-Skill-Edit.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args(argv)
```

`run` must always load initial Skill, optimize or provenance-load one RL Skill, load Test only after freeze, compute common Train/Validation reporting, and call `run_frozen_report`. Remove `SUPPORTED_METHODS`, `--methods`, current import, random branch, and method loop.

Implementation hash:

```python
files = sorted((REPOSITORY_ROOT / "rl_skill_edit").rglob("*.py"))
return sha256_files(files)
```

Dependency hash must read `requirements.txt`.

- [ ] **Step 7: Make both configs self-contained**

Required top-level config mappings:

```yaml
runtime: spreadsheet  # smoke uses mock
seed: 42
openrouter:
  base_url: https://openrouter.ai/api/v1
student:
  model: anthropic/claude-3-haiku
  temperature: 0.0
  max_tokens: 4096
  max_steps: 3
editor:
  model: anthropic/claude-opus-4.5
  temperature: 0.0
  max_tokens: 4096
```

Keep RL policy/reward/patch/evaluation/budget/split/path mappings. Remove `methods`, `repository_config`, every `current_*` path, and Teacher/Reference limits.

- [ ] **Step 8: Replace the smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" -m rl_skill_edit \
  --config "$ROOT/configs/rl_skill_edit_smoke.yaml" \
  --seed 42
```

- [ ] **Step 9: Run CLI tests and smoke**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_rl_skill_edit_end_to_end.py tests/test_data_split_isolation.py -v`

Expected: PASS.

Run: `bash scripts/run_smoke.sh`

Expected: JSON contains exactly `"methods": ["initial_skill", "rl_skill_edit"]` and RL Test reward is greater than initial Test reward.

- [ ] **Step 10: Commit**

```bash
git add -A experiments rl_skill_edit configs scripts tests
git commit -m "refactor: expose one RL workflow"
```

### Task 6: Delete Original-Method Artifacts and Rewrite Public Documentation

**Files:**
- Delete: `src/`
- Delete: `study1_main.py`
- Delete: `config.yaml`
- Delete: `data/spreadsheet/`
- Delete: `data/mock_rl_skill_edit/current_skill.md`
- Delete: `data/mock_rl_skill_edit/current_history.json`
- Delete: `data/mock_rl_skill_edit/current_events.jsonl`
- Delete: `docs/superpowers/plans/2026-07-15-rl-skill-edit.md`
- Delete: `docs/superpowers/specs/2026-07-15-rl-skill-edit-design.md`
- Rewrite: `README.md`
- Rewrite: `ARCHITECTURE.md`
- Rewrite: `CONTEXT.md`
- Rewrite: `docs/rl_skill_edit_implementation_note.md`
- Rename: `requirements-rl.txt` to `requirements.txt`
- Modify: `tests/test_repository_boundary.py`
- Delete or rewrite: tests that import `src.*` or assert current/random behavior

**Interfaces:**
- Consumes: the standalone package and CLI from Tasks 1-5.
- Produces: a repository tree and documentation describing only RL-Skill-Edit.

- [ ] **Step 1: Strengthen the failing repository-boundary test**

```python
FORBIDDEN_PATHS = (
    "src",
    "baselines",
    "experiments",
    "study1_main.py",
    "config.yaml",
    "data/spreadsheet",
    "rl_skill_edit/random_policy.py",
)


def test_original_method_paths_are_absent():
    assert [path for path in FORBIDDEN_PATHS if (ROOT / path).exists()] == []


def test_cli_has_no_method_selector():
    help_text = subprocess.run(
        [sys.executable, "-m", "rl_skill_edit", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--methods" not in help_text
```

- [ ] **Step 2: Run the boundary test and confirm original files still fail it**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_repository_boundary.py -v`

Expected: FAIL listing the original-method paths.

- [ ] **Step 3: Delete the explicitly approved original-method files**

Run `git rm` only for the paths listed in this task. Do not delete anything from the sibling `ooosd` checkout.

- [ ] **Step 4: Rewrite documentation with one method and one command**

README must begin with:

```markdown
# RL-Skill-Edit

RL-Skill-Edit optimizes one external Markdown Skill while the agent model remains frozen.
The actor-critic selects a Skill module and edit operator; an Editor proposes one local
patch; Train reward updates the policy; Validation selects the checkpoint; Test is read
only after the final Skill is frozen.
```

Document only `python -m rl_skill_edit`, `--test-only`, the two configs, output artifacts, API-free smoke, tests, required private manifests/workbooks, and the generated-code isolation warning. Architecture/implementation note must show only Student, Editor, RL policy, Train, Validation, and frozen Test.

CONTEXT must state that the repository contains one optimizer and record the removal decision. Remove every old current/random/OSD run command and directory reference.

- [ ] **Step 5: Rename and trim the dependency lock**

Run: `git mv requirements-rl.txt requirements.txt`

Keep only packages imported by the standalone runtime/tests. Confirm `requirements.txt` contains no large RL framework because the policy remains NumPy.

- [ ] **Step 6: Run boundary and full tests**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests`

Expected: all retained tests PASS and collection imports no `src` or `baselines` modules.

Run: `.venv/bin/ruff check rl_skill_edit tests`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: make repository RL-only"
```

### Task 7: Final Verification, Review, and GitHub Publication

**Files:**
- Modify if needed: `CONTEXT.md`
- Modify after project completion: workspace `../CONTEXT.md`
- No new implementation files.

**Interfaces:**
- Consumes: the completed standalone repository.
- Produces: verified commits on remote default branch `kaggle_data`.

- [ ] **Step 1: Run the complete verification from current HEAD**

Run: `.venv/bin/python -B -m pytest -p no:cacheprovider tests`

Expected: zero failures.

Run: `.venv/bin/ruff check rl_skill_edit tests`

Expected: `All checks passed!`

Run: `.venv/bin/python -B -m compileall -q rl_skill_edit tests`

Expected: exit code 0 and no output.

Run: `bash scripts/run_smoke.sh`

Expected: methods are exactly initial and RL; RL reward is greater than initial reward.

Run: `.venv/bin/python -m rl_skill_edit --config configs/rl_skill_edit_smoke.yaml --seed 42 --test-only`

Expected: the same frozen Test rewards as the preceding smoke run.

- [ ] **Step 2: Verify the method boundary and publication safety**

Run: `git ls-files`

Expected: no `src/`, `baselines/`, `experiments/`, original `config.yaml`, current artifacts, private workbooks, results, caches, or `.env`.

Run a scoped repository search over `rl_skill_edit/`, `configs/`, `scripts/`, `tests/`, and public docs. Any mention of current method, random search, Study 1, or original OSD runtime must either be in this archived design/plan explaining deletion or be removed.

Run a staged secret/path scan for API keys, private-key markers, absolute home-directory prefixes, `.xlsx`, `.pyc`, and result files. Expected: no staged secret, private data, or absolute local path.

- [ ] **Step 3: Perform independent code and scope review**

Review the final diff for runtime correctness and verify the user requirement against the complete tracked file tree. Fix every blocking finding, then rerun Step 1.

- [ ] **Step 4: Update project records**

Update repository `CONTEXT.md` and workspace `CONTEXT.md` with the final commit, verification evidence, exact remote branch, and the decision that initial Skill is input/baseline rather than another method.

- [ ] **Step 5: Commit any final record-only changes**

```bash
git add CONTEXT.md
git commit -m "docs: record standalone RL release"
```

Skip this commit only when Step 4 produced no repository change.

- [ ] **Step 6: Push and verify the remote commit**

Run: `git push origin kaggle_data`

Expected: fast-forward push succeeds.

Read the remote commit back through GitHub and confirm its SHA, branch, message, and tracked tree before reporting completion.

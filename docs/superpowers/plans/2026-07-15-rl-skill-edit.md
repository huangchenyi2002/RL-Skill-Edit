# RL-Skill-Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Complete on 2026-07-15. The detailed boxes below are retained as the original red/green execution record; final verification was 117 tests passed, Ruff/format/compileall passed, and smoke plus test-only replay matched.

**Goal:** Implement an independent reward-only RL baseline that optimizes external skill text, plus a strict unified comparison runner and paper-ready task-level reports.

**Architecture:** Keep `study1_main.py` as the current-method implementation and add an adapter-based RL package beside it. RL receives only train/validation objects, calls the frozen Student and a direct Editor adapter, and exposes a frozen skill to a separate comparison stage that alone may load test data.

**Tech Stack:** Python 3.10+, NumPy, PyYAML, existing OSD Student/OpenRouter/SkillLibrary code, pytest, CSV/JSONL artifacts.

## Global Constraints

- Do not create a branch or worktree; preserve the existing modification to `experiment_plan_20260709.md`.
- RL-Skill-Edit must never instantiate or call Teacher, Expert endpoint, Reference, Parser, lambda projection, or current-method target objects.
- Editor output must be one strict JSON object and one local edit; malformed or mismatched output is rejected without a fallback patch.
- Train drives policy gradients; validation may select checkpoints; test is unavailable until all method skills are frozen.
- All compared methods use the same ordered test tasks, blind Student protocol, model settings, repetitions, and seed.
- Budget exhaustion, split overlap, duplicate tasks, missing artifacts, and incomplete evaluation bundles fail closed.
- Use the workspace `.venv` interpreter at `../.venv/bin/python` for project commands.

---

### Task 1 (complete): Strict manifests and blind Final evaluation

**Files:**
- Create: `baselines/__init__.py`
- Create: `baselines/rl_skill_edit/__init__.py`
- Create: `baselines/rl_skill_edit/manifest.py`
- Modify: `src/agent.py`
- Modify: `src/evaluator.py`
- Modify: `study1_main.py`
- Test: `tests/test_data_split_isolation.py`
- Test: `tests/test_blind_final_evaluation.py`

**Interfaces:**
- `TaskManifest.load(path: Path, split: Split, expected_size: int) -> TaskManifest`
- `validate_manifests(train, validation, test) -> None`
- `StudentAgent.run_task(..., verifier_feedback: bool = True, expose_answer_metadata: bool = True)`
- `Evaluator.witness_estimate(..., verifier_feedback: bool = True, expose_answer_metadata: bool = True)`

- [ ] Write tests proving duplicate IDs, cross-split IDs, canonical task aliases, empty IDs, wrong sizes, and missing spreadsheet files fail.
- [ ] Run the focused tests and confirm they fail because the interfaces do not exist.
- [ ] Implement ordered manifest loading, canonical fingerprints, SHA-256 digests, and strict three-way validation.
- [ ] Write tests proving blind mode hides target sheet/range and stops after the first scored attempt without golden feedback.
- [ ] Run those tests and confirm the current agent leaks metadata/feedback.
- [ ] Add explicit blind flags through Student and Evaluator; use them for `study1_main.py` Final Holdout only.
- [ ] Replace Final overlap filtering with a hard failure.
- [ ] Run both focused test files and the existing tests.

### Task 2 (complete): Skill state, Markdown modules, actions, diagnostics, and reward

**Files:**
- Create: `baselines/rl_skill_edit/types.py`
- Create: `baselines/rl_skill_edit/modules.py`
- Create: `baselines/rl_skill_edit/action_space.py`
- Create: `baselines/rl_skill_edit/state_encoder.py`
- Create: `baselines/rl_skill_edit/reward.py`
- Test: `tests/test_rl_skill_edit_core.py`

**Interfaces:**
- `SkillArtifact.from_file()/to_library()/save()/digest`
- `parse_modules(body: str, max_modules: int) -> tuple[SkillModule, ...]`
- `attribute_failures(modules, EvaluationBatch) -> dict[str, ModuleDiagnostics]`
- `ActionSpace.encode/decode/mask`
- `StateEncoder.encode(...) -> np.ndarray`
- `compute_incremental_reward(before, after, initial_text, current_text, candidate_text, config, invalid) -> RewardBreakdown`

- [ ] Add parser tests for frontmatter-free text, nested headings, duplicate titles, stable IDs, non-overlap, `global`, and max-module hard failure.
- [ ] Add attribution tests using only visible feedback/final answers and `global` for no reliable match.
- [ ] Add mask tests for every operator, absent padded slots, and a single global STOP action.
- [ ] Add fixed-dimension state tests covering every required state field.
- [ ] Add paired-delta and exact token-Levenshtein penalty tests.
- [ ] Run the focused test and observe the missing-module failure.
- [ ] Implement the minimal pure functions and dataclasses.
- [ ] Re-run the focused test until green.

### Task 3 (complete): Lightweight actor-critic, checkpoints, and budget ledger

**Files:**
- Create: `baselines/rl_skill_edit/policy.py`
- Create: `baselines/rl_skill_edit/budget.py`
- Create: `baselines/rl_skill_edit/cache.py`
- Test: `tests/test_rl_skill_edit_policy.py`
- Test: `tests/test_budget_accounting.py`

**Interfaces:**
- `ActorCriticPolicy.select(state, mask, deterministic) -> PolicyDecision`
- `ActorCriticPolicy.update(transitions) -> dict[str, float]`
- `ActorCriticPolicy.save(path)` / `ActorCriticPolicy.load(path)`
- `BudgetLedger.reserve_evaluation()/record_evaluation()/reserve_editor()/record_editor()/snapshot()`
- `JsonFileCache.get/set(namespace, key)`

- [ ] Add forward/mask/deterministic-seed tests and confirm missing implementation failure.
- [ ] Add an update test where the selected action probability changes in the advantage direction.
- [ ] Add checkpoint round-trip tests including RNG state and deterministic output.
- [ ] Implement one tanh hidden layer, actor/value heads, discounted returns, optional advantage normalization, entropy term, value loss, and global gradient clipping.
- [ ] Add budget tests proving a whole evaluation bundle is reserved before work, cache hits consume the same logical search budget but no new API usage, counters are role-separated, and limits fail before partial execution.
- [ ] Implement the thread-safe ledger and atomic JSON cache.
- [ ] Run both focused tests.

### Task 4 (complete): Structured Editor patch generation and local application

**Files:**
- Create: `baselines/rl_skill_edit/patch_generator.py`
- Create: `baselines/rl_skill_edit/patch_validator.py`
- Test: `tests/test_rl_skill_edit_patch.py`

**Interfaces:**
- `PatchGenerator.generate(skill, module, operator, train_batch, edit_history) -> GeneratedPatch`
- `validate_and_apply_patch(skill, modules, selected_action, patch, limits) -> PatchApplication`
- `OpenRouterPatchGenerator` and `MockPatchGenerator`

- [ ] Add schema tests for exact required fields and enum values.
- [ ] Add target/operator mismatch, ambiguous old text, protected region, no-op, oversized change, and whole-skill rewrite rejection tests.
- [ ] Add one valid test for every non-STOP operator.
- [ ] Add a prompt test proving validation/test task results and teacher/reference/lambda fields cannot enter the Editor context.
- [ ] Run the test and observe missing implementation failure.
- [ ] Implement strict `json.loads`, request hashing/caching, direct `editor` profile calls, semantic operator validation, and exact local application.
- [ ] Run the focused test.

### Task 5 (complete): Evaluator adapters and RL optimizer

**Files:**
- Create: `baselines/rl_skill_edit/evaluation.py`
- Create: `baselines/rl_skill_edit/optimizer.py`
- Test: `tests/test_rl_skill_edit_optimizer.py`

**Interfaces:**
- `RepositorySkillEvaluator.evaluate(skill, tasks, split, seed, repetitions, use_cache, blind) -> EvaluationBatch`
- `MockSkillEvaluator.evaluate(...) -> EvaluationBatch`
- `RLSkillEditOptimizer.optimize(initial_skill, train_tasks, validation_tasks, budget, seed) -> OptimizationResult`

- [ ] Add a guard test proving optimizer construction/calls have no test parameter and test evaluation before freeze raises.
- [ ] Add a paired-mini-batch test proving current/candidate use identical ordered task IDs.
- [ ] Add invalid-patch transition, STOP, horizon, per-episode reset, validation-best checkpoint, and no-teacher/reference counter tests.
- [ ] Add output tests for `best_rl_skill.md`, `final_rl_policy.pt`, `rl_training_log.jsonl`, and `rl_episode_summary.csv`, including every required step field and task-level scores.
- [ ] Run the focused test and observe missing implementation failure.
- [ ] Implement cached repository/mock evaluators and the finite-horizon optimizer.
- [ ] Run the focused test.

### Task 6 (complete): Unified comparison, statistics, and imported current method

**Files:**
- Create: `baselines/rl_skill_edit/comparison.py`
- Create: `experiments/run_skill_optimization_comparison.py`
- Test: `tests/test_skill_optimization_comparison.py`

**Interfaces:**
- `run_comparison(config, methods, seed) -> ComparisonResult`
- `paired_bootstrap_ci(initial, candidate, samples, seed)`
- `load_current_method_artifact(skill_path, history_path, jsonl_path) -> MethodResult`

- [ ] Add paired statistics tests for mean, standard error, confidence interval, success rate, and win/tie/loss.
- [ ] Add a current-history import test for Student/Teacher/Reference rollouts, Editor calls, tokens, cost, and wall time.
- [ ] Add a protocol test proving all final methods use the same ordered test IDs/settings and each is evaluated exactly once after freeze; initial must be one of those fresh test calls.
- [ ] Add CSV tests for all requested method-level columns and raw task-level rows.
- [ ] Run the test and observe missing implementation failure.
- [ ] Implement initial/current/RL/random adapters, frozen skill hashes, formal fresh test evaluation, CSV/JSON summaries, and CLI argument validation.
- [ ] Run the focused test.

### Task 7 (complete): Real and mock configurations, smoke data, and run script

**Files:**
- Create: `configs/rl_skill_edit.yaml`
- Create: `configs/rl_skill_edit_smoke.yaml`
- Create: `data/mock_rl_skill_edit/SKILL.md`
- Create: `data/mock_rl_skill_edit/train.json`
- Create: `data/mock_rl_skill_edit/validation.json`
- Create: `data/mock_rl_skill_edit/test.json`
- Create: `scripts/run_rl_skill_edit_smoke.sh`
- Create: `requirements-rl.txt`

- [ ] Add strict configuration values for policy, reward, selection, evaluation, cache, artifacts, budget, models, split sizes, and current-method artifact provenance.
- [ ] Add deterministic mock tasks whose reward improves when the local Editor adds the missing rule.
- [ ] Add a smoke script that resolves the repository root and invokes the workspace virtual environment.
- [ ] Run the smoke experiment and confirm it produces every required output without an API key.

### Task 8 (complete): Documentation, review, and final verification

**Files:**
- Modify: `README.md`
- Create: `ARCHITECTURE.md`
- Modify: `../CONTEXT.md`

- [x] Document real training, all supported baselines, formal test evaluation, reproduction, artifacts, NumPy policy rationale, current-method artifact mode, and limitations.
- [x] Add the skills.sh/GitHub search record and explain why no external optimization framework was adopted.
- [x] Document file responsibilities and the train/validation/freeze/test call graph.
- [x] Review all changes for bugs, then redo the design from first principles and remove unnecessary coupling/fallbacks.
- [x] Run `../.venv/bin/python -m compileall` on the new/modified Python files.
- [x] Run the complete pytest suite with cache disabled.
- [x] Run the mock smoke script without deleting prior cached artifacts.
- [x] Dispatch an independent final code review, fix every important finding, and repeat focused/full verification.
- [x] Update the root `CONTEXT.md` with current state, stop point, decisions, commands, and results.

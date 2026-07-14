# RL-Skill-Edit design

## Goal

Add a paper-ready reinforcement-learning baseline that keeps every language-model parameter frozen and learns only which local external-skill edit to request. It must be independently runnable, auditable at task level, budgeted, cached, deterministic under mocks, and unable to use test outcomes during optimization.

## Considered approaches

1. **Independent adapter-based optimizer (selected).** Define strict skill/evaluator/editor protocols, run RL only on train rewards, use validation only for checkpoint selection, and reuse the repository Student/scorer through an adapter. This cleanly excludes teacher/reference/lambda state and makes leakage tests possible.
2. **Embed RL actions inside `run_study1()`.** This would reuse more orchestration code, but the state and candidate path are built around a teacher-reference `EpochTarget`; an RL baseline there could silently inherit the proposed method's signal. The 2,600-line runner also cannot accept a test-free optimizer boundary without a large refactor.
3. **Adopt a prompt-optimization/RL framework.** TextGrad, PromptWizard, and similar repositories provide useful comparisons, but they add substantial dependencies and optimize through textual feedback rather than the requested masked module/operator policy. A lightweight local implementation is easier to audit and fairer to the current code.

## Architecture

`baselines/rl_skill_edit/` will contain focused modules:

- `types.py`: immutable skill, task-result, action, patch, transition, and optimizer result records.
- `modules.py`: deterministic, non-overlapping Markdown section parser and visible-evidence attribution; unresolved failures map to `global`.
- `action_space.py`: fixed `max_modules × 8 operators` index, dynamic validity mask, and reversible action mapping.
- `state_encoder.py`: fixed numeric state with round, skill length, remaining rollout fraction, last action/reward, initial-skill edit distance, and padded per-module diagnostics.
- `policy.py`: one-hidden-layer NumPy actor-critic, masked categorical sampling, deterministic argmax, discounted returns, normalized advantages, entropy bonus, value loss, global gradient clipping, and loadable checkpoint.
- `patch_generator.py`: direct Editor-profile call with strict JSON, persistent request cache, and a prompt containing only current skill, selected train diagnostics, and edit history.
- `patch_validator.py`: exact target/operator checks plus one local application; invalid, whole-skill, protected-region, duplicate-target, and semantic-operator mismatches fail closed.
- `reward.py`: paired task-score delta and exact token-level Levenshtein penalties, with every component returned separately.
- `budget.py`: preflight structural limits and cumulative Student/Teacher/Reference/Editor/evaluator/token/time counters.
- `evaluation.py`: repository and mock evaluators, persistent rollout cache, strict split guard, and blind-test mode.
- `optimizer.py`: finite-horizon episodes, per-step logs/checkpoints, policy updates, validation selection, and no test argument.
- `comparison.py`: frozen-artifact adapters, paired bootstrap statistics, task-level CSVs, and imported current-method usage metadata.

`experiments/run_skill_optimization_comparison.py` is the only comparison CLI. `configs/rl_skill_edit.yaml` is the real configuration; a separate small mock configuration and script provide an API-free smoke run.

## MDP

At step `t`, the state is the encoded tuple `(current skill, train diagnostics, accepted edit counts, previous operator/reward, t, remaining budget)`. An action is one valid pair `(target module, operator)` where the operators are `ADD_RULE`, `REWRITE_RULE`, `DELETE_RULE`, `ADD_EXAMPLE`, `REWRITE_EXAMPLE`, `MERGE_REDUNDANT_RULES`, `REORDER_CONTENT`, and `STOP`.

For a non-STOP action, the Editor returns one JSON patch. Validation either produces exactly one locally edited next skill or rejects it and leaves the skill unchanged. For the same ordered train mini-batch, the reward is

`mean(candidate - incumbent) - beta_len*C_len - beta_edit*C_edit - beta_invalid*C_invalid`.

Each episode restarts from the identical initial skill and stops at `H` edits or `STOP`. Policy gradients are computed only from train rewards. Validation scores may select the saved skill and may drive configured early stopping, but never enter the policy reward. Test tasks are absent from the optimizer API.

## Data and leakage boundary

The manifest loader requires non-empty unique IDs, exact configured split sizes, pairwise-disjoint IDs, and pairwise-disjoint canonical task fingerprints. It records the ordered IDs and content digests. Train and validation are loaded before optimization; test is loaded only by the comparison stage after all final skill hashes are frozen.

The editor accepts only a train `EvaluationBatch`; passing validation or test results raises. The evaluator refuses test calls before `freeze()`. Formal test calls bypass cache reads so `initial_skill` and every final method are genuinely rerun, while their results are still written to cache afterward. Test rollouts hide answer metadata and do not feed golden-verifier scores into later model attempts.

## Fairness and accounting

Every optimizable method receives the same configured structural limits, initial skill, split manifests, Student model/temperature/max steps, Editor model/temperature, and seed. Each call records role, task count, logical rollout/call usage, cached versus newly executed work, input/output tokens, and elapsed time. Pure RL asserts zero Teacher and Reference rollouts. Imported current-method artifacts retain their original optimization cost metadata and are re-evaluated under the unified validation/test protocol.

## Outputs

The run directory contains the exact requested RL files, frozen skill snapshots, a resolved manifest/config record, task-level train/validation/test scores, method comparison CSV, and per-step JSONL with patch/status/reward/budget fields. Paired test reporting includes mean reward, success rate, standard error, paired bootstrap confidence interval, and win/tie/loss against the initial skill.

## Error policy

Missing manifests, duplicate tasks, budget exhaustion before a complete bundle, malformed Editor JSON, action/operator mismatch, ambiguous text targets, unavailable current-method artifact, and non-blind test invocation are hard errors or explicit invalid actions. There is no hand-written patch fallback, no partial task bundle, and no automatic split filtering.

## Testing

Tests cover the Markdown parser, action mask, patch schema/application, reward components, split isolation, state/policy shapes, policy update, checkpoint round-trip, deterministic seeds, initial-skill Final evaluation, budget preflight/accounting, blind Final behavior, and an API-free end-to-end optimization/comparison run. Verification consists of the complete pytest suite, Python syntax/compile checks, and the mock smoke command.

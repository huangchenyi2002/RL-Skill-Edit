# RL-Skill-Edit

RL-Skill-Edit optimizes one external Markdown Skill while the agent model remains frozen.
The actor-critic selects a Skill module and edit operator; an Editor proposes one local
patch; Train reward updates the policy; Validation selects the checkpoint; Test is read
only after the final Skill is frozen.

The repository contains one optimizer. The initial Skill is required input and is
reported beside the frozen RL result, but it is not another optimization method.

## How it works

1. A frozen Student executes Train tasks with the initial Skill.
2. A small NumPy actor-critic selects one Markdown module and one edit operator.
3. The Editor proposes one strict local patch. Invalid patches are rejected.
4. Paired Train reward updates the policy; Validation reward selects the saved
   checkpoint.
5. After the Skill and provenance are frozen, the CLI runs a fresh blind Test for
   the initial and RL-edited Skills with identical tasks, seeds, and repetitions.

Test data never enters policy updates or checkpoint selection. `--test-only`
accepts an existing result only when its Skill, configuration, splits,
implementation, dependencies, summary, and seed still match the frozen provenance.

## Repository layout

```text
rl_skill_edit/                  RL algorithm, adapters, CLI, and reporting
configs/rl_skill_edit.yaml      Real Spreadsheet configuration
configs/rl_skill_edit_smoke.yaml
                                Deterministic API-free configuration
data/initial_skill.md           Neutral handwritten starting Skill
data/mock_rl_skill_edit/        Small tracked smoke fixtures
scripts/run_smoke.sh            API-free end-to-end check
tests/                          Unit, boundary, isolation, and end-to-end tests
```

The real Train, Validation, and Test manifests and their workbooks are private
inputs and are not included in this repository.

## Setup

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For a real run, provide the API key to the process environment:

```bash
export OPENROUTER_API_KEY='your-key'
```

The API-free smoke does not need this key.

## Private data format

The real configuration expects these user-provided files:

```text
data/private/train.json
data/private/validation.json
data/private/test.json
```

Each manifest is a JSON array with the exact size declared in the configuration.
Every task needs a unique ID, a description, two existing workbook paths, and an
answer range:

```json
{
  "task_id": "unique-task-id",
  "description": "Edit the workbook as requested.",
  "spreadsheet": {
    "init_file": "/absolute/path/input.xlsx",
    "golden_file": "/absolute/path/golden.xlsx",
    "answer_sheet": "Sheet1",
    "answer_position": "A1:C10"
  }
}
```

Relative workbook paths are resolved from the manifest directory. IDs and
workbook content must not overlap across Train, Validation, and Test.

## Run

Train once with the real configuration:

```bash
.venv/bin/python -m rl_skill_edit --config configs/rl_skill_edit.yaml --seed 42
```

Re-run only the frozen blind report:

```bash
.venv/bin/python -m rl_skill_edit --config configs/rl_skill_edit.yaml --seed 42 --test-only
```

Run the API-free end-to-end smoke:

```bash
bash scripts/run_smoke.sh
```

## Outputs

The real run publishes one complete tree under `results/rl_skill_edit/`:

| Path | Purpose |
| --- | --- |
| `rl_skill_edit/best_rl_skill.md` | Validation-selected frozen Skill |
| `rl_skill_edit/final_rl_policy.pt` | NumPy policy weights, configuration, and RNG state |
| `rl_skill_edit/rl_training_log.jsonl` | Per-step actions, patches, rewards, penalties, and usage |
| `rl_skill_edit/rl_episode_summary.csv` | Episode returns and checkpoint paths |
| `rl_skill_edit/rl_optimization_summary.json` | Final Train and Validation summary |
| `rl_skill_edit/freeze_provenance.json` | Bindings checked by `--test-only` |
| `method_comparison.csv` | Initial-versus-RL aggregate metrics and resource use |
| `test_task_level_results.csv` | Ordered blind Test rewards |
| `comparison_report.json` | Machine-readable paired Test report |
| `experiment_manifest.json` | Split, code, dependency, seed, and artifact hashes |
| `.rl-skill-edit-output.json` | Ownership marker bound to the final output path |

Publication is transactional. When a verified result already exists, it is kept
at `results/.rl_skill_edit.previous` during replacement. The CLI validates the
hidden ownership marker and the complete tree before it may replace or clean that
snapshot; unrelated directories are never treated as managed output.

## Generated-code isolation warning

The Spreadsheet runtime executes model-generated Python against a copied
workbook. Its subprocess and resource limits are not a complete security
sandbox. Run real experiments in a disposable VM or container without secrets,
privileged mounts, or access to sensitive networks and files.

## Tests

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider tests
.venv/bin/ruff check rl_skill_edit tests
```

The end-to-end smoke test uses the same CLI, policy update, checkpoint selection,
freeze, and blind Test path as the real runtime, with deterministic local fixtures.

## Search record

- 2026-07-15, GitHub: TextGrad, PromptWizard, Llama Prompt Ops, and EvoAgentX
  provide useful prompt-optimization patterns, but none implements this masked
  module/operator actor-critic with the required split and provenance boundary.
- 2026-07-15, skills.sh: the reviewed prompt-engineering entries did not provide
  a directly reusable RL external-Skill workflow. No large RL framework or new
  runtime dependency was added.

## Status

Completed:

- Standalone NumPy actor-critic and strict local patch validation.
- Forced-Skill OpenRouter/Spreadsheet runtime with explicit budget accounting.
- Train-only learning, Validation selection, frozen blind Test, provenance, and
  transactional output publication.
- Deterministic API-free smoke and repository boundary tests.

Remaining before a real experiment:

- Supply the three private manifests and all declared workbooks.
- Set `OPENROUTER_API_KEY`, confirm model prices, and run the real command above.

# RL-Skill-Edit implementation note

The CLI implements one fixed sequence: load a neutral initial Markdown Skill,
optimize it on Train, select the saved checkpoint on Validation, freeze the Skill
and provenance, then run a fresh blind Test for the initial and RL-edited Skills.
The initial Skill is an input and reporting baseline, not an optimizer.

The frozen Student receives exactly one active Skill. A NumPy actor-critic masks
invalid actions and samples a Markdown module plus edit operator. The frozen
Editor returns one structured local patch; strict validation rejects ambiguous,
out-of-scope, oversized, or malformed edits. Paired Train score change, length,
edit-distance, and invalid-action costs form the transition reward. Validation
never updates the policy.

The Spreadsheet adapter copies each input workbook, extracts one explicit Python
code block, executes it in a restricted subprocess, and compares the declared
golden range with strict cell-type and workbook-structure checks. Empty or invalid
model output, incomplete API usage, execution failure, and unsupported workbook
values fail explicitly. The subprocess controls reduce accidental damage but do
not replace VM or container isolation for generated code.

Freeze provenance binds the initial and selected Skill digests, normalized
configuration, ordered split digests, implementation files, `requirements.txt`,
optimization summary, skill identity, and seed. `--test-only` rejects missing,
unknown, or changed fields. The Test pass uses identical task order, seeds, and
repetitions for both Skills, keeps prompts blind, and disables cache reads.

All optimization artifacts, five paired reports, the experiment manifest, cache,
and hidden ownership marker are built in staging. Publication validates the full
tree and installs it transactionally. An existing verified tree is retained at a
deterministic `.previous` path, and any failed replacement restores it without
touching private input files.

---
name: Spreadsheet Editing Seed
description: Neutral handwritten starting Skill for RL-Skill-Edit.
---

<!-- Repository-authored seed distributed without learned edits. -->

# Workbook handling

- Use the runtime-provided `wb_path` as the input and output workbook path.
- Load the workbook once with `openpyxl` and save it explicitly after editing.
- Select worksheets and columns from task descriptions and workbook labels.

# Cell operations

- Preserve cells that the task does not ask to change.
- Treat blank, numeric, text, date, formula, and merged cells according to their
  actual workbook types.
- Use workbook dimensions and labels instead of assuming a fixed table size.

# Completion

- Return one executable Python code block.
- Ensure the code writes the requested result and calls `wb.save(wb_path)`.

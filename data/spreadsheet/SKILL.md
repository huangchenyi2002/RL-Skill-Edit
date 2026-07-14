# Spreadsheet Manipulation Skill (Haiku-optimized)

## Environment

- Excel file path: variable `wb_path` (already bound by runtime)
- Load: `wb = openpyxl.load_workbook(wb_path)`
- Save: `wb.save(wb_path)`  ← always end with this
- Available libraries: `openpyxl`, `pandas` (nothing else)

---

## Code Template (start from this)

```python
from openpyxl import load_workbook
from openpyxl.utils import (
    get_column_letter, column_index_from_string)

wb = load_workbook(wb_path)
ws = wb.active   # or wb["SheetName"]

# 1) Build header_map from row 1
header_map = {}
for cell in ws[1]:
    if cell.value is not None:
        header_map[str(cell.value).strip()] = (
            cell.column)

# 2) Find target column by header text
target_col = header_map.get("SomeHeader")

# 3) Iterate real rows (no hard-coded max_row)
for row in range(2, ws.max_row + 1):
    v = ws.cell(row=row, column=target_col).value
    if v is None:
        continue
    # ... your transformation here
    ws.cell(row=row, column=target_col).value = (
        new_value)

wb.save(wb_path)
```

---

## Core Rules (checklist, in order)

1. **Load once, save once.** No temp copies. Use `wb_path` on both ends.

2. **Never hard-code column letters.** Always look up by header text via `header_map`. Row 1 is usually the header row.

3. **Never hard-code `max_row`.** Use `ws.max_row` or scan for the first blank row.

4. **Handle None before use.**
   ```python
   v = cell.value
   if v is None:
       continue
   ```

5. **Text compare = normalize first.**
   ```python
   def norm(s):
       return str(s).strip().casefold()
   ```

6. **Numeric parsing must be safe.**
   ```python
   def parse_num(v):
       if v is None or isinstance(v, bool):
           return None
       s = str(v).replace(",", "").replace("$", "")
       s = s.strip()
       try:
           return float(s)
       except ValueError:
           return None
   ```

7. **Formulas → compute in Python, write literal.**
   If a cell has a formula and you need its value,
   load a 2nd workbook with `data_only=True`:
   ```python
   wb_val = load_workbook(wb_path, data_only=True)
   ws_val = wb_val[ws.title]
   cached = ws_val.cell(row=r, column=c).value
   ```

8. **Multi-sheet: use `wb.sheetnames`.**
   ```python
   for name in wb.sheetnames:
       ws = wb[name]
       # ...
   ```

9. **Lookups → build dict once, then reuse.**
   ```python
   lookup = {}
   for row in range(2, ws2.max_row + 1):
       k = norm(ws2.cell(row=row, column=1).value)
       lookup[k] = ws2.cell(row=row, column=2).value
   ```

10. **Dates: use datetime, not strings.**
    ```python
    from datetime import datetime, date
    v = cell.value
    if isinstance(v, (datetime, date)):
        month = v.month
    ```

---

## Common Patterns

### Pattern A: Filter + copy rows

```python
src = wb["Source"]
dst = wb["Target"]
out_row = 2
for row in range(2, src.max_row + 1):
    if src.cell(row=row, column=1).value == "keep":
        for col in range(1, src.max_column + 1):
            dst.cell(row=out_row, column=col).value = (
                src.cell(row=row, column=col).value)
        out_row += 1
```

### Pattern B: Aggregate by group

```python
from collections import defaultdict
totals = defaultdict(float)
for row in range(2, ws.max_row + 1):
    key = norm(ws.cell(row=row, column=1).value)
    n = parse_num(ws.cell(row=row, column=2).value)
    if n is not None:
        totals[key] += n
# write results
for i, (k, v) in enumerate(sorted(totals.items())):
    ws.cell(row=i+2, column=4).value = k
    ws.cell(row=i+2, column=5).value = v
```

### Pattern C: Cross-table lookup

```python
lookup = {}
for row in range(2, ws["Lookup"].max_row + 1):
    k = norm(ws["Lookup"].cell(row=row, column=1).value)
    lookup[k] = ws["Lookup"].cell(row=row, column=2).value

for row in range(2, ws["Main"].max_row + 1):
    k = norm(ws["Main"].cell(row=row, column=1).value)
    if k in lookup:
        ws["Main"].cell(row=row, column=3).value = (
            lookup[k])
```

---

## Do NOT

- ❌ Use `INPUT_PATH` / `OUTPUT_PATH`. Only `wb_path`.
- ❌ Use `pandas.to_excel(...)` to save (destroys formulas).
- ❌ Use `input()` or `sys.argv`.
- ❌ Hard-code `"A2"`, `"B5"`, or specific row numbers.
- ❌ Forget `wb.save(wb_path)` at the end.
- ❌ Use libraries other than openpyxl / pandas.

---

## SLOW_UPDATE

<!-- OSD Slow-Update Area (auto-managed) -->

## APPENDIX

<!-- OSD Appendix Notes (auto-managed) -->

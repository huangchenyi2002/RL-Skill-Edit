#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing repository virtual environment: $PYTHON" >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" experiments/run_skill_optimization_comparison.py \
  --config configs/rl_skill_edit_smoke.yaml \
  --methods initial_skill current_method rl_skill_edit \
  --seed 42

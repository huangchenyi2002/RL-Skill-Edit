#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" -m rl_skill_edit \
  --config "$ROOT/configs/rl_skill_edit_smoke.yaml" \
  --seed 42

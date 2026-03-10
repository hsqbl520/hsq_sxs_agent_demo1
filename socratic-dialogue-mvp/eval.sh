#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

CONDA_PYTHON="/mnt/c/Users/10985/miniconda3/python.exe"

if [[ ! -x "$CONDA_PYTHON" ]]; then
  echo "Conda base python not found: $CONDA_PYTHON"
  exit 1
fi

"$CONDA_PYTHON" tests/evaluate_gold.py \
  --gold tests/fixtures/gold_cases_v1.json \
  --out tests/reports/gold_eval_report_v1.json

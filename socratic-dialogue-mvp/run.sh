#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONDA_PYTHON="/mnt/c/Users/10985/miniconda3/python.exe"

if [[ ! -x "$CONDA_PYTHON" ]]; then
  echo "Conda base python not found: $CONDA_PYTHON"
  exit 1
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[1/2] Installing dependencies (idempotent)..."
"$CONDA_PYTHON" -m pip install -r requirements.txt > /tmp/socratic-pip.log 2>&1 || {
  echo "Dependency installation failed. Check /tmp/socratic-pip.log"
  exit 1
}

echo "[2/2] Starting API on http://127.0.0.1:8000"
exec "$CONDA_PYTHON" -m uvicorn app.main:app --reload --port 8000

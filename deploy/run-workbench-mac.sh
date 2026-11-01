#!/usr/bin/env bash
set -euo pipefail
workbench_project="$(cd "$(dirname "$0")/.." && pwd)"
workbench_default="$workbench_project/.venv-workbench/bin/python"
if [ ! -x "$workbench_default" ]; then workbench_default="$workbench_project/.venv/bin/python"; fi
workbench_python="${DSA_WORKBENCH_PYTHON:-$workbench_default}"
PYTHONPATH="$workbench_project/src" "$workbench_python" -m dsa.workbench.mac_service "${1:-start}" \
  --project "$workbench_project" --data "${DSA_DATA:-$workbench_project/data}" --port "${DSA_WORKBENCH_PORT:-8777}"

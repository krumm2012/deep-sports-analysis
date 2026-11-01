#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export DSA_DATA="$project_root/data"
export DSA_PROJECTS="$project_root/config/source-validation-projects.json"
export RF_HOME="$DSA_DATA/models/rfdetr"
exec "$project_root/.venv/bin/python" -u -m dsa.gpu.local source_validation/phone_2025 "$@"

#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export DSA_DATA="${DSA_DATA:-$project_root/data}"
export DSA_PROJECTS="${DSA_PROJECTS:-$project_root/config/demo-projects.json}"
export RF_HOME="$DSA_DATA/models/rfdetr"
if [[ -n "${DSA_DOWNLOAD_PROXY:-}" ]]; then
  export HTTPS_PROXY="$DSA_DOWNLOAD_PROXY" HTTP_PROXY="$DSA_DOWNLOAD_PROXY"
fi
exec "$project_root/.venv/bin/python" -u -m dsa.gpu.local demo/rally "$@"

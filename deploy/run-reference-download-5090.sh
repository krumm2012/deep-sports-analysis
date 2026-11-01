#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/raw verification
exec 9>verification/reference-download.lock
flock -n 9 || { echo 'A reference download job is already running' >&2; exit 1; }
proxy="${DSA_DOWNLOAD_PROXY:-http://127.0.0.1:18882}"
exec "$project_root/.venv/bin/python" -u -m dsa.ingest.reference_download \
  config/atp-download-manifest.json data/raw --proxy "$proxy" "$@"

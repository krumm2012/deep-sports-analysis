#!/usr/bin/env bash
# Run only after camera_split has frozen the completed 25fps dataset manifest.
set -euo pipefail
cd "${DSA_PROJECT_DIR:-/root/deep-sports-analysis}"
mkdir -p verification
for dsa_seed in 0 1 2; do
  dsa_log="verification/camera05-custom-seed${dsa_seed}.log"
  dsa_exit="verification/camera05-custom-seed${dsa_seed}.exit"
  if PYTHONPATH=src .venv-moseq/bin/python -u -m dsa.measure.moseq_dataset \
    --manifest data/projects/camera05_custom/manifest.json \
    --output "data/projects/camera05_custom/models/seed${dsa_seed}" \
    --seed "$dsa_seed" --num-states 8 --latent-dim 6 \
    --ar-iters 100 --full-iters 500 --apply-iters 200 \
    --kappa-ar 1000000 --kappa-full 100000 >"$dsa_log" 2>&1; then
    printf '0\n' >"$dsa_exit"
  else
    dsa_status=$?
    printf '%s\n' "$dsa_status" >"$dsa_exit"
    exit "$dsa_status"
  fi
done
PYTHONPATH=src .venv/bin/python -m dsa.pages.moseq_dataset_report \
  --manifest data/projects/camera05_custom/manifest.json \
  --runs data/projects/camera05_custom/models/seed0 \
         data/projects/camera05_custom/models/seed1 \
         data/projects/camera05_custom/models/seed2 \
  --output data/projects/camera05_custom/review \
  >verification/camera05-custom-report.log 2>&1

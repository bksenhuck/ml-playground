#!/bin/sh
# entrypoint.sh — Container startup for Cloud Run
set -e

echo "[entrypoint] Syncing models and data from GCS ..."
python /app/deploy/sync_models.py

echo "[entrypoint] Starting Dash app ..."
# Models are already on disk (synced above) — tell HF to never hit the network.
# This prevents hf_pipeline() from hanging on Hub connectivity checks.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec python -m app.app

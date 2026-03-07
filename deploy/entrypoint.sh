#!/bin/sh
# entrypoint.sh — Container startup script for Cloud Run
#
# If GCS_MLFLOW_URI is set, downloads the latest mlflow.db from GCS
# before starting the Dash app. This ensures experiment history persists
# across container restarts and redeployments.
#
# Usage (set in Dockerfile):
#   CMD ["/app/deploy/entrypoint.sh"]

set -e

DB_PATH="/app/mlflow.db"

if [ -n "$GCS_MLFLOW_URI" ]; then
    echo "[entrypoint] Downloading mlflow.db from $GCS_MLFLOW_URI ..."
    if gsutil cp "$GCS_MLFLOW_URI" "$DB_PATH"; then
        echo "[entrypoint] mlflow.db downloaded successfully"
    else
        echo "[entrypoint] WARNING: could not download mlflow.db — starting fresh"
    fi
else
    echo "[entrypoint] GCS_MLFLOW_URI not set — using local mlflow.db (if any)"
fi

# -- Model Weights Sync (Qwen & Llama Guard) ----------------------------------
if [ -n "$GCS_MODEL_PATH" ]; then
    QWEN_DIR="/app/models/Qwen2.5-0.5B-Instruct"
    echo "[entrypoint] Syncing Qwen from $GCS_MODEL_PATH/models/Qwen2.5-0.5B-Instruct ..."
    mkdir -p "$QWEN_DIR"
    gsutil -m rsync -r "$GCS_MODEL_PATH/models/Qwen2.5-0.5B-Instruct" "$QWEN_DIR" || echo "[entrypoint] WARNING: Qwen sync failed"
fi

if [ -n "$GCS_GUARD_PATH" ]; then
    GUARD_DIR="/app/models/meta-llama/Llama-Guard-3-1B"
    echo "[entrypoint] Syncing Llama Guard from $GCS_GUARD_PATH ..."
    mkdir -p "$GUARD_DIR"
    gsutil -m rsync -r "$GCS_GUARD_PATH" "$GUARD_DIR" || echo "[entrypoint] WARNING: Llama Guard sync failed"
fi

echo "[entrypoint] Starting Dash app ..."
exec python -m app.app

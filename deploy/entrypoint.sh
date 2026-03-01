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

echo "[entrypoint] Starting Dash app ..."
exec python -m app.app

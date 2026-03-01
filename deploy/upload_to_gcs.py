"""Upload MLflow tracking database to Google Cloud Storage.

This script:
1. Checkpoints the SQLite WAL into the main DB file (ensures consistency)
2. Uploads mlflow.db to GCS
3. Optionally triggers a Cloud Run redeploy

Usage:
    python -m deploy.upload_to_gcs
    python -m deploy.upload_to_gcs --deploy   # also redeploy Cloud Run

Requirements:
    - GOOGLE_APPLICATION_CREDENTIALS or gcloud auth already configured
    - pip install google-cloud-storage

GCS URI is read from .env:
    GCS_MLFLOW_URI   e.g. gs://my-bucket/mlflow.db
"""
import sys
import sqlite3
import argparse
import logging
import subprocess
from pathlib import Path

from deploy.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket_name, blob_name) from a gs://bucket/path URI."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    _, rest = uri.split("gs://", 1)
    bucket_name, blob_name = rest.split("/", 1)
    return bucket_name, blob_name


def _gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except ImportError:
        raise ImportError(
            "google-cloud-storage is not installed.\n"
            "Run: pip install google-cloud-storage"
        )


def checkpoint_wal(db_path: Path) -> None:
    """Flush the SQLite WAL into the main database file.

    Must be done before uploading mlflow.db so the uploaded file contains
    all committed data (WAL journal is not included in the upload).
    """
    logger.info("[WAL] Checkpointing WAL into %s ...", db_path.name)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.info("[WAL] Checkpoint complete")
    except Exception as exc:
        raise RuntimeError(f"WAL checkpoint failed: {exc}") from exc
    finally:
        conn.close()


def upload_file(local_path: Path, gcs_uri: str) -> None:
    """Upload a single local file to GCS."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(
        "[GCS] Uploading %s (%.1f MB) → %s", local_path.name, size_mb, gcs_uri
    )
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    logger.info("[GCS] %s uploaded successfully", local_path.name)


# ---------------------------------------------------------------------------
# Upload step
# ---------------------------------------------------------------------------

def upload_mlflow_db() -> bool:
    db_path = settings.get_mlflow_db_path()
    if not db_path.exists():
        logger.error("[DB] mlflow.db not found at %s", db_path)
        return False
    checkpoint_wal(db_path)
    upload_file(db_path, settings.GCS_MLFLOW_URI)
    return True


# ---------------------------------------------------------------------------
# Cloud Run deploy
# ---------------------------------------------------------------------------

def trigger_deploy() -> bool:
    """Redeploy Cloud Run using the current image."""
    image = settings.get_docker_image()
    cmd = [
        "gcloud", "run", "deploy", settings.CLOUDRUN_SERVICE,
        "--image", image,
        "--region", settings.GCP_REGION,
        "--platform", "managed",
    ]
    logger.info("[DEPLOY] Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, shell=True)
    if result.returncode == 0:
        logger.info("[DEPLOY] Cloud Run redeployed successfully")
        return True
    logger.error("[DEPLOY] gcloud run deploy failed (exit %d)", result.returncode)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload mlflow.db to GCS before deploying to Cloud Run"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Trigger a Cloud Run redeploy after uploading",
    )
    args = parser.parse_args()

    settings.validate()

    logger.info("=== GCS UPLOAD ===")

    ok = upload_mlflow_db()
    if not ok:
        logger.error("Upload failed — aborting")
        sys.exit(1)

    logger.info("mlflow.db uploaded to %s", settings.GCS_MLFLOW_URI)

    if args.deploy:
        deploy_ok = trigger_deploy()
        if not deploy_ok:
            sys.exit(1)
    else:
        logger.info(
            "Tip: run with --deploy to also trigger a Cloud Run redeploy, "
            "or manually:\n  gcloud run deploy %s --image %s "
            "--region %s --platform managed",
            settings.CLOUDRUN_SERVICE,
            settings.get_docker_image(),
            settings.GCP_REGION,
        )


if __name__ == "__main__":
    main()

"""Download model weights from GCS to local paths at container startup.

Uses google-cloud-storage (already in requirements) instead of gsutil,
so no gcloud SDK install is needed in the container.
"""
import os
import sys

from pathlib import Path
from google.cloud import storage


# Subdirs and suffixes that are not needed for HuggingFace inference
_SKIP_SUBDIRS = {"original", ".cache"}
_SKIP_SUFFIXES = {".lock"}


def _sync_gcs_prefix(bucket_name: str, prefix: str, local_dir: Path) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        print(f"[sync] WARNING: no files found at gs://{bucket_name}/{prefix}")
        return
    for blob in blobs:
        rel = blob.name[len(prefix):].lstrip("/")
        # Skip cache dirs, original weights dir, and lock files
        parts = Path(rel).parts
        if parts and parts[0] in _SKIP_SUBDIRS:
            continue
        if Path(rel).suffix in _SKIP_SUFFIXES:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"[sync] {blob.name} -> {dest}")
            blob.download_to_filename(str(dest))
    print(f"[sync] Done: gs://{bucket_name}/{prefix} -> {local_dir}")


def _parse_gcs(uri: str):
    """Return (bucket, prefix) from a gs://bucket/prefix URI."""
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    parts = uri[5:].split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def main() -> None:
    qwen_uri = os.environ.get("GCS_MODEL_PATH", "")
    guard_uri = os.environ.get("GCS_GUARD_PATH", "")
    mlflow_uri = os.environ.get("GCS_MLFLOW_URI", "")

    if mlflow_uri:
        print(f"[sync] Downloading mlflow.db from {mlflow_uri} ...")
        try:
            bucket_name, blob_name = _parse_gcs(mlflow_uri)
            client = storage.Client()
            client.bucket(bucket_name).blob(blob_name).download_to_filename(
                "/app/mlflow.db"
            )
            print("[sync] mlflow.db downloaded.")
        except Exception as exc:
            print(f"[sync] WARNING: mlflow.db download failed: {exc}")

    if qwen_uri:
        print(f"[sync] Syncing Qwen from {qwen_uri} ...")
        try:
            bucket, prefix = _parse_gcs(qwen_uri)
            _sync_gcs_prefix(bucket, prefix, Path("/app/models/Qwen/Qwen2.5-0.5B-Instruct"))
        except Exception as exc:
            print(f"[sync] WARNING: Qwen sync failed: {exc}")

    if guard_uri:
        print(f"[sync] Syncing Llama Guard from {guard_uri} ...")
        try:
            bucket, prefix = _parse_gcs(guard_uri)
            _sync_gcs_prefix(bucket, prefix, Path("/app/models/meta-llama/Llama-Guard-3-1B"))
        except Exception as exc:
            print(f"[sync] WARNING: Llama Guard sync failed: {exc}")


if __name__ == "__main__":
    main()

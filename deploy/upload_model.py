import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from google.cloud import storage


def _has_weights(dir_path: str) -> bool:
    """Return True if the directory contains actual model weights."""
    p = Path(dir_path)
    return p.is_dir() and (
        any(p.glob("*.safetensors")) or any(p.glob("*.bin")) or
        (p / "config.json").exists()
    )

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

def upload_to_gcs(local_dir, bucket_name, gcs_prefix, project_id):
    """Uploads a local directory recursively to GCS."""
    print(f"Uploading {local_dir} to gs://{bucket_name}/{gcs_prefix}...")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # relative path to GCS
            relative_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{gcs_prefix}/{relative_path.replace(os.sep, '/')}"
            blob = bucket.blob(blob_path)
            print(f"Uploading {local_path} to {blob_path}...")
            blob.upload_from_filename(local_path)

def main():
    parser = argparse.ArgumentParser(description="Download and upload model weights to GCS")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET_NAME"), help="GCS bucket name")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()

    missing = [k for k, v in {"--bucket": args.bucket, "--project": args.project}.items() if not v]
    if missing:
        parser.error(f"Missing required args (not found in .env either): {', '.join(missing)}")

    # 1. Qwen 2.5 0.5B
    if not args.skip_qwen:
        qwen_id = "Qwen/Qwen2.5-0.5B-Instruct"
        qwen_dir = "models/Qwen/Qwen2.5-0.5B-Instruct"
        if not _has_weights(qwen_dir):
            print(f"--- Qwen: Downloading {qwen_id} ---")
            snapshot_download(repo_id=qwen_id, local_dir=qwen_dir)
        else:
            print(f"--- Qwen: Using existing local model at {qwen_dir} ---")
        upload_to_gcs(qwen_dir, args.bucket, "models/Qwen/Qwen2.5-0.5B-Instruct", args.project)

    # 2. Llama Guard 3 1B
    if not args.skip_guard:
        guard_id = "meta-llama/Llama-Guard-3-1B"
        guard_dir = "models/meta-llama/Llama-Guard-3-1B"
        if not _has_weights(guard_dir):
            print(f"--- Llama Guard: Downloading {guard_id} (requires HF token + Meta license) ---")
            snapshot_download(repo_id=guard_id, local_dir=guard_dir)
        else:
            print(f"--- Llama Guard: Using existing local model at {guard_dir} ---")
        upload_to_gcs(guard_dir, args.bucket, "models/meta-llama/Llama-Guard-3-1B", args.project)

    print("All uploads complete!")

if __name__ == "__main__":
    main()

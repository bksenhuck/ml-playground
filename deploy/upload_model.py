import os
import argparse
from huggingface_hub import snapshot_download
from google.cloud import storage

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
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()

    # 1. Qwen 2.5 0.5B
    if not args.skip_qwen:
        qwen_id = "Qwen/Qwen2.5-0.5B-Instruct"
        qwen_dir = "models/Qwen2.5-0.5B-Instruct"
        print(f"--- Qwen: Downloading {qwen_id} ---")
        snapshot_download(repo_id=qwen_id, local_dir=qwen_dir)
        upload_to_gcs(qwen_dir, args.bucket, "models/Qwen2.5-0.5B-Instruct", args.project)

    # 2. Llama Guard 3 1B
    if not args.skip_guard:
        guard_id = "meta-llama/Llama-Guard-3-1B"
        guard_dir = "models/meta-llama/Llama-Guard-3-1B"
        print(f"--- Llama Guard: Downloading {guard_id} ---")
        snapshot_download(repo_id=guard_id, local_dir=guard_dir)
        upload_to_gcs(guard_dir, args.bucket, "models/meta-llama/Llama-Guard-3-1B", args.project)

    print("All uploads complete!")

if __name__ == "__main__":
    main()

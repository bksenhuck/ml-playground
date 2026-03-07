import os
import argparse
from huggingface_hub import snapshot_download

def download_model(model_id: str, local_dir: str):
    print(f"Downloading {model_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Download of {model_id} complete!")

def main():
    parser = argparse.ArgumentParser(description="Download LLM models")
    parser.add_argument(
        "--model", 
        choices=["qwen", "llama-guard", "both"], 
        default="qwen",
        help="Model to download"
    )
    args = parser.parse_args()

    if args.model in ["qwen", "both"]:
        download_model(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            "./models/Qwen/Qwen2.5-0.5B-Instruct"
        )
    
    if args.model in ["llama-guard", "both"]:
        download_model(
            "meta-llama/Llama-Guard-3-1B", 
            "./models/meta-llama/Llama-Guard-3-1B"
        )

if __name__ == "__main__":
    main()

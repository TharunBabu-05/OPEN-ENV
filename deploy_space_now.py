"""
deploy_space_now.py — One-command Space deployment.

Usage:
    python deploy_space_now.py --token hf_YOUR_TOKEN_HERE

This deploys the Gradio Space to HuggingFace immediately.
Space URL will be: https://huggingface.co/spaces/tharu5054/esg-compliance-env
"""

import argparse
import sys
from pathlib import Path

def deploy(token: str):
    from huggingface_hub import HfApi, create_repo

    HF_USERNAME = "tharu5054"
    SPACE_NAME  = "esg-compliance-env"
    REPO_ID     = f"{HF_USERNAME}/{SPACE_NAME}"

    api = HfApi(token=token)

    # Create Space
    print(f"Creating Space: {REPO_ID} ...")
    create_repo(
        REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=token,
        private=False,
    )
    print(f"Space ready: https://huggingface.co/spaces/{REPO_ID}")

    # Files to upload: (local_path, remote_name)
    files = [
        ("space_app.py",         "app.py"),          # HF expects app.py
        ("env.py",               "env.py"),
        ("models.py",            "models.py"),
        ("tasks.py",             "tasks.py"),
        ("dataset_builder.py",   "dataset_builder.py"),
        ("reward_functions.py",  "reward_functions.py"),
        ("space_requirements.txt", "requirements.txt"),
        ("SPACE_README.md",      "README.md"),
    ]

    print("\nUploading files...")
    for local, remote in files:
        if not Path(local).exists():
            print(f"  SKIP (not found): {local}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type="space",
            )
            print(f"  OK: {local} -> {remote}")
        except Exception as e:
            print(f"  ERROR: {local}: {e}")

    # Upload result plots if they exist
    for plot in ["score_comparison.png", "reward_history.png", "esg_metrics.png"]:
        p = Path("results") / plot
        if p.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"results/{plot}",
                    repo_id=REPO_ID,
                    repo_type="space",
                )
                print(f"  OK: results/{plot}")
            except Exception as e:
                print(f"  SKIP plot {plot}: {e}")

    print(f"\n{'='*60}")
    print(f"DEPLOYMENT COMPLETE")
    print(f"Space URL: https://huggingface.co/spaces/{REPO_ID}")
    print(f"{'='*60}")
    return f"https://huggingface.co/spaces/{REPO_ID}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace API token (hf_...)")
    args = parser.parse_args()
    deploy(args.token)

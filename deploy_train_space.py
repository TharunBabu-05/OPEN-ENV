"""Deploy the A100 training Space to HuggingFace."""
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os
TOKEN = os.environ.get("HF_TOKEN", "hf_REPLACE_ME")
REPO = "tharun5054/esg-rl-train"

api = HfApi(token=TOKEN)

# Ensure Space exists with A100 hardware
create_repo(REPO, repo_type="space", space_sdk="gradio", exist_ok=True, token=TOKEN, private=False)

# Files to upload: (local_path, remote_name)
files = [
    ("space_train_app.py",       "app.py"),
    ("train_rl_a100.py",         "train_rl_a100.py"),
    ("train_config_a100.yaml",   "train_config_a100.yaml"),
    ("env.py",                   "env.py"),
    ("models.py",                "models.py"),
    ("tasks.py",                 "tasks.py"),
    ("reward_functions.py",      "reward_functions.py"),
    ("dataset_builder.py",       "dataset_builder.py"),
    ("space_train_requirements.txt", "requirements.txt"),
    ("SPACE_TRAIN_README.md",    "README.md"),
]

print(f"Deploying to: https://huggingface.co/spaces/{REPO}")
for local, remote in files:
    if not Path(local).exists():
        print(f"  SKIP: {local}")
        continue
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=REPO,
        repo_type="space",
    )
    print(f"  OK: {local} -> {remote}")

print(f"\nDone! Visit: https://huggingface.co/spaces/{REPO}")
print("NOTE: Go to Settings -> Hardware -> Select 'A100 Large' to enable GPU")

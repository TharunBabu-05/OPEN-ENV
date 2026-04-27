import os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("Please set HF_TOKEN environment variable.")
    exit(1)
    
REPO_ID = "tharun5054/esg-compliance-env"
api = HfApi(token=TOKEN)

# Create the correct YAML header for Docker SDK
yaml_header = """---
title: Esg Compliance Env
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

"""

# Combine it with the GitHub README.md
github_readme = Path("README.md").read_text(encoding="utf-8")
final_readme_content = yaml_header + github_readme
Path("temp_readme_for_hf.md").write_text(final_readme_content, encoding="utf-8")

print(f"Uploading entire github folder to {REPO_ID} as a Docker Space...")

# Upload the entire folder ignoring unnecessary local git/venv directories
api.upload_folder(
    folder_path=".",
    repo_id=REPO_ID,
    repo_type="space",
    ignore_patterns=[
        ".git/*",
        ".venv/*",
        "__pycache__/*",
        "temp_adapter/*",
        "esg_compliance_env.egg-info/*",
        ".dist/*",
        "uv.lock",
        "temp_readme_for_hf.md",
        "reference/*",  # Ignore the reference clone to prevent HF crashes
        "tutorial/*"    # Ignore tutorials with notebooks
    ]
)

# Overwrite README.md on HuggingFace with the Docker-configured one
print("Uploading combined README.md...")
api.upload_file(
    path_or_fileobj="temp_readme_for_hf.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
)

Path("temp_readme_for_hf.md").unlink(missing_ok=True)
print("Done! GitHub repo perfectly synced to Hugging Face Space using Docker SDK.")

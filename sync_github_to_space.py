import os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "hf_REPLACE_ME")
REPO_ID = "tharun5054/esg-compliance-env"

api = HfApi(token=TOKEN)

# 1. Prepare README.md by combining Space YAML and GitHub README
space_readme = Path("SPACE_README.md").read_text(encoding="utf-8")
yaml_header = ""
if space_readme.startswith("---"):
    yaml_header = space_readme.split("---")[1]
    yaml_header = "---\n" + yaml_header + "---\n\n"

github_readme = Path("README.md").read_text(encoding="utf-8")
final_readme_content = yaml_header + github_readme

# Save a temporary copy to upload
Path("temp_space_readme.md").write_text(final_readme_content, encoding="utf-8")

print(f"Uploading entire folder to {REPO_ID}...")

# 2. Upload the entire folder ignoring unnecessary directories
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
        "temp_space_readme.md",
        "SPACE_README.md"  # Don't need this anymore
    ]
)

# 3. Specifically upload our combined README.md to overwrite the Space's README.md
print("Uploading combined README.md...")
api.upload_file(
    path_or_fileobj="temp_space_readme.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
)

# Also ensure app.py is the gradio app for the space if they want the space UI
if Path("space_app.py").exists():
    print("Ensuring space_app.py is set as the main app.py for HF Spaces...")
    api.upload_file(
        path_or_fileobj="space_app.py",
        path_in_repo="app.py",
        repo_id=REPO_ID,
        repo_type="space",
    )

# Cleanup
Path("temp_space_readme.md").unlink(missing_ok=True)

print("Done! Everything from GitHub is now synced to the Hugging Face Space.")

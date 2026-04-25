# deploy_to_space.ps1
# One-click deploy to HuggingFace Spaces
# Prerequisites: huggingface_hub installed, HF_TOKEN set

param(
    [string]$SpaceName = "esg-compliance-env",
    [string]$HFUsername = "YOUR_HF_USERNAME"   # <-- Update this
)

Write-Host "=== Deploying to HuggingFace Spaces ===" -ForegroundColor Cyan
Write-Host "Space: $HFUsername/$SpaceName" -ForegroundColor White

$python = ".venv\Scripts\python.exe"

# Check HF token
if (-not $env:HF_TOKEN) {
    Write-Host "ERROR: HF_TOKEN environment variable not set." -ForegroundColor Red
    Write-Host "Set it with: `$env:HF_TOKEN = 'hf_...'`" -ForegroundColor Yellow
    exit 1
}

# Install huggingface_hub if needed
& $python -m pip install huggingface_hub -q

# Create the Space and upload files
$deployScript = @"
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = '$HFUsername/$SpaceName'

# Create Space if it doesn't exist
try:
    create_repo(repo_id, repo_type='space', space_sdk='gradio', exist_ok=True, token=os.environ['HF_TOKEN'])
    print(f'Space ready: https://huggingface.co/spaces/{repo_id}')
except Exception as e:
    print(f'Repo creation: {e}')

# Files to upload
files = [
    ('space_app.py', 'app.py'),  # Gradio app (renamed to app.py for HF)
    ('env.py', 'env.py'),
    ('models.py', 'models.py'),
    ('tasks.py', 'tasks.py'),
    ('dataset_builder.py', 'dataset_builder.py'),
    ('reward_functions.py', 'reward_functions.py'),
    ('space_requirements.txt', 'requirements.txt'),
    ('SPACE_README.md', 'README.md'),
]

for local_path, remote_path in files:
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type='space',
        )
        print(f'  Uploaded: {local_path} -> {remote_path}')
    except Exception as e:
        print(f'  ERROR uploading {local_path}: {e}')

# Upload results plots if they exist
import os
for plot in ['score_comparison.png', 'reward_history.png', 'esg_metrics.png']:
    plot_path = f'results/{plot}'
    if os.path.exists(plot_path):
        try:
            api.upload_file(
                path_or_fileobj=plot_path,
                path_in_repo=f'results/{plot}',
                repo_id=repo_id,
                repo_type='space',
            )
            print(f'  Uploaded plot: {plot}')
        except Exception as e:
            print(f'  WARNING: Could not upload {plot}: {e}')

print(f'\nDeployment complete!')
print(f'Visit: https://huggingface.co/spaces/{repo_id}')
"@

$deployScript | & $python

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host "Space URL: https://huggingface.co/spaces/$HFUsername/$SpaceName" -ForegroundColor Cyan

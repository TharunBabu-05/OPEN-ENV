# ESG Compliance Environment - Docker Build & Run Guide

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t esg-env:latest .
```

Expected output:
```
[+] Building 45.2s (14/14) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 1.2kB
 => [internal] load .dockerignore
 => [1/8] FROM docker.io/library/python:3.11-slim
 => [2/8] WORKDIR /app
 => [3/8] RUN apt-get update && apt-get install -y gcc
 => [4/8] COPY pyproject.toml ./
 => [5/8] RUN pip install --no-cache-dir pydantic openai
 => [6/8] COPY models.py env.py tasks.py inference.py openenv.yaml ./
 => [7/8] RUN useradd -m -u 1000 appuser
 => exporting to image
 => => naming to docker.io/library/esg-env:latest
```

### 2. Run Inference

**With OpenAI API:**
```bash
docker run \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="sk-..." \
  esg-env:latest
```

**With Hugging Face Inference:**
```bash
docker run \
  -e API_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Llama-3-8B-Instruct" \
  -e MODEL_NAME="meta-llama/Llama-3-8B-Instruct" \
  -e HF_TOKEN="hf_..." \
  esg-env:latest
```

**With Local LLM (e.g., Ollama):**
```bash
docker run \
  --network host \
  -e API_BASE_URL="http://localhost:11434/v1" \
  -e MODEL_NAME="llama2" \
  -e HF_TOKEN="dummy" \
  esg-env:latest
```

### 3. Interactive Testing

Run the container interactively to test components:

```bash
docker run -it esg-env:latest /bin/bash
```

Inside the container:
```bash
# Test task definitions
python tasks.py

# Test environment manually
python -c "from env import ESGEnvironment; from tasks import TASKS; env = ESGEnvironment(TASKS['basic_compliance']); print(env.reset())"

# Test models
python -c "from models import Action; print([a.name for a in Action])"
```

## Advanced Usage

### Custom Entry Point

Run specific scripts instead of default inference:

```bash
# Test tasks only
docker run esg-env:latest python tasks.py

# Test environment reset
docker run esg-env:latest python -c "from env import ESGEnvironment; from tasks import TASKS; env = ESGEnvironment(TASKS['basic_compliance']); print(env.reset())"
```

### Mount Local Code for Development

```bash
docker run \
  -v $(pwd):/app \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="sk-..." \
  esg-env:latest
```

### Save Logs to File

```bash
docker run \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="sk-..." \
  esg-env:latest > inference_logs.jsonl 2>&1
```

### Resource Constraints (Match OpenEnv Limits)

```bash
docker run \
  --cpus="2" \
  --memory="8g" \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="sk-..." \
  esg-env:latest
```

## Hugging Face Spaces Deployment

### Method 1: Via Web UI

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose "Docker" as SDK
4. Upload these files:
   - `Dockerfile`
   - `models.py`
   - `env.py`
   - `tasks.py`
   - `inference.py`
   - `openenv.yaml`
   - `pyproject.toml`
   - `requirements.txt`
   - `README.md`

5. Set environment variables in Space settings:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

6. Space will auto-build and run

### Method 2: Via Git

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/esg-env
cd esg-env

# Copy all files
cp /path/to/esg-compliance-env/* .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure all Python files are copied in Dockerfile:
```dockerfile
COPY models.py env.py tasks.py inference.py openenv.yaml ./
```

### Issue: Build takes too long

**Solution**: Use `--no-cache` to force fresh build:
```bash
docker build --no-cache -t esg-env:latest .
```

### Issue: Runtime errors with Pydantic

**Solution**: Verify Pydantic v2 is installed:
```bash
docker run esg-env:latest python -c "import pydantic; print(pydantic.__version__)"
```

### Issue: OpenAI API timeout

**Solution**: Increase timeout in inference.py or use faster model:
```python
response = client.chat.completions.create(
    model=model_name,
    timeout=60.0,  # Increase from 30s
)
```

### Issue: Container exits immediately

**Solution**: Check environment variables are set:
```bash
docker run esg-env:latest env | grep -E "API_BASE_URL|MODEL_NAME|HF_TOKEN"
```

## Performance Benchmarks

Expected runtimes on 2 vCPU, 8GB RAM:

| Task | Steps | Avg Runtime |
|------|-------|-------------|
| Basic Compliance | 6 | 2-3 minutes |
| Aggressive Sustainability | 9 | 3-5 minutes |
| Carbon Neutral Excellence | 12 | 5-7 minutes |
| **Total (all 3 tasks)** | **27** | **12-15 minutes** |

## Validation

Validate the environment meets OpenEnv specs:

```bash
# Install openenv CLI
pip install openenv

# Run validation
openenv validate

# Expected output:
# ✓ openenv.yaml is valid
# ✓ Entry points exist
# ✓ Tasks are properly defined
# ✓ Graders are deterministic
# ✓ Environment is compliant
```

## Image Size Optimization

Current image: ~500MB

To reduce further:
```dockerfile
# Use alpine instead of slim (reduces to ~200MB)
FROM python:3.11-alpine

# Or use multi-stage build
FROM python:3.11-slim as builder
# ... build wheels ...
FROM python:3.11-slim
COPY --from=builder ...
```

## Security

The Dockerfile includes security best practices:
- ✓ Non-root user (`appuser`)
- ✓ Minimal base image (python:3.11-slim)
- ✓ No unnecessary packages
- ✓ Health check included
- ✓ Environment variables for secrets (not hardcoded)

## Next Steps

After successful Docker build:

1. ✅ Test locally with `docker run`
2. ✅ Push to Docker Hub (optional)
3. ✅ Deploy to Hugging Face Spaces
4. ✅ Run `openenv validate`
5. ✅ Submit to OpenEnv Hackathon

Good luck! 🚀

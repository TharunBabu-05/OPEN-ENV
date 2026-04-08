# Optimized Dockerfile for ESG Compliance Environment
# Compatible with Hugging Face Spaces and OpenEnv

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py ./
COPY env.py ./
COPY tasks.py ./
COPY inference.py ./
COPY app.py ./
COPY openenv.yaml ./

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from models import Action; from env import ESGEnvironment; from tasks import TASKS; sys.exit(0)"

# Default command runs inference
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

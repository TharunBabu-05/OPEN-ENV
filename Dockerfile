# Optimized Dockerfile for ESG Compliance Environment
# Compatible with Hugging Face Spaces and OpenEnv
# Set SPACE_MODE=gradio to run the Gradio Space UI instead of FastAPI

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SPACE_MODE=fastapi

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml ./
COPY requirements.txt ./
COPY space_requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio>=4.0.0

# Copy application code
COPY models.py ./
COPY env.py ./
COPY tasks.py ./
COPY inference.py ./
COPY app.py ./
COPY space_app.py ./
COPY dataset_builder.py ./
COPY reward_functions.py ./
COPY openenv.yaml ./

# Create results directory
RUN mkdir -p /app/results /app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from env import ESGEnvironment; from tasks import TASKS; print('OK')"

# Startup: choose between FastAPI and Gradio based on SPACE_MODE
CMD if [ "$SPACE_MODE" = "gradio" ]; then \
        python space_app.py; \
    else \
        uvicorn app:app --host 0.0.0.0 --port 7860; \
    fi


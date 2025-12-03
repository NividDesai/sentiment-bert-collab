# Multi-stage Dockerfile for Sentiment Analysis BERT Application
# Stage 1: Build stage - Install dependencies and prepare environment
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage - Minimal runtime image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/models/cache \
    HF_HOME=/app/models/cache

# Copy application code
COPY src/ ./src/
COPY train.py .
COPY inference.py .
COPY evaluate.py .

# Create directories for models and data with proper permissions
RUN mkdir -p /app/models /app/data /app/outputs /app/models/cache && \
    chmod -R 755 /app

# Expose port for API (if using FastAPI/Flask)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - Can be overridden in docker-compose or docker run
CMD ["python", "inference.py", "--help"]

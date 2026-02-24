# ===========================================
# AURA v3 Dockerfile
# Multi-stage build for production deployment
# ===========================================

# Stage 1: Base
FROM python:3.11-slim as base

LABEL maintainer="AURA Team"
LABEL description="AURA v3 - Personal AI Assistant"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    AURA_ENV=production

# Set work directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as deps

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libmp3lame \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Builder
FROM deps as builder

# Copy source code
COPY src/ ./src/
COPY config.example.yaml ./
COPY data/ ./data/

# Stage 4: Production
FROM base as production

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libssl3 \
    libmp3lame0 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from deps stage
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application from builder
COPY --from=builder /app /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/.cache

# Set ownership
RUN useradd -m -u 1000 aura && \
    chown -R aura:aura /app
USER aura:aura

# Expose ports
# API Server
EXPOSE 5000
# Dashboard/Web UI
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Entry point
WORKDIR /app
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--production", "--api-port", "5000", "--dashboard-port", "8080"]

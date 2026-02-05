# Multi-stage Dockerfile for RAG Engine
# =====================================
# Optimized for production deployment

# Stage 1: Builder
# ===============
FROM python:3.10-slim as builder

# Set build arguments
ARG BUILD_VERSION=unknown
ARG BUILD_DATE=unknown

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /opt/app/src/
COPY config/ /opt/app/config/
COPY pyproject.toml /opt/app/

# Build metadata file
RUN echo "version=${BUILD_VERSION}" > /opt/app/version.txt && \
    echo "date=${BUILD_DATE}" >> /opt/app/version.txt

# Stage 2: Runtime
# ================
FROM python:3.10-slim as runtime

# Set labels for metadata
LABEL maintainer="RAG Engine Team" \
      version="${BUILD_VERSION}" \
      build-date="${BUILD_DATE}" \
      org.opencontainers.image.title="RAG Engine" \
      org.opencontainers.image.description="Production-ready RAG API" \
      org.opencontainers.image.version="${BUILD_VERSION}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME="/opt/app" \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /opt/app /opt/app

# Create non-root user for security
RUN useradd -r -u 1000 -m -s /sbin/nologin -c "RAG Engine User" raguser && \
    mkdir -p /opt/app/logs /opt/app/uploads && \
    chown -R raguser:raguser /opt/app

# Switch to non-root user
USER raguser

# Set working directory
WORKDIR /opt/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn (production WSGI server)
CMD ["gunicorn", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--worker-tmp-dir", "/dev/shm", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "src.main:app"]

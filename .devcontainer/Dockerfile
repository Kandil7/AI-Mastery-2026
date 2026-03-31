FROM mcr.microsoft.com/devcontainers/python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml requirements*.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Copy project
COPY . .

# Create non-root user
RUN usermod -a -G vscode vscode

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["bash"]

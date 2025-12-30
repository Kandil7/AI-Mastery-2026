# Dockerfile for AI-Mastery-2026

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.production.api:app", "--host", "0.0.0.0", "--port", "8000"]
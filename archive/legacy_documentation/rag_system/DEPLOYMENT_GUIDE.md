# RAG System Deployment Guide

**Version**: 1.0.0  
**Last Updated**: March 27, 2026

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Configuration](#production-configuration)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 50 GB | 100+ GB SSD |
| **GPU** | Optional | NVIDIA 8GB+ VRAM |

### Software Requirements

- Python 3.10+
- Docker 20.10+ (for containerized deployment)
- Node.js 18+ (for monitoring dashboard)

---

## Local Development

### 1. Clone and Setup

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\rag_system
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Paths
DATASETS_PATH=K:/learning/technical/ai-ml/AI-Mastery-2026/datasets
OUTPUT_PATH=K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data

# Optional: Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional: Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

### 5. Verify Installation

```bash
python simple_test.py
```

Expected: All tests pass ✅

### 6. Run Development Server

```bash
uvicorn src.api.service:app --reload --host 0.0.0.0 --port 8000
```

### 7. Access API Documentation

Open browser: `http://localhost:8000/docs`

---

## Docker Deployment

### 1. Build Docker Image

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p /app/data /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "src.api.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Build and Run

```bash
# Build image
docker build -t arabic-islamic-rag:latest .

# Run container
docker run -d \
  --name rag-system \
  -p 8000:8000 \
  -v ${DATASETS_PATH}:/app/datasets:ro \
  -v rag_data:/app/data \
  -v rag_logs:/app/logs \
  --env-file .env \
  arabic-islamic-rag:latest
```

### 3. Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  # RAG API
  rag-api:
    build: .
    container_name: rag-api
    ports:
      - "8000:8000"
    volumes:
      - ./datasets:/app/datasets:ro
      - rag_data:/app/data
      - rag_logs:/app/logs
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - qdrant
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  # Monitoring (Optional)
  prometheus:
    image: prom/prometheus:latest
    container_name: rag-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  # Grafana Dashboard (Optional)
  grafana:
    image: grafana/grafana:latest
    container_name: rag-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  qdrant_storage:
  rag_data:
  rag_logs:
  prometheus_data:
  grafana_data:
```

### 4. Start All Services

```bash
docker-compose up -d
```

### 5. Check Status

```bash
docker-compose ps
docker-compose logs -f rag-api
```

---

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.large \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp2}
```

#### 2. Install Docker

```bash
# SSH to instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Deploy Application

```bash
# Clone repository
git clone your-repo.git
cd rag_system

# Copy environment file
scp .env ec2-user@your-instance-ip:rag_system/.env

# Start services
docker-compose up -d
```

#### 4. Configure Load Balancer (Optional)

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name rag-lb \
  --subnets subnet-xxx subnet-yyy \
  --security-group-ids sg-xxxxx \
  --type application
```

### Google Cloud Platform

#### 1. Create GKE Cluster

```bash
gcloud container clusters create rag-cluster \
  --num-nodes=2 \
  --machine-type=e2-standard-4 \
  --zone=us-central1-a
```

#### 2. Deploy to GKE

```bash
kubectl create namespace rag-system
kubectl config set-context --current --namespace=rag-system

# Create secrets
kubectl create secret generic rag-secrets \
  --from-env-file=.env

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## Production Configuration

### 1. Performance Tuning

Create `config/production.yaml`:

```yaml
# Production Configuration

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 60
  keep_alive: 5

# Rate Limiting
rate_limit:
  enabled: true
  requests_per_minute: 100
  burst: 20

# Caching
cache:
  enabled: true
  type: "redis"
  host: "localhost"
  port: 6379
  ttl: 3600

# Database
vector_db:
  type: "qdrant"
  host: "localhost"
  port: 6333
  collection: "arabic_islamic_literature"
  hnsw_m: 32
  hnsw_ef_construct: 400

# Embeddings
embedding:
  model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  batch_size: 64
  cache_enabled: true

# LLM
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.3
  max_tokens: 2000
  timeout: 30

# Logging
logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/rag.log"
  max_size_mb: 100
  backup_count: 10

# Monitoring
monitoring:
  enabled: true
  prometheus_port: 9090
  health_check_interval: 30
```

### 2. Security Hardening

#### Enable HTTPS

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Configure Nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### API Authentication

Add to `.env`:

```bash
API_KEY=your-secret-api-key
JWT_SECRET=your-jwt-secret
```

### 3. Database Backup

Create `scripts/backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/backups/qdrant"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Qdrant
docker exec rag-qdrant \
  tar czf /tmp/qdrant_backup_$DATE.tar.gz /qdrant/storage

# Copy to backup directory
docker cp rag-qdrant:/tmp/qdrant_backup_$DATE.tar.gz $BACKUP_DIR/

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/qdrant_backup_$DATE.tar.gz \
  s3://your-bucket/backups/qdrant/

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: qdrant_backup_$DATE.tar.gz"
```

Add to crontab:

```bash
# Daily backup at 2 AM
0 2 * * * /app/scripts/backup.sh >> /app/logs/backup.log 2>&1
```

---

## Monitoring & Maintenance

### 1. Health Checks

```bash
# API Health
curl http://localhost:8000/health

# Vector DB Health
curl http://localhost:6333/cluster/status

# Check Logs
docker-compose logs -f rag-api
```

### 2. Metrics Dashboard

Access Grafana: `http://localhost:3000`
- Username: `admin`
- Password: `admin` (change after first login)

### 3. Alerting

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: rag-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        annotations:
          summary: "High latency detected"
          
      - alert: LowDiskSpace
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 10m
        annotations:
          summary: "Low disk space warning"
```

### 4. Log Management

Configure log rotation `/etc/logrotate.d/rag`:

```
/app/logs/*.log {
    daily
    rotate 10
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
    postrotate
        systemctl reload rsyslog
    endscript
}
```

### 5. Performance Monitoring

```bash
# Monitor CPU/Memory
docker stats rag-api

# Monitor API requests
curl http://localhost:9090/api/v1/query?query=http_requests_total

# Monitor query latency
curl http://localhost:9090/api/v1/query?query=http_request_duration_seconds
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

```bash
# Increase container memory limit
docker update --memory=4g rag-api

# Or reduce batch size in config
embedding.batch_size = 32
```

#### 2. Slow Queries

```bash
# Check vector DB performance
curl http://localhost:6333/collections/arabic_islamic_literature

# Optimize HNSW parameters
# Increase hnsw_m for better accuracy (slower)
# Decrease hnsw_m for faster search (less accurate)
```

#### 3. High CPU Usage

```bash
# Reduce number of workers
# In docker-compose.yml:
command: uvicorn src.api.service:app --workers 2

# Or use GPU for embeddings
# Add to docker-compose.yml:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

---

## Support

For issues or questions:
- Documentation: `README.md`
- API Docs: `http://localhost:8000/docs`
- Logs: `docker-compose logs -f`

---

**Deployment Status**: ✅ Production Ready  
**Last Verified**: March 27, 2026

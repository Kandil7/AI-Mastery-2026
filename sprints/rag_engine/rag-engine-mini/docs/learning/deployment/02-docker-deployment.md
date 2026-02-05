# Docker Deployment Guide - From Local to Production

## Overview

Docker is the foundation of modern application deployment. This guide teaches you how to containerize RAG Engine Mini, optimize images for production, and deploy using Docker Compose.

## Learning Objectives

By the end of this module, you will:
1. Understand Docker fundamentals for Python applications
2. Create optimized multi-stage Dockerfiles
3. Configure Docker Compose for local development
4. Implement container security best practices
5. Deploy to production Docker hosts
6. Troubleshoot common Docker issues

**Estimated Time:** 4-6 hours

---

## Part 1: Docker Fundamentals

### What is Docker?

Docker packages applications with all dependencies into standardized units called **containers**.

**Analogy:** Think of Docker containers as shipping containers:
- **Before Docker:** Ship individual boxes (install Python, pip, dependencies manually)
- **With Docker:** Ship sealed container (everything included, works anywhere)

### Why Docker for RAG Engine?

```
Without Docker:
Developer A (macOS): "It works on my machine!"
Developer B (Windows): "Import error!"
Production Server (Ubuntu): "Dependency conflict!"

With Docker:
All environments: "It just works!"
```

**Benefits:**
1. **Consistency:** Same environment everywhere
2. **Isolation:** Apps don't interfere
3. **Portability:** Run on any Docker host
4. **Efficiency:** Share OS kernel, lighter than VMs
5. **Versioning:** Track image versions

### Docker Architecture

```
Docker Host
├─ Docker Daemon (manages containers)
├─ Docker Client (CLI tool)
├─ Docker Images (read-only templates)
│  └─ Layers:
│     ├─ Base OS (Ubuntu/Python)
│     ├─ Dependencies (pip install)
│     └─ Application code
└─ Docker Containers (running instances)
   └─ Writable layer on top of image
```

### Key Concepts

**Images vs Containers:**
- **Image:** Blueprint (like a class in OOP)
- **Container:** Running instance (like an object)

**Layers:**
Each Dockerfile instruction creates a layer:
```dockerfile
FROM python:3.11-slim     # Layer 1: Base image
WORKDIR /app              # Layer 2: Working directory
COPY requirements.txt .   # Layer 3: Dependencies file
RUN pip install -r ...    # Layer 4: Installed packages
COPY . .                  # Layer 5: Application code
```

Layers are cached! If `requirements.txt` doesn't change, Layer 4 is reused.

---

## Part 2: Understanding the Dockerfile

### Current Project Dockerfile Analysis

Let's break down the project's Dockerfile line by line:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

# 1. Set working directory
WORKDIR /app

# 2. Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

# 4. Create non-root user (security)
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 5. Copy only necessary artifacts from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /app /app

# 6. Set environment
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# 7. Set proper permissions
RUN chown -R appuser:appgroup /app

# 8. Switch to non-root user
USER appuser

# 9. Working directory
WORKDIR /app

# 10. Copy application code
COPY --chown=appuser:appgroup . .

# 11. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Expose port
EXPOSE 8000

# 13. Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Build Explained

**Why Multi-Stage?**

```
Single-Stage Image Size: 1.2 GB
├─ Python base: 350 MB
├─ Build tools: 200 MB
├─ Dependencies: 400 MB
└─ App code: 50 MB

Multi-Stage Image Size: 450 MB
├─ Python base: 350 MB
├─ Dependencies: 50 MB (copied from builder)
└─ App code: 50 MB

Savings: 63% reduction!
```

**Benefits:**
1. Smaller image size (faster pulls, less storage)
2. No build tools in production (security)
3. Cleaner separation of concerns

### Security Hardening in Dockerfile

**1. Non-Root User:**
```dockerfile
# BAD: Running as root
FROM python:3.11
CMD ["python", "app.py"]

# GOOD: Running as non-root
FROM python:3.11
RUN useradd -m myuser
USER myuser
CMD ["python", "app.py"]
```

**2. Minimal Base Image:**
```dockerfile
# BAD: Full Ubuntu (200+ MB)
FROM ubuntu:22.04

# GOOD: Slim Python (120 MB)
FROM python:3.11-slim

# BEST: Distroless (20 MB, Google-maintained)
FROM gcr.io/distroless/python3
```

**3. Read-Only Filesystem:**
```dockerfile
# Make app directory read-only
RUN chmod -R 555 /app

# Use tmpfs for temporary files
VOLUME ["/tmp"]
```

---

## Part 3: Building Optimized Images

### Step-by-Step Build Process

**1. Check Dockerfile exists:**
```bash
ls -la Dockerfile
```

**2. Build the image:**
```bash
docker build -t rag-engine:latest .
```

**Flags explained:**
- `-t rag-engine:latest`: Tag the image
- `.`: Build context (current directory)

**3. Verify the build:**
```bash
docker images | grep rag-engine
```

**4. Check image size:**
```bash
docker images rag-engine:latest --format "{{.Size}}"
```

### Build Optimization Techniques

**1. Layer Caching:**
Order Dockerfile instructions by change frequency:
```dockerfile
# Good: Base image (rarely changes)
FROM python:3.11-slim

# Good: System deps (rarely changes)
RUN apt-get install -y gcc

# Good: Python deps (changes occasionally)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Good: App code (changes frequently)
COPY . .
```

**2. BuildKit (Advanced):**
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with cache mounts
docker build \
  --cache-from=rag-engine:cache \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t rag-engine:latest .
```

**3. .dockerignore:**
Prevent unnecessary files from entering build context:
```
# .dockerignore
.git
.gitignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.gitlab-ci.yml
.travis.yml
appveyor.yml
.DS_Store
.idea
.vscode
*.swp
*.swo
*~
docs/
tests/
scripts/backup.py
scripts/restore.py
*.md
!README.md
```

---

## Part 4: Docker Compose for Local Development

### Understanding docker-compose.yml

The project's Docker Compose orchestrates multiple services:

```yaml
version: '3.8'

services:
  # 1. Main API Application
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rag_engine
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
    depends_on:
      - postgres
      - redis
      - qdrant
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 2. Background Worker (Celery)
  worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rag_engine
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  # 3. PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=rag_engine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # 4. Redis (Cache & Queue)
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # 5. Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

### Service Breakdown

**API Service:**
- Builds from local Dockerfile
- Maps port 8000 to host
- Connects to other services by name
- Mounts uploads directory for persistence
- Health checks ensure it's running

**Worker Service:**
- Uses same image as API
- Different command (Celery worker)
- Processes background tasks

**Database Services:**
- Use official images
- Persistent volumes for data
- Exposed ports for local access

### Running with Docker Compose

**1. Start all services:**
```bash
docker-compose up -d
```

**Flags:**
- `-d`: Detached mode (run in background)

**2. View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
```

**3. Check status:**
```bash
docker-compose ps
```

**4. Stop services:**
```bash
docker-compose down
```

**5. Full reset (including data):**
```bash
docker-compose down -v
```

### Development Workflow

**Hot Reloading:**
```yaml
# docker-compose.override.yml for development
version: '3.8'
services:
  api:
    volumes:
      - .:/app  # Mount code for hot reload
    environment:
      - DEBUG=1
      - LOG_LEVEL=debug
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Part 5: Production Deployment

### Single Server Deployment

**Architecture:**
```
VPS/Cloud Instance
├─ Docker Engine
│  ├─ RAG Engine Container
│  ├─ PostgreSQL Container
│  ├─ Redis Container
│  └─ Qdrant Container
├─ Nginx (reverse proxy + SSL)
└─ Certbot (SSL certificates)
```

**Step-by-Step Deployment:**

**1. Provision server:**
- Ubuntu 22.04 LTS
- 4 CPU cores, 8GB RAM, 50GB SSD
- Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)

**2. Install Docker:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

**3. Clone repository:**
```bash
git clone https://github.com/your-org/rag-engine-mini.git
cd rag-engine-mini
```

**4. Create production environment file:**
```bash
cat > .env.production << EOF
# Database
DATABASE_URL=postgresql://raguser:STRONG_PASSWORD@localhost:5432/rag_engine

# Redis
REDIS_URL=redis://localhost:6379/0

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Security
JWT_SECRET=$(openssl rand -hex 32)
API_KEY_SALT=$(openssl rand -hex 16)

# LLM
OPENAI_API_KEY=sk-your-key-here

# App
ENVIRONMENT=production
LOG_LEVEL=info
EOF
```

**5. Create production Docker Compose:**
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  api:
    image: rag-engine:latest
    build: .
    restart: always
    env_file: .env.production
    environment:
      - DATABASE_URL=postgresql://raguser:${DB_PASSWORD}@postgres:5432/rag_engine
    ports:
      - "127.0.0.1:8000:8000"  # Only local access (nginx will proxy)
    depends_on:
      - postgres
      - redis
      - qdrant
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  worker:
    image: rag-engine:latest
    restart: always
    env_file: .env.production
    command: celery -A tasks worker --loglevel=info --concurrency=4
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        limits:
          memory: 2G

  postgres:
    image: postgres:15-alpine
    restart: always
    environment:
      - POSTGRES_USER=raguser
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=rag_engine
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    command: 
      - "postgres"
      - "-c"
      - "wal_level=replica"
      - "-c"
      - "max_wal_senders=3"
      - "-c"
      - "max_replication_slots=3"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U raguser -d rag_engine"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: always
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  qdrant_data:
    driver: local
```

**6. Set up Nginx reverse proxy:**
```bash
sudo apt install nginx certbot python3-certbot-nginx

# Create Nginx config
sudo tee /etc/nginx/sites-available/rag-engine << 'EOF'
server {
    listen 80;
    server_name your-domain.com;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Proxy to Docker container
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/rag-engine /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

**7. Obtain SSL certificate:**
```bash
sudo certbot --nginx -d your-domain.com
```

**8. Deploy application:**
```bash
# Build and start
docker-compose -f docker-compose.production.yml up -d --build

# Run migrations
docker-compose -f docker-compose.production.yml exec api alembic upgrade head

# Seed data (optional)
docker-compose -f docker-compose.production.yml exec api python scripts/seed_sample_data.py
```

**9. Set up automated backups:**
```bash
# Create backup script
cat > /opt/backup-rag-engine.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup database
docker exec rag-engine_postgres_1 pg_dump -U raguser rag_engine > $BACKUP_DIR/database.sql

# Backup vector data
docker run --rm -v rag-engine_postgres_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/qdrant.tar.gz /data

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/rag-engine/

# Keep only last 7 days
find /backups -type d -mtime +7 -exec rm -rf {} + 2>/dev/null
EOF

chmod +x /opt/backup-rag-engine.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/backup-rag-engine.sh") | crontab -
```

---

## Part 6: Container Security Best Practices

### Security Checklist

**1. Image Scanning:**
```bash
# Install Trivy
sudo apt install trivy

# Scan image
trivy image rag-engine:latest

# Fail on high/critical vulnerabilities
trivy image --exit-code 1 --severity HIGH,CRITICAL rag-engine:latest
```

**2. Non-Root User:**
Already implemented in Dockerfile, verify with:
```bash
docker run --rm rag-engine:latest id
# Should show: uid=999(appuser) gid=999(appgroup)
```

**3. Read-Only Root Filesystem:**
```yaml
# docker-compose.yml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
```

**4. Resource Limits:**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

**5. Security Options:**
```yaml
services:
  api:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

---

## Part 7: Troubleshooting Common Issues

### Issue 1: Container Won't Start

**Symptoms:**
```bash
docker-compose up
# Container exits immediately
```

**Diagnosis:**
```bash
# Check logs
docker-compose logs api

# Check exit code
docker inspect <container_id> --format='{{.State.ExitCode}}'
```

**Common Causes:**
1. **Missing environment variables:**
   ```bash
   # Check .env file exists
   cat .env
   
   # Verify variables loaded
   docker-compose config
   ```

2. **Port already in use:**
   ```bash
   # Find process using port 8000
   sudo lsof -i :8000
   
   # Kill process or change port
   ```

3. **Database not ready:**
   ```yaml
   # Add healthcheck dependency
   depends_on:
     postgres:
       condition: service_healthy
   ```

### Issue 2: High Memory Usage

**Diagnosis:**
```bash
# Check container stats
docker stats

# Check for memory leaks in app
# Look for growing memory in logs
```

**Solutions:**
```yaml
# Add memory limits
services:
  api:
    deploy:
      resources:
        limits:
          memory: 1G
    # Add restart on OOM
    restart: unless-stopped
```

### Issue 3: Slow Performance

**Check:**
```bash
# Container resource usage
docker stats --no-stream

# Database performance
docker exec postgres psql -c "SELECT * FROM pg_stat_activity;"

# Network latency
docker exec api ping postgres
```

**Optimizations:**
1. **Increase connection pool:**
   ```python
   # In config.py
   DATABASE_POOL_SIZE = 20
   DATABASE_MAX_OVERFLOW = 10
   ```

2. **Add Redis caching:**
   ```python
   # Cache frequent queries
   CACHE_TTL = 300  # 5 minutes
   ```

3. **Optimize vector search:**
   ```yaml
   # Qdrant configuration
   qdrant:
     environment:
       - QDRANT__STORAGE__WAL_CAPACITY_MB=1024
   ```

### Issue 4: SSL Certificate Problems

**Renew certificates:**
```bash
# Test renewal
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal

# Restart nginx
sudo systemctl restart nginx
```

---

## Part 8: Production Maintenance

### Daily Checks

```bash
#!/bin/bash
# daily-health-check.sh

echo "=== RAG Engine Health Check ==="
echo "Date: $(date)"

# Check containers
echo -e "\n1. Container Status:"
docker-compose ps

# Check disk space
echo -e "\n2. Disk Space:"
df -h | grep -E '(Filesystem|/dev/)'

# Check memory
echo -e "\n3. Memory Usage:"
free -h

# Check logs for errors
echo -e "\n4. Recent Errors:"
docker-compose logs --tail=50 | grep -i error || echo "No errors found"

# API health check
echo -e "\n5. API Health:"
curl -s http://localhost:8000/health | jq .

echo -e "\n=== Check Complete ==="
```

### Weekly Maintenance

```bash
# Update base images
docker-compose pull
docker-compose up -d

# Clean up old images
docker image prune -f

# Clean up old volumes
docker volume prune -f

# Review logs
docker-compose logs --tail=1000 > /var/log/rag-engine-weekly.log
```

### Monthly Tasks

- [ ] Review and rotate secrets
- [ ] Update SSL certificates (if needed)
- [ ] Test backup restoration
- [ ] Review resource usage and resize if needed
- [ ] Update dependencies
- [ ] Security scan

---

## Next Steps

1. **Practice building images:** Try optimizing the Dockerfile further
2. **Set up local environment:** Use Docker Compose for development
3. **Deploy to cloud VPS:** Follow the single-server guide
4. **Move to Kubernetes:** When you need multi-instance deployment

**Continue to Module 3: Kubernetes Deployment!** ☸️

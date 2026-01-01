# Cloud Deployment Guide

Deploy AI-Mastery-2026 to production on popular cloud platforms.

---

## Quick Deploy Options

| Platform | Deploy Command | Cost |
|----------|----------------|------|
| **Render** | Connect GitHub → Auto-deploy | Free tier |
| **Railway** | `railway up` | $5/mo |
| **Fly.io** | `fly launch` | Free tier |
| **AWS ECS** | See below | Pay-per-use |

---

## 1. Render Deployment (Recommended)

### Steps

1. **Connect Repository**
   - Go to [render.com](https://render.com)
   - New → Web Service → Connect GitHub
   - Select `AI-Mastery-2026` repo

2. **Configure Service**
   ```
   Name: ai-mastery-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.production.api:app --host 0.0.0.0 --port $PORT
   ```

3. **Environment Variables**
   ```
   OPENAI_API_KEY=sk-...
   ENVIRONMENT=production
   ```

4. **Deploy**
   - Auto-deploys on every push to main

### render.yaml (Optional)
```yaml
services:
  - type: web
    name: ai-mastery-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.production.api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ENVIRONMENT
        value: production
```

---

## 2. Railway Deployment

### Steps

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# Open dashboard
railway open
```

### railway.json
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn src.production.api:app --host 0.0.0.0 --port $PORT"
  }
}
```

---

## 3. Docker Deployment (Any Cloud)

### Build & Push
```bash
# Build
docker build -t ai-mastery:latest .

# Tag for registry
docker tag ai-mastery:latest your-registry.com/ai-mastery:latest

# Push
docker push your-registry.com/ai-mastery:latest
```

### Run on AWS ECS
```bash
# Create cluster
aws ecs create-cluster --cluster-name ai-mastery-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task.json

# Create service
aws ecs create-service \
  --cluster ai-mastery-cluster \
  --service-name ai-mastery-api \
  --task-definition ai-mastery-task \
  --desired-count 2
```

---

## 4. Environment Configuration

### Required Variables
```bash
# Core
ENVIRONMENT=production
LOG_LEVEL=INFO

# API Keys (if using external LLMs)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...

# Database (optional)
DATABASE_URL=postgresql://...

# Redis (optional, for caching)
REDIS_URL=redis://...
```

### Production Settings
```python
# config/production.py
CORS_ORIGINS = ["https://yourdomain.com"]
RATE_LIMIT = 100  # requests per minute
CACHE_TTL = 3600  # 1 hour
```

---

## 5. Monitoring Setup

### Prometheus Endpoint
```
GET /metrics → Prometheus-compatible metrics
```

### Health Check
```bash
curl https://your-api.com/health
# {"status": "healthy", "version": "1.0.0"}
```

### Recommended Alerts
```yaml
# prometheus/alerts.yml
groups:
  - name: api
    rules:
      - alert: HighLatency
        expr: http_request_duration_seconds_p95 > 1
        for: 5m
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) > 0.05
        for: 2m
```

---

## 6. Scaling Considerations

| Load | Instances | Memory |
|------|-----------|--------|
| < 100 req/min | 1 | 512MB |
| 100-1000 req/min | 2-4 | 1GB |
| > 1000 req/min | 4+ | 2GB+ |

### Auto-scaling (AWS)
```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/ai-mastery-cluster/ai-mastery-api \
  --min-capacity 1 \
  --max-capacity 10
```

---

## Quick Verification

After deployment:

```bash
# Health check
curl https://your-api.com/health

# Test prediction
curl -X POST https://your-api.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4, 5]}'

# Check metrics
curl https://your-api.com/metrics
```

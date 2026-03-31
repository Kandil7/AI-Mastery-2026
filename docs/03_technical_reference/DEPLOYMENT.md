# Deployment Guide

Deploying AI-Mastery-2026 to production environments.

---

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Kubernetes Deployment](#kubernetes-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Checklist](#production-checklist)

---

## Docker Deployment

### Basic Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml requirements*.txt ./
RUN pip install --no-cache-dir -e ".[prod]"

# Copy application
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.production.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t ai-mastery:latest .

# Run container
docker run -d -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e API_HOST=0.0.0.0 \
  ai-mastery:latest

# With volume for models
docker run -d -p 8000:8000 \
  -v ./models:/app/models \
  -v ./data:/app/data \
  ai-mastery:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=ai_mastery
      - POSTGRES_USER=ai_mastery
      - POSTGRES_PASSWORD=changeme
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  redis_data:
  postgres_data:
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-mastery-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-mastery
  template:
    metadata:
      labels:
        app: ai-mastery
    spec:
      containers:
      - name: api
        image: ai-mastery:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: ai-mastery-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Manifest

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-mastery-service
spec:
  selector:
    app: ai-mastery
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-mastery-config
data:
  redis-url: "redis://redis-master:6379"
  log-level: "INFO"
  max-workers: "4"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-mastery-secrets
type: Opaque
stringData:
  database-password: "your-secure-password"
  api-key: "your-api-key"
```

### Deploy

```bash
# Apply manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/ai-mastery-api
```

---

## Cloud Deployment

### AWS (ECS/Fargate)

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag ai-mastery:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ai-mastery:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ai-mastery:latest

# Deploy to ECS
aws ecs update-service --cluster ai-mastery --service api --force-new-deployment
```

### GCP (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/ai-mastery

# Deploy to Cloud Run
gcloud run deploy ai-mastery \
  --image gcr.io/<project-id>/ai-mastery \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure (Container Apps)

```bash
# Build and push to ACR
az acr build --registry <registry-name> --image ai-mastery:latest .

# Deploy to Container Apps
az containerapp up \
  --name ai-mastery \
  --resource-group my-resource-group \
  --image <registry-name>.azurecr.io/ai-mastery:latest \
  --target-port 8000 \
  --ingress external
```

---

## Production Checklist

### Security

- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up authentication/authorization
- [ ] Rotate API keys and secrets
- [ ] Enable rate limiting
- [ ] Configure security headers
- [ ] Set up WAF (Web Application Firewall)
- [ ] Enable audit logging

### Performance

- [ ] Configure connection pooling
- [ ] Set up caching (Redis/Memcached)
- [ ] Enable compression
- [ ] Configure worker threads/processes
- [ ] Set up CDN for static assets
- [ ] Configure database indexes
- [ ] Enable query optimization

### Monitoring

- [ ] Set up health check endpoints
- [ ] Configure metrics collection (Prometheus)
- [ ] Set up log aggregation (ELK/Loki)
- [ ] Configure alerting (PagerDuty/Slack)
- [ ] Set up distributed tracing
- [ ] Enable error tracking (Sentry)

### Reliability

- [ ] Configure auto-scaling
- [ ] Set up load balancing
- [ ] Enable circuit breakers
- [ ] Configure retry logic
- [ ] Set up backup strategy
- [ ] Test disaster recovery
- [ ] Configure graceful shutdown

### Database

- [ ] Enable connection pooling
- [ ] Set up read replicas
- [ ] Configure backup schedule
- [ ] Enable point-in-time recovery
- [ ] Set up migration strategy

---

## Environment Variables

### Required

```bash
# Environment
ENVIRONMENT=production

# API
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379
```

### Optional

```bash
# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
MAX_WORKERS=4
BATCH_SIZE=32

# Features
ENABLE_CACHING=true
ENABLE_MONITORING=true
```

---

## Scaling

### Horizontal Scaling

```bash
# Kubernetes
kubectl scale deployment ai-mastery-api --replicas=5

# Docker Swarm
docker service scale ai-mastery_api=5
```

### Vertical Scaling

Increase resources in deployment manifest:
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

---

## Troubleshooting

### Check pod status
```bash
kubectl describe pod <pod-name>
```

### View logs
```bash
kubectl logs -f <pod-name>
```

### Execute into container
```bash
kubectl exec -it <pod-name> -- /bin/bash
```

### Check resource usage
```bash
kubectl top pods
kubectl top nodes
```

---

**Last Updated:** March 31, 2026

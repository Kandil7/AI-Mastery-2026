# Infrastructure: Complete Guide

## Table of Contents
1. [Secrets Management](#secrets-management)
2. [Monitoring Stack](#monitoring-stack)
3. [CDN Integration](#cdn-integration)
4. [Connection Pooling](#connection-pooling)
5. [Horizontal Scaling](#horizontal-scaling)
6. [Disaster Recovery](#disaster-recovery)
7. [Cost Monitoring](#cost-monitoring)
8. [Compliance Auditing](#compliance-auditing)

---

## Secrets Management

### AWS Secrets Manager

```python
import boto3
import json
from typing import Any

def get_secret(secret_name: str) -> dict[str, Any]:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    
    response = client.get_secret_value(SecretId=secret_name)
    secret = response['SecretString']
    
    return json.loads(secret)

# Usage
OPENAI_API_KEY = get_secret('prod/openai/api_key')['api_key']
DATABASE_URL = get_secret('prod/database/url')['url']
```

### GCP Secret Manager

```python
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str) -> str:
    """Retrieve secret from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    
    return response.payload.data.decode('UTF-8')

# Usage
API_KEY = get_secret('my-project', 'openai-api-key')
```

### Azure Key Vault

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def get_secret(vault_url: str, secret_name: str) -> str:
    """Retrieve secret from Azure Key Vault."""
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    
    secret = client.get_secret(secret_name)
    return secret.value

# Usage
API_KEY = get_secret(
    'https://my-vault.vault.azure.net/',
    'openai-api-key'
)
```

---

## Monitoring Stack

### Docker Compose Stack

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    ports:
      - 9090:9090

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - 3000:3000

  loki:
    image: grafana/loki:latest
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./config/loki:/etc/loki
      - loki-data:/loki
    ports:
      - 3100:3100

  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=9411
    ports:
      - 5775:5775/udp
      - 6831:6831/udp
      - 6832:6832/udp
      - 16686:16686

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./config/prometheus/alerts.yml:/etc/alertmanager/config.yml
    ports:
      - 9093:9093

volumes:
  prometheus-data:
  grafana-data:
  loki-data:
```

### Prometheus Configuration

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-engine'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'
```

### Grafana Provisioning

```yaml
# config/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false

---
# config/grafana/provisioning/dashboards/rag-engine.yml
apiVersion: 1

providers:
  - name: 'RAG Engine'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

---

## CDN Integration

### CloudFlare CDN

```python
import requests

def invalidate_cdn_cache(file_paths: list[str], api_key: str):
    """Invalidate CloudFlare CDN cache."""
    zone_id = "your_zone_id"
    url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    data = {
        'files': file_paths,
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()

# Usage
invalidate_cdn_cache(
    ['/documents/test.pdf', '/documents/example.pdf'],
    api_key='your_api_key'
)
```

### AWS CloudFront

```python
import boto3

def invalidate_cloudfront(distribution_id: str, paths: list[str]):
    """Invalidate AWS CloudFront cache."""
    client = boto3.client('cloudfront')
    
    response = client.create_invalidation(
        DistributionId=distribution_id,
        InvalidationBatch={
            'Paths': {'Quantity': len(paths), 'Items': paths},
            'CallerReference': f'invalidation-{time.time()}',
        },
    )
    
    return response['Invalidation']['Id']

# Usage
invalidation_id = invalidate_cloudfront(
    'E1234567890ABCD',
    ['/documents/*', '/static/*']
)
```

---

## Connection Pooling

### SQLAlchemy Connection Pool

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,           # Max connections
    max_overflow=10,         # Overflow connections
    pool_recycle=3600,        # Recycle after 1 hour
    pool_pre_ping=True,       # Verify connections
    pool_use_lifo=True,       # Use LIFO
    echo=False,
)
```

### Redis Connection Pool

```python
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

client = redis.Redis(connection_pool=pool)
```

---

## Horizontal Scaling

### Kubernetes Deployment

```yaml
# config/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine
  labels:
    app: rag-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-engine
  template:
    metadata:
      labels:
        app: rag-engine
    spec:
      containers:
      - name: rag-engine
        image: ghcr.io/user/rag-engine:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: rag-engine-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: rag-engine-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-engine-service
spec:
  selector:
    app: rag-engine
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-engine-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-engine
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Terraform Infrastructure

```hcl
# config/terraform/main.tf
provider "aws" {
  region = var.aws_region
}

resource "aws_ecs_cluster" "rag_engine" {
  name = "rag-engine-cluster"
}

resource "aws_ecs_task_definition" "rag_engine" {
  family                   = "rag-engine"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  
  container_definitions = jsonencode([
    {
      name      = "rag-engine"
      image     = "ghcr.io/user/rag-engine:latest"
      cpu       = 256
      memory    = 512
      essential = true
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "DATABASE_URL"
          value = aws_secretsmanager_secret.database_url.secret_string
        }
      ]
    }
  ])
}

resource "aws_ecs_service" "rag_engine" {
  name            = "rag-engine-service"
  cluster         = aws_ecs_cluster.rag_engine.id
  task_definition = aws_ecs_task_definition.rag_engine.arn
  desired_count   = 3
  
  load_balancer {
    target_group_arn = aws_lb_target_group.rag_engine.arn
    container_name   = "rag-engine"
    container_port   = 8000
  }
}

resource "aws_appautoscaling_target" "rag_engine" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = aws_ecs_service.rag_engine.name
  scalable_dimension = aws_appautoscaling_policy.rag_engine.resource_id
  service_namespace  = aws_ecs_service.rag_engine.id
}

resource "aws_appautoscaling_policy" "rag_engine" {
  name               = "rag-engine-autoscaling"
  policy_type         = "TargetTrackingScaling"
  target_tracking_scaling_policy_configuration {
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

---

## Disaster Recovery

### Automated Backup Script

```python
import os
import subprocess
from datetime import datetime
import boto3

def backup_to_s3(bucket_name: str, database_url: str):
    """Backup database to S3."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"/tmp/rag_engine_backup_{timestamp}.sql"
    
    # Dump database
    subprocess.run([
        'pg_dump',
        database_url,
        '-f', backup_file,
    ], check=True)
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(
        backup_file,
        bucket_name,
        f"backups/rag_engine_{timestamp}.sql"
    )
    
    # Cleanup local file
    os.remove(backup_file)
    
    print(f"Backup completed: {backup_file}")

# Schedule with cron: 0 2 * * * (daily at 2 AM)
```

### Restore Script

```python
def restore_from_s3(bucket_name: str, backup_key: str, database_url: str):
    """Restore database from S3 backup."""
    # Download from S3
    s3 = boto3.client('s3')
    backup_file = f"/tmp/restore_{backup_key}"
    s3.download_file(bucket_name, backup_key, backup_file)
    
    # Restore database
    subprocess.run([
        'psql',
        database_url,
        '-f', backup_file,
    ], check=True)
    
    # Cleanup
    os.remove(backup_file)
    
    print(f"Restore completed: {backup_file}")
```

---

## Cost Monitoring

### AWS Cost Alerts

```python
import boto3

def setup_cost_alerts():
    """Setup AWS cost alerts."""
    client = boto3.client('budgets')
    
    # Create budget
    budget = client.create_budget(
        AccountId='123456789012',
        Budget={
            'BudgetName': 'rag-engine-monthly',
            'BudgetLimit': {'Amount': '500.00'},
            'TimeUnit': 'MONTHLY',
        }
    )
    
    # Create alert
    alert = client.create_notification(
        AccountId='123456789012',
        BudgetName='rag-engine-monthly',
        Notification={
            'NotificationType': 'ACTUAL',
            'NotificationThreshold': {
                'Absolute': {'Amount': '400.00'},
            },
            'Subscriber': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'admin@example.com',
                },
            ],
        }
    )
    
    return budget, alert
```

---

## Summary

| Component | Purpose | Tool |
|-----------|----------|-------|
| **Secrets** | Secure credential storage | AWS/GCP/Azure Secret Managers |
| **Monitoring** | Metrics, logs, traces | Prometheus + Grafana + Loki + Jaeger |
| **CDN** | Content delivery | CloudFlare, CloudFront |
| **Scaling** | Auto-scaling | Kubernetes HPA, AWS Auto Scaling |
| **Backup** | Disaster recovery | Automated scripts, S3/GCS |
| **Cost** | Expense monitoring | AWS Budgets, Cost Explorer |

---

## Further Reading

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform Documentation](https://www.terraform.io/docs/)
- [AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/)
- [GCP Secret Manager](https://cloud.google.com/secret-manager)

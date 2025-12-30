# Cloud Deployment Guide

Deploy AI-Mastery-2026 on major cloud providers.

---

## 1. AWS Deployment

### 1.1 ECS with Fargate

```yaml
# aws/task-definition.json
{
  "family": "ai-mastery-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-ecr-repo/ai-mastery:latest",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "DB_HOST", "value": "your-rds-endpoint"},
        {"name": "REDIS_HOST", "value": "your-elasticache-endpoint"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-mastery",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### 1.2 Terraform Infrastructure

```hcl
# aws/main.tf
provider "aws" {
  region = "us-east-1"
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "ai-mastery-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "ai-mastery-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "ai-mastery-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier        = "ai-mastery-db"
  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = "db.t3.medium"
  allocated_storage = 20
  
  db_name  = "ai_mastery"
  username = "postgres"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  skip_final_snapshot = true
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "ai-mastery-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
}
```

### 1.3 Deploy Commands

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO
docker build -t ai-mastery:latest .
docker tag ai-mastery:latest $ECR_REPO:latest
docker push $ECR_REPO:latest

# Deploy with Terraform
cd aws/
terraform init
terraform plan
terraform apply

# Update ECS service
aws ecs update-service --cluster ai-mastery-cluster --service ai-mastery-api --force-new-deployment
```

### 1.4 Cost Optimization (AWS)

| Component | On-Demand | Spot/Reserved | Savings |
|-----------|-----------|---------------|---------|
| ECS Fargate | $73/mo | $44/mo (Spot) | 40% |
| RDS t3.medium | $49/mo | $31/mo (RI) | 37% |
| ElastiCache | $12/mo | $8/mo (RI) | 33% |
| **Total** | **$134/mo** | **$83/mo** | **38%** |

---

## 2. Google Cloud Platform (GCP)

### 2.1 Cloud Run Deployment

```yaml
# gcp/service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ai-mastery-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: gcr.io/your-project/ai-mastery:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: "2"
              memory: "2Gi"
          env:
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: host
```

### 2.2 Terraform for GCP

```hcl
# gcp/main.tf
provider "google" {
  project = "your-project-id"
  region  = "us-central1"
}

# Cloud Run Service
resource "google_cloud_run_service" "api" {
  name     = "ai-mastery-api"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "gcr.io/${var.project}/ai-mastery:latest"
        
        resources {
          limits = {
            cpu    = "2"
            memory = "2Gi"
          }
        }
        
        env {
          name  = "DB_HOST"
          value = google_sql_database_instance.main.private_ip_address
        }
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.main.id
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "main" {
  name             = "ai-mastery-db"
  database_version = "POSTGRES_15"
  region           = "us-central1"

  settings {
    tier = "db-f1-micro"
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
    }
  }
}

# Memorystore Redis
resource "google_redis_instance" "cache" {
  name           = "ai-mastery-redis"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = "us-central1"
}
```

### 2.3 Deploy Commands (GCP)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/$PROJECT_ID/ai-mastery:latest

# Deploy to Cloud Run
gcloud run deploy ai-mastery-api \
  --image gcr.io/$PROJECT_ID/ai-mastery:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 3. Azure Deployment

### 3.1 Azure Container Apps

```yaml
# azure/container-app.yaml
name: ai-mastery-api
location: eastus
properties:
  managedEnvironmentId: /subscriptions/.../managedEnvironments/ai-mastery-env
  configuration:
    ingress:
      external: true
      targetPort: 8000
      traffic:
        - latestRevision: true
          weight: 100
    secrets:
      - name: db-connection
        value: ${DB_CONNECTION_STRING}
  template:
    containers:
      - name: api
        image: youracr.azurecr.io/ai-mastery:latest
        resources:
          cpu: 1
          memory: 2Gi
        env:
          - name: DB_HOST
            secretRef: db-connection
    scale:
      minReplicas: 1
      maxReplicas: 10
      rules:
        - name: http-rule
          http:
            metadata:
              concurrentRequests: "100"
```

### 3.2 Deploy Commands (Azure)

```bash
# Login and set subscription
az login
az account set --subscription "Your Subscription"

# Create resource group
az group create --name ai-mastery-rg --location eastus

# Create Container Registry
az acr create --resource-group ai-mastery-rg --name aimastery2026 --sku Basic

# Build and push
az acr build --registry aimastery2026 --image ai-mastery:latest .

# Deploy Container App
az containerapp create \
  --name ai-mastery-api \
  --resource-group ai-mastery-rg \
  --environment ai-mastery-env \
  --image aimastery2026.azurecr.io/ai-mastery:latest \
  --target-port 8000 \
  --ingress external
```

---

## 4. Kubernetes Deployment (Any Cloud)

### 4.1 Helm Chart

```yaml
# helm/ai-mastery/values.yaml
replicaCount: 3

image:
  repository: your-registry/ai-mastery
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 2000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: true
  auth:
    database: ai_mastery
    
redis:
  enabled: true
  architecture: standalone
```

### 4.2 Deploy with Helm

```bash
# Add Helm repo and update
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install
helm install ai-mastery ./helm/ai-mastery \
  --namespace ai-mastery \
  --create-namespace \
  --set image.tag=v1.0.0

# Upgrade
helm upgrade ai-mastery ./helm/ai-mastery \
  --namespace ai-mastery \
  --set image.tag=v1.1.0
```

---

## 5. Cost Comparison

| Provider | Small (Dev) | Medium (Prod) | Large (Scale) |
|----------|-------------|---------------|---------------|
| **AWS** | $50/mo | $200/mo | $800/mo |
| **GCP** | $40/mo | $180/mo | $750/mo |
| **Azure** | $55/mo | $210/mo | $820/mo |

*Includes: Compute, Database, Cache, Load Balancer, Bandwidth*

---

## 6. Monitoring Setup

### CloudWatch (AWS)
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "API-High-Latency" \
  --metric-name Latency \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 60 \
  --threshold 500 \
  --comparison-operator GreaterThanThreshold
```

### Cloud Monitoring (GCP)
```bash
gcloud monitoring policies create \
  --display-name="API Latency Alert" \
  --condition-display-name="High Latency" \
  --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_latencies"'
```

---

## Quick Reference

| Task | AWS | GCP | Azure |
|------|-----|-----|-------|
| Deploy | `aws ecs update-service` | `gcloud run deploy` | `az containerapp update` |
| Logs | CloudWatch Logs | Cloud Logging | Log Analytics |
| Secrets | Secrets Manager | Secret Manager | Key Vault |
| Registry | ECR | GCR/Artifact Registry | ACR |

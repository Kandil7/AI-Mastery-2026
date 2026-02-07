# GCP Deployment Guide - Production RAG Engine on Google Cloud Platform

## Overview

This guide walks you through deploying RAG Engine Mini on Google Cloud Platform (GCP) using industry best practices. We'll cover multiple deployment options from simple serverless to enterprise-grade Kubernetes.

## Learning Objectives

By the end of this guide, you will:
1. Understand GCP services for container deployment
2. Deploy using Cloud Run (serverless containers)
3. Set up GKE (Google Kubernetes Engine) for Kubernetes
4. Configure Cloud SQL for managed PostgreSQL
5. Implement Cloud Storage for document storage
6. Set up Cloud Load Balancing with SSL
7. Configure Cloud Monitoring and Logging
8. Implement auto-scaling policies
9. Manage costs effectively

**Estimated Time:** 6-8 hours
**Cost:** $150-600/month (depending on scale)

---

## Part 1: GCP Architecture Options

### Option 1: Cloud Run (Recommended for Beginners)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud Run Service                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Cloud Load Balancer (HTTPS)                 │   │
│  │            (SSL termination, routing)               │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              Cloud Run Revision                     │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Container│ │  API Container│ │  API Container│  │   │
│  │  │  (Instance 1)│ │  (Instance 2)│ │  (Instance 3)│  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  │       ┌──────────────┐                               │   │
│  │       │ Worker Job   │                               │   │
│  │       │ (Container)  │                               │   │
│  │       └──────────────┘                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────┼──────────────────────────────────┐   │
│  │                  │          VPC Network             │   │
│  │  ┌───────────────▼────────┐ ┌─────────────────────┐  │   │
│  │  │    Cloud SQL           │ │   Memorystore       │  │   │
│  │  │    (PostgreSQL)        │ │   (Redis)           │  │   │
│  │  └────────────────────────┘ └─────────────────────┘  │   │
│  │  ┌────────────────────────┐                          │   │
│  │  │    Cloud Storage       │                          │   │
│  │  │    (Documents)         │                          │   │
│  │  └────────────────────────┘                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Fully serverless (no infrastructure management)
- Automatic scaling to zero (cost-effective for low traffic)
- Simple deployment with `gcloud run deploy`
- Good for startups and small teams
- Supports 1,000-100,000 requests per second

**Pros:**
- No cluster management
- Pay only for requests (scales to zero)
- Automatic HTTPS
- Built-in traffic splitting

**Cons:**
- Request timeout limits (60 min max)
- Limited persistent storage options
- Cold start latency (~1-2 seconds)

### Option 2: GKE (Google Kubernetes Engine)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     GKE Cluster                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Cloud Load Balancer (Ingress)               │   │
│  │              (SSL, path routing)                    │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              GKE Autopilot / Standard               │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Pod     │ │  API Pod     │ │ Worker Pod   │  │   │
│  │  │  (Node 1)    │ │  (Node 2)    │ │ (Node 3)     │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  External GCP Services:                                     │
│  • Cloud SQL for PostgreSQL                                 │
│  • Memorystore for Redis                                    │
│  • Cloud Storage for document storage                       │
│  • Cloud Monitoring for observability                       │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Need Kubernetes ecosystem and portability
- Complex microservices architecture
- Multi-cloud strategy planned
- Need persistent volumes and StatefulSets

**Pros:**
- Full Kubernetes feature set
- GKE Autopilot manages nodes automatically
- Better for complex workloads
- Multi-region deployment support

**Cons:**
- Higher complexity
- Always running (minimum cost ~$70/month)
- Requires Kubernetes expertise

### Option 3: Cloud Run + Cloud Tasks (Hybrid)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     Hybrid Architecture                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Cloud Load Balancer                         │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              Cloud Run (API Layer)                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Inst. 1 │ │  API Inst. 2 │ │  API Inst. 3 │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Cloud Tasks Queue                       │   │
│  │         (Async job processing)                       │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │              Cloud Run Jobs                          │   │
│  │         (Background processing)                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Async processing needed (document ingestion)
- Mix of synchronous API and batch workloads
- Cost optimization for background jobs

---

## Part 2: Prerequisites

### Required Tools

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com \
    container.googleapis.com \
    sqladmin.googleapis.com \
    redis.googleapis.com \
    storage.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    vpcaccess.googleapis.com

# Verify installation
gcloud --version
```

### GCP Project Setup

```bash
# Create a new project (optional)
gcloud projects create rag-engine-prod \
    --name="RAG Engine Production"

# Set billing account
gcloud billing projects link rag-engine-prod \
    --billing-account=YOUR_BILLING_ACCOUNT_ID

# Set as default
gcloud config set project rag-engine-prod
```

### Service Account Setup

```bash
# Create service account for CI/CD
gcloud iam service-accounts create rag-engine-deployer \
    --display-name="RAG Engine Deployment Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding rag-engine-prod \
    --member="serviceAccount:rag-engine-deployer@rag-engine-prod.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding rag-engine-prod \
    --member="serviceAccount:rag-engine-deployer@rag-engine-prod.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding rag-engine-prod \
    --member="serviceAccount:rag-engine-deployer@rag-engine-prod.iam.gserviceaccount.com" \
    --role="roles/cloudsql.admin"

# Create and download key
gcloud iam service-accounts keys create deployer-key.json \
    --iam-account=rag-engine-deployer@rag-engine-prod.iam.gserviceaccount.com
```

---

## Part 3: Cloud Run Deployment

### Step 1: Build and Push Container

```bash
# Set variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="rag-engine-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Build container with Cloud Build
gcloud builds submit --tag ${IMAGE_NAME}:latest \
    --machine-type=e2-highcpu-8

# Or build locally and push
docker build -t ${IMAGE_NAME}:latest .
docker push ${IMAGE_NAME}:latest
```

### Step 2: Set Up Cloud SQL (PostgreSQL)

```bash
# Create Cloud SQL instance
INSTANCE_NAME="rag-engine-db"
gcloud sql instances create ${INSTANCE_NAME} \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=${REGION} \
    --storage-size=10GB \
    --storage-auto-increase \
    --backup-start-time="03:00" \
    --maintenance-window-day=SUN \
    --maintenance-window-hour=4

# Create database
gcloud sql databases create rag_engine --instance=${INSTANCE_NAME}

# Create user
gcloud sql users create rag_user \
    --instance=${INSTANCE_NAME} \
    --password=$(openssl rand -base64 32)

# Get connection name
CONNECTION_NAME=$(gcloud sql instances describe ${INSTANCE_NAME} \
    --format='value(connectionName)')

echo "Connection Name: ${CONNECTION_NAME}"
```

### Step 3: Set Up Memorystore (Redis)

```bash
# Create Redis instance
REDIS_NAME="rag-engine-redis"
gcloud redis instances create ${REDIS_NAME} \
    --tier=standard \
    --size=5 \
    --region=${REGION} \
    --redis-version=redis_6_x

# Get connection info
REDIS_HOST=$(gcloud redis instances describe ${REDIS_NAME} \
    --region=${REGION} \
    --format='value(host)')

REDIS_PORT=$(gcloud redis instances describe ${REDIS_NAME} \
    --region=${REGION} \
    --format='value(port)')

echo "Redis Host: ${REDIS_HOST}"
echo "Redis Port: ${REDIS_PORT}"
```

### Step 4: Set Up Cloud Storage

```bash
# Create storage bucket
BUCKET_NAME="${PROJECT_ID}-rag-documents"
gcloud storage buckets create gs://${BUCKET_NAME} \
    --location=${REGION} \
    --uniform-bucket-level-access

# Set lifecycle policy (optional - delete old versions)
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesStorageClass": "STANDARD"
        }
      }
    ]
  }
}
EOF

gcloud storage buckets set-lifecycle lifecycle.json gs://${BUCKET_NAME}
```

### Step 5: Create Secrets

```bash
# Store database password in Secret Manager
gcloud secrets create db-password \
    --data-file=<(gcloud sql users list --instance=${INSTANCE_NAME} \
        --format='table(name, password)' | grep rag_user | awk '{print $2}')

# Store other secrets
echo -n "your-jwt-secret" | gcloud secrets create jwt-secret --data-file=-
echo -n "your-llm-api-key" | gcloud secrets create llm-api-key --data-file=-
```

### Step 6: Deploy to Cloud Run

```bash
# Create service account for the service
gcloud iam service-accounts create rag-engine-sa \
    --display-name="RAG Engine Service Account"

# Grant access to secrets
gcloud secrets add-iam-policy-binding db-password \
    --member="serviceAccount:rag-engine-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding jwt-secret \
    --member="serviceAccount:rag-engine-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --region=${REGION} \
    --platform=managed \
    --service-account=rag-engine-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-cloudsql-instances=${CONNECTION_NAME} \
    --set-secrets="DB_PASSWORD=db-password:latest,JWT_SECRET=jwt-secret:latest,LLM_API_KEY=llm-api-key:latest" \
    --set-env-vars="ENVIRONMENT=production,DB_HOST=/cloudsql/${CONNECTION_NAME},DB_NAME=rag_engine,DB_USER=rag_user,REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},STORAGE_BUCKET=${BUCKET_NAME},GCP_PROJECT_ID=${PROJECT_ID}" \
    --memory=2Gi \
    --cpu=2 \
    --concurrency=80 \
    --max-instances=10 \
    --min-instances=1 \
    --timeout=300 \
    --allow-unauthenticated

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --format='value(status.url)')

echo "Service URL: ${SERVICE_URL}"
```

### Step 7: Configure VPC Access (for Redis)

```bash
# Create VPC connector
CONNECTOR_NAME="rag-engine-connector"
gcloud compute networks vpc-access connectors create ${CONNECTOR_NAME} \
    --region=${REGION} \
    --network=default \
    --range=10.8.0.0/28 \
    --min-instances=2 \
    --max-instances=10

# Update service with VPC connector
gcloud run services update ${SERVICE_NAME} \
    --region=${REGION} \
    --vpc-connector=${CONNECTOR_NAME} \
    --vpc-egress=private-ranges-only
```

---

## Part 4: GKE Deployment

### Step 1: Create GKE Cluster

```bash
# Option A: GKE Autopilot (Recommended)
gcloud container clusters create-auto rag-engine-cluster \
    --region=${REGION} \
    --release-channel=regular \
    --enable-vertical-pod-autoscaling \
    --workload-policies=allowed-repos

# Option B: GKE Standard (More control)
gcloud container clusters create rag-engine-cluster \
    --region=${REGION} \
    --release-channel=regular \
    --machine-type=e2-standard-4 \
    --num-nodes=3 \
    --enable-autoscaling \
    --min-nodes=3 \
    --max-nodes=10 \
    --enable-vertical-pod-autoscaling \
    --enable-autorepair \
    --enable-autoupgrade \
    --workload-pool=${PROJECT_ID}.svc.id.goog

# Get credentials
gcloud container clusters get-credentials rag-engine-cluster \
    --region=${REGION}
```

### Step 2: Install Ingress Controller

```bash
# Install GKE Ingress Controller (managed)
# Already included with GKE, just need to configure

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Wait for cert-manager
echo "Waiting for cert-manager..."
sleep 60
```

### Step 3: Configure Workload Identity

```bash
# Create Kubernetes service account
kubectl create serviceaccount rag-engine-ksa \
    --namespace=default

# Bind to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
    rag-engine-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/rag-engine-ksa]"

# Annotate Kubernetes SA
kubectl annotate serviceaccount rag-engine-ksa \
    --namespace=default \
    iam.gke.io/gcp-service-account=rag-engine-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### Step 4: Deploy Application

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-engine
  labels:
    istio-injection: enabled

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-engine-config
  namespace: rag-engine
data:
  ENVIRONMENT: "production"
  DB_NAME: "rag_engine"
  DB_USER: "rag_user"
  REDIS_PORT: "6379"
  STORAGE_BUCKET: "rag-engine-documents"
  LOG_LEVEL: "INFO"
  
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-engine-secrets
  namespace: rag-engine
type: Opaque
stringData:
  DB_PASSWORD: "<from-secret-manager>"
  JWT_SECRET: "<from-secret-manager>"
  LLM_API_KEY: "<from-secret-manager>"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine-api
  namespace: rag-engine
  labels:
    app: rag-engine
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-engine
      component: api
  template:
    metadata:
      labels:
        app: rag-engine
        component: api
    spec:
      serviceAccountName: rag-engine-ksa
      containers:
      - name: api
        image: gcr.io/${PROJECT_ID}/rag-engine-api:latest
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: rag-engine-config
        - secretRef:
            name: rag-engine-secrets
        env:
        - name: DB_HOST
          value: "10.0.0.3"  # Cloud SQL private IP
        - name: REDIS_HOST
          value: "10.0.0.4"  # Memorystore IP
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
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
        volumeMounts:
        - name: cloudsql-proxy
          mountPath: /cloudsql
      - name: cloudsql-proxy
        image: gcr.io/cloudsql-docker/gce-proxy:1.33.0
        command:
        - "/cloud_sql_proxy"
        - "-instances=${CONNECTION_NAME}=tcp:5432"
        - "-credential_file=/secrets/service-account.json"
        securityContext:
          runAsNonRoot: true
        volumeMounts:
        - name: service-account-key
          mountPath: /secrets
          readOnly: true
      volumes:
      - name: service-account-key
        secret:
          secretName: cloudsql-service-account-key

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-engine-service
  namespace: rag-engine
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: rag-engine
    component: api

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-engine-ingress
  namespace: rag-engine
  annotations:
    kubernetes.io/ingress.class: gce
    kubernetes.io/ingress.global-static-ip-name: rag-engine-ip
    networking.gke.io/managed-certificates: rag-engine-cert
    networking.gke.io/v1beta1.FrontendConfig: rag-engine-frontend-config
spec:
  rules:
  - host: api.rag-engine.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-engine-service
            port:
              number: 80

---
# k8s/managedcertificate.yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: rag-engine-cert
  namespace: rag-engine
spec:
  domains:
  - api.rag-engine.com

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-engine-hpa
  namespace: rag-engine
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-engine-api
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### Step 5: Deploy to GKE

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/managedcertificate.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=120s \
    deployment/rag-engine-api -n rag-engine

# Check status
kubectl get pods -n rag-engine
kubectl get svc -n rag-engine
kubectl get ingress -n rag-engine
```

---

## Part 5: Monitoring and Observability

### Cloud Monitoring Setup

```bash
# Create monitoring workspace (automatic with GCP)
# Navigate to: https://console.cloud.google.com/monitoring

# Create uptime check
gcloud monitoring uptime create rag-engine-health-check \
    --display-name="RAG Engine Health" \
    --protocol=https \
    --resource-type=cloud-run-revision \
    --resource-labels=service_name=rag-engine-api \
    --resource-labels=location=${REGION} \
    --request-path=/health

# Create alert policy
cat > alert-policy.json <<EOF
{
  "displayName": "High Error Rate",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "Error rate > 5%",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class!=\"2xx\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0.05,
        "duration": "0s",
        "trigger": {
          "count": 1
        }
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "86400s"
  },
  "severity": "ERROR",
  "notificationChannels": [
    "projects/${PROJECT_ID}/notificationChannels/YOUR_CHANNEL_ID"
  ]
}
EOF

gcloud alpha monitoring policies create --policy-from-file=alert-policy.json
```

### Cloud Logging

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-engine-api" \
    --limit=50 \
    --format="table(timestamp,severity,textPayload)"

# Create log-based metric
gcloud logging metrics create error-count \
    --description="Count of error logs" \
    --log-filter="severity>=ERROR AND resource.type=cloud_run_revision"

# Export logs to BigQuery (for analysis)
gcloud logging sinks create bigquery-export \
    bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/rag_logs \
    --log-filter="resource.type=cloud_run_revision"
```

### Custom Dashboard

```bash
# Create dashboard via gcloud
cat > dashboard.json <<EOF
{
  "displayName": "RAG Engine Dashboard",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Request Count",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_RATE"
                }
              }
            }
          }]
        }
      },
      {
        "title": "Latency",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_PERCENTILE_99"
                }
              }
            }
          }]
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=dashboard.json
```

---

## Part 6: Security Best Practices

### Cloud Armor (WAF)

```bash
# Create security policy
gcloud compute security-policies create rag-engine-policy \
    --description="WAF policy for RAG Engine"

# Add rules
gcloud compute security-policies rules create 1000 \
    --security-policy=rag-engine-policy \
    --expression="true" \
    --action="allow"

# Add SQL injection protection
gcloud compute security-policies rules create 2000 \
    --security-policy=rag-engine-policy \
    --expression="evaluatePreconfiguredExpr('sqli-stable')" \
    --action="deny(403)"

# Add XSS protection
gcloud compute security-policies rules create 2001 \
    --security-policy=rag-engine-policy \
    --expression="evaluatePreconfiguredExpr('xss-stable')" \
    --action="deny(403)"

# Add rate limiting
gcloud compute security-policies rules create 3000 \
    --security-policy=rag-engine-policy \
    --expression="true" \
    --action="rate_based_ban" \
    --rate-limit-threshold=100 \
    --ban-duration=3600 \
    --conform-action=allow \
    --exceed-action=deny(429)

# Attach to backend service (for GKE/Load Balancer)
gcloud compute backend-services update rag-engine-backend \
    --security-policy=rag-engine-policy \
    --global
```

### Private Google Access

```bash
# Enable Private Google Access for subnet
gcloud compute networks subnets update default \
    --region=${REGION} \
    --enable-private-ip-google-access

# Create Cloud NAT for outbound internet (if needed)
gcloud compute routers create nat-router \
    --region=${REGION} \
    --network=default

gcloud compute routers nats create nat-config \
    --router=nat-router \
    --region=${REGION} \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
```

### Binary Authorization

```bash
# Enable Binary Authorization
gcloud services enable binaryauthorization.googleapis.com

# Create policy
gcloud container binauthz policy create-standard \
    --platform=cloud-run

# Require attestations
gcloud container binauthz policy update-policy \
    --platform=cloud-run \
    --require-attestations
```

---

## Part 7: CI/CD Pipeline with Cloud Build

### cloudbuild.yaml

```yaml
steps:
  # Build container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/rag-engine-api:$COMMIT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/rag-engine-api:latest'
      - '.'
    id: 'build'

  # Run security scan
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud artifacts docker images scan \
          gcr.io/$PROJECT_ID/rag-engine-api:$COMMIT_SHA \
          --format='value(response.scan)'
    id: 'security-scan'
    waitFor: ['build']

  # Push to registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'gcr.io/$PROJECT_ID/rag-engine-api:$COMMIT_SHA'
    id: 'push'
    waitFor: ['build']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: bash
    args:
      - '-c'
      - |
        gcloud run deploy rag-engine-api \
          --image gcr.io/$PROJECT_ID/rag-engine-api:$COMMIT_SHA \
          --region us-central1 \
          --platform managed \
          --no-traffic \
          --tag $COMMIT_SHA
    id: 'deploy-staging'
    waitFor: ['push']

  # Integration tests
  - name: 'gcr.io/cloud-builders/curl'
    entrypoint: bash
    args:
      - '-c'
      - |
        # Run smoke tests
        curl -f https://staging-api.rag-engine.com/health || exit 1
    id: 'integration-tests'
    waitFor: ['deploy-staging']

  # Promote to production
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: bash
    args:
      - '-c'
      - |
        gcloud run services update-traffic rag-engine-api \
          --region us-central1 \
          --to-revisions $COMMIT_SHA=100
    id: 'promote-production'
    waitFor: ['integration-tests']

options:
  machineType: 'E2_HIGHCPU_8'
  
timeout: '1200s'

images:
  - 'gcr.io/$PROJECT_ID/rag-engine-api:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/rag-engine-api:latest'
```

### Trigger Setup

```bash
# Create build trigger
gcloud builds triggers create github \
    --name="rag-engine-deploy" \
    --repo-owner=your-org \
    --repo-name=rag-engine-mini \
    --branch-pattern="main" \
    --build-config="cloudbuild.yaml"

# Or use Cloud Source Repositories
gcloud builds triggers create cloud-source-repositories \
    --name="rag-engine-deploy" \
    --repo=rag-engine-mini \
    --branch-pattern="main" \
    --build-config="cloudbuild.yaml"
```

---

## Part 8: Cost Optimization

### Cost Breakdown (Monthly Estimates)

```
Cloud Run (1-10 instances, scales to zero):
- Compute: $0.000024/vCPU-s + $0.0000025/GiB-s
- Estimated: $50-200/month (varies by traffic)

Cloud SQL (db-f1-micro):
- Instance: $7-15/month
- Storage: $0.10/GB/month
- Estimated: $15-30/month

Memorystore (5GB):
- Instance: $30-40/month

Cloud Storage (100GB):
- Storage: $2.30/month
- Operations: $1-5/month
- Estimated: $5-10/month

Load Balancer:
- Forwarding rules: $18/month
- Processing: $0.008/GB
- Estimated: $20-40/month

Cloud Monitoring:
- First 1B metrics free
- Estimated: $0-20/month

Total Estimated: $110-320/month
```

### Cost Optimization Strategies

```bash
# 1. Use Cloud Run min-instances wisely
# Set to 0 for dev/test, 1-2 for production

# 2. Right-size Cloud SQL
gcloud sql instances patch rag-engine-db \
    --tier=db-f1-micro  # For dev
    # --tier=db-g1-small  # For production

# 3. Use committed use discounts for GKE
gcloud compute commitments create rag-engine-commitment \
    --region=${REGION} \
    --resources=vcpu=24,memory=96 \
    --plan=12-month

# 4. Set budget alerts
gcloud billing budgets create \
    --display-name="RAG Engine Budget" \
    --budget-amount=500USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100

# 5. Use Cloud Scheduler to scale down dev environments
# Create a job to set min-instances=0 at night
```

---

## Part 9: Troubleshooting Common Issues

### Issue 1: Cloud Run Cold Start

**Symptoms:** First request after idle is slow (2-5 seconds)

**Solution:**
```bash
# Set minimum instances to keep warm
gcloud run services update rag-engine-api \
    --min-instances=2 \
    --region=${REGION}

# Or implement keep-alive ping
# Add Cloud Scheduler job
gcloud scheduler jobs create http keep-alive \
    --schedule="*/5 * * * *" \
    --uri="${SERVICE_URL}/health" \
    --http-method=GET
```

### Issue 2: Cloud SQL Connection Failures

**Symptoms:** Database connection errors, "Cloud SQL instance not found"

**Solution:**
```bash
# Verify connection name
gcloud sql instances describe ${INSTANCE_NAME} --format='value(connectionName)'

# Test connectivity from local
cloud-sql-proxy ${CONNECTION_NAME} &
psql -h 127.0.0.1 -p 5432 -U rag_user -d rag_engine

# Check service account permissions
gcloud projects get-iam-policy ${PROJECT_ID} \
    --flatten="bindings[].members" \
    --format='table(bindings.role)' \
    --filter="bindings.members:rag-engine-sa"
```

### Issue 3: Memorystore Connection Timeout

**Symptoms:** Redis connection errors, timeout on cache operations

**Solution:**
```bash
# Verify Redis is in same region
gcloud redis instances describe ${REDIS_NAME} --region=${REGION}

# Check VPC connector status
gcloud compute networks vpc-access connectors describe ${CONNECTOR_NAME} \
    --region=${REGION}

# Test connectivity
gcloud compute ssh test-vm --zone=${REGION}-a \
    --command="redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} ping"
```

### Issue 4: High Latency on GKE

**Symptoms:** Slow response times, pods throttled

**Solution:**
```bash
# Check resource usage
kubectl top pods -n rag-engine

# View HPA status
kubectl get hpa -n rag-engine

# Describe pod for events
kubectl describe pod <pod-name> -n rag-engine

# Increase resources if needed
kubectl patch deployment rag-engine-api -n rag-engine \
    -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"requests":{"memory":"2Gi","cpu":"1000m"}}}]}}}}'
```

---

## Part 10: Production Checklist

### Pre-Deployment

- [ ] GCP project created and billing enabled
- [ ] Required APIs enabled
- [ ] Service accounts created with minimal permissions
- [ ] Secrets stored in Secret Manager
- [ ] Container image built and tested locally
- [ ] Health check endpoints implemented (`/health`, `/ready`)

### Infrastructure

- [ ] Cloud SQL instance created with backups enabled
- [ ] Memorystore Redis instance created
- [ ] Cloud Storage bucket created with lifecycle policies
- [ ] VPC connector created (for private services)
- [ ] SSL certificate configured (for custom domain)

### Security

- [ ] Cloud Armor security policy applied
- [ ] Binary Authorization enabled (optional)
- [ ] Service accounts use Workload Identity
- [ ] No hardcoded secrets in code
- [ ] Private Google Access enabled
- [ ] Cloud IAM roles follow least privilege

### Monitoring

- [ ] Uptime checks configured
- [ ] Alert policies created
- [ ] Log-based metrics defined
- [ ] Dashboard created
- [ ] Error reporting enabled

### CI/CD

- [ ] Cloud Build trigger configured
- [ ] Automated tests in pipeline
- [ ] Security scanning enabled
- [ ] Staging environment deployed
- [ ] Rollback procedure documented

### Documentation

- [ ] Runbooks created for common issues
- [ ] Architecture diagrams updated
- [ ] On-call rotation configured
- [ ] Cost estimates documented
- [ ] Disaster recovery plan tested

---

## Summary

You now have a production-ready RAG Engine deployment on Google Cloud Platform with:

✅ **Cloud Run** serverless deployment (auto-scaling, pay-per-use)
✅ **GKE Autopilot** option for Kubernetes workloads
✅ **Cloud SQL** for managed PostgreSQL
✅ **Memorystore** for Redis caching
✅ **Cloud Storage** for document storage
✅ **Cloud Monitoring** and alerting
✅ **Cloud Armor** for security
✅ **Cloud Build** CI/CD pipeline

**Next Steps:**
1. Monitor costs and optimize based on usage
2. Set up automated backups and disaster recovery
3. Implement multi-region deployment for high availability
4. Add more sophisticated autoscaling based on custom metrics

**Support Resources:**
- [GCP Documentation](https://cloud.google.com/docs)
- [Cloud Run Docs](https://cloud.google.com/run/docs)
- [GKE Docs](https://cloud.google.com/kubernetes-engine/docs)
- [Cloud SQL Docs](https://cloud.google.com/sql/docs)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

# Azure Deployment Guide - Production RAG Engine on Microsoft Azure

## Overview

This guide walks you through deploying RAG Engine Mini on Microsoft Azure using industry best practices. We'll cover multiple deployment options from simple container instances to enterprise-grade Kubernetes.

## Learning Objectives

By the end of this guide, you will:
1. Understand Azure services for container deployment
2. Deploy using Container Apps (serverless containers)
3. Set up AKS (Azure Kubernetes Service)
4. Configure Azure Database for PostgreSQL
5. Implement Azure Blob Storage for document storage
6. Set up Application Gateway with SSL
7. Configure Azure Monitor and Log Analytics
8. Implement auto-scaling policies
9. Manage costs effectively

**Estimated Time:** 6-8 hours
**Cost:** $180-750/month (depending on scale)

---

## Part 1: Azure Architecture Options

### Option 1: Container Apps (Recommended for Beginners)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                   Azure Container Apps                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Application Gateway (WAF)                   │   │
│  │         (SSL termination, load balancing)           │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              Container Apps Environment             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Replica │ │  API Replica │ │  API Replica │  │   │
│  │  │  (Revision)  │ │  (Revision)  │ │  (Revision)  │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  │       ┌──────────────┐                               │   │
│  │       │ Background   │                               │   │
│  │       │ Job (KEDA)   │                               │   │
│  │       └──────────────┘                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────┼──────────────────────────────────┐   │
│  │                  │          VNet Integration        │   │
│  │  ┌───────────────▼────────┐ ┌─────────────────────┐  │   │
│  │  │    Azure Database      │ │   Azure Cache       │  │   │
│  │  │    for PostgreSQL      │ │   for Redis         │  │   │
│  │  │    (Flexible Server)   │ │   (Enterprise)      │  │   │
│  │  └────────────────────────┘ └─────────────────────┘  │   │
│  │  ┌────────────────────────┐                          │   │
│  │  │    Blob Storage        │                          │   │
│  │  │    (Hot Tier)          │                          │   │
│  │  └────────────────────────┘                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Fully managed serverless containers
- KEDA-based event-driven scaling
- Built-in Dapr support for microservices
- Good for microservices and background jobs

**Pros:**
- No Kubernetes complexity
- Automatic scaling to zero
- Built-in ingress with HTTPS
- Integrated with Azure ecosystem

**Cons:**
- Newer service (some limitations)
- Limited customization compared to AKS
- Regional availability constraints

### Option 2: AKS (Azure Kubernetes Service)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     AKS Cluster                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Application Gateway Ingress Controller      │   │
│  │              (AGIC with WAF)                        │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              Node Pool (VMSS)                       │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Pod     │ │  API Pod     │ │ Worker Pod   │  │   │
│  │  │  (Node 1)    │ │  (Node 2)    │ │ (Node 3)     │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  External Azure Services:                                   │
│  • Azure Database for PostgreSQL                            │
│  • Azure Cache for Redis                                    │
│  • Azure Blob Storage                                       │
│  • Azure Monitor                                            │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Full Kubernetes feature set needed
- Complex multi-tier applications
- Need for specific Kubernetes versions
- Advanced networking requirements

**Pros:**
- Full Kubernetes control
- Best for complex workloads
- Multi-region deployment support
- Extensive ecosystem

**Cons:**
- Higher complexity
- Requires Kubernetes expertise
- Always running (minimum cost)

### Option 3: Container Instances (Quick Testing)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│              Azure Container Instances                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Container Group                             │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │           RAG Engine API                     │  │   │
│  │  │         (Quick Demo/Testing)                 │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │       ┌──────────────┐                             │   │
│  │       │ Sidecar:     │                             │   │
│  │       │ SQL Proxy    │                             │   │
│  │       └──────────────┘                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  External Azure Services:                                   │
│  • Azure Database for PostgreSQL                            │
│  • Azure Blob Storage                                       │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Quick demos and testing
- Simple workloads without orchestration
- Burst scenarios
- CI/CD integration

---

## Part 2: Prerequisites

### Required Tools

```bash
# Install Azure CLI
# Windows (PowerShell)
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'

# macOS
brew install azure-cli

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Verify installation
az --version

# Login
az login

# Set subscription
az account set --subscription "Your Subscription Name"
```

### Azure Resource Group Setup

```bash
# Variables
RESOURCE_GROUP="rag-engine-rg"
LOCATION="eastus"
APP_NAME="rag-engine"

# Create resource group
az group create \
    --name ${RESOURCE_GROUP} \
    --location ${LOCATION}

# Verify
az group show --name ${RESOURCE_GROUP}
```

### Service Principal Setup

```bash
# Create service principal for CI/CD
az ad sp create-for-rbac \
    --name "rag-engine-deployer" \
    --role contributor \
    --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/${RESOURCE_GROUP} \
    --sdk-auth

# Output will be JSON credentials - save this for CI/CD
# Store in GitHub Secrets as AZURE_CREDENTIALS
```

### Enable Required Providers

```bash
# Register resource providers
az provider register --namespace Microsoft.ContainerInstance
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.DBforPostgreSQL
az provider register --namespace Microsoft.Cache
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.Insights

# Wait for registration (check status)
az provider show --namespace Microsoft.App --query "registrationState"
```

---

## Part 3: Container Apps Deployment

### Step 1: Build and Push Container

```bash
# Create Azure Container Registry
ACR_NAME="ragengine$(date +%s)"  # Must be globally unique
az acr create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${ACR_NAME} \
    --sku Standard \
    --location ${LOCATION} \
    --admin-enabled true

# Login to ACR
az acr login --name ${ACR_NAME}

# Build and push image
az acr build \
    --registry ${ACR_NAME} \
    --image ${APP_NAME}/api:latest \
    --file Dockerfile .

# Or build locally
docker build -t ${ACR_NAME}.azurecr.io/${APP_NAME}/api:latest .
docker push ${ACR_NAME}.azurecr.io/${APP_NAME}/api:latest
```

### Step 2: Set Up Azure Database for PostgreSQL

```bash
# Create PostgreSQL Flexible Server
POSTGRES_SERVER="${APP_NAME}-postgres"
POSTGRES_USER="ragadmin"
POSTGRES_DB="rag_engine"

az postgres flexible-server create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${POSTGRES_SERVER} \
    --location ${LOCATION} \
    --admin-user ${POSTGRES_USER} \
    --admin-password $(openssl rand -base64 24) \
    --sku-name Standard_B1ms \
    --tier Burstable \
    --storage-size 32 \
    --version 14 \
    --backup-retention 7 \
    --geo-redundant-backup Disabled \
    --public-access None \
    --database-name ${POSTGRES_DB}

# Get connection details
POSTGRES_HOST=$(az postgres flexible-server show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${POSTGRES_SERVER} \
    --query fullyQualifiedDomainName \
    --output tsv)

echo "Database Host: ${POSTGRES_HOST}"

# Create database
echo "CREATE DATABASE rag_engine;" | az postgres flexible-server execute \
    --name ${POSTGRES_SERVER} \
    --admin-user ${POSTGRES_USER} \
    --admin-password $(az postgres flexible-server show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${POSTGRES_SERVER} \
        --query administratorLoginPassword \
        -o tsv) \
    --database-name postgres
```

### Step 3: Set Up Azure Cache for Redis

```bash
# Create Redis Cache
REDIS_NAME="${APP_NAME}-redis"

az redis create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${REDIS_NAME} \
    --location ${LOCATION} \
    --sku Basic \
    --vm-size C0 \
    --enable-non-ssl-port false

# Get connection details
REDIS_HOST=$(az redis show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${REDIS_NAME} \
    --query hostName \
    --output tsv)

REDIS_KEY=$(az redis list-keys \
    --resource-group ${RESOURCE_GROUP} \
    --name ${REDIS_NAME} \
    --query primaryKey \
    --output tsv)

echo "Redis Host: ${REDIS_HOST}"
```

### Step 4: Set Up Blob Storage

```bash
# Create storage account
STORAGE_ACCOUNT="ragengine$(date +%s)"  # Must be globally unique

az storage account create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${STORAGE_ACCOUNT} \
    --location ${LOCATION} \
    --sku Standard_LRS \
    --kind StorageV2 \
    --access-tier Hot \
    --enable-large-file-share \
    --min-tls-version TLS1_2

# Create container
az storage container create \
    --name documents \
    --account-name ${STORAGE_ACCOUNT}

# Get connection string
STORAGE_CONNECTION=$(az storage account show-connection-string \
    --resource-group ${RESOURCE_GROUP} \
    --name ${STORAGE_ACCOUNT} \
    --query connectionString \
    --output tsv)

echo "Storage Account: ${STORAGE_ACCOUNT}"
```

### Step 5: Create Container Apps Environment

```bash
# Create Log Analytics workspace
WORKSPACE_NAME="${APP_NAME}-logs"

az monitor log-analytics workspace create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${WORKSPACE_NAME} \
    --location ${LOCATION}

# Get workspace ID and key
WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${WORKSPACE_NAME} \
    --query customerId \
    --output tsv)

WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group ${RESOURCE_GROUP} \
    --name ${WORKSPACE_NAME} \
    --query primarySharedKey \
    --output tsv)

# Create Container Apps environment
ENVIRONMENT_NAME="${APP_NAME}-env"

az containerapp env create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${ENVIRONMENT_NAME} \
    --location ${LOCATION} \
    --logs-workspace-id ${WORKSPACE_ID} \
    --logs-workspace-key ${WORKSPACE_KEY}

# Create managed identity for service
az identity create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-identity

IDENTITY_ID=$(az identity show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-identity \
    --query id \
    --output tsv)

IDENTITY_CLIENT_ID=$(az identity show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-identity \
    --query clientId \
    --output tsv)
```

### Step 6: Deploy to Container Apps

```bash
# Create Container App
az containerapp create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-api \
    --environment ${ENVIRONMENT_NAME} \
    --image ${ACR_NAME}.azurecr.io/${APP_NAME}/api:latest \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 10 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --env-vars "ENVIRONMENT=production" \
               "DB_HOST=${POSTGRES_HOST}" \
               "DB_NAME=${POSTGRES_DB}" \
               "DB_USER=${POSTGRES_USER}" \
               "REDIS_HOST=${REDIS_HOST}" \
               "REDIS_PORT=6380" \
               "STORAGE_ACCOUNT=${STORAGE_ACCOUNT}" \
               "AZURE_CLIENT_ID=${IDENTITY_CLIENT_ID}" \
    --secrets "db-password=$(openssl rand -base64 24)" \
              "jwt-secret=$(openssl rand -base64 32)" \
              "redis-key=${REDIS_KEY}" \
              "storage-connection=${STORAGE_CONNECTION}" \
    --registry-server ${ACR_NAME}.azurecr.io \
    --user-assigned ${IDENTITY_ID}

# Get app URL
APP_URL=$(az containerapp show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-api \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo "App URL: https://${APP_URL}"
```

### Step 7: Configure Scaling Rules

```bash
# Create KEDA-based scaling with custom rules
az containerapp update \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-api \
    --scale-rule-name http-rule \
    --scale-rule-type http \
    --scale-rule-metadata "concurrentRequests=50" \
    --scale-rule-http-concurrency 50

# Or with Azure CLI scale configuration
az containerapp update \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-api \
    --min-replicas 1 \
    --max-replicas 20
```

### Step 8: Configure VNet Integration

```bash
# Create virtual network
VNET_NAME="${APP_NAME}-vnet"

az network vnet create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${VNET_NAME} \
    --address-prefix 10.0.0.0/16 \
    --subnet-name containerapp-subnet \
    --subnet-prefix 10.0.0.0/21

# Create private endpoints for database
az network private-endpoint create \
    --resource-group ${RESOURCE_GROUP} \
    --name postgres-private-endpoint \
    --vnet-name ${VNET_NAME} \
    --subnet containerapp-subnet \
    --private-connection-resource-id $(az postgres flexible-server show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${POSTGRES_SERVER} \
        --query id \
        --output tsv) \
    --group-id postgresqlServer \
    --connection-name postgres-connection
```

---

## Part 4: AKS Deployment

### Step 1: Create AKS Cluster

```bash
# Create AKS cluster
AKS_NAME="${APP_NAME}-aks"

az aks create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${AKS_NAME} \
    --location ${LOCATION} \
    --node-count 3 \
    --node-vm-size Standard_D4s_v3 \
    --enable-cluster-autoscaler \
    --min-count 3 \
    --max-count 10 \
    --enable-managed-identity \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --network-plugin azure \
    --network-policy azure \
    --enable-azure-policy

# Get credentials
az aks get-credentials \
    --resource-group ${RESOURCE_GROUP} \
    --name ${AKS_NAME} \
    --overwrite-existing

# Verify connection
kubectl get nodes
```

### Step 2: Install Ingress Controller

```bash
# Install NGINX Ingress Controller with Helm
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install with Azure integration
helm install ingress-nginx ingress-nginx/ingress-nginx \
    --create-namespace \
    --namespace ingress-nginx \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-health-probe-request-path"=/healthz \
    --set controller.service.externalTrafficPolicy=Local

# Wait for load balancer IP
echo "Waiting for load balancer IP..."
sleep 60
kubectl get service ingress-nginx-controller -n ingress-nginx

# Get external IP
INGRESS_IP=$(kubectl get service ingress-nginx-controller \
    -n ingress-nginx \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Ingress IP: ${INGRESS_IP}"
```

### Step 3: Install cert-manager

```bash
# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Wait for cert-manager
kubectl wait --for=condition=available --timeout=120s deployment/cert-manager -n cert-manager
kubectl wait --for=condition=available --timeout=120s deployment/cert-manager-webhook -n cert-manager

# Create ClusterIssuer for Let's Encrypt
cat > cluster-issuer.yaml <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@rag-engine.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

kubectl apply -f cluster-issuer.yaml
```

### Step 4: Create Secrets

```bash
# Create namespace
kubectl create namespace rag-engine

# Create secrets
cat > secrets.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: rag-engine-secrets
  namespace: rag-engine
type: Opaque
stringData:
  DB_PASSWORD: "$(openssl rand -base64 24)"
  JWT_SECRET: "$(openssl rand -base64 32)"
  REDIS_KEY: "${REDIS_KEY}"
  STORAGE_CONNECTION: "${STORAGE_CONNECTION}"
EOF

kubectl apply -f secrets.yaml

# Create managed identity for pod
az identity create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-pod-identity

# Get identity details
POD_IDENTITY_ID=$(az identity show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-pod-identity \
    --query id \
    --output tsv)

POD_IDENTITY_CLIENT_ID=$(az identity show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-pod-identity \
    --query clientId \
    --output tsv)
```

### Step 5: Deploy Application

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-engine-config
  namespace: rag-engine
data:
  ENVIRONMENT: "production"
  DB_HOST: "rag-engine-postgres.postgres.database.azure.com"
  DB_NAME: "rag_engine"
  DB_USER: "ragadmin"
  REDIS_HOST: "rag-engine-redis.redis.cache.windows.net"
  REDIS_PORT: "6380"
  STORAGE_ACCOUNT: "ragenginestorage"
  LOG_LEVEL: "INFO"

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
      containers:
      - name: api
        image: ragengine.azurecr.io/rag-engine/api:latest
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: rag-engine-config
        - secretRef:
            name: rag-engine-secrets
        env:
        - name: PORT
          value: "8000"
        - name: AZURE_CLIENT_ID
          value: "${POD_IDENTITY_CLIENT_ID}"
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
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.rag-engine.com
    secretName: rag-engine-tls
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
```

### Step 6: Deploy to AKS

```bash
# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=120s \
    deployment/rag-engine-api -n rag-engine

# Apply ingress
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n rag-engine
kubectl get svc -n rag-engine
kubectl get ingress -n rag-engine
```

---

## Part 5: Monitoring and Observability

### Azure Monitor Setup

```bash
# Create Application Insights
APP_INSIGHTS_NAME="${APP_NAME}-insights"

az monitor app-insights component create \
    --resource-group ${RESOURCE_GROUP} \
    --app ${APP_INSIGHTS_NAME} \
    --location ${LOCATION} \
    --kind web \
    --application-type web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --resource-group ${RESOURCE_GROUP} \
    --app ${APP_INSIGHTS_NAME} \
    --query instrumentationKey \
    --output tsv)

echo "Instrumentation Key: ${INSTRUMENTATION_KEY}"
```

### Container Insights

```bash
# Enable Container Insights (already enabled with monitoring addon)
# View logs
az monitor log-analytics query \
    --workspace ${WORKSPACE_ID} \
    --analytics-query "ContainerInventory | where Name contains 'rag-engine' | project Name, Image, State, StartedTime" \
    --timespan PT1H

# View metrics
az monitor metrics list \
    --resource $(az containerapp show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${APP_NAME}-api \
        --query id \
        --output tsv) \
    --metric "UsageNanoCores" \
    --interval PT1M
```

### Create Alerts

```bash
# Create action group
ACTION_GROUP_NAME="${APP_NAME}-alerts"

az monitor action-group create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${ACTION_GROUP_NAME} \
    --short-name ragengine \
    --email-receivers admin@rag-engine.com \
    --email-receiver-names Admin

# Create alert for high CPU
cat > alert-cpu.json <<EOF
{
  "condition": {
    "allOf": [
      {
        "metricName": "UsageNanoCores",
        "metricNamespace": "Microsoft.App/containerApps",
        "operator": "GreaterThan",
        "threshold": 80,
        "timeAggregation": "Average"
      }
    ]
  },
  "description": "CPU usage is above 80%",
  "severity": 2,
  "evaluationFrequency": "PT1M",
  "windowSize": "PT5M"
}
EOF

az monitor metrics alert create \
    --resource-group ${RESOURCE_GROUP} \
    --name "High CPU Alert" \
    --scopes $(az containerapp show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${APP_NAME}-api \
        --query id \
        --output tsv) \
    --condition @alert-cpu.json \
    --action ${ACTION_GROUP_NAME}

# Create alert for HTTP errors
az monitor metrics alert create \
    --resource-group ${RESOURCE_GROUP} \
    --name "High Error Rate" \
    --scopes $(az containerapp show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${APP_NAME}-api \
        --query id \
        --output tsv) \
    --condition "avg http_requests > 10 where http_status_code == 5xx" \
    --action ${ACTION_GROUP_NAME}
```

### Custom Dashboard

```bash
# Create dashboard via Azure CLI
cat > dashboard.json <<EOF
{
  "lenses": {
    "0": {
      "order": 0,
      "parts": {
        "0": {
          "position": {
            "x": 0,
            "y": 0,
            "rowSpan": 4,
            "colSpan": 6
          },
          "metadata": {
            "inputs": [
              {
                "name": "options",
                "value": {
                  "chart": {
                    "metrics": [
                      {
                        "resourceMetadata": {
                          "id": "$(az containerapp show --resource-group ${RESOURCE_GROUP} --name ${APP_NAME}-api --query id -o tsv)"
                        },
                        "name": "Requests",
                        "aggregationType": 1,
                        "namespace": "Microsoft.App/containerApps"
                      }
                    ]
                  }
                }
              }
            ],
            "type": "Extension/HubsExtension/PartType/MonitorChartPart"
          }
        }
      }
    }
  }
}
EOF

# Create dashboard in portal manually or use Azure Resource Manager template
```

---

## Part 6: Security Best Practices

### Application Gateway with WAF

```bash
# Create Application Gateway
VNET_NAME="${APP_NAME}-vnet"
SUBNET_NAME="appgw-subnet"
APPGW_NAME="${APP_NAME}-appgw"

# Create subnet for App Gateway
az network vnet subnet create \
    --resource-group ${RESOURCE_GROUP} \
    --vnet-name ${VNET_NAME} \
    --name ${SUBNET_NAME} \
    --address-prefix 10.0.1.0/24

# Create public IP
az network public-ip create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-appgw-ip \
    --allocation-method Static \
    --sku Standard

# Create Application Gateway with WAF
az network application-gateway create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APPGW_NAME} \
    --location ${LOCATION} \
    --sku WAF_v2 \
    --public-ip-address ${APP_NAME}-appgw-ip \
    --vnet-name ${VNET_NAME} \
    --subnet ${SUBNET_NAME} \
    --servers $(az containerapp show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${APP_NAME}-api \
        --query properties.configuration.ingress.fqdn \
        --output tsv) \
    --waf-policy ${APP_NAME}-waf-policy

# Configure WAF policy
az network application-gateway waf-policy create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-waf-policy

# Enable OWASP rules
az network application-gateway waf-policy managed-rule rule-set add \
    --policy-name ${APP_NAME}-waf-policy \
    --resource-group ${RESOURCE_GROUP} \
    --type OWASP \
    --version 3.2
```

### Azure Key Vault

```bash
# Create Key Vault
KEYVAULT_NAME="ragengine$(date +%s)"  # Must be globally unique

az keyvault create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${KEYVAULT_NAME} \
    --location ${LOCATION} \
    --sku standard \
    --enable-soft-delete \
    --retention-days 90

# Add secrets
az keyvault secret set \
    --vault-name ${KEYVAULT_NAME} \
    --name db-password \
    --value $(openssl rand -base64 24)

az keyvault secret set \
    --vault-name ${KEYVAULT_NAME} \
    --name jwt-secret \
    --value $(openssl rand -base64 32)

az keyvault secret set \
    --vault-name ${KEYVAULT_NAME} \
    --name redis-key \
    --value ${REDIS_KEY}

# Grant access to managed identity
az keyvault set-policy \
    --name ${KEYVAULT_NAME} \
    --object-id $(az identity show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${APP_NAME}-identity \
        --query principalId \
        --output tsv) \
    --secret-permissions get list
```

### Network Security

```bash
# Create Network Security Group
NSG_NAME="${APP_NAME}-nsg"

az network nsg create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${NSG_NAME} \
    --location ${LOCATION}

# Add security rules
az network nsg rule create \
    --resource-group ${RESOURCE_GROUP} \
    --nsg-name ${NSG_NAME} \
    --name AllowHTTPS \
    --priority 100 \
    --source-address-prefixes Internet \
    --destination-port-ranges 443 \
    --access Allow \
    --protocol Tcp

az network nsg rule create \
    --resource-group ${RESOURCE_GROUP} \
    --nsg-name ${NSG_NAME} \
    --name DenyAllInbound \
    --priority 4096 \
    --source-address-prefixes * \
    --destination-port-ranges * \
    --access Deny \
    --protocol *

# Associate NSG with subnet
az network vnet subnet update \
    --resource-group ${RESOURCE_GROUP} \
    --vnet-name ${VNET_NAME} \
    --name containerapp-subnet \
    --network-security-group ${NSG_NAME}
```

---

## Part 7: CI/CD Pipeline with Azure DevOps

### azure-pipelines.yaml

```yaml
trigger:
  - main

variables:
  dockerRegistryServiceConnection: 'rag-engine-acr'
  imageRepository: 'rag-engine/api'
  containerRegistry: 'ragengine.azurecr.io'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'
  
  # Agent VM image name
  vmImageName: 'ubuntu-latest'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    pool:
      vmImage: $(vmImageName)
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: SecurityScan
  displayName: Security Scan
  dependsOn: Build
  jobs:
  - job: Scan
    displayName: Scan Image
    pool:
      vmImage: $(vmImageName)
    steps:
    - task: AzureCLI@2
      inputs:
        azureSubscription: 'Azure-RAG-Engine'
        scriptType: 'bash'
        scriptLocation: 'inlineScript'
        inlineScript: |
          az acr run --cmd "trivy image $(containerRegistry)/$(imageRepository):$(tag)" \
            --registry $(containerRegistry) \
            /dev/null

- stage: DeployStaging
  displayName: Deploy to Staging
  dependsOn: SecurityScan
  jobs:
  - job: Deploy
    displayName: Deploy to Container Apps
    pool:
      vmImage: $(vmImageName)
    steps:
    - task: AzureCLI@2
      inputs:
        azureSubscription: 'Azure-RAG-Engine'
        scriptType: 'bash'
        scriptLocation: 'inlineScript'
        inlineScript: |
          az containerapp update \
            --resource-group rag-engine-rg \
            --name rag-engine-api-staging \
            --image $(containerRegistry)/$(imageRepository):$(tag)

- stage: IntegrationTests
  displayName: Integration Tests
  dependsOn: DeployStaging
  jobs:
  - job: Test
    displayName: Run Tests
    pool:
      vmImage: $(vmImageName)
    steps:
    - script: |
        curl -f https://staging-api.rag-engine.com/health || exit 1
        pytest tests/integration/ -v

- stage: DeployProduction
  displayName: Deploy to Production
  dependsOn: IntegrationTests
  condition: succeeded()
  jobs:
  - deployment: DeployProd
    displayName: Deploy to Production
    environment: 'RAG-Engine-Production'
    pool:
      vmImage: $(vmImageName)
    strategy:
      canary:
        increments: [10, 50, 100]
        preDeploy:
          steps:
          - script: echo "Starting canary deployment..."
        deploy:
          steps:
          - task: AzureCLI@2
            inputs:
              azureSubscription: 'Azure-RAG-Engine'
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az containerapp update \
                  --resource-group rag-engine-rg \
                  --name rag-engine-api \
                  --image $(containerRegistry)/$(imageRepository):$(tag)
```

### GitHub Actions Alternative

```yaml
# .github/workflows/azure-deploy.yml
name: Deploy to Azure

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  AZURE_WEBAPP_NAME: rag-engine-api
  AZURE_WEBAPP_PACKAGE_PATH: '.'
  NODE_VERSION: '18.x'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to ACR
      uses: azure/docker-login@v1
      with:
        login-server: ${{ secrets.ACR_LOGIN_SERVER }}
        username: ${{ secrets.ACR_USERNAME }}
        password: ${{ secrets.ACR_PASSWORD }}

    - name: Build and push image
      run: |
        docker build . -t ${{ secrets.ACR_LOGIN_SERVER }}/rag-engine/api:${{ github.sha }}
        docker push ${{ secrets.ACR_LOGIN_SERVER }}/rag-engine/api:${{ github.sha }}

    - name: Deploy to Container Apps
      uses: azure/container-apps-deploy-action@v1
      with:
        appSourcePath: ${{ github.workspace }}
        acrName: ragengine
        containerAppName: rag-engine-api
        resourceGroup: rag-engine-rg
        imageToDeploy: ${{ secrets.ACR_LOGIN_SERVER }}/rag-engine/api:${{ github.sha }}
```

---

## Part 8: Cost Optimization

### Cost Breakdown (Monthly Estimates)

```
Container Apps (1-10 replicas, scales to zero):
- vCPU: $0.000024/vCPU-s
- Memory: $0.0000025/GiB-s
- Requests: $0.40/million
- Estimated: $40-150/month

Azure Database for PostgreSQL (Flexible Server - B1ms):
- Compute: $15-25/month
- Storage: $0.10/GB/month
- Backup: Included
- Estimated: $25-40/month

Azure Cache for Redis (C0 Basic):
- Instance: $15-20/month

Azure Blob Storage (100GB Hot tier):
- Storage: $2.30/month
- Operations: $1-5/month
- Estimated: $5-10/month

Application Gateway (WAF v2):
- Fixed cost: $25/month
- Capacity units: $0.016/hour
- Estimated: $40-60/month

AKS (if using Standard tier):
- Control plane: $73/month
- Nodes (3x D4s_v3): $420/month
- Estimated: $493/month

Monitoring:
- Application Insights: $2.30/GB ingested
- Log Analytics: $2.99/GB ingested
- Estimated: $20-50/month

Total Estimated: $145-808/month
(Container Apps vs AKS makes big difference)
```

### Cost Optimization Strategies

```bash
# 1. Use Container Apps for dev/test
# Scale to zero when not in use

# 2. Right-size PostgreSQL tier
az postgres flexible-server update \
    --resource-group ${RESOURCE_GROUP} \
    --name ${POSTGRES_SERVER} \
    --tier Burstable \
    --sku-name Standard_B1ms

# 3. Use Azure Reservations for long-term AKS
cat > reservation.json <<EOF
{
  "sku": {
    "name": "Standard_D4s_v3"
  },
  "location": "${LOCATION}",
  "properties": {
    "reservedResourceType": "VirtualMachines",
    "billingScopeId": "/subscriptions/$(az account show --query id -o tsv)",
    "term": "P1Y",
    "billingPlan": "Upfront",
    "quantity": 3,
    "displayName": "RAG Engine Reservation"
  }
}
EOF

# 4. Set up budget alerts
az consumption budget create \
    --resource-group ${RESOURCE_GROUP} \
    --budget-name "RAG Engine Budget" \
    --amount 500 \
    --category Cost \
    --time-grain Monthly \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --notifications "[
      {\"operator\":\"GreaterThan\",\"threshold\":80,\"contactEmails\":[\"admin@rag-engine.com\"]},
      {\"operator\":\"GreaterThan\",\"threshold\":100,\"contactEmails\":[\"admin@rag-engine.com\"]}
    ]"

# 5. Use lifecycle management for Blob Storage
az storage account management-policy create \
    --resource-group ${RESOURCE_GROUP} \
    --account-name ${STORAGE_ACCOUNT} \
    --policy @lifecycle-policy.json

# lifecycle-policy.json
{
  "rules": [
    {
      "name": "archiveOldDocuments",
      "enabled": true,
      "type": "Lifecycle",
      "definition": {
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["documents/"]
        },
        "actions": {
          "baseBlob": {
            "tierToCool": { "daysAfterModificationGreaterThan": 30 },
            "tierToArchive": { "daysAfterModificationGreaterThan": 90 },
            "delete": { "daysAfterModificationGreaterThan": 365 }
          }
        }
      }
    }
  ]
}
```

---

## Part 9: Troubleshooting Common Issues

### Issue 1: Container Apps Connection to PostgreSQL

**Symptoms:** Database connection failures, "no pg_hba.conf entry"

**Solution:**
```bash
# Check PostgreSQL firewall rules
az postgres flexible-server firewall-rule list \
    --resource-group ${RESOURCE_GROUP} \
    --name ${POSTGRES_SERVER}

# Add Azure services rule
az postgres flexible-server firewall-rule create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${POSTGRES_SERVER} \
    --rule-name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0

# Or better, use private endpoint
az network private-endpoint create \
    --resource-group ${RESOURCE_GROUP} \
    --name postgres-pe \
    --vnet-name ${VNET_NAME} \
    --subnet containerapp-subnet \
    --private-connection-resource-id $(az postgres flexible-server show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${POSTGRES_SERVER} \
        --query id -o tsv) \
    --group-id postgresqlServer
```

### Issue 2: AKS Pod Startup Failures

**Symptoms:** Pods stuck in Pending, CrashLoopBackOff

**Solution:**
```bash
# Check pod status
kubectl get pods -n rag-engine -o wide

# Describe pod for events
kubectl describe pod <pod-name> -n rag-engine

# Check logs
kubectl logs <pod-name> -n rag-engine --previous

# Check resource quotas
kubectl describe resourcequota -n rag-engine

# Verify image pull
kubectl get events -n rag-engine --field-selector reason=FailedToPullImage

# Fix image pull secret
kubectl create secret docker-registry acr-secret \
    --docker-server=${ACR_NAME}.azurecr.io \
    --docker-username=${ACR_NAME} \
    --docker-password=$(az acr credential show \
        --name ${ACR_NAME} \
        --query passwords[0].value \
        -o tsv) \
    --namespace rag-engine
```

### Issue 3: Redis SSL/TLS Errors

**Symptoms:** Redis connection errors, SSL handshake failures

**Solution:**
```bash
# Check Redis SSL settings
az redis show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${REDIS_NAME} \
    --query sslPort

# Connect with SSL
redis-cli -h ${REDIS_HOST} -p 6380 -a ${REDIS_KEY} --tls

# Or disable SSL for testing (not recommended for production)
az redis update \
    --resource-group ${RESOURCE_GROUP} \
    --name ${REDIS_NAME} \
    --set "enableNonSslPort=true"
```

### Issue 4: Application Gateway 502 Bad Gateway

**Symptoms:** 502 errors, backend health unhealthy

**Solution:**
```bash
# Check backend health
az network application-gateway show-backend-health \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APPGW_NAME}

# Verify backend settings
az network application-gateway address-pool show \
    --resource-group ${RESOURCE_GROUP} \
    --gateway-name ${APPGW_NAME} \
    --name appGatewayBackendPool

# Check probe configuration
az network application-gateway probe show \
    --resource-group ${RESOURCE_GROUP} \
    --gateway-name ${APPGW_NAME} \
    --name defaultprobe

# Test backend directly
curl -v http://$(az containerapp show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${APP_NAME}-api \
    --query properties.configuration.ingress.fqdn \
    --output tsv)/health
```

---

## Part 10: Production Checklist

### Pre-Deployment

- [ ] Azure subscription with billing enabled
- [ ] Resource group created
- [ ] Service principal for CI/CD
- [ ] Container Registry configured
- [ ] Container image built and tested
- [ ] Health check endpoints implemented

### Infrastructure

- [ ] Azure Database for PostgreSQL created
- [ ] Azure Cache for Redis provisioned
- [ ] Blob Storage account created
- [ ] Container Apps Environment configured
- [ ] Virtual Network with proper subnets
- [ ] Private endpoints configured (for security)

### Security

- [ ] Application Gateway with WAF enabled
- [ ] Key Vault secrets configured
- [ ] Managed identities for services
- [ ] Network Security Groups applied
- [ ] SSL/TLS certificates configured
- [ ] Azure Policy compliance verified

### Monitoring

- [ ] Application Insights integrated
- [ ] Container Insights enabled
- [ ] Alert rules configured
- [ ] Dashboard created
- [ ] Log retention policies set
- [ ] Health probes configured

### CI/CD

- [ ] Azure DevOps pipeline configured
- [ ] GitHub Actions workflow (alternative)
- [ ] Automated tests in pipeline
- [ ] Security scanning enabled
- [ ] Staging environment deployed
- [ ] Blue-green deployment ready

### Documentation

- [ ] Runbooks for common issues
- [ ] Architecture diagrams updated
- [ ] On-call procedures documented
- [ ] Cost estimates and budgets set
- [ ] Disaster recovery plan tested

---

## Summary

You now have a production-ready RAG Engine deployment on Microsoft Azure with:

✅ **Container Apps** serverless deployment (auto-scaling, pay-per-use)
✅ **AKS** option for full Kubernetes control
✅ **Azure Database for PostgreSQL** managed database
✅ **Azure Cache for Redis** for caching
✅ **Azure Blob Storage** for document storage
✅ **Application Gateway** with WAF protection
✅ **Azure Monitor** and Application Insights
✅ **Azure Key Vault** for secrets management
✅ **CI/CD pipeline** with Azure DevOps/GitHub Actions

**Next Steps:**
1. Monitor costs with Azure Cost Management + Billing
2. Set up geo-redundancy for high availability
3. Implement Azure Front Door for global load balancing
4. Add Azure AD authentication for enterprise security

**Support Resources:**
- [Azure Documentation](https://docs.microsoft.com/azure)
- [Container Apps Docs](https://docs.microsoft.com/azure/container-apps)
- [AKS Docs](https://docs.microsoft.com/azure/aks)
- [Azure Database Docs](https://docs.microsoft.com/azure/postgresql)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator)

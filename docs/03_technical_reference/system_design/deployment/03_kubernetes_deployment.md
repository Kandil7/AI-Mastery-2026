# Kubernetes Deployment Guide - From Zero to Production

## Overview

Kubernetes (K8s) is the industry standard for container orchestration. This guide teaches you how to deploy RAG Engine Mini on Kubernetes, from basic concepts to production-ready clusters with auto-scaling, monitoring, and high availability.

## Learning Objectives

By the end of this module, you will:
1. Understand Kubernetes architecture and core concepts
2. Write and deploy Kubernetes manifests
3. Implement ConfigMaps and Secrets for configuration
4. Set up persistent storage with volumes
5. Configure auto-scaling (HPA) and load balancing
6. Deploy to managed Kubernetes services (EKS, GKE, AKS)
7. Implement monitoring and logging
8. Troubleshoot common Kubernetes issues

**Estimated Time:** 8-10 hours

**Prerequisites:**
- Docker knowledge (Module 2)
- kubectl installed
- Access to a Kubernetes cluster (local or cloud)

---

## Part 1: Kubernetes Fundamentals

### What is Kubernetes?

**Analogy:** If Docker containers are shipping containers, Kubernetes is the port authority that:
- Decides where containers go
- Monitors container health
- Replaces failed containers
- Scales containers based on demand

**Official Definition:** Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.

### Why Kubernetes for RAG Engine?

**Without Kubernetes (Single Server):**
```
Server Crashes → Manual restart
High Load → Manual scaling (buy bigger server)
Update → Downtime during deployment
Failure → Data loss risk
```

**With Kubernetes:**
```
Pod Crashes → Auto-restart (self-healing)
High Load → Auto-scale (add more pods)
Update → Rolling update (zero downtime)
Failure → Distributed (data safe)
```

### Core Kubernetes Concepts

**1. Cluster Architecture:**
```
Kubernetes Cluster
├─ Control Plane (Master)
│  ├─ API Server (receives commands)
│  ├─ etcd (database for cluster state)
│  ├─ Scheduler (places pods on nodes)
│  └─ Controller Manager (maintains desired state)
│
└─ Worker Nodes
   ├─ kubelet (agent, runs pods)
   ├─ Container Runtime (Docker/containerd)
   └─ kube-proxy (networking)
```

**2. Key Resources:**

| Resource | Purpose | Analogy |
|----------|---------|---------|
| **Pod** | Smallest deployable unit (1+ containers) | Apartment (one or more roommates) |
| **Deployment** | Manages pod replicas and updates | Building manager (handles apartments) |
| **Service** | Exposes pods to network | Reception desk (directs visitors) |
| **ConfigMap** | Configuration data | Bulletin board (shared info) |
| **Secret** | Sensitive data (encrypted) | Safe (secure storage) |
| **Ingress** | HTTP/HTTPS routing | Building entrance (with directory) |
| **PersistentVolume** | Storage | Storage unit (survives moves) |

### RAG Engine on Kubernetes - Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Ingress Controller (nginx)              │    │
│  │         Routes: /api/* → API Service                │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐    │
│  │                  API Service                         │    │
│  │         (Load balancer for API pods)                │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│       ┌─────────────┼─────────────┐                         │
│       │             │             │                         │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐                     │
│  │ API Pod │  │ API Pod │  │ API Pod │  (3 replicas)       │
│  │  - App  │  │  - App  │  │  - App  │                     │
│  │  - Logs │  │  - Logs │  │  - Logs │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │                           │
│       └────────────┼────────────┘                           │
│                    │                                        │
│  ┌─────────────────▼──────────────────┐                    │
│  │         External Services           │                    │
│  │  ┌──────────┐ ┌──────────┐        │                    │
│  │  │ PostgreSQL│ │  Redis   │        │                    │
│  │  │ (RDS)    │ │(ElastiCache)       │                    │
│  │  └──────────┘ └──────────┘        │                    │
│  │  ┌──────────┐                      │                    │
│  │  │ Qdrant   │                      │                    │
│  │  │ (Managed)│                      │                    │
│  │  └──────────┘                      │                    │
│  └───────────────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 2: Project Kubernetes Manifests

### File Structure

```
config/kubernetes/
├── namespace.yaml       # Isolation boundary
├── config.yaml          # ConfigMap (non-sensitive config)
├── secrets.yaml         # Secrets (sensitive data)
├── deployment.yaml      # Main application deployment
├── service.yaml         # Service discovery
├── ingress.yaml         # HTTP routing
├── hpa.yaml            # Horizontal Pod Autoscaler
├── pdb.yaml            # Pod Disruption Budget
└── network-policy.yaml # Security rules
```

### 1. Namespace

**What:** Logical isolation within cluster

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-engine
  labels:
    environment: production
    app: rag-engine
```

**Why:**
- Resource isolation (dev/staging/prod)
- Access control boundaries
- Resource quota management

**Commands:**
```bash
# Create namespace
kubectl apply -f namespace.yaml

# List namespaces
kubectl get namespaces

# Switch to namespace
kubectl config set-context --current --namespace=rag-engine
```

### 2. ConfigMap

**What:** Non-sensitive configuration data

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-engine-config
  namespace: rag-engine
data:
  # Application settings
  ENVIRONMENT: "production"
  LOG_LEVEL: "info"
  
  # Service endpoints
  QDRANT_HOST: "qdrant.rag-engine.svc.cluster.local"
  QDRANT_PORT: "6333"
  
  # Performance settings
  DATABASE_POOL_SIZE: "20"
  CACHE_TTL: "300"
  
  # Feature flags
  ENABLE_HYBRID_SEARCH: "true"
  ENABLE_RERANKING: "true"
```

**Usage in Pod:**
```yaml
# As environment variables
env:
  - name: QDRANT_HOST
    valueFrom:
      configMapKeyRef:
        name: rag-engine-config
        key: QDRANT_HOST

# Or mount as files
volumeMounts:
  - name: config
    mountPath: /app/config
volumes:
  - name: config
    configMap:
      name: rag-engine-config
```

### 3. Secrets

**What:** Sensitive data (base64 encoded, encrypted at rest)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-engine-secrets
  namespace: rag-engine
type: Opaque
data:
  # Base64 encoded values
  # echo -n 'postgresql://...' | base64
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0Bsb2NhbGhvc3Q6NTQzMi9yYWdfZW5naW5l
  redis-url: cmVkaXM6Ly9sb2NhbGhvc3Q6NjM3OS8w
  openai-api-key: c2stdGVzdC1rZXktMTIzNDU2Nzg5MA==
  jwt-secret: c3VwZXItc2VjcmV0LWp3dC1rZXktY2hhbmdlLW1l
```

**Create from command line:**
```bash
# Create secret imperatively
kubectl create secret generic rag-engine-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=openai-api-key='sk-...' \
  -n rag-engine

# Or from file
kubectl create secret generic rag-engine-secrets \
  --from-file=./secrets.txt \
  -n rag-engine
```

### 4. Deployment

**What:** Manages pod replicas and updates

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine-api
  namespace: rag-engine
  labels:
    app: rag-engine-api
spec:
  # Number of pod replicas
  replicas: 3
  
  # How to identify pods managed by this deployment
  selector:
    matchLabels:
      app: rag-engine-api
  
  # Pod template
  template:
    metadata:
      labels:
        app: rag-engine-api
    spec:
      # Security
      serviceAccountName: rag-engine
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      containers:
      - name: api
        image: your-registry/rag-engine:v1.0.0
        imagePullPolicy: Always
        
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        
        # Environment variables
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
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: rag-engine-config
              key: ENVIRONMENT
        
        # Resource limits
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # Security
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        
        # Volume mounts
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      
      # Volumes
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir:
          sizeLimit: 1Gi
      
      # Pod lifecycle
      terminationGracePeriodSeconds: 30
```

**Key Features Explained:**

**Replicas:**
- 3 pods running simultaneously
- If one fails, 2 remain (high availability)

**Resource Requests vs Limits:**
```yaml
requests:  # Guaranteed resources
  memory: "256Mi"  # K8s reserves this
  cpu: "250m"      # 0.25 CPU cores

limits:    # Maximum allowed
  memory: "512Mi"  # OOM if exceeded
  cpu: "500m"      # Throttled if exceeded
```

**Probes:**
- **Liveness:** Is the app running? (restart if not)
- **Readiness:** Is the app ready for traffic? (remove from load balancer if not)

### 5. Service

**What:** Exposes pods to network (internal or external)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-engine-api
  namespace: rag-engine
  labels:
    app: rag-engine-api
spec:
  type: ClusterIP  # Internal only
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: rag-engine-api  # Routes to pods with this label
```

**Service Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| **ClusterIP** | Internal cluster access only | Default, internal APIs |
| **NodePort** | Exposes on each node's IP | Direct access, debugging |
| **LoadBalancer** | Cloud load balancer | Public exposure (cloud only) |
| **ExternalName** | DNS alias | External services |

### 6. Ingress

**What:** HTTP/HTTPS routing to services

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-engine-ingress
  namespace: rag-engine
  annotations:
    # NGINX-specific settings
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    
    # Cert-manager for SSL
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: rag-engine-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-engine-api
            port:
              number: 80
```

**Benefits:**
- Single entry point for all HTTP traffic
- SSL termination
- Path-based routing
- Rate limiting

### 7. Horizontal Pod Autoscaler (HPA)

**What:** Automatically scales pods based on metrics

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-engine-api-hpa
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
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

**Scaling Triggers:**
- CPU > 70% → Add pods
- Memory > 80% → Add pods
- Requests/sec > 100 → Add pods
- Scale up fast (15s), scale down slow (5min cooldown)

### 8. Pod Disruption Budget (PDB)

**What:** Ensures minimum availability during disruptions

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rag-engine-api-pdb
  namespace: rag-engine
spec:
  minAvailable: 2  # At least 2 pods must always be running
  selector:
    matchLabels:
      app: rag-engine-api
```

**Use Cases:**
- Node maintenance (draining)
- Cluster upgrades
- Prevents accidental mass pod deletion

---

## Part 3: Step-by-Step Deployment

### Prerequisites

**1. Install kubectl:**
```bash
# macOS
brew install kubectl

# Windows
choco install kubernetes-cli

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

**2. Access a Kubernetes cluster:**

**Option A: Local (Minikube/kind):**
```bash
# Install minikube
brew install minikube

# Start cluster
minikube start --driver=docker --memory=4096 --cpus=2

# Verify
kubectl get nodes
```

**Option B: Cloud (EKS/GKE/AKS):**
```bash
# EKS example
aws eks update-kubeconfig --region us-west-2 --name my-cluster

# Verify
kubectl get nodes
```

**3. Install Helm (optional but recommended):**
```bash
brew install helm
```

### Deployment Steps

**Step 1: Create Namespace**
```bash
kubectl apply -f config/kubernetes/namespace.yaml

# Verify
kubectl get namespaces
```

**Step 2: Create ConfigMap and Secrets**
```bash
# Apply config
kubectl apply -f config/kubernetes/config.yaml

# Create secrets (use imperative for security)
kubectl create secret generic rag-engine-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=openai-api-key='sk-...' \
  -n rag-engine

# Verify
kubectl get configmaps -n rag-engine
kubectl get secrets -n rag-engine
```

**Step 3: Deploy Application**
```bash
kubectl apply -f config/kubernetes/deployment.yaml

# Watch rollout
kubectl rollout status deployment/rag-engine-api -n rag-engine

# Verify pods
kubectl get pods -n rag-engine
```

**Step 4: Create Service**
```bash
kubectl apply -f config/kubernetes/service.yaml

# Verify
kubectl get svc -n rag-engine
```

**Step 5: Setup Ingress (requires Ingress Controller)**
```bash
# Install NGINX Ingress Controller (if not present)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Wait for it to be ready
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

# Apply ingress
kubectl apply -f config/kubernetes/ingress.yaml

# Verify
kubectl get ingress -n rag-engine
```

**Step 6: Enable Autoscaling**
```bash
# Metrics server required for HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Apply HPA
kubectl apply -f config/kubernetes/hpa.yaml

# Verify
kubectl get hpa -n rag-engine
```

---

## Part 4: Production Best Practices

### 1. Resource Management

**Right-sizing:**
```bash
# Check current usage
kubectl top pods -n rag-engine

# Adjust requests/limits based on metrics
```

**Quality of Service Classes:**

| QoS | Requirements | Use Case |
|-----|--------------|----------|
| **Guaranteed** | Limits = Requests | Critical pods |
| **Burstable** | Limits > Requests | Most apps |
| **BestEffort** | No limits set | Low priority |

**Example (Burstable):**
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### 2. Security Hardening

**Pod Security Standards:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

**Network Policies:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: rag-engine
spec:
  podSelector:
    matchLabels:
      app: rag-engine-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

### 3. Monitoring Setup

**Prometheus ServiceMonitor:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: rag-engine-metrics
  namespace: rag-engine
spec:
  selector:
    matchLabels:
      app: rag-engine-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### 4. Backup Strategy

**Velero for Cluster Backup:**
```bash
# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.6.0 \
  --bucket my-backups \
  --backup-location-config region=us-west-2 \
  --snapshot-location-config region=us-west-2 \
  --secret-file ./credentials-velero

# Create backup
velero backup create rag-engine-backup \
  --include-namespaces rag-engine \
  --ttl 720h0m0s

# Schedule daily backups
velero schedule create rag-engine-daily \
  --schedule="0 2 * * *" \
  --include-namespaces rag-engine
```

---

## Part 5: Troubleshooting

### Common Commands

```bash
# Check pod status
kubectl get pods -n rag-engine

# Describe pod (events, errors)
kubectl describe pod <pod-name> -n rag-engine

# View logs
kubectl logs <pod-name> -n rag-engine
kubectl logs <pod-name> -n rag-engine --previous  # Previous container
kubectl logs -l app=rag-engine-api -n rag-engine --tail=100  # All pods

# Execute command in pod
kubectl exec -it <pod-name> -n rag-engine -- /bin/sh

# Port forward for local testing
kubectl port-forward svc/rag-engine-api 8080:80 -n rag-engine

# Check resource usage
kubectl top pods -n rag-engine
kubectl top nodes

# Check events
kubectl get events -n rag-engine --sort-by='.lastTimestamp'
```

### Common Issues

**Issue 1: Pod stuck in Pending**
```bash
# Check events
kubectl describe pod <pod-name> | grep -A 5 Events

# Common causes:
# - Insufficient resources
# - Node selector mismatch
# - PVC not bound
# - Image pull errors
```

**Issue 2: CrashLoopBackOff**
```bash
# Check logs
kubectl logs <pod-name> --previous

# Common causes:
# - App crashing on start
# - Missing environment variables
# - Database connection failure
```

**Issue 3: ImagePullBackOff**
```bash
# Verify image exists
docker pull your-registry/rag-engine:v1.0.0

# Check imagePullSecrets
kubectl get secrets -n rag-engine
```

---

## Next Steps

1. **Install Ingress Controller** on your cluster
2. **Deploy RAG Engine** using manifests above
3. **Setup monitoring** with Prometheus/Grafana
4. **Configure autoscaling** with HPA
5. **Practice troubleshooting** common issues

**Continue to Module 4: Cloud Deployments!** ☁️

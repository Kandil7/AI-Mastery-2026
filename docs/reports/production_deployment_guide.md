# Production LLM Deployment Guide

## Complete Guide to Deploying LLMs at Scale

**Version:** 1.0  
**Last Updated:** March 24, 2026

---

## Table of Contents

1. [Deployment Architecture Overview](#deployment-architecture-overview)
2. [vLLM Production Deployment](#vllm-production-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Docker Compose Stack](#docker-compose-stack)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Scaling Strategies](#scaling-strategies)
7. [Security Hardening](#security-hardening)
8. [Troubleshooting](#troubleshooting)

---

## Deployment Architecture Overview

### Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
│                    (NGINX / HAProxy / ALB)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│                  (Kong / AWS API Gateway)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Rate Limiting│  │   Auth       │  │  Monitoring  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Service Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  vLLM Pod 1  │  │  vLLM Pod 2  │  │  vLLM Pod N  │         │
│  │  (GPU)       │  │  (GPU)       │  │  (GPU)       │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Support Services                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Redis   │  │ Postgres │  │  Qdrant  │  │  Prometheus│      │
│  │  Cache   │  │ Metadata │  │  Vector  │  │  Grafana  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Load Balancer** | Traffic distribution | NGINX, HAProxy, AWS ALB |
| **API Gateway** | Auth, rate limiting, routing | Kong, AWS API Gateway |
| **Inference** | LLM serving | vLLM, TGI, Ollama |
| **Cache** | Response caching | Redis |
| **Vector DB** | RAG embeddings | Qdrant, Pinecone |
| **Monitoring** | Metrics, alerts | Prometheus, Grafana |

---

## vLLM Production Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
RUN pip3 install vllm

# Install additional dependencies
RUN pip3 install \
    prometheus-client \
    redis \
    psycopg2-binary

# Create non-root user
RUN useradd -m -u 1000 vllm
USER vllm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start vLLM
CMD ["python3", "-m", "vllm.entrypoints.api_server", \
     "--model", "meta-llama/Meta-Llama-3-8B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tensor-parallel-size", "1", \
     "--max-num-batched-tokens", "32768", \
     "--enable-prefix-caching"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  vllm-inference:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    volumes:
      - model-cache:/home/vllm/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-storage:/qdrant/storage

volumes:
  model-cache:
  redis-data:
  qdrant-storage:
```

### vLLM Configuration Guide

**Optimal Settings for Production:**

```python
# vLLM production configuration
from vllm import LLM, SamplingParams

llm = LLM(
    # Model configuration
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    
    # Performance optimization
    tensor_parallel_size=4,  # Number of GPUs
    gpu_memory_utilization=0.9,  # 90% GPU memory
    max_num_batched_tokens=32768,  # Max tokens per batch
    max_num_seqs=256,  # Max sequences per batch
    
    # Caching
    enable_prefix_caching=True,  # Critical for RAG
    max_model_len=8192,  # Context length
    
    # Quantization (optional)
    # quantization="awq",  # or "gptq", "fp8"
    
    # Scheduling
    scheduler="continuous",  # Continuous batching
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
    repetition_penalty=1.1,
)
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- NVIDIA GPU nodes
- NVIDIA Container Toolkit
- kubectl configured

### Namespace and RBAC

**namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-inference
  labels:
    name: llm-inference
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: llm-quota
  namespace: llm-inference
spec:
  hard:
    requests.cpu: "32"
    requests.memory: 128Gi
    requests.nvidia.com/gpu: "8"
    limits.cpu: "64"
    limits.memory: 256Gi
    limits.nvidia.com/gpu: "16"
```

### ConfigMap and Secrets

**configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
  namespace: llm-inference
data:
  MODEL_NAME: "meta-llama/Meta-Llama-3-8B-Instruct"
  TENSOR_PARALLEL_SIZE: "2"
  GPU_MEMORY_UTILIZATION: "0.9"
  MAX_NUM_BATCHED_TOKENS: "32768"
  ENABLE_PREFIX_CACHING: "true"
  MAX_MODEL_LEN: "8192"
  LOG_LEVEL: "info"
```

**secret.yaml:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: vllm-secrets
  namespace: llm-inference
type: Opaque
stringData:
  HUGGING_FACE_HUB_TOKEN: "hf_xxx"
  REDIS_URL: "redis://redis:6379"
  DATABASE_URL: "postgresql://user:pass@postgres:5432/llm"
```

### Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: llm-inference
  labels:
    app: vllm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
    spec:
      serviceAccountName: vllm-service-account
      containers:
      - name: vllm
        image: vllm-inference:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: MODEL_NAME
        - name: TENSOR_PARALLEL_SIZE
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: TENSOR_PARALLEL_SIZE
        - name: GPU_MEMORY_UTILIZATION
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: GPU_MEMORY_UTILIZATION
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: vllm-secrets
              key: HUGGING_FACE_HUB_TOKEN
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 24Gi
            cpu: 4
          limits:
            nvidia.com/gpu: 1
            memory: 48Gi
            cpu: 8
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: model-cache
          mountPath: /home/vllm/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu
                operator: Exists
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: llm-inference
spec:
  selector:
    app: vllm-inference
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service-external
  namespace: llm-inference
spec:
  selector:
    app: vllm-inference
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

**hpa.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: llm-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
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
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

### Ingress

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: llm-inference
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - llm-api.example.com
    secretName: llm-tls-secret
  rules:
  - host: llm-api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-service
            port:
              number: 80
```

---

## Monitoring and Observability

### Prometheus Configuration

**prometheus-config.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: llm-inference
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
    - job_name: 'vllm'
      static_configs:
      - targets: ['vllm-service:8000']
      metrics_path: '/metrics'
    
    - job_name: 'node-exporter'
      static_configs:
      - targets: ['node-exporter:9100']
```

### Key Metrics to Monitor

```yaml
# Important vLLM metrics
metrics:
  # Performance
  - vllm:num_requests_running
  - vllm:num_requests_waiting
  - vllm:time_to_first_token_seconds
  - vllm:time_per_output_token_seconds
  - vllm:gpu_cache_usage_perc
  
  # Resource
  - vllm:gpu_memory_usage_bytes
  - vllm:cpu_memory_usage_bytes
  
  # Custom metrics (implement in your app)
  - custom:requests_per_second
  - custom:average_latency_ms
  - custom:error_rate
  - custom:cost_per_request
```

### Grafana Dashboard

**dashboard.json:**
```json
{
  "dashboard": {
    "title": "LLM Inference Dashboard",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [
          {
            "expr": "rate(vllm:num_requests_running[1m])"
          }
        ]
      },
      {
        "title": "P99 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "vllm:gpu_memory_usage_bytes / vllm:gpu_memory_total_bytes * 100"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(custom:cache_hits[5m]) / rate(custom:cache_requests[5m]) * 100"
          }
        ]
      }
    ]
  }
}
```

---

## Scaling Strategies

### Vertical Scaling

**GPU Selection Guide:**

| Model Size | Minimum GPU | Recommended | Multi-GPU |
|------------|-------------|-------------|-----------|
| **7B** | RTX 3090 (24GB) | A10 (24GB) | No |
| **13B** | A100 (40GB) | A100 (80GB) | Optional |
| **30B** | A100 (80GB) | 2× A100 (80GB) | Yes |
| **70B** | 2× A100 (80GB) | 4× A100 (80GB) | Yes |
| **175B+** | 8× A100 (80GB) | 8× H100 (80GB) | Yes |

### Horizontal Scaling

**Auto-scaling Configuration:**

```yaml
# KEDA ScaledObject for advanced autoscaling
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaledobject
  namespace: llm-inference
spec:
  scaleTargetRef:
    name: vllm-inference
  minReplicaCount: 2
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: requests_per_second
      query: sum(rate(vllm:num_requests_running[1m]))
      threshold: '100'
  - type: cpu
    metadata:
      value: '70'
  advanced:
    restoreToOriginalReplicaCount: true
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
```

### Cost Optimization

**Spot Instance Strategy:**

```yaml
# Node pool with spot instances
apiVersion: v1
kind: NodePool
metadata:
  name: gpu-spot-pool
spec:
  template:
    spec:
      taints:
      - key: spot
        value: "true"
        effect: NoSchedule
  autoscaling:
    enabled: true
    minNodeCount: 2
    maxNodeCount: 10
  nodeConfig:
    instanceTypes:
    - g5.2xlarge  # On-demand fallback
    - g5.2xlarge-spot  # Spot primary
```

---

## Security Hardening

### Network Policies

**network-policy.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-network-policy
  namespace: llm-inference
spec:
  podSelector:
    matchLabels:
      app: vllm-inference
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: llm-inference
    ports:
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 5432  # Postgres
```

### Pod Security

**pod-security-policy.yaml:**
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: vllm-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
  - ALL
  volumes:
  - 'persistentVolumeClaim'
  - 'secret'
  - 'configMap'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: false
```

### Secrets Management

```yaml
# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: vllm-external-secret
  namespace: llm-inference
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: vllm-secrets
    creationPolicy: Owner
  data:
  - secretKey: HUGGING_FACE_HUB_TOKEN
    remoteRef:
      key: llm/huggingface-token
  - secretKey: DATABASE_URL
    remoteRef:
      key: llm/database-url
```

---

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory (OOM)**

```yaml
# Solution: Reduce GPU memory utilization
env:
- name: GPU_MEMORY_UTILIZATION
  value: "0.8"  # Reduce from 0.9 to 0.8
- name: MAX_MODEL_LEN
  value: "4096"  # Reduce context length
```

**Issue 2: Slow Inference**

```bash
# Check GPU utilization
kubectl top pods -n llm-inference

# Check vLLM metrics
curl http://vllm-service:8000/metrics | grep vllm

# Enable continuous batching
# Ensure: --scheduler=continuous
```

**Issue 3: High Latency**

```yaml
# Solution: Increase replicas
spec:
  replicas: 5  # Increase from 3 to 5
  
# Or enable prefix caching
env:
- name: ENABLE_PREFIX_CACHING
  value: "true"
```

### Debugging Commands

```bash
# Check pod status
kubectl get pods -n llm-inference

# View logs
kubectl logs -f deployment/vllm-inference -n llm-inference

# Exec into pod
kubectl exec -it deployment/vllm-inference -n llm-inference -- bash

# Check GPU allocation
kubectl describe node <gpu-node-name>

# Monitor metrics
kubectl port-forward svc/prometheus 9090 -n llm-inference
```

---

## Complete Deployment Script

**deploy.sh:**
```bash
#!/bin/bash

set -e

NAMESPACE="llm-inference"
IMAGE_NAME="vllm-inference"
IMAGE_TAG="latest"

echo "🚀 Starting LLM Deployment..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Push to registry (optional)
# docker push ${IMAGE_NAME}:${IMAGE_TAG}

# Create namespace
echo "🏷️  Creating namespace..."
kubectl apply -f kubernetes/namespace.yaml

# Apply ConfigMap and Secrets
echo "🔐 Applying ConfigMap and Secrets..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secret.yaml

# Deploy application
echo "🚀 Deploying application..."
kubectl apply -f kubernetes/deployment.yaml

# Create service
echo "🌐 Creating service..."
kubectl apply -f kubernetes/service.yaml

# Setup HPA
echo "📊 Setting up autoscaling..."
kubectl apply -f kubernetes/hpa.yaml

# Wait for deployment
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/vllm-inference -n ${NAMESPACE}

# Check status
echo "✅ Deployment complete!"
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}

echo "🎉 LLM inference is ready!"
```

---

## Conclusion

This guide provides a complete production deployment setup for LLMs using vLLM and Kubernetes. Key takeaways:

1. **Use vLLM** for high-throughput inference with PagedAttention
2. **Enable prefix caching** for RAG workloads (400%+ improvement)
3. **Monitor everything** - GPU memory, latency, cache hit rate
4. **Auto-scale** based on demand (requests per second)
5. **Secure** with network policies, RBAC, and secrets management
6. **Optimize costs** with spot instances and right-sizing

---

**Last Updated:** March 24, 2026  
**Version:** 1.0  
**Maintained by:** LLM Engineering Team

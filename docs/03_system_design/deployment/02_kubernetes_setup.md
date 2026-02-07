# Kubernetes Setup Guide - Complete Implementation
# =============================================

## Overview

This guide covers Kubernetes deployment of RAG Engine.

## Prerequisites

- kubectl CLI installed
- Kubernetes cluster running (minikube, GKE, EKS, AKS)
- Docker image built and pushed to registry

## Quick Start

```bash
# Apply all manifests
kubectl apply -f config/kubernetes/

# Verify deployment
kubectl get pods -n rag-engine

# Check services
kubectl get svc -n rag-engine

# View logs
kubectl logs -f deployment/rag-engine -n rag-engine

# Port forward for local access
kubectl port-forward svc/rag-engine-service 8000:80 -n rag-engine
```

## Components

- **Deployment**: Pod spec with HPA
- **Service**: LoadBalancer
- **Ingress**: TLS with Let's Encrypt
- **Secrets**: Database, Redis, Qdrant, OpenAI keys
- **ConfigMap**: Environment configuration
- **PVC**: Persistent storage
- **NetworkPolicy**: Pod-to-pod communication

## Scaling

HPA configured to scale:
- Min replicas: 2
- Max replicas: 10
- Metric: CPU utilization > 70%

## Monitoring

Access metrics at:
```
http://loadbalancer-url/metrics
```

## Troubleshooting

**Pod not starting:**
```bash
kubectl describe pod <pod-name> -n rag-engine
kubectl logs <pod-name> -n rag-engine
```

**Service not accessible:**
```bash
kubectl describe svc rag-engine-service -n rag-engine
kubectl get endpoints rag-engine-service -n rag-engine
```

---
**Document Version:** 1.0
**Last Updated:** 2026-01-31
**Author:** AI-Mastery-2026

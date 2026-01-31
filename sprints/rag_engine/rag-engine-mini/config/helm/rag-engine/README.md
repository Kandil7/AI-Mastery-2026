# Helm Chart for RAG Engine
# مخطط Helm لمحرك RAG

---

## Overview / نظرة عامة

This Helm chart deploys the RAG Engine application with all necessary Kubernetes resources.

هذا المخطط ينشر تطبيق محرك RAG مع جميع الموارد Kubernetes الضرورية.

## Installation / التثبيت

```bash
# Add the chart
helm repo add rag-engine ./config/helm/rag-engine

# Install
helm install rag-engine ./config/helm/rag-engine --namespace rag-engine
```

## Values / القيم

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Docker image repository | `rag-engine` |
| `image.tag` | Docker image tag | `latest` |
| `replicaCount` | Number of replicas | `3` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `2Gi` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Min replicas for HPA | `2` |
| `autoscaling.maxReplicas` | Max replicas for HPA | `10` |
| `ingress.enabled` | Enable Ingress | `true` |
| `ingress.host` | Ingress host | `rag.example.com` |
| `persistence.enabled` | Enable PVC | `true` |
| `persistence.storageSize` | Storage size | `10Gi` |

| المعلمة | الوصف | الافتراضي |
|-----------|-------------|---------|
| `image.repository` | مستودع الصورة Docker | `rag-engine` |
| `image.tag` | وسم الصورة Docker | `latest` |
| `replicaCount` | عدد النسخ المكررة | `3` |
| `resources.requests.cpu` | طلب وحدة المعالجة | `500m` |
| `resources.requests.memory` | طلب الذاكرة | `1Gi` |
| `resources.limits.cpu` | حد وحدة المعالجة | `1000m` |
| `resources.limits.memory` | حد الذاكرة | `2Gi` |
| `autoscaling.enabled` | تفعيل HPA | `true` |
| `autoscaling.minReplicas` | الحد الأدنى للنسخ | `2` |
| `autoscaling.maxReplicas` | الحد الأقصى للنسخ | `10` |
| `ingress.enabled` | تفعيل الدخول | `true` |
| `ingress.host` | اسم المضيف | `rag.example.com` |
| `persistence.enabled` | تفعيل التخزين الدائم | `true` |
| `persistence.storageSize` | حجم التخزين | `10Gi` |

---

## Resources / الموارد

The chart creates the following Kubernetes resources:
- Namespace
- ConfigMap
- Secret
- PersistentVolumeClaim
- Deployment
- Service
- HorizontalPodAutoscaler
- PodDisruptionBudget
- NetworkPolicy

ينشئ هذا المخطط موارد Kubernetes التالية:
- Namespace
- ConfigMap
- Secret
- PersistentVolumeClaim
- Deployment
- Service
- HorizontalPodAutoscaler
- PodDisruptionBudget
- NetworkPolicy

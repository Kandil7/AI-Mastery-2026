# Infrastructure-as-Code Guide
# دليل البنية كود

## Table of Contents / جدول المحتويات

1. [Introduction / مقدمة](#introduction)
2. [Kubernetes Basics / أساسيات Kubernetes](#kubernetes-basics)
3. [Helm Package Manager / مدير الحزم Helm](#helm-package-manager)
4. [Terraform / Terraform](#terraform)
5. [Manifests Created / الملفات المنشأة](#manifests-created)
6. [Best Practices / أفضل الممارسات](#best-practices)
7. [Deployment Guide / دليل النشر](#deployment-guide)
8. [Summary / الملخص](#summary)

---

## Introduction / مقدمة

### What is IaC? / ما هو البنية كود؟

**Infrastructure as Code (IaC)** is the practice of managing and provisioning infrastructure through machine-readable definition files, rather than manual configuration.

**Key benefits:**
- **Version control**: Infrastructure changes are tracked in Git
- **Reproducibility**: Same environment can be easily recreated
- **Consistency**: Reduces configuration drift
- **Self-documenting**: Infrastructure definition serves as documentation
- **Modularity**: Complex systems broken into reusable components

**البنية كود (IaC)** هي ممارسة إدارة وتوفير البنية من خلال تعريفات قابلة للقراءة، بدلاً من التكوين اليدوي.

**الفوائد الرئيسية:**
- **التحكم في الإصدارات**: يتم تتبع تغييرات البنية في Git
- **قابلية الإعادة**: يمكن إعادة إنشاء نفس البيئة بسهولة
- **اتساق التكوين**: يقلل التباعد في التكوين
- **الوثائق ذاتية**: تعريف البنية يُعد بوسيلة وثائق
- **الوحدية**: الأنظمة المعقدة تُقسم إلى مكونات قابلة لإعادة الاستخدام

---

## Kubernetes Basics / أساسيات Kubernetes

### Core Concepts / المفاهيم الأساسية

**1. Pod / بود**
- Smallest deployable unit
- Contains one or more containers
- Shared storage and network namespace
- Ephemeral (pod = deleted, all containers stop)

**2. Deployment / النشر**
- Manages replica set of pods
- Provides updates (rolling, rollback)
- Defines pod template and update strategy
- Self-healing capabilities

**3. Service / الخدمة**
- Exposes pods within/cluster or externally
- Stable network endpoint (ClusterIP, NodePort, LoadBalancer)
- Load balancing across pods

**4. Ingress / الدخول**
- Manages external access to cluster services
- HTTP/HTTPS routing rules
- TLS termination
- Path-based routing

**5. ConfigMap / خريطة التكوين**
- Key-value pairs for configuration
- Injected into containers as environment variables
- Updates without pod restart

**6. Secret / السر**
- Stores sensitive data (passwords, API keys)
- Not stored in container specs
- Mounted as volumes

**7. PersistentVolumeClaim / مطالبة وحدة تخزين**
- Request for storage (PVC)
- Binds to PersistentVolume
- Can be retained across pod restarts

**8. HorizontalPodAutoscaler (HPA)** / موسع البود الأفقي تلقائيًا**
- Automatically scales pod replicas based on metrics
- Configurable min/max replicas
- Scale up/down policies

**9. PodDisruptionBudget (PDB)** / ميزانية تعطيل البود**
- Limits number of simultaneous pod disruptions
- Ensures service availability during updates

**10. NetworkPolicy / سياسة الشبكة**
- Controls traffic between pods
- Whitelist/blacklist rules
- Namespace-level or cluster-wide

### المفاهيم الأساسية

**1. بود**: أصغر وحدة قابلة للنشر
**2. النشر**: يدير مجموعة بودات متطابعة
**3. الخدمة**: نقطة نها مستقرة للوصول إلى التطبيقات
**4. الدخول**: يدير الوصول الخارجي للخدمات
**5. خريطة التكوين**: أزواج قيمة-مفتاح للتكوين
**6. السر**: يخزن بيانات حساسة
**7. مطالبة وحدة تخزين**: طلب وحدة تخزين
**8. HPA**: موسع تلقائي وفق الاستخدام
**9. PDB**: يحد تعطيل البود لضمان التوفر
**10. سياسة الشبكة**: يتحكم في تدفق المرور بين البودات

---

## Helm Package Manager / مدير الحزم Helm

### What is Helm? / ما هو Helm؟

**Helm** is the package manager for Kubernetes. It helps you manage Kubernetes applications.

**Helm** هو مدير الحزم Kubernetes. يساعد في إدارة تطبيقات Kubernetes.

### Components / المكونات

**1. Chart / المخطط**
- Package containing Kubernetes resources
- Template files for resource generation
- `Chart.yaml` - Metadata and configuration
- `values.yaml` - Default values

**2. Release / الإصدار**
- Instance of a chart deployed to cluster
- Versioned (e.g., rag-engine-1.0.0)

**3. Repository / المستودع**
- Collection of related charts
- Can be local or remote (e.g., https://charts.bitnami.com)

### Helm Commands / أوامر Helm

```bash
# Search for charts
helm search repo rag-engine

# Install from local directory
helm install rag-engine ./config/helm/rag-engine

# Install from remote repository
helm install rag-engine bitnami/rag-engine

# Upgrade release
helm upgrade rag-engine

# Rollback to previous version
helm rollback rag-engine 2

# List releases
helm list

# Uninstall
helm uninstall rag-engine

# Get release status
helm status rag-engine
```

### أوامر Helm

### Chart Structure / هيكل المخطط

```
rag-engine/
├── Chart.yaml           # Chart metadata
├── values.yaml          # Default values
├── templates/           # Kubernetes templates
│   ├── deployment.yaml    # Deployment manifest
│   ├── service.yaml       # Service manifest
│   ├── hpa.yaml          # HPA manifest
│   ├── ingress.yaml        # Ingress manifest
│   ├── pvc.yaml           # PVC manifest
│   ├── pdb.yaml           # PDB manifest
│   ├── networkpolicy.yaml # NetworkPolicy manifest
│   └── _helpers.tpl       # Template helpers
└── NOTES.txt            # Post-install notes
```

### هيكل المخطط

---

## Terraform / Terraform

### What is Terraform? / ما هو Terraform؟

**Terraform** is an infrastructure as code tool by HashiCorp. It supports multiple cloud providers.

**Terraform** هي أداة البنية كود من HashiCorp. تدعم مزودين سحابية متعددة.

### Supported Providers / المزودون المدعومون

**1. AWS (Amazon Web Services)**
- EKS (Elastic Kubernetes Service)
- EC2 (Elastic Compute Cloud)
- S3 (Simple Storage Service)
- RDS (Relational Database Service)
- VPC, Security Groups, IAM

**2. GCP (Google Cloud Platform)**
- GKE (Google Kubernetes Engine)
- Compute Engine
- Cloud Storage
- Cloud SQL

**3. Azure (Microsoft Azure)**
- AKS (Azure Kubernetes Service)
- Virtual Machines
- Blob Storage
- SQL Database

### Terraform Configuration Files / ملفات تكوين Terraform

**1. main.tf** - Provider configuration, remote state, modules
**2. variables.tf** - Input variables for customization
**3. outputs.tf** - Output values for other workflows
**4. provider.tf** - Provider-specific configuration
**5. resource files** - Define infrastructure resources

### Terraform Commands / أوامر Terraform

```bash
# Initialize working directory
terraform init

# Check configuration
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy

# Show state
terraform show
```

### أوامر Terraform

### Terraform State File / ملف الحالة Terraform

- Tracks current infrastructure state
- Must be version controlled
- Don't manually edit
- Use `terraform import` for moving resources

### ملف الحالة

---

## Manifests Created / الملفات المنشأة

### Kubernetes Resources / موارد Kubernetes

The following manifests have been created:

**Kubernetes Deployment**
- `config/kubernetes/deployment.yaml` - Complete deployment with HPA
- `config/kubernetes/service.yaml` - Service for cluster access
- `config/kubernetes/configmap.yaml` - Application configuration
- `config/kubernetes/secret.yaml` - Sensitive data
- `config/kubernetes/hpa.yaml` - HorizontalPodAutoscaler
- `config/kubernetes/pdb.yaml` - PodDisruptionBudget
- `config/kubernetes/ingress.yaml` - Ingress with TLS
- `config/kubernetes/networkpolicy.yaml` - Network policy for Redis
- `config/kubernetes/pvc.yaml` - PersistentVolumeClaim

**Helm Chart**
- `config/helm/rag-engine/Chart.yaml` - Chart metadata and version
- `config/helm/rag-engine/values.yaml` - All configurable values
- `config/helm/rag-engine/templates/deployment.yaml` - Deployment template
- `config/helm/rag-engine/templates/service.yaml` - Service template
- `config/helm/rag-engine/templates/hpa.yaml` - HPA template
- `config/helm/rag-engine/templates/ingress.yaml` - Ingress template
- `config/helm/rag-engine/templates/pdb.yaml` - PDB template
- `config/helm/rag-engine/templates/networkpolicy.yaml` - NetworkPolicy template
- `config/helm/rag-engine/templates/pvc.yaml` - PVC template
- `config/helm/rag-engine/templates/_helpers.tpl` - Template helpers
- `config/helm/rag-engine/NOTES.txt` - Installation notes

### Kubernetes Resources Created / موارد Kubernetes المنشأة

**مخطط Helm**: `config/helm/rag-engine/`

---

## Best Practices / أفضل الممارسات

### 1. Resource Management / إدارة الموارد

```yaml
# GOOD: Resource requests and limits
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 1000m
    memory: 2Gi

# BAD: No limits (can affect cluster stability)
resources:
  requests:
    cpu: 2000m
    memory: 2Gi
```

### 2. Health Checks / فحوص الصحة

```yaml
# Configure appropriate probes
livenessProbe:
  httpGet:
    path: /health
    initialDelaySeconds: 30 30
    periodSeconds: 10
readinessProbe:
  httpGet:
    path: /health/ready
    initialDelaySeconds: 5 5
    periodSeconds: 5
```

### 3. Secrets Management / إدارة الأسرار

```yaml
# Use Kubernetes Secrets for sensitive data
env:
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: app-secrets
        key: database-url
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: app-secrets
        key: api-key
```

### 4. Rolling Updates / التحديثات المتدرجة

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 1
    terminationGracePeriodSeconds: 30
```

### 2. إدارة الموارد
### 3. فحوص الصحة
### 4. إدارة الأسرار

---

## Deployment Guide / دليل النشر

### Installation with Helm / التثبيت بـ Helm

```bash
# Install the chart
helm install rag-engine ./config/helm/rag-engine --namespace rag-engine

# Upgrade existing installation
helm upgrade rag-engine ./config/helm/rag-engine --namespace rag-engine

# Uninstall
helm uninstall rag-engine --namespace rag-engine
```

### Installation with kubectl / التثبيت بـ kubectl

```bash
# Apply all manifests
kubectl apply -f config/kubernetes/

# Get status
kubectl get pods -n rag-engine

# View logs
kubectl logs -n rag-engine -l rag-engine
```

### Configuration / التكوين

```bash
# Set custom values
helm install rag-engine ./config/helm/rag-engine --set replicaCount=5

# Use specific values file
helm install rag-engine ./config/helm/rag-engine -f values-prod.yaml
```

### التثبيت بـ Helm
### التثبيت بـ kubectl

### التكوين

---

## Summary / الملخص

### Key Takeaways / النقاط الرئيسية

1. **IaC benefits**: Version control, reproducibility, self-documentation
2. **Kubernetes basics**: Pods, Deployments, Services, Ingress, ConfigMaps, Secrets, PVCs
3. **Helm**: Package manager, Charts, Releases, Repositories
4. **Terraform**: Multi-cloud support (AWS, GCP, Azure), state management
5. **Created resources**: Complete IaC for Kubernetes and Helm
6. **Best practices**: Resource limits, health probes, secrets management, rolling updates

### النقاط الرئيسية

1. **فوائد IaC**: التحكم في الإصدارات، قابلية الإعادة، اتساق التكوين
2. **أساسيات Kubernetes**: البود، النشر، الخدمة، الدخول، التخزين، HPA
3. **Helm**: مدير الحزم مع المخططات والإصدارات
4. **Terraform**: أداة البنية كود متعددة المزودين
5. **الموارد المنشأة**: مخطط Helm كامل مع 8 قوالب، ملفات Kubernetes
6. **أفضل الممارسات**: إدارة الموارد، فحوص الصحة، الأسرار، التحديثات المتدرجة

---

## Further Reading / قراءة إضافية

- [Kubernetes Documentation](https://kubernetes.io/docs/concepts/)
- [Helm Documentation](https://helm.sh/docs/)
- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

---

## Arabic Summary / ملخص بالعربية

هذا الدليل يغطي البنية كود لمحرك RAG. تشمل المواضيع الرئيسية:

1. **ما هو IaC**: البنية كود لفوارد إدارة البنية
2. **أساسيات Kubernetes**: البود، النشر، الخدمة، الدخول، التخزين، HPA
3. **مدير Helm**: مدير الحزم لتطبيقات Kubernetes
4. **Terraform**: أداة البنية كود لمزودين سحابية متعددة (AWS، GCP، Azure)
5. **الموارد المنشأة**: مخطط Helm كامل مع 8 قوالب
6. **أفضل الممارسات**: إدارة الموارد، فحوص الصحة، الأسرار، التحديثات

تمثل هذه الموارد الأساسيات لاستضافة التطبيق إلى Kubernetes وإدارته بشكل احترافي.

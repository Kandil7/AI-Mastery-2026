# GCP Deployment Guide
# =====================
# Complete guide for deploying RAG Engine to GCP using Terraform.

# دليل نشر RAG Engine إلى Google Cloud باستخدام Terraform

## Prerequisites / المتطلبات

1. **GCP Project** with appropriate IAM roles
2. **Terraform** >= 1.3.0 installed locally
3. **gcloud CLI** configured with credentials
4. **kubectl** for cluster management
5. **Helm** for application deployment

## Installation / التثبيت

```bash
# Install Terraform
brew install terraform  # macOS
wget -O- https://releases.hashicorp.com/terraform/pool/main/main.html  # Linux

# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## Configuration / التكوين

### 1. Authenticate with GCP

```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project rag-engine-prod

# Get application default credentials
gcloud auth application-default login
```

### 2. Customize Variables

Create `terraform.tfvars`:

```hcl
# Project
gcp_project_id = "rag-engine-prod"
gcp_region = "us-central1"
gcp_zone = "us-central1-a"
environment = "prod"

# GKE Cluster
gke_node_type = "e2-large"
gke_min_nodes = 3
gke_max_nodes = 10
gke_pod_cidr = "10.64.0.0/14"
gke_master_cidr = "10.128.0.0/28"
node_disk_size_gb = 100
preemptible = true

# Database
postgres_version = "POSTGRES_14"
cloudsql_tier = "db-custom-2-3840"
storage_gb = 500
disk_autoresize = true

# Redis
redis_memory_gb = 16
redis_tier = "STANDARD_HA"
redis_node_count = 2
```

## Deployment / النشر

### 1. Initialize Terraform

```bash
cd terraform/gcp

terraform init
```

### 2. Plan Changes

```bash
terraform plan \
  -var-file="terraform.tfvars" \
  -out="tfplan"
```

### 3. Apply Configuration

```bash
terraform apply \
  -var-file="terraform.tfvars" \
  -auto-approve
```

Expected outputs:
```
gke_cluster_endpoint = "https://34.123.45.67"
gke_cluster_name = "gke-rag-engine-prod"
cloudsql_connection_name = "rag-engine-prod"
redis_endpoint = "10.1.2.3:6379"
gcs_bucket_name = "rag-engine-prod-uploads"
```

### 4. Configure kubectl

```bash
# Get cluster credentials
gcloud container clusters get-credentials rag-engine-prod \
  --region us-central1

# Verify connection
kubectl get nodes
kubectl get namespaces
```

## Application Deployment / نشر التطبيق

### 1. Create Namespace

```bash
kubectl create namespace rag-engine
```

### 2. Deploy RAG Engine with Helm

```bash
cd helm/rag-engine

helm install rag-engine \
  --namespace rag-engine \
  --values values-prod.yaml \
  .
```

## Cost Estimation / تقدير التكلفة

| Resource | Type | Monthly Cost (USD) |
|----------|------|-------------------|
| VPC & Networking | N/A | ~$20 |
| GKE Cluster | Control Plane | $73/month |
| GKE Nodes (3x e2-large) | Preemptible | ~$86/month |
| Cloud SQL | db-custom-2-3840 (500GB) | ~$156/month |
| Memorystore Redis | 2x STANDARD_HA (16GB) | ~$88/month |
| GCS Storage | 1TB Standard | ~$20/month |
| **Total** | | **~$443/month** |

*Prices are estimates as of 2026-01-31.*

## Security Considerations / اعتبارات الأمان

1. **Network Isolation**
   - Private subnets for Cloud SQL and Memorystore
   - VPC native networking for GKE
   - Service account with least privilege

2. **Encryption**
   - GCS bucket encryption by default
   - Cloud SQL encryption at rest enabled
   - Memorystore encryption in transit

3. **Access Control**
   - IAM roles with least privilege
   - GCS bucket policies restrict to GKE service account
   - Database credentials in Secret Manager

## Monitoring / المراقبة

### Cloud Monitoring Dashboards

```bash
# Create monitoring dashboard
gcloud monitoring dashboards create \
  --config-file dashboard.json
```

### Alerts

```bash
# Create alert for high CPU
gcloud alpha monitoring policies create \
  --display-name "GKE High CPU" \
  --policy-yaml gke-high-cpu-policy.yaml
```

## Troubleshooting / استكشاف الأخطاء

### 1. GKE Cluster Issues

```bash
# Check cluster status
gcloud container clusters describe rag-engine-prod

# Check node status
kubectl get nodes -o wide

# Check control plane logs
gcloud container clusters logs rag-engine-prod
```

### 2. Database Connection Issues

```bash
# Check SQL instance status
gcloud sql instances describe rag-engine-prod

# Test connection
telnet 10.1.2.3 5432

# Check logs
gcloud sql instances logs tail rag-engine-prod
```

### 3. GCS Access Issues

```bash
# Check bucket IAM
gcloud storage buckets describe gs://rag-engine-prod-uploads \
  --format json

# Verify access
gsutil ls gs://rag-engine-prod-uploads
```

## Cleanup / التنظيف

```bash
# Destroy Helm release
helm uninstall rag-engine --namespace rag-engine

# Destroy infrastructure
terraform destroy \
  -var-file="terraform.tfvars" \
  -auto-approve
```

## Best Practices / أفضل الممارسات

1. **Preemptible Nodes** - Use preemptible GKE nodes for cost savings
2. **Regional Clusters** - Multi-zone GKE for availability
3. **Connection Pooling** - Configure database connection pooling
4. **Auto Scaling** - Enable GKE cluster autoscaling
5. **Backup Strategy** - Enable Cloud SQL automated backups
6. **Monitoring** - Set up Cloud Monitoring dashboards and alerts
7. **Security** - Rotate credentials regularly, audit IAM policies
8. **Documentation** - Document all changes and procedures

---

**Last Updated:** 2026-01-31
**Terraform Version:** 1.3.0+

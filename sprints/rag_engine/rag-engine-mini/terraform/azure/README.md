# Azure Deployment Guide
# ======================
# Complete guide for deploying RAG Engine to Azure using Terraform.

# دليل نشر RAG Engine إلى Azure باستخدام Terraform

## Prerequisites / المتطلبات

1. **Azure Account** with appropriate permissions
2. **Terraform** >= 1.3.0 installed locally
3. **Azure CLI** configured with credentials
4. **kubectl** for cluster management
5. **Helm** for application deployment

## Installation / التثبيت

```bash
# Install Terraform (macOS)
brew install terraform

# Install Terraform (Linux)
wget -O- https://apt.releases.hashicorp.com/terraform/pool/main/main.html
sudo unzip main && sudo mv terraform /usr/local/bin/

# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCli | bash

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## Configuration / التكوين

### 1. Create Service Principal

```bash
# Create service principal for Terraform
az ad sp create-for-rbac \
  --name rag-engine-terraform \
  --role contributor \
  --scopes /subscriptions/{subscription-id} \
  --sdk-auth

# Note the output values:
# AppId (client_id)
# Password (client_secret)
```

### 2. Customize Variables

Create `terraform.tfvars`:

```hcl
# Azure Configuration
subscription_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
tenant_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
client_id = "your-client-id"
client_secret = "your-client-secret"

# Environment
environment = "prod"
project_name = "rag-engine"
azure_location = "eastus"

# Networking
vnet_cidr = "10.0.0.0/16"
subnet_prefixes = ["private", "public"]

# AKS Cluster
aks_node_type = "Standard_DS3_v2"
aks_min_nodes = 3
aks_max_nodes = 10
aks_desired_nodes = 3
aks_pod_cidr = "10.244.0.0/14"
aks_service_cidr = "10.245.0.0/16"

# Database
database_server_name = "ragengine-sql"
database_sku_name = "GP_Gen5_2_v8"
storage_mb = 512000

# Redis
cache_family = "P"
cache_sku_name = "Basic"
cache_capacity_gb = 16

# Storage
storage_account_name = "ragengineuploads"
container_name = "uploads"
replication_type = "LRS"
```

## Deployment / النشر

### 1. Initialize Terraform

```bash
cd terraform/azure

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
aks_cluster_endpoint = "rag-engine-prod.eastus.cloudapp.azure.com"
aks_cluster_name = "rag-engine-prod"
database_fqdn = "ragengine-sql.postgres.database.azure.com"
cache_host = "rag-engine-prod-redis.redis.cache.windows.net"
storage_account_name = "ragengineuploadsprod"
```

### 4. Configure kubectl

```bash
# Get AKS credentials
az aks get-credentials \
  --name rag-engine-prod \
  --resource-group rag-engine-prod-rg

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
| AKS Cluster | Control Plane | $73/month |
| AKS Nodes (3x DS3_v2) | Standard | ~$408/month |
| Database (PostgreSQL) | GP_Gen5_2_v8 (500GB) | ~$345/month |
| Redis Cache (Basic) | 16GB | ~$105/month |
| Blob Storage | LRS, 1TB | ~$20/month |
| **Total** | | **~$971/month** |

*Prices are estimates as of 2026-01-31.*

## Security Considerations / اعتبارات الأمان

1. **Network Isolation**
   - Private subnets for database and cache
   - NSG rules restrict access to AKS
   - VNet integration for private communication

2. **Encryption**
   - Storage account encryption by default
   - Database SSL/TLS enabled
   - Redis SSL/TLS enabled
   - AKS secrets manager for sensitive data

3. **Access Control**
   - Managed identity for AKS
   - RBAC for Azure resources
   - Role assignments for storage access
   - Database credentials in Key Vault

## Monitoring / المراقبة

### Azure Monitor Dashboards

```bash
# Create monitoring dashboard
az monitor dashboard create \
  --name rag-engine-metrics \
  --resource-group rag-engine-prod-rg
```

### Alerts

```bash
# Create alert for high CPU
az monitor metrics alert create \
  --name aks-high-cpu \
  --resource-id /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.ContainerService/managedClusters/{cluster} \
  --condition "avg Percentage CPU > 80" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action-groups /subscriptions/{sub}/resourceGroups/{rg}/providers/microsoft.insights/actionGroups/{name}
```

## Troubleshooting / استكشاف الأخطاء

### 1. AKS Cluster Issues

```bash
# Check cluster status
az aks show \
  --name rag-engine-prod \
  --resource-group rag-engine-prod-rg

# Check node status
kubectl get nodes -o wide

# Check control plane logs
az aks browse \
  --name rag-engine-prod \
  --resource-group rag-engine-prod-rg
```

### 2. Database Connection Issues

```bash
# Check database status
az sql server show \
  --name ragengine-sql \
  --resource-group rag-engine-prod-rg

# Check firewall rules
az sql server firewall-rule list \
  --server ragengine-sql

# Test connection
telnet ragengine-sql.postgres.database.azure.com 5432
```

### 3. Storage Access Issues

```bash
# Check storage account
az storage account show \
  --name ragengineuploadsprod \
  --resource-group rag-engine-prod-rg

# Check container access
az storage container show \
  --name uploads \
  --account-name ragengineuploadsprod
```

## Cleanup / التنظيف

```bash
# Destroy Helm release
helm uninstall rag-engine --namespace rag-engine

# Destroy infrastructure
terraform destroy \
  -var-file="terraform.tfvars" \
  -auto-approve

# Clean up terraform state
rm -rf .terraform
```

## Best Practices / أفضل الممارسات

1. **Use Terraform Workspaces** - Separate dev/staging/prod environments
2. **State Management** - Use Terraform Cloud or Azure Storage for state
3. **Version Control** - Commit Terraform files to Git
4. **Cost Optimization** - Use Reserved Instances for predictable workloads
5. **Backup Strategy** - Enable database backups and storage replication
6. **Monitoring** - Set up Azure Monitor dashboards and alerts
7. **Security** - Use managed identities, rotate credentials regularly
8. **Documentation** - Document all changes and procedures

---

**Last Updated:** 2026-01-31
**Terraform Version:** 1.3.0+

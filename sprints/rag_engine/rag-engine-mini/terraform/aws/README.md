# AWS Deployment Guide
# ======================
# Complete guide for deploying RAG Engine to AWS using Terraform.

# دليل نشر RAG Engine إلى AWS باستخدام Terraform

## Prerequisites / المتطلبات

1. **AWS Account** with appropriate IAM permissions
2. **Terraform** >= 1.3.0 installed locally
3. **AWS CLI** configured with credentials
4. **kubectl** for cluster management
5. **Helm** for application deployment

## Installation / التثبيت

```bash
# Install Terraform (macOS)
brew install terraform

# Install Terraform (Linux)
wget -O- https://apt.releases.hashicorp.com/terraform/pool/main/main.html
sudo unzip main && sudo mv terraform /usr/local/bin/

# Install AWS CLI
pip install awscli

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## Configuration / التكوين

### 1. Create Terraform Workspace

```bash
cd terraform/aws

# Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"

# Or use AWS CLI profile
export AWS_PROFILE="rag-engine-profile"
```

### 2. Customize Variables

Create `terraform.tfvars`:

```hcl
# Environment
environment = "prod"
project_name = "rag-engine"

# Networking
vpc_cidr           = "10.0.0.0/16"
availability_zones  = ["us-east-1a", "us-east-1b"]

# EKS Cluster
eks_node_type      = "t3.large"
eks_min_nodes      = 3
eks_max_nodes      = 10
eks_desired_nodes  = 3
k8s_version        = "1.28"

# Database
rds_instance_class = "db.r5.xlarge"
rds_storage_gb     = 500
postgres_version    = "15.4"

# Redis
redis_node_type = "cache.m6g.large"
redis_num_nodes = 3
redis_version   = "7.0"
```

## Deployment / النشر

### 1. Initialize Terraform

```bash
terraform init
```

### 2. Plan Changes

```bash
# Review planned changes
terraform plan \
  -var-file="terraform.tfvars" \
  -out="tfplan"
```

### 3. Apply Configuration

```bash
# Deploy infrastructure
terraform apply \
  -var-file="terraform.tfvars" \
  -auto-approve
```

Expected outputs:
```
eks_cluster_endpoint = "https://ABC123.gr7.us-east-1.eks.amazonaws.com"
eks_cluster_name = "rag-engine-prod"
rds_endpoint = "rag-engine-prod.xxxxxx.us-east-1.rds.amazonaws.com"
redis_endpoint = "rag-engine-prod-redis.xxxxxx.0001.use1.cache.amazonaws.com"
s3_bucket_name = "rag-engine-prod-uploads"
```

### 4. Configure kubectl

```bash
# Update kubeconfig
aws eks update-kubeconfig \
  --name rag-engine-prod \
  --region us-east-1

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

# Deploy
helm install rag-engine \
  --namespace rag-engine \
  --values values-prod.yaml \
  .
```

### 3. Verify Deployment

```bash
# Check pods
kubectl -n rag-engine get pods

# Check services
kubectl -n rag-engine get services

# Check ingress
kubectl -n rag-engine get ingress
```

## Cost Estimation / تقدير التكلفة

| Resource | Type | Monthly Cost (USD) |
|----------|------|-------------------|
| VPC & Networking | N/A | ~$20 |
| EKS Cluster | Control Plane | $73/month |
| EKS Nodes (3x t3.large) | On-Demand | ~$108/month |
| RDS PostgreSQL | db.t3.medium (100GB) | ~$42/month |
| ElastiCache Redis | 2x cache.t3.medium | ~$60/month |
| S3 Storage | 1TB Standard | ~$23/month |
| **Total** | | **~$326/month** |

*Prices are estimates as of 2026-01-31. Actual costs may vary.*

## Security Considerations / اعتبارات الأمان

1. **Network Isolation**
   - Private subnets for RDS and ElastiCache
   - NAT gateways for outbound traffic
   - Security groups with least privilege

2. **Encryption**
   - S3 server-side encryption enabled
   - RDS encryption at rest enabled
   - ElastiCache transit encryption enabled
   - EKS secrets manager for sensitive data

3. **Access Control**
   - IAM roles with least privilege
   - S3 bucket policies restrict to EKS cluster
   - Database credentials in AWS Secrets Manager

## Monitoring / المراقبة

### CloudWatch Dashboards

```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name rag-engine-metrics

# Add metrics:
# - EKS CPU/Memory utilization
# - RDS connections and latency
# - ElastiCache evictions and memory
# - S3 request counts
```

### Alarms

```bash
# Create alarm for high CPU
aws cloudwatch put-metric-alarm \
  --alarm-name eks-high-cpu \
  --metric-name CPUUtilization \
  --namespace AWS/EKS \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:rag-engine-alerts
```

## Troubleshooting / استكشاف الأخطاء

### 1. EKS Cluster Not Responding

```bash
# Check cluster status
aws eks describe-cluster \
  --name rag-engine-prod \
  --region us-east-1

# Check node status
kubectl get nodes -o wide

# Check control plane logs
aws logs group /aws/eks/rag-engine-prod
```

### 2. Database Connection Issues

```bash
# Check RDS status
aws rds describe-db-instances \
  --db-instance-identifier rag-engine-prod

# Check security group rules
aws ec2 describe-security-groups \
  --group-ids sg-xxxxx

# Test connection
telnet rag-engine-prod.xxxxxx.us-east-1.rds.amazonaws.com 5432
```

### 3. S3 Access Issues

```bash
# Check bucket policy
aws s3api get-bucket-policy \
  --bucket rag-engine-prod-uploads

# Verify encryption
aws s3api get-bucket-encryption \
  --bucket rag-engine-prod-uploads
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
2. **State Management** - Use Terraform Cloud or S3 for state storage
3. **Version Control** - Commit Terraform files to Git
4. **Cost Optimization** - Use Spot instances for EKS nodes when possible
5. **Backup Strategy** - Enable RDS backups and S3 versioning
6. **Monitoring** - Set up CloudWatch dashboards and alarms
7. **Security** - Regularly rotate credentials and review IAM policies
8. **Documentation** - Document all changes and procedures

---

**Last Updated:** 2026-01-31
**Terraform Version:** 1.3.0+

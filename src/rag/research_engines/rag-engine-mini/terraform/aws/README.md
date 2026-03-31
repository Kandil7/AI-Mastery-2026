# RAG Engine on AWS with Terraform

This Terraform configuration sets up the infrastructure for the RAG Engine on AWS, including an EKS cluster and all required resources.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- kubectl
- AWS IAM permissions for EKS, VPC, EC2, and related services


## Quick Start

1. Initialize Terraform:
```bash
terraform init
```

2. Create a `terraform.tfvars` file with your variables:
```hcl
aws_region         = "us-west-2"
cluster_name       = "rag-engine-prod"
openai_api_key     = "your-openai-api-key"
environment        = "prod"
```

3. Review the execution plan:
```bash
terraform plan
```

4. Apply the configuration:
```bash
terraform apply
```

5. Configure kubectl for the new cluster:
```bash
aws eks --region $(terraform output -raw region) update-kubeconfig --name $(terraform output -raw cluster_name)
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

## Security

- Private subnets for worker nodes
- Public subnets for load balancers
- IAM roles with least-privilege access
- Kubernetes RBAC controls

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

## Scaling

Node groups are configured with:
- Minimum: 1 node
- Desired: 2 nodes
- Maximum: 5 nodes

Adjust these values in your `terraform.tfvars` file as needed.

---

**Last Updated:** 2026-01-31
**Terraform Version:** 1.3.0+

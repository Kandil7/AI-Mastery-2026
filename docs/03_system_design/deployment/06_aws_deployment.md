# AWS Deployment Guide - Production RAG Engine on Amazon Web Services

## Overview

This guide walks you through deploying RAG Engine Mini on AWS using industry best practices. We'll cover multiple deployment options from simple to enterprise-grade.

## Learning Objectives

By the end of this guide, you will:
1. Understand AWS services for container deployment
2. Deploy using ECS (Elastic Container Service) with Fargate
3. Set up EKS (Elastic Kubernetes Service) for Kubernetes
4. Configure RDS for managed PostgreSQL
5. Implement S3 for document storage
6. Set up Application Load Balancer with SSL
7. Configure CloudWatch monitoring
8. Implement auto-scaling policies
9. Manage costs effectively

**Estimated Time:** 6-8 hours
**Cost:** $200-800/month (depending on scale)

---

## Part 1: AWS Architecture Options

### Option 1: ECS with Fargate (Recommended for Beginners)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                        VPC                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Application Load Balancer (ALB)             │   │
│  │            (SSL termination, routing)               │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              ECS Cluster (Fargate)                   │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Task    │ │  API Task    │ │  API Task    │  │   │
│  │  │  (Container) │ │  (Container) │ │  (Container) │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  │       ┌──────────────┐                               │   │
│  │       │ Worker Task  │                               │   │
│  │       │ (Container)  │                               │   │
│  │       └──────────────┘                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────┼──────────────────────────────────┐   │
│  │                  │          Private Subnets          │   │
│  │  ┌───────────────▼────────┐ ┌─────────────────────┐  │   │
│  │  │    RDS PostgreSQL      │ │   ElastiCache       │  │   │
│  │  │    (Multi-AZ)          │ │   (Redis)           │  │   │
│  │  └────────────────────────┘ └─────────────────────┘  │   │
│  │  ┌────────────────────────┐                          │   │
│  │  │    S3 Bucket           │                          │   │
│  │  │    (Documents)         │                          │   │
│  │  └────────────────────────┘                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Simpler than Kubernetes
- Serverless containers (no EC2 management)
- Automatic scaling
- Good for 1,000-50,000 users

### Option 2: EKS (Elastic Kubernetes Service)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     EKS Cluster                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         AWS Load Balancer Controller                │   │
│  │              (ALB for Ingress)                      │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │              Managed Node Group                     │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │  API Pod     │ │  API Pod     │ │ Worker Pod   │  │   │
│  │  │  (Node 1)    │ │  (Node 2)    │ │ (Node 3)     │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  External AWS Services:                                     │
│  • RDS for PostgreSQL                                       │
│  • ElastiCache for Redis                                    │
│  • S3 for document storage                                  │
│  • CloudWatch for monitoring                                │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- Need Kubernetes features
- Multi-region deployment
- Complex microservices
- 50,000+ users

---

## Part 2: Prerequisites

### AWS Account Setup

**1. Create AWS Account:**
```bash
# Sign up at https://aws.amazon.com/
# Enable MFA on root account
# Set up billing alerts
```

**2. Install AWS CLI:**
```bash
# macOS
brew install awscli

# Windows
choco install awscli

# Linux
pip install awscli

# Verify
aws --version
```

**3. Configure Credentials:**
```bash
# Create IAM user with programmatic access
# Attach policies: AmazonECS_FullAccess, AmazonEKSClusterPolicy, etc.

# Configure CLI
aws configure
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region name: us-west-2
# Default output format: json

# Verify
aws sts get-caller-identity
```

**4. Install Additional Tools:**
```bash
# Install eksctl for EKS management
brew install eksctl

# Install ecs-cli for ECS
brew install amazon-ecs-cli

# Install kubectl
brew install kubectl
```

---

## Part 3: VPC and Network Setup

### Create VPC with CloudFormation

```yaml
# vpc-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'VPC for RAG Engine'

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: rag-engine-vpc

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: rag-engine-igw

  AttachGateway:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      VpcId: !Ref VPC
      InternetGatewayId: !Ref InternetGateway

  # Public Subnets (for ALB)
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: rag-engine-public-1

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: rag-engine-public-2

  # Private Subnets (for containers, RDS)
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.3.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      Tags:
        - Key: Name
          Value: rag-engine-private-1

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.4.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      Tags:
        - Key: Name
          Value: rag-engine-private-2

  # NAT Gateway for private subnets
  NatGateway1:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatEIP1.AllocationId
      SubnetId: !Ref PublicSubnet1

  NatEIP1:
    Type: AWS::EC2::EIP
    Properties:
      Domain: vpc

  # Route Tables
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: rag-engine-public-rt

  PublicRoute:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet1
      RouteTableId: !Ref PublicRouteTable

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet2
      RouteTableId: !Ref PublicRouteTable

Outputs:
  VPCId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: rag-engine-vpc-id

  PublicSubnets:
    Description: Public subnets
    Value: !Join [',', [!Ref PublicSubnet1, !Ref PublicSubnet2]]
    Export:
      Name: rag-engine-public-subnets

  PrivateSubnets:
    Description: Private subnets
    Value: !Join [',', [!Ref PrivateSubnet1, !Ref PrivateSubnet2]]
    Export:
      Name: rag-engine-private-subnets
```

**Deploy VPC:**
```bash
# Create stack
aws cloudformation create-stack \
  --stack-name rag-engine-vpc \
  --template-body file://vpc-template.yaml \
  --region us-west-2

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name rag-engine-vpc

# Get outputs
aws cloudformation describe-stacks \
  --stack-name rag-engine-vpc \
  --query 'Stacks[0].Outputs'
```

---

## Part 4: ECS Fargate Deployment

### Step 1: Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster \
  --cluster-name rag-engine-cluster \
  --region us-west-2 \
  --settings name=containerInsights,value=enabled

# Verify
aws ecs describe-clusters \
  --clusters rag-engine-cluster \
  --region us-west-2
```

### Step 2: Create Task Definition

```json
{
  "family": "rag-engine-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "YOUR_ECR_REPO/rag-engine:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "LOG_LEVEL",
          "value": "info"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:ACCOUNT_ID:secret:rag-engine/db-url"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:ACCOUNT_ID:secret:rag-engine/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-engine-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

**Register Task Definition:**
```bash
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json \
  --region us-west-2
```

### Step 3: Create Application Load Balancer

```bash
# Create security group for ALB
ALB_SG=$(aws ec2 create-security-group \
  --group-name rag-engine-alb-sg \
  --description "ALB Security Group" \
  --vpc-id $(aws cloudformation describe-stacks \
    --stack-name rag-engine-vpc \
    --query 'Stacks[0].Outputs[?OutputKey==`VPCId`].OutputValue' \
    --output text) \
  --query 'GroupId' \
  --output text)

# Allow HTTPS inbound
aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Create ALB
aws elbv2 create-load-balancer \
  --name rag-engine-alb \
  --subnets $(aws cloudformation describe-stacks \
    --stack-name rag-engine-vpc \
    --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnets`].OutputValue' \
    --output text | tr ',' ' ') \
  --security-groups $ALB_SG \
  --scheme internet-facing \
  --type application \
  --region us-west-2
```

### Step 4: Create ECS Service with Auto Scaling

```bash
# Get ALB ARN and create target group
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names rag-engine-alb \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

# Create target group
TG_ARN=$(aws elbv2 create-target-group \
  --name rag-engine-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id $(aws cloudformation describe-stacks \
    --stack-name rag-engine-vpc \
    --query 'Stacks[0].Outputs[?OutputKey==`VPCId`].OutputValue' \
    --output text) \
  --target-type ip \
  --health-check-path /health \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

# Create ECS service
aws ecs create-service \
  --cluster rag-engine-cluster \
  --service-name rag-engine-api \
  --task-definition rag-engine-api:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$(aws cloudformation describe-stacks \
    --stack-name rag-engine-vpc \
    --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnets`].OutputValue' \
    --output text)],securityGroups=[sg-xxxx],assignPublicIp=DISABLED}" \
  --load-balancers targetGroupArn=$TG_ARN,containerName=api,containerPort=8000 \
  --region us-west-2

# Configure auto scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/rag-engine-cluster/rag-engine-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10 \
  --region us-west-2

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --policy-name rag-engine-cpu-scaling \
  --service-namespace ecs \
  --resource-id service/rag-engine-cluster/rag-engine-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "TargetValue": 70.0,
    "ScaleOutCooldown": 60,
    "ScaleInCooldown": 300
  }' \
  --region us-west-2
```

---

## Part 5: EKS Deployment (Alternative)

### Step 1: Create EKS Cluster

```bash
# Create cluster with eksctl (easiest method)
eksctl create cluster \
  --name rag-engine-cluster \
  --region us-west-2 \
  --vpc-private-subnets $(aws cloudformation describe-stacks \
    --stack-name rag-engine-vpc \
    --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnets`].OutputValue' \
    --output text | tr ',' ' ') \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed \
  --asg-access \
  --external-dns-access \
  --full-ecr-access

# Verify
kubectl get nodes
```

### Step 2: Install AWS Load Balancer Controller

```bash
# Download IAM policy
curl -O https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.6.2/docs/install/iam_policy.json

# Create IAM policy
aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file://iam_policy.json

# Create IAM OIDC provider
eksctl utils associate-iam-oidc-provider \
  --cluster rag-engine-cluster \
  --approve

# Create service account
eksctl create iamserviceaccount \
  --cluster rag-engine-cluster \
  --namespace kube-system \
  --name aws-load-balancer-controller \
  --attach-policy-arn arn:aws:iam::ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

# Install controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=rag-engine-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

### Step 3: Deploy Application

```bash
# Update Kubernetes manifests for AWS
kubectl apply -f config/kubernetes/

# Verify deployment
kubectl get pods -n rag-engine
kubectl get svc -n rag-engine
kubectl get ingress -n rag-engine
```

---

## Part 6: Database and Cache Setup

### RDS PostgreSQL

```bash
# Create subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name rag-engine-db-subnet \
  --db-subnet-group-description "Subnet group for RAG Engine DB" \
  --subnet-ids '["subnet-xxxxx","subnet-yyyyy"]' \
  --region us-west-2

# Create database
aws rds create-db-instance \
  --db-instance-identifier rag-engine-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.4 \
  --allocated-storage 100 \
  --storage-type gp3 \
  --storage-encrypted \
  --master-username ragadmin \
  --master-user-password 'STRONG_PASSWORD_HERE' \
  --vpc-security-group-ids sg-xxxxx \
  --db-subnet-group-name rag-engine-db-subnet \
  --multi-az \
  --publicly-accessible false \
  --backup-retention-period 7 \
  --preferred-backup-window 03:00-04:00 \
  --enable-performance-insights \
  --performance-insights-retention-period 7 \
  --enable-cloudwatch-logs-exports '["postgresql"]' \
  --deletion-protection \
  --region us-west-2

# Get endpoint
aws rds describe-db-instances \
  --db-instance-identifier rag-engine-db \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text
```

### ElastiCache Redis

```bash
# Create subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name rag-engine-cache-subnet \
  --cache-subnet-group-description "Subnet group for RAG Engine cache" \
  --subnet-ids '["subnet-xxxxx","subnet-yyyyy"]' \
  --region us-west-2

# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id rag-engine-cache \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --num-cache-nodes 1 \
  --cache-subnet-group-name rag-engine-cache-subnet \
  --security-group-ids sg-xxxxx \
  --preferred-maintenance-window sun:05:00-sun:06:00 \
  --region us-west-2
```

---

## Part 7: Monitoring with CloudWatch

### Container Insights

Already enabled in cluster creation. View metrics:

```bash
# Container metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ClusterName,Value=rag-engine-cluster \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average

# Set alarms
aws cloudwatch put-metric-alarm \
  --alarm-name rag-engine-high-cpu \
  --alarm-description "CPU utilization > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=ClusterName,Value=rag-engine-cluster \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-west-2:ACCOUNT_ID:rag-engine-alerts
```

### Application Logs

```bash
# View logs
aws logs tail /ecs/rag-engine-api --follow

# Query logs
aws logs start-query \
  --log-group-name /ecs/rag-engine-api \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --end-time $(date +%s)000 \
  --query-string 'fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20'
```

---

## Part 8: Cost Optimization

### Right-Sizing

```bash
# Monitor actual usage
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name MemoryUtilization \
  --dimensions Name=ServiceName,Value=rag-engine-api \
  --statistics Average \
  --start-time $(date -d '7 days ago' +%Y-%m-%d) \
  --end-time $(date +%Y-%m-%d) \
  --period 86400

# Adjust based on metrics
# If avg < 50%, reduce memory
# If avg > 80%, increase memory
```

### Spot Instances (EKS)

```bash
# Add spot node group
eksctl create nodegroup \
  --cluster rag-engine-cluster \
  --name spot-workers \
  --spot \
  --instance-types t3.medium,t3a.medium \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed
```

### Reserved Capacity

For predictable workloads:
- ECS: Compute Savings Plans
- RDS: Reserved Instances (1-3 year commitment)
- ElastiCache: Reserved Nodes

---

## Part 9: Security Best Practices

### 1. IAM Roles and Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-west-2:ACCOUNT_ID:secret:rag-engine/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::rag-engine-documents/*"
    }
  ]
}
```

### 2. Security Groups

```bash
# ALB Security Group: Allow HTTPS from anywhere
# Container Security Group: Allow 8000 from ALB only
# RDS Security Group: Allow 5432 from containers only
# ElastiCache Security Group: Allow 6379 from containers only
```

### 3. Secrets Management

```bash
# Store secrets in Secrets Manager
aws secretsmanager create-secret \
  --name rag-engine/database-url \
  --description "Database connection string" \
  --secret-string 'postgresql://user:pass@host:5432/db' \
  --kms-key-id alias/aws/secretsmanager

# Rotate automatically
aws secretsmanager rotate-secret \
  --secret-id rag-engine/database-url \
  --rotation-lambda-arn arn:aws:lambda:us-west-2:ACCOUNT_ID:function:rotate-postgres-secret \
  --automatically-rotate-after-days 30
```

---

## Part 10: Backup and Disaster Recovery

### Automated Backups

```bash
# RDS automated backups (already enabled)
# Point-in-time recovery available

# Create manual snapshots
aws rds create-db-snapshot \
  --db-instance-identifier rag-engine-db \
  --db-snapshot-identifier rag-engine-db-$(date +%Y%m%d)

# Cross-region backup
aws rds copy-db-snapshot \
  --source-db-snapshot-identifier rag-engine-db-20240115 \
  --target-db-snapshot-identifier rag-engine-db-20240115-west \
  --source-region us-west-2 \
  --destination-region us-east-1
```

### Document Storage Backup

```bash
# S3 versioning and replication
aws s3api put-bucket-versioning \
  --bucket rag-engine-documents \
  --versioning-configuration Status=Enabled

# Enable replication to another region
aws s3api put-bucket-replication \
  --bucket rag-engine-documents \
  --replication-configuration file://replication.json
```

---

## Summary: AWS Deployment Checklist

### Pre-Deployment:
- [ ] AWS CLI configured
- [ ] VPC created with public/private subnets
- [ ] IAM roles and policies configured
- [ ] ECR repository created
- [ ] SSL certificate requested (ACM)

### Deployment:
- [ ] ECS cluster created OR EKS cluster created
- [ ] Task definitions registered
- [ ] RDS database provisioned
- [ ] ElastiCache cluster created
- [ ] Application Load Balancer configured
- [ ] ECS service deployed with auto-scaling
- [ ] DNS records updated (Route 53)

### Post-Deployment:
- [ ] CloudWatch alarms configured
- [ ] Log groups verified
- [ ] Backup policies confirmed
- [ ] Security groups audited
- [ ] Cost alerts set up
- [ ] Documentation updated

### Monthly:
- [ ] Review CloudWatch metrics
- [ ] Check AWS costs (Budgets)
- [ ] Verify backups
- [ ] Rotate secrets
- [ ] Security scan

---

**Next Steps:**
1. Choose ECS or EKS based on requirements
2. Deploy following step-by-step guide
3. Configure monitoring and alerting
4. Document your architecture
5. Train team on AWS management

**Estimated Monthly Cost:**
- ECS Fargate: $150-400
- EKS: $200-600
- RDS: $100-300
- ElastiCache: $50-150
- ALB: $25-50
- Data Transfer: $20-100
- **Total: $400-1,600/month**

**Continue to GCP and Azure guides for multi-cloud knowledge!** ☁️

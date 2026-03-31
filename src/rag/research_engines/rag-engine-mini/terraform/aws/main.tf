# Terraform configuration for RAG Engine on AWS# Kubernetes provider for EKS
data "aws_eks_cluster" "cluster" {
  name = module.eks.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
  token                  = data.aws_eks_cluster_auth.cluster.token
  load_config_file       = false
  version                = "~> 2.24"
}

# Helm provider for EKS
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Create namespace for RAG Engine
resource "kubernetes_namespace" "rag_engine" {
  metadata {
    name = var.namespace
    labels = {
      name = var.namespace
    }
  }
}

# Deploy RAG Engine using Helm
resource "helm_release" "rag_engine" {
  name       = "rag-engine"
  repository = ""  # Using local chart
  chart      = "../../../config/helm/rag-engine"
  namespace  = kubernetes_namespace.rag_engine.metadata[0].name

  set {
    name  = "ragEngine.replicaCount"
    value = var.rag_engine_replicas
  }

  set_sensitive {
    name  = "ragEngine.env.OPENAI_API_KEY"
    value = var.openai_api_key
  }

  depends_on = [kubernetes_namespace.rag_engine]
}
# Sets up EKS cluster, VPC, and related infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Create VPC for the cluster
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr_block

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  single_nat_gateway = true
  enable_vpn_gateway = false

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = "1"
  }
}

# Create EKS cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = var.k8s_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = concat(module.vpc.private_subnets, module.vpc.public_subnets)

  # EKS managed node group
  eks_managed_node_group_defaults = {
    ami_type = "AL2_x86_64"
  }

  eks_managed_node_groups = {
    initial = {
      name = "${var.cluster_name}-ng-default"

      instance_types = var.node_instance_types

      min_size     = var.node_min_size
      max_size     = var.node_max_size
      desired_size = var.node_desired_size

      # Attach additional policies to the node IAM role
      attach_additional_iam_policies = [
        "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
      ]
    }
  }

  # Enable cluster creator admin role
  enable_cluster_creator_admin_permissions = true

  # Manage aws-auth configmap
  manage_aws_auth_configmap = true

  # Workers role additional policies
  worker_role_additional_policies = {
    AmazonEBSCSIDriverPolicy = "arn:aws:iam::aws:policy/AmazonEBSCSIDriverPolicy"
  }
}

# Module: RDS PostgreSQL
module "rds" {
  source = "./modules/rds"

  identifier           = "${var.project_name}-${var.environment}"
  engine               = "postgres"
  engine_version       = var.postgres_version
  instance_class       = var.rds_instance_class
  allocated_storage    = var.rds_storage_gb
  database_name       = var.database_name
  master_username     = var.database_username
  vpc_id             = module.networking.vpc_id
  subnet_ids          = module.networking.private_subnet_ids
  environment         = var.environment

  tags = var.default_tags
}

# Module: ElastiCache Redis
module "elasticache" {
  source = "./modules/elasticache"

  cluster_id       = "${var.project_name}-${var.environment}-redis"
  node_type        = var.redis_node_type
  num_cache_nodes  = var.redis_num_nodes
  engine_version   = var.redis_version
  vpc_id          = module.networking.vpc_id
  subnet_ids       = module.networking.private_subnet_ids
  environment      = var.environment

  tags = var.default_tags
}

# Module: S3 Buckets
module "s3" {
  source = "./modules/s3"

  bucket_name     = "${var.project_name}-${var.environment}-uploads"
  environment     = var.environment
  sse_enabled    = var.s3_sse_enabled
  versioning      = var.s3_versioning
  tags            = var.default_tags
}

# Outputs
output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks_cluster.cluster_endpoint
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks_cluster.cluster_name
}

output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks_cluster.security_group_id
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
}

output "rds_port" {
  description = "RDS PostgreSQL port"
  value       = module.rds.port
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.elasticache.endpoint
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = module.elasticache.port
}

output "s3_bucket_name" {
  description = "S3 bucket name for uploads"
  value       = module.s3.bucket_name
}

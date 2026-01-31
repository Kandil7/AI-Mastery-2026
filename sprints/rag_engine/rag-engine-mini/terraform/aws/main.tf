# AWS Terraform Configuration
# ==========================
# Infrastructure as Code for AWS deployment.
# البنية التحتية ككود لنشر AWS

# AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    Environment = var.environment
    Project     = "rag-engine"
    ManagedBy  = "terraform"
  }
}

# Module: VPC and Networking
module "networking" {
  source = "./modules/networking"

  vpc_cidr           = var.vpc_cidr
  availability_zones  = var.availability_zones
  environment         = var.environment
  project_name        = var.project_name

  tags = var.default_tags
}

# Module: EKS Cluster
module "eks_cluster" {
  source = "./modules/eks"

  cluster_name    = "${var.project_name}-${var.environment}"
  vpc_id         = module.networking.vpc_id
  subnet_ids      = module.networking.private_subnet_ids
  environment     = var.environment
  node_type      = var.eks_node_type
  min_nodes      = var.eks_min_nodes
  max_nodes      = var.eks_max_nodes
  desired_nodes  = var.eks_desired_nodes

  tags = var.default_tags
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

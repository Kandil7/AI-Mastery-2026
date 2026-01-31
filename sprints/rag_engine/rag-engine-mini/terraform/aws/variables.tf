# AWS Terraform Variables
# ========================
# Configuration variables for AWS infrastructure.

# متغيرات تكوين Terraform لـ AWS

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "rag-engine"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "eks_node_type" {
  description = "EC2 instance type for EKS nodes"
  type        = string
  default     = "t3.medium"
}

variable "eks_min_nodes" {
  description = "Minimum number of EKS nodes"
  type        = number
  default     = 2
}

variable "eks_max_nodes" {
  description = "Maximum number of EKS nodes"
  type        = number
  default     = 10
}

variable "eks_desired_nodes" {
  description = "Desired number of EKS nodes"
  type        = number
  default     = 3
}

variable "eks_public_access_cidrs" {
  description = "CIDR blocks allowed to access EKS API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "k8s_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.28"
}

variable "postgres_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_storage_gb" {
  description = "Allocated storage for RDS in GB"
  type        = number
  default     = 100
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "ragengine"
}

variable "database_username" {
  description = "Database master username"
  type        = string
  default     = "ragengine_admin"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}

variable "redis_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "redis_auth_token" {
  description = "Redis auth token"
  type        = string
  sensitive   = true
  default     = null
}

variable "s3_sse_enabled" {
  description = "Enable server-side encryption for S3"
  type        = bool
  default     = true
}

variable "s3_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "default_tags" {
  description = "Default tags for all resources"
  type        = map(string)
  default = {
    Environment = "dev"
    ManagedBy  = "terraform"
  }
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

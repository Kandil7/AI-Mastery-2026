# Variables for RAG Engine AWS Infrastructure

# General configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rag-engine"
}

variable "environment" {
  description = "Environment name (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-west-2"
}

# VPC configuration
variable "vpc_cidr_block" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones to use for the subnets"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# EKS configuration
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "rag-engine-cluster"
}

variable "k8s_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "Instance types for the EKS nodes"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "node_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 5
}

variable "node_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 2
}

# RAG Engine configuration
variable "namespace" {
  description = "Kubernetes namespace for RAG Engine"
  type        = string
  default     = "rag-engine"
}

variable "rag_engine_replicas" {
  description = "Number of replicas for RAG Engine API"
  type        = number
  default     = 3
}

# Secrets
variable "openai_api_key" {
  description = "OpenAI API key for RAG Engine"
  type        = string
  sensitive   = true
}

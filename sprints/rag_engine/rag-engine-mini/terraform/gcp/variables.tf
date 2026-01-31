# GCP Terraform Variables
# ==========================
# Configuration variables for GCP infrastructure.

# متغيرات تكوين Terraform لـ Google Cloud

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = "rag-engine-prod"
}

variable "gcp_region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone for compute resources"
  type        = string
  default     = "us-central1-a"
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

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "gke_node_type" {
  description = "Compute instance type for GKE nodes"
  type        = string
  default     = "e2-medium"
}

variable "gke_min_nodes" {
  description = "Minimum number of GKE nodes"
  type        = number
  default     = 2
}

variable "gke_max_nodes" {
  description = "Maximum number of GKE nodes"
  type        = number
  default     = 10
}

variable "gke_pod_cidr" {
  description = "CIDR block for GKE pods"
  type        = string
  default     = "10.64.0.0/14"
}

variable "gke_master_cidr" {
  description = "CIDR block for GKE master"
  type        = string
  default     = "10.128.0.0/28"
}

variable "node_disk_size_gb" {
  description = "Disk size for GKE nodes in GB"
  type        = number
  default     = 100
}

variable "preemptible" {
  description = "Use preemptible nodes for cost savings"
  type        = bool
  default     = true
}

variable "postgres_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "POSTGRES_14"
}

variable "cloudsql_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-custom-2-3840"
}

variable "storage_gb" {
  description = "Allocated storage for Cloud SQL in GB"
  type        = number
  default     = 100
}

variable "disk_autoresize" {
  description = "Enable automatic disk resizing"
  type        = bool
  default     = true
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

variable "vpc_peering_cidr" {
  description = "CIDR block for VPC peering"
  type        = string
  default     = "172.16.0.0/12"
}

variable "redis_memory_gb" {
  description = "Memory size for Redis in GB"
  type        = number
  default     = 16
}

variable "redis_tier" {
  description = "Memorystore Redis tier"
  type        = string
  default     = "STANDARD_HA"
}

variable "redis_version" {
  description = "Redis engine version"
  type        = string
  default     = "REDIS_7_0"
}

variable "redis_node_count" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}

variable "redis_source_cidrs" {
  description = "CIDR blocks allowed to access Redis"
  type        = list(string)
  default     = ["10.64.0.0/14"]
}

variable "authorized_network" {
  description = "Network for Redis access"
  type        = string
  default     = ""
}

variable "gke_service_account" {
  description = "Service account for GKE nodes"
  type        = string
  default     = "rag-engine-sa@rag-engine-prod.iam.gserviceaccount.com"
}

variable "bucket_name" {
  description = "GCS bucket name"
  type        = string
  default     = "rag-engine-uploads"
}

variable "uniform_access" {
  description = "Enable uniform bucket-level access"
  type        = bool
  default     = true
}

variable "versioning" {
  description = "Enable GCS bucket versioning"
  type        = bool
  default     = true
}

variable "default_labels" {
  description = "Default labels for all resources"
  type        = map(string)
  default = {
    Environment = "dev"
    ManagedBy  = "terraform"
  }
}

variable "labels" {
  description = "Additional labels for resources"
  type        = map(string)
  default     = {}
}

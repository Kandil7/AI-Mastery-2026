# Google Cloud Terraform Configuration
# ====================================
# Infrastructure as Code for GCP deployment.
# البنية التحتية ككود لنشر Google Cloud

# GCP Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone

  default_labels = {
    environment = var.environment
    project     = "rag-engine"
    managed-by  = "terraform"
  }
}

# Module: VPC and Networking
module "networking" {
  source = "./modules/networking"

  vpc_cidr        = var.vpc_cidr
  region           = var.gcp_region
  environment      = var.environment
  project_name     = var.project_name

  labels = var.default_labels
}

# Module: GKE Cluster
module "gke_cluster" {
  source = "./modules/gke"

  cluster_name = "${var.project_name}-${var.environment}"
  network_id   = module.networking.network_id
  subnet_ids   = module.networking.subnet_ids
  environment  = var.environment
  node_type    = var.gke_node_type
  min_nodes    = var.gke_min_nodes
  max_nodes    = var.gke_max_nodes

  labels = var.default_labels
}

# Module: Cloud SQL PostgreSQL
module "cloudsql" {
  source = "./modules/cloudsql"

  instance_name = "${var.project_name}-${var.environment}"
  database_name = var.database_name
  region        = var.gcp_region
  network_id    = module.networking.network_id
  environment   = var.environment
  tier          = var.cloudsql_tier
  storage_gb    = var.cloudsql_storage_gb

  labels = var.default_labels
}

# Module: Cloud Memorystore Redis
module "memorystore" {
  source = "./modules/memorystore"

  cluster_id       = "${var.project_name}-${var.environment}-redis"
  region          = var.gcp_region
  network_id      = module.networking.network_id
  environment     = var.environment
  node_type       = var.redis_node_type
  node_count      = var.redis_node_count
  redis_version   = var.redis_version

  labels = var.default_labels
}

# Module: GCS Bucket
module "gcs" {
  source = "./modules/bucket"

  bucket_name   = "${var.project_name}-${var.environment}-uploads"
  location     = var.gcp_region
  environment  = var.environment
  uniform      = var.gcs_uniform_access
  versioning   = var.gcs_versioning

  labels = var.default_labels
}

# Outputs
output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke_cluster.endpoint
}

output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = module.gke_cluster.name
}

output "cloudsql_connection_name" {
  description = "Cloud SQL connection name"
  value       = module.cloudsql.connection_name
}

output "redis_endpoint" {
  description = "Memorystore Redis endpoint"
  value       = module.memorystore.endpoint
}

output "gcs_bucket_name" {
  description = "GCS bucket name for uploads"
  value       = module.gcs.bucket_name
}

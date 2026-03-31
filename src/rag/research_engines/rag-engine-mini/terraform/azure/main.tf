# Azure Terraform Configuration
# ============================
# Infrastructure as Code for Azure deployment.
# البنية التحتية ككود لنشر Azure

# Azure Provider
provider "azurerm" {
  features {}

  skip_provider_registration = false

  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id

  client_id     = var.client_id
  client_secret = var.client_secret
}

# Module: VPC (Virtual Network) and Networking
module "networking" {
  source = "./modules/networking"

  vnet_name           = "${var.project_name}-${var.environment}"
  vnet_cidr           = var.vnet_cidr
  location            = var.azure_location
  environment          = var.environment
  project_name         = var.project_name

  tags = var.default_tags
}

# Module: AKS Cluster
module "aks_cluster" {
  source = "./modules/aks"

  cluster_name    = "${var.project_name}-${var.environment}"
  vnet_id         = module.networking.vnet_id
  subnet_ids      = module.networking.subnet_ids
  location        = var.azure_location
  environment     = var.environment
  node_type      = var.aks_node_type
  min_nodes      = var.aks_min_nodes
  max_nodes      = var.aks_max_nodes
  desired_nodes  = var.aks_desired_nodes

  tags = var.default_tags
}

# Module: Azure Database for PostgreSQL
module "database" {
  source = "./modules/database"

  server_name    = "${var.project_name}-${var.environment}"
  database_name  = var.database_name
  vnet_id        = module.networking.vnet_id
  subnet_ids     = module.networking.subnet_ids
  location       = var.azure_location
  environment    = var.environment
  tier           = var.database_tier
  storage_mb     = var.database_storage_mb

  tags = var.default_tags
}

# Module: Azure Cache for Redis
module "cache" {
  source = "./modules/cache"

  cache_name    = "${var.project_name}-${var.environment}-redis"
  vnet_id        = module.networking.vnet_id
  subnet_ids     = module.networking.subnet_ids
  location       = var.azure_location
  environment    = var.environment
  tier           = var.cache_tier
  capacity_gb    = var.cache_capacity_gb

  tags = var.default_tags
}

# Module: Azure Blob Storage
module "blob" {
  source = "./modules/blob"

  storage_account_name = "${var.project_name}${var.environment}"
  container_name     = "uploads"
  location          = var.azure_location
  environment       = var.environment
  replication_type  = var.blob_replication_type

  tags = var.default_tags
}

# Outputs
output "aks_cluster_endpoint" {
  description = "AKS cluster endpoint"
  value       = module.aks_cluster.cluster_endpoint
}

output "aks_cluster_name" {
  description = "AKS cluster name"
  value       = module.aks_cluster.cluster_name
}

output "aks_cluster_id" {
  description = "AKS cluster ID"
  value       = module.aks_cluster.cluster_id
}

output "database_fqdn" {
  description = "Azure Database FQDN"
  value       = module.database.fqdn
}

output "database_port" {
  description = "Azure Database port"
  value       = module.database.port
}

output "cache_host" {
  description = "Azure Cache host"
  value       = module.cache.host
}

output "cache_port" {
  description = "Azure Cache port"
  value       = module.cache.port
}

output "storage_account_name" {
  description = "Azure Storage account name"
  value       = module.blob.storage_account_name
}

output "storage_container_name" {
  description = "Azure Storage container name"
  value       = module.blob.container_name
}

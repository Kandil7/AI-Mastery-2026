# Azure Terraform Variables
# ==========================
# Configuration variables for Azure infrastructure.

# متغيرات تكوين Terraform لـ Azure

variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
  default     = ""
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
  default     = ""
}

variable "client_id" {
  description = "Azure Service Principal App ID"
  type        = string
  default     = ""
}

variable "client_secret" {
  description = "Azure Service Principal Secret"
  type        = string
  sensitive   = true
  default     = null
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

variable "azure_location" {
  description = "Azure region for all resources"
  type        = string
  default     = "East US"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "rag-engine-prod-rg"
}

variable "vnet_cidr" {
  description = "CIDR block for Virtual Network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_prefixes" {
  description = "Subnet name prefixes"
  type        = list(string)
  default     = ["private", "public"]
}

variable "aks_node_type" {
  description = "Azure VM size for AKS nodes"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "aks_min_nodes" {
  description = "Minimum number of AKS nodes"
  type        = number
  default     = 2
}

variable "aks_max_nodes" {
  description = "Maximum number of AKS nodes"
  type        = number
  default     = 10
}

variable "aks_desired_nodes" {
  description = "Desired number of AKS nodes"
  type        = number
  default     = 3
}

variable "aks_pod_cidr" {
  description = "CIDR block for AKS pods"
  type        = string
  default     = "10.244.0.0/14"
}

variable "aks_service_cidr" {
  description = "CIDR block for AKS services"
  type        = string
  default     = "10.245.0.0/16"
}

variable "aks_cluster_cidr" {
  description = "CIDR block for AKS cluster API server"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_auto_scaling" {
  description = "Enable AKS cluster auto-scaling"
  type        = bool
  default     = true
}

variable "additional_node_pools" {
  description = "Number of additional node pools"
  type        = number
  default     = 0
}

variable "node_disk_size_gb" {
  description = "Disk size for AKS nodes in GB"
  type        = number
  default     = 100
}

variable "database_server_name" {
  description = "Database server name"
  type        = string
  default     = "ragengine-sql"
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "ragengine"
}

variable "database_admin_username" {
  description = "Database admin username"
  type        = string
  default     = "ragengine_admin"
}

variable "database_admin_password" {
  description = "Database admin password"
  type        = string
  sensitive   = true
  default     = null
}

variable "database_tier" {
  description = "Database tier"
  type        = string
  default     = "GeneralPurpose"
}

variable "database_sku_name" {
  description = "Database SKU name"
  type        = string
  default     = "GP_Gen5_2_v8"
}

variable "storage_mb" {
  description = "Storage allocation in MB"
  type        = number
  default     = 512000  # 500GB
}

variable "cache_family" {
  description = "Cache family"
  type        = string
  default     = "P"
}

variable "cache_name" {
  description = "Redis cache name"
  type        = string
  default     = "ragengine-redis"
}

variable "cache_tier" {
  description = "Cache tier"
  type        = string
  default     = "Premium"
}

variable "cache_sku_name" {
  description = "Cache SKU name"
  type        = string
  default     = "Basic"
}

variable "cache_capacity_gb" {
  description = "Cache capacity in GB"
  type        = number
  default     = 16
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "6"
}

variable "storage_account_name" {
  description = "Storage account name"
  type        = string
  default     = "ragengineuploads"
}

variable "container_name" {
  description = "Storage container name"
  type        = string
  default     = "uploads"
}

variable "replication_type" {
  description = "Storage replication type"
  type        = string
  default     = "GRS"
}

variable "aks_managed_identity" {
  description = "AKS managed identity principal ID"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

variable "default_tags" {
  description = "Default tags for all resources"
  type        = map(string)
  default = {
    Environment = "dev"
    ManagedBy  = "terraform"
  }
}

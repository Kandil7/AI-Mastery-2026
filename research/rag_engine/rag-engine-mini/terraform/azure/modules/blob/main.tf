# Azure Blob Storage Module
# ============================
# Azure Blob Storage for file uploads.

# وحدة تخزين الكائنات - تخزين الملفات على Azure

resource "azurerm_storage_account" "main" {
  name                     = "${var.storage_account_name}${var.environment}"
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type  = var.replication_type
  access_tier             = "Hot"

  enable_https_traffic_only = true

  depends_on = [
    var.resource_group,
  ]

  tags = merge(
    var.tags,
    {
      Name = "${var.storage_account_name}${var.environment}"
    }
  )
}

resource "azurerm_storage_container" "main" {
  name                  = var.container_name
  storage_account_name    = azurerm_storage_account.main.name
  container_access_type = "Private"

  depends_on = [
    azurerm_storage_account.main,
  ]
}

resource "azurerm_storage_container" "static" {
  name                  = "static"
  storage_account_name    = azurerm_storage_account.main.name
  container_access_type = "Blob"

  depends_on = [
    azurerm_storage_account.main,
  ]
}

resource "azurerm_storage_management_policy" "main" {
  storage_account_id = azurerm_storage_account.main.id

  depends_on = [
    azurerm_storage_account.main,
  ]
}

resource "azurerm_role_assignment" "aks_reader" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.aks_managed_identity

  depends_on = [
    azurerm_storage_account.main,
    var.aks_managed_identity,
  ]
}

# Outputs
output "storage_account_name" {
  description = "Storage account name"
  value       = azurerm_storage_account.main.name
}

output "storage_account_id" {
  description = "Storage account ID"
  value       = azurerm_storage_account.main.id
}

output "primary_blob_endpoint" {
  description = "Primary blob storage endpoint"
  value       = azurerm_storage_account.main.primary_blob_endpoint
}

output "container_name" {
  description = "Storage container name"
  value       = azurerm_storage_container.main.name
}

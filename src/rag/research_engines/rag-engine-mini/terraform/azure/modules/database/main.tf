# Azure Database Module (PostgreSQL)
# =====================================
# Azure Database for PostgreSQL server.

# وحدة قاعدة البيانات - خادم PostgreSQL على Azure

resource "azurerm_mssql_server" "main" {
  name                         = "${var.server_name}-${var.environment}"
  location                     = var.location
  resource_group_name          = var.resource_group_name
  administrator_login          = var.database_admin_username
  administrator_password      = var.database_admin_password
  version                      = "12.0"

  public_network_access_enabled = false

  storage_mb                   = var.storage_mb
  sku_name                     = var.sku_name

  tags = merge(
    var.tags,
    {
      Name = "${var.server_name}-${var.environment}"
    }
  )
}

resource "azurerm_mssql_database" "main" {
  name                = var.database_name
  server_id           = azurerm_mssql_server.main.id
  collation           = "SQL_Latin1_General_CP1_CI_AS"

  depends_on = [
    azurerm_mssql_server.main,
  ]
}

resource "azurerm_mssql_firewall_rule" "aks_cluster" {
  name                = "allow-aks-cluster"
  server_id           = azurerm_mssql_server.main.id
  start_ip_address    = var.aks_cluster_cidr
  end_ip_address      = var.aks_cluster_cidr

  depends_on = [
    azurerm_mssql_server.main,
  ]
}

resource "azurerm_mssql_virtual_network_rule" "vnet_access" {
  name      = "allow-vnet"
  server_id = azurerm_mssql_server.main.id
  subnet_id  = var.database_subnet_id

  depends_on = [
    azurerm_mssql_server.main,
  ]
}

resource "azurerm_mssql_database" "audit_log" {
  name       = "audit_log"
  server_id  = azurerm_mssql_server.main.id
  sku_name   = "Basic"

  depends_on = [
    azurerm_mssql_server.main,
  ]
}

# Outputs
output "fqdn" {
  description = "Database FQDN"
  value       = azurerm_mssql_server.main.fully_qualified_domain_name
}

output "port" {
  description = "Database port"
  value       = 1433
}

output "server_id" {
  description = "Database server ID"
  value       = azurerm_mssql_server.main.id
}

output "database_id" {
  description = "Database ID"
  value       = azurerm_mssql_database.main.id
}

output "subnet_id" {
  description = "Database subnet ID"
  value       = var.database_subnet_id
}

# Azure Cache Module (Redis)
# ================================
# Azure Cache for Redis Enterprise.

# وحدة التخزين المؤقت - Redis Enterprise على Azure

resource "azurerm_redis_cache" "main" {
  name                = "${var.cache_name}-${var.environment}"
  location            = var.location
  resource_group_name = var.resource_group_name
  capacity            = var.capacity_gb
  family              = var.cache_family
  sku_name            = var.sku_name
  minimum_tls_version = "1.2"

  redis_version        = var.redis_version

  enable_non_ssl_port = false
  public_network_access_enabled = false

  subnet_id = var.cache_subnet_id

  static_ip {
    subnet_id = var.cache_subnet_id
  }

  depends_on = [
    var.resource_group,
  ]

  tags = merge(
    var.tags,
    {
      Name = "${var.cache_name}-${var.environment}"
    }
  )
}

# Outputs
output "host" {
  description = "Redis cache host"
  value       = azurerm_redis_cache.main.host_name
}

output "port" {
  description = "Redis cache port"
  value       = azurerm_redis_cache.main.enable_non_ssl_port ? 6379 : 6380
}

output "ssl_port" {
  description = "Redis SSL port"
  value       = azurerm_redis_cache.main.ssl_port
}

output "cache_id" {
  description = "Redis cache ID"
  value       = azurerm_redis_cache.main.id
}

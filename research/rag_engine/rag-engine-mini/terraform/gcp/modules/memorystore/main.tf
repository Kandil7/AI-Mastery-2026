# GCP Cloud Memorystore Module
# ============================
# Managed Redis cache on Memorystore.

# وحدة Redis المُدارة على Memorystore

resource "google_redis_instance" "main" {
  name           = "${var.cluster_id}-${var.environment}"
  region         = var.gcp_region
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_gb

  display_name   = "Redis cluster for RAG Engine"
  redis_version  = var.redis_version

  location_id = var.network_id

  redis_configs {
    redis_version = var.redis_version
  maxmemory_policy = "allkeys-lru"
    notify_keyspace_events = 1
  }

  authorized_network = var.authorized_network
  connect_mode     = "PRIVATE_SERVICE_ACCESS"

  depends_on = [
    google_project_service.enabled_redis,
  ]
}

resource "google_compute_global_address" "redis" {
  name = "${var.project_name}-${var.environment}-redis-ip"
}

resource "google_compute_firewall" "redis" {
  name    = "${var.project_name}-${var.environment}-redis"
  network = var.network_id

  allow {
    protocol = "tcp"
    ports    = ["6379"]
  }

  source_ranges = var.redis_source_cidrs

  depends_on = [
    google_redis_instance.main,
  ]
}

# Outputs
output "host" {
  description = "Memorystore Redis host"
  value       = google_redis_instance.main.host
}

output "port" {
  description = "Memorystore Redis port"
  value       = google_redis_instance.main.port
}

output "cluster_id" {
  description = "Memorystore cluster ID"
  value       = google_redis_instance.main.id
}

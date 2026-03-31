# GCP Cloud SQL Module
# =======================
# Managed PostgreSQL database on Cloud SQL.

# وحدة PostgreSQL المُدارة على Cloud SQL

resource "google_sql_database_instance" "main" {
  name             = "${var.instance_name}-${var.environment}"
  database_version = var.postgres_version
  region           = var.gcp_region

  settings {
    tier              = var.cloudsql_tier
    availability_type  = "REGIONAL"
    activation_policy  = "ALWAYS"
    disk_autoresize  = var.disk_autoresize
    disk_size         = var.storage_gb
    disk_type         = "PD_SSD"
    ip_configuration {
      ipv4_enabled = true
      private_network = true
    }

    backup_configuration {
      enabled            = true
      start_time         = "03:00"
      location          = var.gcp_region
      transaction_log_retention_days = 7
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }

    database_flags {
      name  = "shared_buffers"
      value = "{DBInstanceClassMemory * 32768/16}"
    }
  }

  deletion_protection = true

  depends_on = [
    google_project_service.enabled_sqladmin,
    google_service_networking_connection.main,
  ]
}

resource "google_sql_database" "main" {
  name     = var.database_name
  instance = google_sql_database_instance.main.name
  charset  = "UTF8"

  depends_on = [
    google_sql_database_instance.main,
  ]
}

resource "google_sql_user" "main" {
  name     = var.database_username
  instance = google_sql_database_instance.main.name

  depends_on = [
    google_sql_database_instance.main,
  ]
}

resource "google_service_networking_connection" "main" {
  name                    = "${var.project_name}-${var.environment}-sql-connection"
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges   = [var.vpc_peering_cidr]

  depends_on = [
    google_compute_network.main,
  ]
}

# Outputs
output "connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "ip_address" {
  description = "Cloud SQL private IP address"
  value       = google_sql_database_instance.main.ip_address.0.ip_address
}

output "instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.main.name
}

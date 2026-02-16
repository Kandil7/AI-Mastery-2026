# GCS Bucket Module
# =================
# Google Cloud Storage bucket.

# وحدة دلو GCS - Google Cloud Storage

resource "google_storage_bucket" "main" {
  name          = "${var.bucket_name}-${var.environment}"
  location      = var.gcp_region
  force_destroy = false

  uniform_bucket_level_access = var.uniform_access

  versioning {
    enabled = var.versioning
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }

    condition {
      with_state = "ARCHIVED"
    }

    abort_incomplete_multipart_upload_days = 7
  }

  depends_on = [
    google_project_service.enabled_storage_component,
  ]
}

resource "google_storage_bucket_iam" "main" {
  bucket = google_storage_bucket.main.name
  role    = "roles/storage.objectViewer"

  members = [
    "serviceAccount:${var.gke_service_account}",
  ]

  depends_on = [
    google_storage_bucket.main,
  ]
}

resource "google_storage_bucket_iam" "write" {
  bucket = google_storage_bucket.main.name
  role    = "roles/storage.objectCreator"

  members = [
    "serviceAccount:${var.gke_service_account}",
  ]

  depends_on = [
    google_storage_bucket.main,
  ]
}

# Outputs
output "bucket_name" {
  description = "GCS bucket name"
  value       = google_storage_bucket.main.name
}

output "bucket_url" {
  description = "GCS bucket URL"
  value       = google_storage_bucket.main.url
}

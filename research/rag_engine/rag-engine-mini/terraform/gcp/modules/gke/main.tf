# GCP GKE Module
# ==================
# Google Kubernetes Engine cluster.

# وحدة مجموعة GKE - نظام K8s على Google Cloud

resource "google_container_cluster" "main" {
  name     = "${var.cluster_name}-${var.environment}"
  location = var.gcp_region

  initial_node_count = var.min_nodes
  node_count          = null  # Use autoscaling

  networking_mode = "VPC_NATIVE"

  network    = var.network_id
  subnetwork = element(var.subnet_ids, 0)

  ip_allocation_policy {
    cluster_ipv4_cidr_block = var.gcp_pod_cidr
  }

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "DEFAULT"
  }

  addons_config {
    network_policy_config {
      disabled = true
    }
  }

  remove_default_node_pool = true

  depends_on = [
    google_project_service.enabled_container,
  ]
}

resource "google_container_node_pool" "main" {
  name       = "${var.cluster_name}-${var.environment}-nodes"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name

  node_count = var.min_nodes

  node_config {
    machine_type = var.node_type
    disk_size_gb = var.node_disk_size_gb
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      environment = var.environment
      project     = var.project_name
    }

    preemptible  = var.preemptible
  }

  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade  = true
  }

  depends_on = [
    google_container_cluster.main,
  ]
}

resource "google_compute_firewall" "gke_nodes" {
  name    = "${var.project_name}-${var.environment}-gke-nodes"
  network = var.network_id

  allow {
    protocol = "tcp"
    ports    = ["443", "10250"]
  }

  source_ranges = var.gke_master_cidr

  target_tags = ["gke-${var.project_name}-${var.environment}"]
}

# Outputs
output "endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.main.endpoint
}

output "name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "location" {
  description = "GKE cluster location"
  value       = google_container_cluster.main.location
}

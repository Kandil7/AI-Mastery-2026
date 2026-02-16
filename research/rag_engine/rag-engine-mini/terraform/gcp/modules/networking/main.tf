# GCP Networking Module
# ========================
# VPC, subnets, and routing for GCP.

# وحدة شبكة GCP - VPC، الشبكات الفرعية، والتوجيه

resource "google_compute_network" "main" {
  name                    = "${var.project_name}-${var.environment}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  description = "VPC for RAG Engine"

  depends_on = [
    google_project_service.enabled_compute,
  ]
}

resource "google_compute_subnetwork" "private" {
  count                     = 3
  name                      = "${var.project_name}-${var.environment}-private-${count.index + 1}"
  network                   = google_compute_network.main.id
  ip_cidr_range            = cidrsubnet(google_compute_network.main.ip_cidr_range, 8, count.index)
  region                    = var.gcp_region
  private_ip_google_access   = false

  description = "Private subnet for RAG Engine"
}

resource "google_compute_router" "main" {
  name    = "${var.project_name}-${var.environment}-router"
  network = google_compute_network.main.id
  region  = var.gcp_region
}

resource "google_compute_nat" "main" {
  count   = 3
  name    = "${var.project_name}-${var.environment}-nat-${count.index + 1}"
  router  = google_compute_router.main.id
  region  = var.gcp_region

  nat_ip_allocate_option = "MANUAL_ONLY"
}

resource "google_compute_address" "nat" {
  count   = 3
  name    = "${var.project_name}-${var.environment}-nat-ip-${count.index + 1}"
  region  = var.gcp_region
}

resource "google_compute_router_nat" "main" {
  count              = 3
  name               = "${var.project_name}-${var.environment}-nat-rule-${count.index + 1}"
  router             = google_compute_router.main.id
  region             = var.gcp_region
  nat_ip             = element(google_compute_address.nat, count.index).address
  min_ports_per_vm  = 64
  max_ports_per_vm  = 65536

  depends_on = [
    google_compute_nat.main,
    google_compute_address.nat,
  ]
}

resource "google_compute_firewall" "gke_cluster" {
  name    = "${var.project_name}-${var.environment}-gke-cluster"
  network = google_compute_network.main.id

  allow {
    protocol = "tcp"
    ports    = ["10250"]
  }

  source_ranges = ["10.0.0.0/8"]

  target_tags = ["gke-${var.project_name}-${var.environment}"]
}

# Outputs
output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.main.id
}

output "subnet_ids" {
  description = "Subnet IDs"
  value       = google_compute_subnetwork.private[*].id
}

output "router_id" {
  description = "Cloud Router ID"
  value       = google_compute_router.main.id
}

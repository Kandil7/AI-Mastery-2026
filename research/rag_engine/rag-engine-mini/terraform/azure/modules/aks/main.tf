# Azure AKS Module
# =================
# Azure Kubernetes Service cluster.

# وحدة مجموعة AKS - نظام K8s على Azure

resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.cluster_name}-${var.environment}"
  location            = var.location
  resource_group_name = var.resource_group_name
  dns_prefix          = "${var.cluster_name}-${var.environment}"

  default_node_pool {
    name           = "nodepool"
    vm_size        = var.node_type
    node_count     = var.desired_nodes
    os_disk_size_gb = var.node_disk_size_gb

    vnet_subnet_id = element(var.subnet_ids, 0)

    enable_auto_scaling = var.enable_auto_scaling

    min_count = var.min_nodes
    max_count = var.max_nodes

    max_pods = 110
  }

  network_profile {
    network_plugin = "kubenet"
    pod_cidr     = var.aks_pod_cidr
    service_cidr = var.aks_service_cidr
  }

  identity {
    type = "SystemAssigned"
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-${var.environment}"
    }
  )
}

resource "azurerm_kubernetes_cluster_node_pool" "additional" {
  count              = var.additional_node_pools
  name               = "nodepool-${count.index}"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size            = var.node_type
  node_count         = 1

  vnet_subnet_id = element(var.subnet_ids, 0)

  os_disk_size_gb = var.node_disk_size_gb

  tags = merge(
    var.tags,
    {
      Name = "${var.cluster_name}-${var.environment}-nodepool-${count.index}"
    }
  )
}

# Outputs
output "cluster_endpoint" {
  description = "AKS cluster endpoint"
  value       = azurerm_kubernetes_cluster.main.fqdn
}

output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.main.name
}

output "cluster_id" {
  description = "AKS cluster ID"
  value       = azurerm_kubernetes_cluster.main.id
}

output "node_resource_group" {
  description = "Node pool resource group name"
  value       = azurerm_kubernetes_cluster.main.node_resource_group
}

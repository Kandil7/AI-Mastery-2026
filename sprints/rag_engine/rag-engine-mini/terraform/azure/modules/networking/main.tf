# Azure Networking Module
# ==========================
# Virtual Network, subnets, and routing.

# وحدة شبكة Azure - الشبكة الافتراضية، الشبكات الفرعية، والتوجيه

resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-${var.environment}-vnet"
  location            = var.location
  resource_group_name = var.resource_group_name
  address_space       = [var.vnet_cidr]

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-vnet"
    }
  )
}

resource "azurerm_subnet" "private" {
  count                 = length(var.subnet_prefixes)
  name                 = "${var.project_name}-${var.environment}-private-${element(var.subnet_prefixes, count.index)}"
  resource_group_name  = var.resource_group_name
  virtual_network_name  = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 4, count.index)]

  depends_on = [
    azurerm_virtual_network.main,
  ]
}

resource "azurerm_subnet" "public" {
  count                 = length(var.subnet_prefixes)
  name                 = "${var.project_name}-${var.environment}-public-${element(var.subnet_prefixes, count.index)}"
  resource_group_name  = var.resource_group_name
  virtual_network_name  = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 4, length(var.subnet_prefixes) + count.index)]

  depends_on = [
    azurerm_virtual_network.main,
  ]
}

resource "azurerm_network_security_group" "aks_cluster" {
  name                = "${var.project_name}-${var.environment}-aks-cluster"
  location            = var.location
  resource_group_name = var.resource_group_name

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-aks-cluster-sg"
    }
  )
}

resource "azurerm_network_security_rule" "aks_egress" {
  name                        = "allow_all_outbound"
  priority                    = 100
  direction                  = "Outbound"
  access                     = "Allow"
  protocol                   = "*"
  source_port_range          = "*"
  destination_port_range     = "*"
  source_address_prefixes    = "*"
  destination_address_prefixes = "*"
  resource_group_name        = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.aks_cluster.name
}

resource "azurerm_network_security_group" "aks_nodes" {
  name                = "${var.project_name}-${var.environment}-aks-nodes"
  location            = var.location
  resource_group_name = var.resource_group_name

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-aks-nodes-sg"
    }
  )
}

resource "azurerm_network_security_rule" "aks_ingress" {
  name                        = "allow_control_plane"
  priority                    = 200
  direction                  = "Inbound"
  access                     = "Allow"
  protocol                   = "Tcp"
  source_port_range          = "*"
  destination_port_range     = "10250"
  source_address_prefixes    = ["${var.vnet_cidr}"]
  destination_address_prefixes = ["*"]
  resource_group_name        = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.aks_nodes.name
}

# Outputs
output "vnet_id" {
  description = "Virtual Network ID"
  value       = azurerm_virtual_network.main.id
}

output "subnet_ids" {
  description = "Subnet IDs"
  value       = azurerm_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = azurerm_subnet.public[*].id
}

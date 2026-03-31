# AWS ElastiCache Redis Module
# ===============================
# Managed Redis cache on ElastiCache.

# وحدة Redis المُدارة على ElastiCache

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.cluster_id}-subnet-group"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "redis" {
  name_prefix = var.cluster_id
  description = "Security group for Redis"
  vpc_id      = var.vpc_id

  tags = var.tags
}

resource "aws_security_group_rule" "redis_ingress" {
  description              = "Allow Redis from EKS cluster"
  type                    = "ingress"
  from_port               = 6379
  to_port                 = 6379
  protocol                = "tcp"
  security_group_id       = aws_security_group.redis.id
  source_security_group_id = var.cluster_security_group_id
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id          = var.cluster_id
  description                  = "Redis cluster for RAG Engine"
  node_type                   = var.node_type
  num_cache_clusters           = 1
  engine                      = "redis"
  engine_version              = var.engine_version
  at_rest_encryption         = true
  transit_encryption_enabled = true
  auth_token                  = var.auth_token

  automatic_failover_enabled = true
  multi_az_enabled           = true
  auto_minor_version_upgrade  = true
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]

  tags = var.tags

  depends_on = [
    aws_elasticache_subnet_group.main,
    aws_security_group.redis,
  ]
}

resource "aws_elasticache_parameter_group" "main" {
  family      = "redis7"
  name        = "${var.cluster_id}-parameter-group"
  description = "Parameter group for RAG Engine Redis"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "maxmemory"
    value = "{node_memory * 0.75}"
  }

  tags = var.tags
}

resource "aws_elasticache_replication_group" "modify" {
  count = 0  # Set to 1 to modify parameter group

  replication_group_id    = aws_elasticache_replication_group.main.id
  parameter_group_name   = aws_elasticache_parameter_group.main.name

  apply_immediately = true

  depends_on = [
    aws_elasticache_parameter_group.main,
  ]
}

# Outputs
output "endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.main.primary_endpoint_port
}

output "cluster_arn" {
  description = "ElastiCache cluster ARN"
  value       = aws_elasticache_replication_group.main.arn
}

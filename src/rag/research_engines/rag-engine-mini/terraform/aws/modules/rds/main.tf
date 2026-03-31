# AWS RDS PostgreSQL Module
# ===========================
# Managed PostgreSQL database on RDS.

# وحدة PostgreSQL المُدارة على RDS

resource "aws_db_subnet_group" "main" {
  name       = "${var.identifier}-subnet-group"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "rds" {
  name_prefix = var.identifier
  description = "Security group for RDS PostgreSQL"
  vpc_id      = var.vpc_id

  tags = var.tags
}

resource "aws_security_group_rule" "rds_ingress" {
  description              = "Allow PostgreSQL from EKS cluster"
  type                    = "ingress"
  from_port               = 5432
  to_port                 = 5432
  protocol                = "tcp"
  security_group_id       = aws_security_group.rds.id
  source_security_group_id = var.cluster_security_group_id
}

resource "aws_db_instance" "main" {
  identifier              = var.identifier
  engine                 = var.engine
  engine_version         = var.engine_version
  instance_class         = var.instance_class
  allocated_storage       = var.allocated_storage
  storage_encrypted      = true
  storage_type           = "gp3"
  db_name                = var.database_name
  username               = var.master_username
  port                   = 5432

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.rds.id]

  multi_az               = true
  backup_retention_period  = 7
  skip_final_snapshot    = false
  publicly_accessible     = false

  tags = var.tags

  depends_on = [
    aws_db_subnet_group.main,
    aws_security_group.rds,
  ]
}

resource "aws_db_parameter_group" "main" {
  name   = "${var.identifier}-parameter-group"
  family = "postgres14"

  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory * 32768/16}"
  }

  parameter {
    name  = "max_connections"
    value = "200"
  }

  tags = var.tags
}

resource "aws_db_instance" "modify" {
  count = 0  # Set to 1 to modify parameter group

  identifier     = aws_db_instance.main.identifier
  parameter_group_name = aws_db_parameter_group.main.name

  apply_immediately = true

  depends_on = [
    aws_db_parameter_group.main,
  ]
}

# Outputs
output "endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.main.endpoint
}

output "port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.main.port
}

output "db_instance_arn" {
  description = "RDS DB instance ARN"
  value       = aws_db_instance.main.arn
}

output "db_subnet_group_name" {
  description = "DB subnet group name"
  value       = aws_db_subnet_group.main.name
}

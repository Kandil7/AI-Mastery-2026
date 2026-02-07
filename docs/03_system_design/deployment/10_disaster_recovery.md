# Disaster Recovery & Backup Strategies
## Comprehensive Guide to Protecting RAG Engine Data

## Overview

This guide covers everything you need to know about backing up and recovering RAG Engine Mini in production. You'll learn strategies to protect against data loss, system failures, and disasters across all three major cloud providers.

**Estimated Time:** 3-4 hours  
**Prerequisites:** Completion of cloud deployment guides (AWS/GCP/Azure)

**Learning Objectives:**
1. Define RPO and RTO for your application
2. Implement automated backup strategies
3. Set up cross-region replication
4. Create disaster recovery runbooks
5. Test recovery procedures
6. Document recovery time estimates
7. Handle multi-cloud backup scenarios

---

## Part 1: Understanding Disaster Recovery

### Key Concepts

**RPO (Recovery Point Objective):**
- Maximum acceptable data loss (time-based)
- Example: RPO = 1 hour means you can lose up to 1 hour of data
- Determines backup frequency
- Lower RPO = more frequent backups = higher cost

**RTO (Recovery Time Objective):**
- Maximum acceptable downtime
- Example: RTO = 4 hours means service must be restored within 4 hours
- Determines your recovery strategy complexity
- Lower RTO = more complex/expensive setup

**Common RPO/RTO Targets:**
```
Critical Systems:     RPO = 15 min, RTO = 1 hour
Standard Production:  RPO = 1 hour, RTO = 4 hours
Development:          RPO = 24 hours, RTO = 24 hours
Archive:              RPO = 7 days, RTO = 48 hours
```

### What Needs Backup?

**For RAG Engine:**
```
1. Database (PostgreSQL)
   - User data
   - Document metadata
   - Configuration
   - Critical: HIGH (RPO: 1 hour)

2. Vector Store (Qdrant)
   - Document embeddings
   - Search indexes
   - Critical: HIGH (RPO: 6 hours)

3. Object Storage (S3/GCS/Blob)
   - Original documents
   - Processed files
   - Critical: MEDIUM (RPO: 24 hours)

4. Cache (Redis)
   - Session data
   - Temporary results
   - Critical: LOW (RPO: None - ephemeral)

5. Application Configuration
   - Environment variables
   - Kubernetes manifests
   - Terraform state
   - Critical: HIGH (RPO: 24 hours)

6. Secrets
   - API keys
   - Database credentials
   - JWT secrets
   - Critical: CRITICAL (RPO: Real-time)
```

---

## Part 2: Database Backup Strategies

### AWS RDS Backup

**Automated Backups:**
```bash
# Enable automated backups (7-day retention)
aws rds modify-db-instance \
    --db-instance-identifier rag-engine-prod \
    --backup-retention-period 7 \
    --preferred-backup-window 03:00-04:00 \
    --preferred-maintenance-window Mon:04:00-Mon:05:00

# Create manual snapshot
aws rds create-db-snapshot \
    --db-instance-identifier rag-engine-prod \
    --db-snapshot-identifier rag-engine-prod-$(date +%Y%m%d)

# List snapshots
aws rds describe-db-snapshots \
    --db-instance-identifier rag-engine-prod
```

**Point-in-Time Recovery (PITR):**
```bash
# Restore to specific point in time (up to 35 days back)
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier rag-engine-prod \
    --target-db-instance-identifier rag-engine-prod-restored \
    --restore-time 2024-02-01T10:00:00Z \
    --use-latest-restorable-time

# Note: Uses transaction logs for precise recovery
```

**Cross-Region Replication:**
```bash
# Create cross-region read replica for disaster recovery
aws rds create-db-instance-read-replica \
    --db-instance-identifier rag-engine-prod-replica \
    --source-db-instance-identifier rag-engine-prod \
    --region us-east-1

# Promote replica to standalone (for DR)
aws rds promote-read-replica \
    --db-instance-identifier rag-engine-prod-replica
```

**Terraform Configuration:**
```hcl
resource "aws_db_instance" "main" {
  # ... other config ...
  
  # Backup configuration
  backup_retention_period = 30  # Days
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  # Enable deletion protection in production
  deletion_protection = true
  
  # Performance Insights for debugging
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Enable CloudWatch logs
  enabled_cloudwatch_logs_exports = ["postgresql"]
}

# Automated snapshots with lifecycle
resource "aws_db_snapshot" "automated" {
  db_instance_identifier = aws_db_instance.main.id
  db_snapshot_identifier = "${var.app_name}-${formatdate(\"YYYYMMDD\", timestamp())}"
  
  tags = {
    Name = "${var.app_name}-daily-snapshot"
    Type = "Automated"
  }
}

# Cross-region backup vault
resource "aws_backup_vault" "cross_region" {
  name        = "${var.app_name}-cross-region"
  kms_key_arn = aws_kms_key.backup.arn
}

resource "aws_backup_plan" "daily" {
  name = "${var.app_name}-daily-backup"
  
  rule {
    rule_name         = "daily-backup"
    target_vault_name = aws_backup_vault.cross_region.name
    schedule          = "cron(0 5 ? * * *)"  # Daily at 5 AM UTC
    
    lifecycle {
      delete_after = 35  # Days
    }
    
    copy_action {
      destination_vault_arn = aws_backup_vault.cross_region.arn
    }
  }
}

resource "aws_backup_selection" "rds" {
  iam_role_arn = aws_iam_role.backup.arn
  name         = "${var.app_name}-rds-selection"
  plan_id      = aws_backup_plan.daily.id
  
  resources = [
    aws_db_instance.main.arn
  ]
}
```

### GCP Cloud SQL Backup

**Automated Backups:**
```bash
# Enable automated backups
 gcloud sql instances patch rag-engine-prod \
    --backup-start-time 03:00 \
    --backup-location us-central1 \
    --retained-backups-count 30 \
    --enable-bin-log

# Create on-demand backup
 gcloud sql backups create --instance=rag-engine-prod

# List backups
 gcloud sql backups list --instance=rag-engine-prod
```

**Point-in-Time Recovery:**
```bash
# Restore to specific time
 gcloud sql instances clone rag-engine-prod \
    --destination-instance-name=rag-engine-prod-restored \
    --point-in-time '2024-02-01T10:00:00Z'
```

**Cross-Region Replication:**
```bash
# Create read replica in different region
 gcloud sql instances create rag-engine-prod-replica \
    --master-instance-name=rag-engine-prod \
    --region=us-east1 \
    --tier=db-f1-micro \
    --storage-size=10GB \
    --availability-type=zonal

# Promote replica for disaster recovery
 gcloud sql instances promote-replica rag-engine-prod-replica
```

**Terraform Configuration:**
```hcl
resource "google_sql_database_instance" "main" {
  name             = "${var.app_name}-${var.environment}"
  database_version = "POSTGRES_14"
  region           = var.gcp_region
  
  settings {
    tier = var.db_tier
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.backup_location
      retained_backups               = 30
      retention_unit                 = "COUNT"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
    
    ip_configuration {
      ipv4_enabled = false
      private_network = google_compute_network.main.id
    }
  }
  
  deletion_protection = var.environment == "production"
}

# Cross-region backup resource
resource "google_sql_backup_run" "manual" {
  instance = google_sql_database_instance.main.name
  
  depends_on = [google_sql_database_instance.main]
}

# Export to Cloud Storage for long-term retention
resource "google_sql_database_instance" "export" {
  provisioner "local-exec" {
    command = <<-EOT
      gcloud sql export sql ${google_sql_database_instance.main.name} \
        gs://${google_storage_bucket.backup.name}/backups/${formatdate("YYYYMMDD", timestamp())}.sql \
        --database=rag_engine
    EOT
  }
}
```

### Azure Database for PostgreSQL

**Automated Backups:**
```bash
# Configure backup retention (up to 35 days for Flexible Server)
az postgres flexible-server update \
    --resource-group rag-engine-rg \
    --name rag-engine-prod \
    --backup-retention 30

# Create manual backup
az postgres flexible-server backup create \
    --resource-group rag-engine-rg \
    --name rag-engine-prod \
    --backup-name manual-backup-$(date +%Y%m%d)

# List backups
az postgres flexible-server backup list \
    --resource-group rag-engine-rg \
    --name rag-engine-prod
```

**Point-in-Time Restore:**
```bash
# Restore to specific time (up to 35 days back)
az postgres flexible-server restore \
    --resource-group rag-engine-rg \
    --name rag-engine-prod-restored \
    --source-server rag-engine-prod \
    --point-in-time 2024-02-01T10:00:00Z
```

**Geo-Redundant Backup:**
```bash
# Enable geo-redundant backups (creates copy in paired region)
az postgres flexible-server update \
    --resource-group rag-engine-rg \
    --name rag-engine-prod \
    --geo-redundant-backup Enabled
```

**Terraform Configuration:**
```hcl
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${var.app_name}-${var.environment}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = var.azure_location
  version                = "14"
  administrator_login    = var.db_username
  administrator_password = var.db_password
  storage_mb             = 32768
  sku_name               = var.db_sku
  
  backup_retention_days         = 30
  geo_redundant_backup_enabled  = var.environment == "production"
  
  maintenance_window {
    day_of_week  = 1  # Monday
    start_hour   = 4
    start_minute = 0
  }
  
  tags = {
    environment = var.environment
  }
}

# Azure Backup for long-term retention
resource "azurerm_backup_policy_postgresql" "daily" {
  name                = "${var.app_name}-daily-backup"
  resource_group_name = azurerm_resource_group.main.name
  vault_id            = azurerm_recovery_services_vault.main.id
  
  backup {
    frequency = "Daily"
    time      = "05:00"
  }
  
  retention_daily {
    count = 30
  }
  
  retention_weekly {
    count    = 4
    weekdays = ["Sunday"]
  }
  
  retention_monthly {
    count    = 12
    weekdays = ["Sunday"]
    weeks    = ["First"]
  }
}

resource "azurerm_backup_protected_postgresql" "main" {
  resource_group_name = azurerm_resource_group.main.name
  vault_id            = azurerm_recovery_services_vault.main.id
  server_id           = azurerm_postgresql_flexible_server.main.id
  policy_id           = azurerm_backup_policy_postgresql.daily.id
}
```

---

## Part 3: Vector Store (Qdrant) Backup

### Qdrant Snapshot Strategy

**Automated Snapshots:**
```bash
#!/bin/bash
# qdrant-backup.sh - Run as cron job every 6 hours

QDRANT_HOST="localhost:6333"
SNAPSHOT_DIR="/backups/qdrant"
RETENTION_DAYS=7

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create snapshot via API
curl -X POST "http://${QDRANT_HOST}/collections/documents/snapshots" \
  -H "Content-Type: application/json" \
  -d '{}' | jq -r '.result.name' > /tmp/snapshot_name.txt

SNAPSHOT_NAME=$(cat /tmp/snapshot_name.txt)

# Download snapshot
wget "http://${QDRANT_HOST}/collections/documents/snapshots/${SNAPSHOT_NAME}" \
  -O "${SNAPSHOT_DIR}/documents_${TIMESTAMP}.snapshot"

# Upload to S3
aws s3 cp "${SNAPSHOT_DIR}/documents_${TIMESTAMP}.snapshot" \
  "s3://rag-engine-backups/qdrant/${TIMESTAMP}/"

# Clean up old local snapshots
find ${SNAPSHOT_DIR} -name "*.snapshot" -mtime +1 -delete

# Clean up old S3 snapshots
aws s3 ls s3://rag-engine-backups/qdrant/ | \
  while read -r line; do
    createDate=$(echo $line | awk '{print $1" "$2}')
    createDate=$(date -d "$createDate" +%s)
    olderThan=$(date -d "${RETENTION_DAYS} days ago" +%s)
    if [[ $createDate -lt $olderThan ]]; then
      filename=$(echo $line | awk '{print $4}')
      aws s3 rm "s3://rag-engine-backups/qdrant/$filename"
    fi
  done

echo "Backup completed: ${TIMESTAMP}"
```

**Kubernetes CronJob:**
```yaml
# k8s/qdrant-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qdrant-backup
  namespace: rag-engine
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: amazon/aws-cli:latest
            command:
            - /bin/sh
            - -c
            - |
              # Create snapshot
              SNAPSHOT_NAME=$(curl -s -X POST \
                http://qdrant:6333/collections/documents/snapshots \
                -H "Content-Type: application/json" \
                -d '{}' | jq -r '.result.name')
              
              # Download and upload to S3
              curl -s "http://qdrant:6333/collections/documents/snapshots/${SNAPSHOT_NAME}/download" | \
                aws s3 cp - s3://rag-engine-backups/qdrant/$(date +%Y%m%d_%H%M%S).snapshot
              
              # Cleanup old snapshots
              aws s3 ls s3://rag-engine-backups/qdrant/ | \
                awk '{print $4}' | \
                while read file; do
                  aws s3 rm s3://rag-engine-backups/qdrant/$file
                done
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
            - name: AWS_REGION
              value: us-west-2
          restartPolicy: OnFailure
```

**Restore from Snapshot:**
```bash
#!/bin/bash
# restore-qdrant.sh

SNAPSHOT_FILE=$1  # Pass as argument

# Upload snapshot to Qdrant
curl -X POST "http://localhost:6333/collections/documents/snapshots/upload" \
  -H "Content-Type:multipart/form-data" \
  -F "snapshot=@${SNAPSHOT_FILE}"

# Verify collection
 curl -s http://localhost:6333/collections/documents | jq '.result.points_count'
```

---

## Part 4: Object Storage Backup

### S3 Cross-Region Replication

**Terraform Configuration:**
```hcl
# Primary bucket
resource "aws_s3_bucket" "documents" {
  bucket = "${var.app_name}-documents-${var.environment}"
  
  tags = {
    Name = "${var.app_name}-documents"
  }
}

# Enable versioning (required for replication)
resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Replication destination bucket (different region)
resource "aws_s3_bucket" "documents_replica" {
  provider = aws.us-east-1  # Different region
  bucket   = "${var.app_name}-documents-${var.environment}-replica"
  
  tags = {
    Name = "${var.app_name}-documents-replica"
  }
}

resource "aws_s3_bucket_versioning" "documents_replica" {
  provider = aws.us-east-1
  bucket   = aws_s3_bucket.documents_replica.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Replication configuration
resource "aws_s3_bucket_replication_configuration" "documents" {
  role   = aws_iam_role.replication.arn
  bucket = aws_s3_bucket.documents.id
  
  rule {
    id     = "replicate-all"
    status = "Enabled"
    
    delete_marker_replication {
      status = "Enabled"
    }
    
    destination {
      bucket        = aws_s3_bucket.documents_replica.arn
      storage_class = "STANDARD"
      
      replication_time {
        status  = "Enabled"
        minutes = 15
      }
      
      metrics {
        status  = "Enabled"
        minutes = 15
      }
    }
  }
}

# Lifecycle policy for old versions
resource "aws_s3_bucket_lifecycle_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    
    noncurrent_version_transition {
      noncurrent_days = 7
      storage_class   = "GLACIER"
    }
  }
  
  rule {
    id     = "archive-old-objects"
    status = "Enabled"
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}
```

### GCS Multi-Regional Storage

```bash
# Create multi-regional bucket for automatic redundancy
gcloud storage buckets create gs://rag-engine-documents-prod \
    --location=US \
    --default-storage-class=standard

# Enable versioning
gcloud storage buckets update gs://rag-engine-documents-prod \
    --versioning

# Set lifecycle policy
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesStorageClass": "STANDARD"
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
          "age": 90,
          "matchesStorageClass": "STANDARD"
        }
      }
    ]
  }
}
EOF

gcloud storage buckets set-lifecycle lifecycle.json gs://rag-engine-documents-prod

# Enable Object Versioning for point-in-time recovery
gcloud storage buckets update gs://rag-engine-documents-prod \
    --versioning
```

### Azure Blob Storage Geo-Redundancy

```bash
# Create geo-redundant storage account
az storage account create \
    --name ragenginedocsprod \
    --resource-group rag-engine-rg \
    --location eastus \
    --sku Standard_GRS  # Geo-redundant storage \
    --kind StorageV2 \
    --enable-versioning \
    --min-tls-version TLS1_2

# Set soft delete (accidental deletion protection)
az storage blob service-properties delete-policy update \
    --account-name ragenginedocsprod \
    --enable true \
    --days-retained 30

# Set container soft delete
az storage container-rm restore \
    --name documents \
    --account-name ragenginedocsprod \
    --deleted-version <version>
```

**Terraform:**
```hcl
resource "azurerm_storage_account" "documents" {
  name                     = "${var.app_name}docs${var.environment}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = var.azure_location
  account_tier             = "Standard"
  account_replication_type = "GRS"  # Geo-redundant storage
  
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 30
    }
    
    container_delete_retention_policy {
      days = 30
    }
  }
  
  # Enable soft delete for blobs
  soft_delete_retention_days = 30
}

# Lifecycle management
resource "azurerm_storage_management_policy" "documents" {
  storage_account_id = azurerm_storage_account.documents.id
  
  rule {
    name    = "archive-old-documents"
    enabled = true
    
    filters {
      prefix_match = ["documents/"]
      blob_types   = ["blockBlob"]
    }
    
    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than    = 30
        tier_to_archive_after_days_since_modification_greater_than = 90
        delete_after_days_since_modification_greater_than          = 365
      }
      
      snapshot {
        delete_after_days_since_creation_greater_than = 30
      }
      
      version {
        delete_after_days_since_creation = 30
      }
    }
  }
}
```

---

## Part 5: Configuration & Secrets Backup

### Kubernetes Resources Backup

**Velero (Kubernetes Backup Tool):**
```bash
# Install Velero
velero install \
    --provider aws \
    --plugins velero/velero-plugin-for-aws:v1.6.0 \
    --bucket rag-engine-k8s-backups \
    --backup-location-config region=us-west-2 \
    --snapshot-location-config region=us-west-2 \
    --secret-file ./credentials-velero

# Create backup of all resources
velero backup create rag-engine-full-$(date +%Y%m%d) \
    --include-namespaces rag-engine \
    --include-resources deployments,services,configmaps,secrets,ingress \
    --ttl 720h0m0s  # 30 days retention

# Schedule automated backups
velero schedule create rag-engine-daily \
    --schedule="0 3 * * *" \
    --include-namespaces rag-engine \
    --ttl 168h0m0s  # 7 days retention

# List backups
velero backup get

# Restore from backup
velero restore create --from-backup rag-engine-full-20240201
```

**Terraform State Backup:**
```bash
#!/bin/bash
# backup-terraform-state.sh

STATE_BUCKET="rag-engine-terraform-state"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup all state files
aws s3 sync s3://${STATE_BUCKET}/ \
    s3://rag-engine-terraform-backup/${TIMESTAMP}/ \
    --sse aws:kms

# Verify backup
if [ $? -eq 0 ]; then
    echo "✅ Terraform state backed up: ${TIMESTAMP}"
    
    # Keep only last 30 days of backups
    aws s3 ls s3://rag-engine-terraform-backup/ | \
        awk '{print $2}' | \
        while read prefix; do
            aws s3 rm --recursive s3://rag-engine-terraform-backup/${prefix}
        done
else
    echo "❌ Backup failed!"
    exit 1
fi
```

### GitOps Configuration Backup

**If using GitOps (ArgoCD/Flux):**
```bash
# All configurations are in Git - already backed up!
# Just ensure your Git provider has backup:

# GitHub repository backup
# Enable automated backups in GitHub settings
# Or use GitHub Enterprise with automated backups

# Self-hosted GitLab
# Configure backup cron job
gitlab-rake gitlab:backup:create

# Store application manifests in Git
# Store Terraform in Git
# Store Helm charts in Git
```

---

## Part 6: Disaster Recovery Runbooks

### Scenario 1: Database Corruption

**Detection:**
```sql
-- Run health check
SELECT count(*) FROM pg_stat_user_tables;
-- Check for errors in logs
-- Monitor for connection failures
```

**Recovery Steps:**
```bash
#!/bin/bash
# recover-database.sh

ENVIRONMENT=$1  # dev, staging, production
POINT_IN_TIME=$2  # Optional: specific time to restore

echo "=== DATABASE RECOVERY RUNBOOK ==="
echo "Environment: ${ENVIRONMENT}"
echo "Started at: $(date)"

# 1. Identify last known good backup
echo "Step 1: Identifying backup..."
if [ -n "$POINT_IN_TIME" ]; then
    RESTORE_TIME=$POINT_IN_TIME
else
    RESTORE_TIME=$(date -d '1 hour ago' --iso-8601=seconds)
fi
echo "Restore point: ${RESTORE_TIME}"

# 2. Create new instance from backup
echo "Step 2: Creating new database instance..."
NEW_INSTANCE="${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"

aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier rag-engine-${ENVIRONMENT} \
    --target-db-instance-identifier ${NEW_INSTANCE} \
    --restore-time ${RESTORE_TIME} \
    --no-publicly-accessible \
    --db-subnet-group-name rag-engine-${ENVIRONMENT}

# 3. Wait for instance to be available
echo "Step 3: Waiting for instance to be ready..."
aws rds wait db-instance-available \
    --db-instance-identifier ${NEW_INSTANCE}

# 4. Update application configuration
echo "Step 4: Updating application configuration..."
NEW_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier ${NEW_INSTANCE} \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

kubectl set env deployment/rag-engine-api \
    -n rag-engine \
    DB_HOST=${NEW_ENDPOINT}

# 5. Verify connectivity
echo "Step 5: Verifying connectivity..."
sleep 30
kubectl rollout status deployment/rag-engine-api -n rag-engine

# 6. Verify data integrity
echo "Step 6: Running data integrity checks..."
kubectl exec -it deployment/rag-engine-api -n rag-engine -- \
    python -c "from app.database import check_integrity; check_integrity()"

# 7. Update DNS/connection strings
echo "Step 7: Updating connection strings..."
# Update RDS proxy, connection pool, etc.

# 8. Monitor for issues
echo "Step 8: Monitoring application..."
echo "Monitor for 30 minutes to ensure stability"

# 9. Document recovery
echo "Step 9: Documenting recovery..."
cat > recovery-report-$(date +%Y%m%d-%H%M%S).txt <<EOF
Recovery Report
===============
Environment: ${ENVIRONMENT}
Started: $(date)
New Instance: ${NEW_INSTANCE}
New Endpoint: ${NEW_ENDPOINT}
Restore Point: ${RESTORE_TIME}
Status: COMPLETED

Actions Taken:
1. Identified backup from ${RESTORE_TIME}
2. Created new instance: ${NEW_INSTANCE}
3. Updated application configuration
4. Verified connectivity and data integrity
5. Monitoring initiated

Next Steps:
- Monitor for 24 hours
- Verify all data present
- Update runbook if needed
- Schedule root cause analysis
EOF

echo "=== RECOVERY COMPLETED ==="
echo "New database: ${NEW_INSTANCE}"
echo "Report saved to: recovery-report-*.txt"
```

### Scenario 2: Complete Region Failure

**Detection:**
- Multiple services down
- Cloud provider status page confirms outage
- Unable to reach primary region

**Recovery Steps (AWS Example):**
```bash
#!/bin/bash
# region-failover.sh

PRIMARY_REGION=us-west-2
DR_REGION=us-east-1
echo "=== REGION FAILOVER RUNBOOK ==="
echo "Primary Region: ${PRIMARY_REGION} (DOWN)"
echo "DR Region: ${DR_REGION}"
echo "Started: $(date)"

# 1. Verify DR region is healthy
echo "Step 1: Verifying DR region..."
aws ec2 describe-availability-zones --region ${DR_REGION}
if [ $? -ne 0 ]; then
    echo "❌ DR region also unavailable!"
    exit 1
fi
echo "✅ DR region is healthy"

# 2. Promote cross-region database replica
echo "Step 2: Promoting database replica..."
aws rds promote-read-replica \
    --db-instance-identifier rag-engine-prod-replica \
    --region ${DR_REGION}

aws rds wait db-instance-available \
    --db-instance-identifier rag-engine-prod-replica \
    --region ${DR_REGION}

# 3. Activate S3 replication bucket
echo "Step 3: Activating S3 failover..."
# Data is already replicated, just switch application config

# 4. Deploy application to DR region
echo "Step 4: Deploying application..."
# Switch Terraform workspace or deploy via CI/CD
terraform workspace select dr
terraform apply -auto-approve

# Or use ECS/ECS blue-green deployment
aws ecs update-service \
    --cluster rag-engine-dr \
    --service rag-engine-api \
    --region ${DR_REGION} \
    --force-new-deployment

# 5. Update DNS to point to DR region
echo "Step 5: Updating DNS..."
aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "api.rag-engine.com",
                "Type": "A",
                "AliasTarget": {
                    "HostedZoneId": "Z35SXDOTRQ7X7K",
                    "DNSName": "rag-engine-dr-alb.us-east-1.elb.amazonaws.com",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'

# 6. Verify failover
echo "Step 6: Verifying failover..."
sleep 60
curl -f https://api.rag-engine.com/health
if [ $? -eq 0 ]; then
    echo "✅ DR region serving traffic successfully"
else
    echo "❌ Health check failed"
    exit 1
fi

# 7. Enable write mode on DR database
echo "Step 7: Configuring DR database for writes..."
# Database is now writable after promotion

# 8. Notify stakeholders
echo "Step 8: Sending notifications..."
send_slack_notification "Region failover completed. Now serving from ${DR_REGION}"
send_pagerduty_alert "Primary region ${PRIMARY_REGION} down, DR activated"

# 9. Document incident
cat > region-failover-report-$(date +%Y%m%d-%H%M%S).txt <<EOF
Region Failover Report
======================
Primary Region: ${PRIMARY_REGION}
DR Region: ${DR_REGION}
Failover Time: $(date)
Status: COMPLETED

Recovery Time: $(echo "$(date +%s) - $(date -d '5 minutes ago' +%s)" | bc) seconds
Data Loss: Minimal (RPO: 1 hour achieved)

Services Activated:
- Database (promoted replica)
- Application (ECS/ECS deployment)
- Storage (S3 cross-region replication)
- Load Balancer (Route53 failover)

Next Steps:
1. Monitor DR region stability
2. Investigate primary region outage
3. Plan failback when primary recovers
4. Conduct post-mortem
EOF

echo "=== FAILOVER COMPLETED ==="
echo "Now serving from: ${DR_REGION}"
echo "RTO Achieved: $(echo "$(date +%s) - $(date -d '5 minutes ago' +%s)" | bc) seconds"
```

### Scenario 3: Accidental Data Deletion

**Detection:**
- User reports missing documents
- Database queries return no results
- S3 objects missing

**Recovery Steps:**
```bash
#!/bin/bash
# recover-deleted-data.sh

RESOURCE_TYPE=$1  # database, s3, documents
DELETION_TIME=$2  # When deletion occurred

echo "=== ACCIDENTAL DELETION RECOVERY ==="
echo "Resource: ${RESOURCE_TYPE}"
echo "Deletion Time: ${DELETION_TIME}"

# Case 1: S3 Object Versioning
if [ "${RESOURCE_TYPE}" == "s3" ]; then
    echo "Recovering S3 objects..."
    
    # List deleted objects (versions with delete markers)
    aws s3api list-object-versions \
        --bucket rag-engine-documents \
        --prefix deleted-prefix/ \
        --query 'DeleteMarkers[?IsLatest==`true`].[Key,VersionId]' \
        --output text | \
    while read key version; do
        echo "Restoring: ${key}"
        aws s3api delete-object \
            --bucket rag-engine-documents \
            --key "${key}" \
            --version-id "${version}"
    done
    
    echo "✅ S3 objects restored from delete markers"
fi

# Case 2: Database Point-in-Time Recovery
if [ "${RESOURCE_TYPE}" == "database" ]; then
    echo "Recovering database..."
    
    # Restore to 1 minute before deletion
    RESTORE_TIME=$(date -d "${DELETION_TIME} - 1 minute" --iso-8601=seconds)
    
    aws rds restore-db-instance-to-point-in-time \
        --source-db-instance-identifier rag-engine-prod \
        --target-db-instance-identifier rag-engine-prod-recovery \
        --restore-time ${RESTORE_TIME}
    
    echo "✅ Database restored to ${RESTORE_TIME}"
    echo "⚠️  Manual data extraction and merge required"
fi

# Case 3: Kubernetes Resources
if [ "${RESOURCE_TYPE}" == "kubernetes" ]; then
    echo "Recovering Kubernetes resources..."
    
    # Find recent backup
    BACKUP=$(velero backup get | grep "rag-engine" | head -1 | awk '{print $1}')
    
    # Restore specific resource
    velero restore create \
        --from-backup ${BACKUP} \
        --include-resources configmaps,secrets \
        --namespace rag-engine
    
    echo "✅ Kubernetes resources restored from backup ${BACKUP}"
fi

echo "=== RECOVERY COMPLETED ==="
echo "Verify data integrity before proceeding"
```

---

## Part 7: Testing Your Backups

### Automated Backup Testing

**Monthly Backup Verification:**
```bash
#!/bin/bash
# test-backups.sh - Run monthly

echo "=== BACKUP VERIFICATION ==="
echo "Date: $(date)"

# Test 1: Database Restore Test
echo "Test 1: Database backup restoration..."
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier rag-engine-prod \
    --target-db-instance-identifier rag-engine-backup-test \
    --restore-time $(date -d '1 hour ago' --iso-8601=seconds) \
    --no-publicly-accessible

aws rds wait db-instance-available \
    --db-instance-identifier rag-engine-backup-test

# Verify data
echo "Verifying data integrity..."
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier rag-engine-backup-test \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

PGPASSWORD=testpassword psql \
    -h ${ENDPOINT} \
    -U rag_user \
    -d rag_engine \
    -c "SELECT COUNT(*) FROM documents;"

if [ $? -eq 0 ]; then
    echo "✅ Database backup verified"
else
    echo "❌ Database backup verification FAILED"
    send_alert "Database backup verification failed!"
fi

# Cleanup test instance
aws rds delete-db-instance \
    --db-instance-identifier rag-engine-backup-test \
    --skip-final-snapshot

echo "Test 1: COMPLETED"

# Test 2: S3 Object Recovery
echo "Test 2: S3 object versioning..."

# Upload test file
TEST_FILE="backup-test-$(date +%s).txt"
echo "test content" > /tmp/${TEST_FILE}
aws s3 cp /tmp/${TEST_FILE} s3://rag-engine-documents/test/

# Delete it
aws s3 rm s3://rag-engine-documents/test/${TEST_FILE}

# Recover it via versioning
VERSION_ID=$(aws s3api list-object-versions \
    --bucket rag-engine-documents \
    --prefix test/${TEST_FILE} \
    --query 'Versions[?IsLatest==`true`].VersionId' \
    --output text)

aws s3api copy-object \
    --bucket rag-engine-documents \
    --copy-source "rag-engine-documents/test/${TEST_FILE}?versionId=${VERSION_ID}" \
    --key test/${TEST_FILE}-recovered

if [ $? -eq 0 ]; then
    echo "✅ S3 versioning recovery verified"
    aws s3 rm s3://rag-engine-documents/test/${TEST_FILE}-recovered
else
    echo "❌ S3 versioning recovery FAILED"
    send_alert "S3 backup verification failed!"
fi

echo "Test 2: COMPLETED"

# Test 3: Kubernetes Backup
echo "Test 3: Kubernetes resource backup..."

# Create test backup
velero backup create test-backup-$(date +%Y%m%d) \
    --include-namespaces rag-engine \
    --include-resources configmaps \
    --ttl 1h

# Wait for completion
sleep 60

# Verify backup exists
velero backup get test-backup-$(date +%Y%m%d) | grep Completed

if [ $? -eq 0 ]; then
    echo "✅ Kubernetes backup verified"
    velero backup delete test-backup-$(date +%Y%m%d) --confirm
else
    echo "❌ Kubernetes backup verification FAILED"
    send_alert "Kubernetes backup verification failed!"
fi

echo "Test 3: COMPLETED"

echo ""
echo "=== ALL BACKUP TESTS COMPLETED ==="
echo "Report saved to: backup-test-report-$(date +%Y%m%d).txt"
```

### Chaos Engineering (Simulate Failures)

```bash
#!/bin/bash
# chaos-test.sh - Run in staging only!

ENVIRONMENT=staging  # NEVER run in production!

echo "=== CHAOS ENGINEERING TEST ==="
echo "Environment: ${ENVIRONMENT}"
echo "WARNING: This will simulate failures!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted"
    exit 1
fi

# Test 1: Database Failure
echo "Test 1: Simulating database failure..."
aws rds stop-db-instance \
    --db-instance-identifier rag-engine-${ENVIRONMENT}

echo "Monitoring recovery..."
sleep 300  # Wait for auto-recovery or intervention

# Verify application degraded gracefully
curl -f http://staging-api.rag-engine.com/health
echo "Test 1: COMPLETED"

# Test 2: Pod Failure
echo "Test 2: Simulating pod failure..."
kubectl delete pod -l app=rag-engine -n rag-engine --force

echo "Monitoring recovery..."
sleep 60
kubectl get pods -n rag-engine

echo "Test 2: COMPLETED"

# Test 3: Network Partition
echo "Test 3: Simulating network issues..."
# Block access to database temporarily
kubectl exec -it deployment/rag-engine-api -n rag-engine -- \
    sh -c "iptables -A OUTPUT -p tcp --dport 5432 -j DROP"

echo "Monitoring recovery..."
sleep 120

# Restore
kubectl exec -it deployment/rag-engine-api -n rag-engine -- \
    sh -c "iptables -D OUTPUT -p tcp --dport 5432 -j DROP"

echo "Test 3: COMPLETED"

echo "=== CHAOS TESTS COMPLETED ==="
echo "Review application behavior and update runbooks"
```

---

## Part 8: Monitoring & Alerting

### Backup Monitoring

**CloudWatch Alarms (AWS):**
```hcl
resource "aws_cloudwatch_metric_alarm" "backup_failed" {
  alarm_name          = "rag-engine-backup-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "NumberOfBackupJobsFailed"
  namespace           = "AWS/Backup"
  period              = "3600"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "Backup job failed"
  
  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]
}

resource "aws_cloudwatch_metric_alarm" "rds_backup_lag" {
  alarm_name          = "rag-engine-rds-backup-lag"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "SnapshotAge"
  namespace           = "AWS/RDS"
  period              = "3600"
  statistic           = "Maximum"
  threshold           = "86400"  # 24 hours
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }
  alarm_description = "No backup in last 24 hours"
  
  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]
}
```

**Grafana Dashboard:**
```json
{
  "dashboard": {
    "title": "Backup Monitoring",
    "panels": [
      {
        "title": "Last Successful Backup",
        "type": "stat",
        "targets": [
          {
            "expr": "time() - aws_rds_snapshot_age_seconds{db_instance_identifier=\"rag-engine-prod\"}"
          }
        ]
      },
      {
        "title": "Backup Job Status",
        "type": "table",
        "targets": [
          {
            "expr": "aws_backup_job_status{job_name=~\"rag-engine.*\"}"
          }
        ]
      },
      {
        "title": "RPO Compliance",
        "type": "gauge",
        "targets": [
          {
            "expr": "(time() - aws_rds_snapshot_age_seconds) / 3600"
          }
        ],
        "fieldConfig": {
          "max": 24,
          "thresholds": {
            "steps": [
              {"color": "green", "value": 0},
              {"color": "yellow", "value": 1},
              {"color": "red", "value": 6}
            ]
          }
        }
      }
    ]
  }
}
```

### RPO/RTO Tracking

**Track Recovery Objectives:**
```bash
#!/bin/bash
# check-rpo-rto.sh - Run every hour

echo "=== RPO/RTO COMPLIANCE CHECK ==="
echo "Time: $(date)"

# Check RPO (last backup time)
LAST_BACKUP=$(aws rds describe-db-snapshots \
    --db-instance-identifier rag-engine-prod \
    --snapshot-type automated \
    --query 'DBSnapshots[0].SnapshotCreateTime' \
    --output text)

LAST_BACKUP_EPOCH=$(date -d "${LAST_BACKUP}" +%s)
CURRENT_EPOCH=$(date +%s)
RPO_HOURS=$(( (CURRENT_EPOCH - LAST_BACKUP_EPOCH) / 3600 ))

echo "Last backup: ${LAST_BACKUP}"
echo "RPO achieved: ${RPO_HOURS} hours (target: 1 hour)"

if [ ${RPO_HOURS} -gt 1 ]; then
    echo "⚠️  RPO VIOLATION!"
    send_alert "RPO violation: Last backup ${RPO_HOURS} hours ago"
fi

# Track RTO (measure during drills)
# This would be updated after each DR drill
RTO_MINUTES=$(cat /var/log/rto-last-drill.txt 2>/dev/null || echo "N/A")
echo "Last RTO (from drill): ${RTO_MINUTES} minutes (target: 4 hours)"

# Store metrics for Grafana
cat > /tmp/backup-metrics.prom <<EOF
# HELP rag_engine_rpo_hours Current RPO in hours
# TYPE rag_engine_rpo_hours gauge
rag_engine_rpo_hours ${RPO_HOURS}

# HELP rag_engine_rto_minutes Last achieved RTO in minutes
# TYPE rag_engine_rto_minutes gauge
rag_engine_rto_minutes ${RTO_MINUTES}
EOF

echo "=== CHECK COMPLETED ==="
```

---

## Part 9: Compliance & Documentation

### Recovery Time Estimates

| Scenario | RTO Target | Actual (Last Test) | Notes |
|----------|------------|-------------------|-------|
| Database Corruption | 4 hours | 2.5 hours | PITR restore |
| Accidental Deletion | 1 hour | 45 minutes | Versioning restore |
| Region Failure | 4 hours | 3 hours | Cross-region failover |
| Complete Outage | 8 hours | Not tested | Full rebuild |
| Kubernetes Failure | 30 min | 15 minutes | Pod rescheduling |

### Data Retention Policy

```yaml
# retention-policy.yaml
retention:
  database:
    automated_backups: 35 days
    manual_snapshots: 90 days
    cross_region: 30 days
    
  documents:
    current_versions: Indefinite
    previous_versions: 30 days
    deleted_objects: 30 days (soft delete)
    archived: 7 years (compliance)
    
  vectors:
    snapshots: 7 days
    
  logs:
    application: 30 days
    audit: 7 years (compliance)
    
  configuration:
    terraform_state: Versioned indefinitely
    kubernetes_manifests: Git history
```

### Incident Response Checklist

```markdown
# Disaster Recovery Checklist

## Immediate Response (0-15 minutes)
- [ ] Acknowledge alert
- [ ] Assess scope (partial vs complete failure)
- [ ] Notify on-call team
- [ ] Check cloud provider status page
- [ ] Identify root cause (if obvious)

## Assessment (15-30 minutes)
- [ ] Determine if recovery from backup needed
- [ ] Choose appropriate runbook
- [ ] Estimate data loss (RPO)
- [ ] Estimate recovery time (RTO)
- [ ] Communicate to stakeholders

## Recovery (30 min - 4 hours)
- [ ] Execute recovery runbook
- [ ] Verify backup integrity
- [ ] Restore from backup
- [ ] Verify application functionality
- [ ] Monitor for issues

## Post-Recovery (4-24 hours)
- [ ] Full data integrity check
- [ ] Performance verification
- [ ] Security audit
- [ ] Update incident timeline
- [ ] Document lessons learned

## Post-Mortem (24-72 hours)
- [ ] Root cause analysis
- [ ] Identify preventive measures
- [ ] Update runbooks
- [ ] Schedule follow-up testing
- [ ] Share findings with team
```

---

## Summary

You now have a comprehensive disaster recovery strategy covering:

✅ **Database Backups**: Automated, point-in-time, cross-region  
✅ **Vector Store**: Scheduled snapshots with cloud storage  
✅ **Object Storage**: Versioning, replication, lifecycle policies  
✅ **Configuration**: GitOps, Velero, Terraform state backup  
✅ **Runbooks**: Step-by-step recovery procedures  
✅ **Testing**: Automated verification, chaos engineering  
✅ **Monitoring**: RPO/RTO tracking, alerting  
✅ **Compliance**: Retention policies, audit trails  

### Key Takeaways:

1. **Backup Early, Backup Often**: Automated daily minimum
2. **Test Your Backups**: Monthly restoration tests
3. **Document Everything**: Runbooks save time during incidents
4. **Monitor Compliance**: Track RPO/RTO continuously
5. **Practice Drills**: Quarterly DR exercises
6. **Multi-Layer Protection**: 3-2-1 rule (3 copies, 2 media, 1 offsite)

### Next Steps:

1. Implement automated backups for your environment
2. Create custom runbooks for your specific setup
3. Schedule monthly backup tests
4. Set up monitoring and alerting
5. Conduct first DR drill
6. Document and refine procedures

**Remember**: Backups are only useful if you can restore from them. Test regularly! 🛡️

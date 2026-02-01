#!/bin/bash
# backup-routine.sh
# Automated backup script for RAG Engine
# Usage: ./backup-routine.sh [full|incremental]

set -e

# Configuration
BACKUP_TYPE=${1:-incremental}
BACKUP_DIR="/backups/rag-engine"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30
S3_BUCKET="${S3_BACKUP_BUCKET:-rag-engine-backups}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting ${BACKUP_TYPE} backup at $(date)${NC}"

# Create backup directory
mkdir -p ${BACKUP_DIR}/${TIMESTAMP}

# Function to log messages
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a ${BACKUP_DIR}/backup.log
}

# Function to handle errors
error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a ${BACKUP_DIR}/backup.log
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed"
    fi
    
    if ! command -v pg_dump &> /dev/null; then
        error "pg_dump is not installed"
    fi
    
    log "✅ Prerequisites check passed"
}

# Backup PostgreSQL
backup_postgres() {
    log "Backing up PostgreSQL database..."
    
    local backup_file="${BACKUP_DIR}/${TIMESTAMP}/postgres.sql.gz"
    
    pg_dump \
        -h ${DB_HOST:-localhost} \
        -p ${DB_PORT:-5432} \
        -U ${DB_USER:-rag_user} \
        -d ${DB_NAME:-rag_engine} \
        --verbose \
        --format=custom \
        2>/dev/null | gzip > ${backup_file}
    
    if [ $? -eq 0 ]; then
        local size=$(du -h ${backup_file} | cut -f1)
        log "✅ PostgreSQL backup completed: ${size}"
    else
        error "PostgreSQL backup failed"
    fi
}

# Backup Qdrant snapshots
backup_qdrant() {
    log "Creating Qdrant snapshots..."
    
    local qdrant_host="${QDRANT_HOST:-localhost}"
    local qdrant_port="${QDRANT_PORT:-6333}"
    local snapshot_dir="${BACKUP_DIR}/${TIMESTAMP}/qdrant"
    
    mkdir -p ${snapshot_dir}
    
    # Create snapshots via API
    collections=("documents" "embeddings")
    
    for collection in "${collections[@]}"; do
        log "Creating snapshot for collection: ${collection}"
        
        snapshot_info=$(curl -s -X POST \
            "http://${qdrant_host}:${qdrant_port}/collections/${collection}/snapshots" \
            -H "Content-Type: application/json" \
            -d '{}')
        
        snapshot_name=$(echo ${snapshot_info} | jq -r '.result.name')
        
        if [ ! -z "${snapshot_name}" ] && [ "${snapshot_name}" != "null" ]; then
            # Download snapshot
            curl -s "http://${qdrant_host}:${qdrant_port}/collections/${collection}/snapshots/${snapshot_name}/download" \
                -o "${snapshot_dir}/${collection}_${TIMESTAMP}.snapshot"
            
            log "✅ Qdrant snapshot for ${collection} saved"
        else
            log "⚠️  Failed to create snapshot for ${collection}"
        fi
    done
}

# Backup Redis (if persistence is enabled)
backup_redis() {
    log "Backing up Redis data..."
    
    if [ -f /data/redis/dump.rdb ]; then
        cp /data/redis/dump.rdb "${BACKUP_DIR}/${TIMESTAMP}/redis_dump.rdb"
        log "✅ Redis backup completed"
    else
        log "⚠️  Redis persistence not enabled, skipping backup"
    fi
}

# Backup documents from S3
backup_documents() {
    log "Syncing documents from S3..."
    
    local doc_backup_dir="${BACKUP_DIR}/${TIMESTAMP}/documents"
    mkdir -p ${doc_backup_dir}
    
    aws s3 sync \
        s3://${DOCUMENTS_BUCKET:-rag-engine-documents}/ \
        ${doc_backup_dir}/ \
        --storage-class STANDARD_IA
    
    log "✅ Documents backup completed"
}

# Backup Kubernetes manifests
backup_k8s_manifests() {
    log "Backing up Kubernetes manifests..."
    
    if command -v kubectl &> /dev/null; then
        kubectl get all -n rag-engine -o yaml > "${BACKUP_DIR}/${TIMESTAMP}/k8s_manifests.yaml"
        kubectl get configmaps -n rag-engine -o yaml >> "${BACKUP_DIR}/${TIMESTAMP}/k8s_manifests.yaml"
        kubectl get secrets -n rag-engine -o yaml >> "${BACKUP_DIR}/${TIMESTAMP}/k8s_manifests.yaml"
        log "✅ Kubernetes manifests backed up"
    else
        log "⚠️  kubectl not available, skipping K8s backup"
    fi
}

# Create backup metadata
create_metadata() {
    log "Creating backup metadata..."
    
    cat > "${BACKUP_DIR}/${TIMESTAMP}/metadata.json" <<EOF
{
    "backup_type": "${BACKUP_TYPE}",
    "timestamp": "${TIMESTAMP}",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.0",
    "components": [
        "postgresql",
        "qdrant",
        "redis",
        "documents",
        "kubernetes"
    ],
    "retention_days": ${RETENTION_DAYS},
    "size": "$(du -sh ${BACKUP_DIR}/${TIMESTAMP} | cut -f1)"
}
EOF
    
    log "✅ Metadata created"
}

# Upload to S3
upload_to_s3() {
    log "Uploading backup to S3..."
    
    aws s3 sync \
        ${BACKUP_DIR}/${TIMESTAMP}/ \
        s3://${S3_BUCKET}/backups/${TIMESTAMP}/ \
        --storage-class STANDARD_IA
    
    if [ $? -eq 0 ]; then
        log "✅ Backup uploaded to s3://${S3_BUCKET}/backups/${TIMESTAMP}/"
    else
        error "Failed to upload backup to S3"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Local cleanup
    find ${BACKUP_DIR} -maxdepth 1 -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} + 2>/dev/null || true
    
    # S3 cleanup
    aws s3 ls s3://${S3_BUCKET}/backups/ | \
        awk '{print $2}' | \
        while read prefix; do
            # Extract date from prefix
            backup_date=$(echo ${prefix} | grep -oP '\d{8}_\d{6}' || true)
            if [ ! -z "${backup_date}" ]; then
                backup_epoch=$(date -d "${backup_date:0:8} ${backup_date:9:2}:${backup_date:11:2}:${backup_date:13:2}" +%s 2>/dev/null || echo 0)
                cutoff_epoch=$(date -d "${RETENTION_DAYS} days ago" +%s)
                
                if [ ${backup_epoch} -lt ${cutoff_epoch} ]; then
                    log "Deleting old backup: ${prefix}"
                    aws s3 rm --recursive s3://${S3_BUCKET}/backups/${prefix} 2>/dev/null || true
                fi
            fi
        done
    
    log "✅ Cleanup completed"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    local issues=0
    
    # Check PostgreSQL backup
    if [ ! -f "${BACKUP_DIR}/${TIMESTAMP}/postgres.sql.gz" ]; then
        log "❌ PostgreSQL backup missing"
        issues=$((issues + 1))
    fi
    
    # Check metadata
    if [ ! -f "${BACKUP_DIR}/${TIMESTAMP}/metadata.json" ]; then
        log "❌ Metadata file missing"
        issues=$((issues + 1))
    fi
    
    if [ ${issues} -eq 0 ]; then
        log "✅ Backup integrity verified"
    else
        error "Backup verification failed with ${issues} issues"
    fi
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Slack notification
    if [ ! -z "${SLACK_WEBHOOK_URL}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${message}\"}" \
            ${SLACK_WEBHOOK_URL} \
            2>/dev/null || true
    fi
    
    # Email notification (if configured)
    if [ ! -z "${ALERT_EMAIL}" ] && command -v mail &> /dev/null; then
        echo "${message}" | mail -s "RAG Engine Backup ${status}" ${ALERT_EMAIL} \
            2>/dev/null || true
    fi
}

# Main execution
main() {
    log "=========================================="
    log "RAG Engine Backup Routine"
    log "Type: ${BACKUP_TYPE}"
    log "Started at: $(date)"
    log "=========================================="
    
    check_prerequisites
    
    case ${BACKUP_TYPE} in
        full)
            backup_postgres
            backup_qdrant
            backup_redis
            backup_documents
            backup_k8s_manifests
            ;;
        incremental)
            backup_postgres
            backup_qdrant
            ;;
        *)
            error "Unknown backup type: ${BACKUP_TYPE}. Use 'full' or 'incremental'"
            ;;
    esac
    
    create_metadata
    verify_backup
    upload_to_s3
    cleanup_old_backups
    
    # Calculate duration
    end_time=$(date +%s)
    start_time=$(date -d "${TIMESTAMP:0:8} ${TIMESTAMP:9:2}:${TIMESTAMP:11:2}:${TIMESTAMP:13:2}" +%s 2>/dev/null || echo ${end_time})
    duration=$((end_time - start_time))
    
    log "=========================================="
    log "Backup completed successfully!"
    log "Duration: ${duration} seconds"
    log "Backup location: s3://${S3_BUCKET}/backups/${TIMESTAMP}/"
    log "=========================================="
    
    send_notification "SUCCESS" "✅ RAG Engine backup completed in ${duration}s. Location: s3://${S3_BUCKET}/backups/${TIMESTAMP}/"
}

# Run main function
main "$@"

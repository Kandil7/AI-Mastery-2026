# RAG Engine Deployment Scripts

This directory contains production-ready scripts for deploying, managing, and maintaining RAG Engine.

## 📁 Directory Structure

```
scripts/
├── deployment/          # Deployment scripts
├── backup/             # Backup automation scripts
├── monitoring/         # Health check and monitoring scripts
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Docker Deployment (Easiest)

```bash
# One-command deployment
./scripts/deployment/quick-start-docker.sh production

# Or for development
./scripts/deployment/quick-start-docker.sh development
```

This will:
- ✅ Check prerequisites (Docker, Docker Compose)
- ✅ Create necessary directories
- ✅ Generate secure secrets
- ✅ Download docker-compose.yml
- ✅ Start all services
- ✅ Verify deployment health

### 2. Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deployment/deploy-to-kubernetes.sh production rag-engine

# Or with custom namespace
./scripts/deployment/deploy-to-kubernetes.sh staging staging-ns
```

This will:
- ✅ Check kubectl connectivity
- ✅ Create namespace
- ✅ Generate and apply secrets
- ✅ Deploy PostgreSQL, Redis, Qdrant
- ✅ Deploy RAG Engine API
- ✅ Configure Horizontal Pod Autoscaler
- ✅ Verify deployment

## 📦 Deployment Scripts

### `quick-start-docker.sh`

One-command Docker deployment.

**Usage:**
```bash
./scripts/deployment/quick-start-docker.sh [environment]
```

**Parameters:**
- `environment`: `production` (default) or `development`

**What it does:**
1. Checks Docker and Docker Compose installation
2. Creates data directories (postgres, redis, qdrant, documents)
3. Generates secure environment variables (.env)
4. Downloads/creates docker-compose.yml
5. Creates nginx configuration
6. Pulls Docker images
7. Starts services
8. Health checks

**Output:**
- Running RAG Engine on http://localhost:8000
- All data persisted in ./data/
- Logs in ./logs/

### `deploy-to-kubernetes.sh`

Deploy to any Kubernetes cluster.

**Usage:**
```bash
./scripts/deployment/deploy-to-kubernetes.sh [environment] [namespace] [image-tag]
```

**Parameters:**
- `environment`: `production` (default) or `staging`
- `namespace`: `rag-engine` (default)
- `image-tag`: `latest` (default)

**Requirements:**
- kubectl configured and connected to cluster
- Kubernetes cluster running (minikube, EKS, GKE, AKS, etc.)
- Helm (optional but recommended)

**What it deploys:**
- Namespace
- Secrets (JWT, DB password, Redis password)
- ConfigMap (environment variables)
- PostgreSQL StatefulSet
- Redis Deployment
- Qdrant StatefulSet
- RAG Engine API Deployment (3 replicas)
- Services for all components
- Ingress configuration
- Horizontal Pod Autoscaler

## 💾 Backup Scripts

### `backup-routine.sh`

Automated backup script for all RAG Engine data.

**Usage:**
```bash
# Full backup (all components)
./scripts/backup/backup-routine.sh full

# Incremental backup (database + vectors)
./scripts/backup/backup-routine.sh incremental
```

**Features:**
- PostgreSQL database dump
- Qdrant vector store snapshots
- Redis data (if persistence enabled)
- Documents from S3
- Kubernetes manifests
- Automatic S3 upload
- 30-day retention policy
- Integrity verification
- Slack/email notifications

**Environment Variables:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=rag_user
export DB_NAME=rag_engine
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export S3_BACKUP_BUCKET=rag-engine-backups
export SLACK_WEBHOOK_URL=https://hooks.slack.com/...
export ALERT_EMAIL=ops@example.com
```

**Scheduling:**
```bash
# Add to crontab for automated backups
# Daily at 3 AM
0 3 * * * /path/to/backup-routine.sh full

# Or every 6 hours (incremental)
0 */6 * * * /path/to/backup-routine.sh incremental
```

## 🔍 Monitoring Scripts

### `health-check.sh`

Comprehensive health check for RAG Engine.

**Usage:**
```bash
# Quick check (essential services only)
./scripts/monitoring/health-check.sh quick

# Full check (all components)
./scripts/monitoring/health-check.sh full
```

**Checks Performed:**

**Quick Check:**
- ✅ API health endpoint
- ✅ Database connectivity
- ✅ Docker container status

**Full Check:**
- ✅ API health and response time
- ✅ Database connectivity
- ✅ Redis cache
- ✅ Qdrant vector store
- ✅ Disk space
- ✅ Memory usage
- ✅ CPU usage
- ✅ Container health
- ✅ SSL certificate expiry
- ✅ API endpoints
- ✅ Log errors
- ✅ Backup status

**Exit Codes:**
- `0`: All checks passed (or warnings only)
- `1`: Some checks failed

**Integration:**
```bash
# Add to monitoring system
./scripts/monitoring/health-check.sh full || alert_ops_team

# Or in CI/CD
./scripts/monitoring/health-check.sh quick
```

## 🔧 Usage Examples

### Complete Deployment Workflow

```bash
# 1. Deploy with Docker
./scripts/deployment/quick-start-docker.sh production

# 2. Check health
./scripts/monitoring/health-check.sh full

# 3. Setup automated backups
crontab -e
# Add: 0 3 * * * /path/to/scripts/backup/backup-routine.sh full

# 4. Monitor regularly
./scripts/monitoring/health-check.sh quick
```

### Kubernetes Deployment Workflow

```bash
# 1. Ensure kubectl is configured
kubectl cluster-info

# 2. Deploy
./scripts/deployment/deploy-to-kubernetes.sh production

# 3. Verify
kubectl get pods -n rag-engine

# 4. Check health
kubectl port-forward svc/rag-engine-api 8000:80 -n rag-engine &
./scripts/monitoring/health-check.sh full

# 5. Setup backup CronJob
kubectl apply -f k8s/backup-cronjob.yaml
```

### Disaster Recovery

```bash
# If database corruption detected
./scripts/monitoring/health-check.sh full
# Shows: ❌ Database connection

# Restore from backup
./scripts/backup/restore-database.sh s3://rag-engine-backups/20240201_030000/

# Verify restoration
./scripts/monitoring/health-check.sh full
# Shows: ✅ Database connection
```

## 📝 Script Requirements

### Common Requirements
- Bash 4.0+
- curl
- openssl (for secret generation)

### Docker Deployment
- Docker Engine 20.10+
- Docker Compose 2.0+

### Kubernetes Deployment
- kubectl 1.24+
- Connected to Kubernetes cluster

### Backup Script
- AWS CLI (for S3 upload)
- pg_dump (PostgreSQL client)
- curl
- jq (optional, for parsing)

### Monitoring Script
- bc (for calculations)
- docker-compose (for container checks)
- openssl (for SSL checks)
- Standard Unix tools (free, top, df, du)

## 🎯 Best Practices

### 1. Security
- Never commit `.env` files to version control
- Rotate secrets regularly
- Use least-privilege AWS credentials
- Enable SSL/TLS in production

### 2. Monitoring
- Run health checks every 5 minutes
- Review logs daily
- Monitor disk space weekly
- Check SSL expiry 30 days before expiration

### 3. Backups
- Full backup: Daily at 3 AM (low traffic)
- Incremental: Every 6 hours
- Test restore monthly
- Keep offsite copies (S3)

### 4. Updates
- Test updates in staging first
- Use rolling deployments
- Keep rollback plan ready
- Monitor error rates after updates

## 🆘 Troubleshooting

### Docker Deployment Issues

**Port already in use:**
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Change port in docker-compose.yml
# Or stop the conflicting service
```

**Permission denied:**
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data
```

### Kubernetes Issues

**Image pull errors:**
```bash
# Check if image exists
kubectl describe pod <pod-name> -n rag-engine

# Or use local image
kubectl set image deployment/rag-engine-api api=rag-engine:local -n rag-engine
```

**Pending pods:**
```bash
# Check resource constraints
kubectl describe pod <pod-name> -n rag-engine

# Check node capacity
kubectl top nodes
```

### Backup Failures

**S3 upload fails:**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://rag-engine-backups/
```

**Database connection fails:**
```bash
# Check if database is running
docker-compose ps

# Or for Kubernetes
kubectl get pods -n rag-engine | grep postgres
```

## 📚 Additional Resources

- [Docker Deployment Guide](../../docs/learning/deployment/02-docker-deployment.md)
- [Kubernetes Guide](../../docs/learning/deployment/03-kubernetes-deployment.md)
- [Disaster Recovery Guide](../../docs/learning/deployment/10-disaster-recovery.md)
- [Troubleshooting Playbook](../../docs/learning/deployment/05-troubleshooting-playbook.md)

## 🤝 Contributing

To add new scripts:
1. Place in appropriate subdirectory
2. Make executable: `chmod +x script.sh`
3. Add header comment with usage
4. Update this README
5. Test thoroughly

## 📄 License

These scripts are part of the RAG Engine Mini project and follow the same license terms.

---

**Need help?** Check the troubleshooting playbook or open an issue on GitHub.

**Ready to deploy?** Start with `./scripts/deployment/quick-start-docker.sh` 🚀

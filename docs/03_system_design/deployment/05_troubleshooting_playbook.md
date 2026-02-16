# Production Troubleshooting Playbook

## Overview

When things go wrong in production, you need systematic approaches to diagnose and fix issues quickly. This playbook provides step-by-step troubleshooting procedures for common production problems.

## Quick Reference: Emergency Commands

```bash
# Check system health
kubectl get pods -n rag-engine
kubectl top nodes

# Check logs (last 100 lines)
kubectl logs -l app=rag-engine-api -n rag-engine --tail=100

# Restart pod
kubectl delete pod <pod-name> -n rag-engine

# Scale up quickly
kubectl scale deployment rag-engine-api --replicas=10 -n rag-engine

# Rollback deployment
kubectl rollout undo deployment/rag-engine-api -n rag-engine

# Port forward for debugging
kubectl port-forward svc/rag-engine-api 8080:80 -n rag-engine
```

---

## Issue Categories

### 1. Application Not Responding

#### Symptoms
- 502/503 errors from load balancer
- Health checks failing
- High response latency

#### Diagnosis Steps

**Step 1: Check Pod Status**
```bash
kubectl get pods -n rag-engine

# Look for:
# - CrashLoopBackOff
# - ImagePullBackOff
# - Pending
# - Evicted
```

**Step 2: Describe Pod**
```bash
kubectl describe pod <pod-name> -n rag-engine

# Check:
# - Events section
# - Resource constraints
# - Node assignment
```

**Step 3: Check Logs**
```bash
# Current logs
kubectl logs <pod-name> -n rag-engine

# Previous container (if crashed)
kubectl logs <pod-name> -n rag-engine --previous

# All pods matching label
kubectl logs -l app=rag-engine-api -n rag-engine --tail=500
```

**Common Causes & Solutions:**

**A. CrashLoopBackOff**
```bash
# Cause: App crashes on start
# Solution: Check logs

kubectl logs <pod-name> -n rag-engine --previous | tail -50

# Common fixes:
# 1. Missing environment variables
kubectl get pod <pod-name> -n rag-engine -o yaml | grep -A 5 env

# 2. Database connection failure
# Check secrets
kubectl get secrets -n rag-engine

# 3. Port conflict
kubectl describe pod <pod-name> | grep Port
```

**B. ImagePullBackOff**
```bash
# Cause: Can't download image
# Solutions:

# 1. Check image exists
docker pull your-registry/rag-engine:v1.0

# 2. Check imagePullSecrets
kubectl get serviceaccount default -n rag-engine -o yaml

# 3. Re-create with correct image
kubectl set image deployment/rag-engine-api \
  api=correct-registry/rag-engine:v1.0 \
  -n rag-engine
```

**C. OOMKilled (Out of Memory)**
```bash
# Cause: Exceeded memory limit
# Check:
kubectl describe pod <pod-name> | grep -A 5 "Reason"

# Solution: Increase limit or optimize
kubectl patch deployment rag-engine-api -n rag-engine -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"1Gi"}}}]}}}}'

# Or set in manifest:
resources:
  limits:
    memory: "1Gi"  # Increased from 512Mi
```

---

### 2. High Latency / Slow Response

#### Symptoms
- Response time > 2 seconds
- Timeout errors
- Queue buildup

#### Diagnosis Steps

**Step 1: Check Resource Usage**
```bash
# Pod resources
kubectl top pods -n rag-engine

# Node resources
kubectl top nodes

# If high CPU:
# - Scale horizontally (add pods)
kubectl scale deployment rag-engine-api --replicas=10 -n rag-engine

# If high memory:
# - Check for memory leaks
# - Profile application
```

**Step 2: Check Database Performance**
```bash
# Connect to database pod
kubectl exec -it <postgres-pod> -n rag-engine -- psql -U postgres

# Check active queries
SELECT * FROM pg_stat_activity WHERE state = 'active';

# Check slow queries
SELECT query, mean_exec_time FROM pg_stat_statements 
ORDER BY mean_exec_time DESC LIMIT 10;

# Check connection count
SELECT count(*) FROM pg_stat_activity;
```

**Step 3: Check External Services**
```bash
# Test database connectivity
kubectl exec <api-pod> -n rag-engine -- nc -zv postgres 5432

# Test Redis
kubectl exec <api-pod> -n rag-engine -- redis-cli -h redis ping

# Test Qdrant
kubectl exec <api-pod> -n rag-engine -- curl http://qdrant:6333/healthz
```

**Step 4: Enable Debug Logging**
```bash
# Temporarily increase log level
kubectl set env deployment/rag-engine-api LOG_LEVEL=DEBUG -n rag-engine

# Watch logs
kubectl logs -f <pod-name> -n rag-engine

# After debugging, reset
kubectl set env deployment/rag-engine-api LOG_LEVEL=INFO -n rag-engine
```

**Common Solutions:**

**Database Connection Pool Exhaustion:**
```python
# Increase pool size in config
DATABASE_POOL_SIZE = 50  # Was 20
DATABASE_MAX_OVERFLOW = 20
```

**Redis Cache Misses:**
```python
# Increase cache TTL
CACHE_TTL = 600  # Was 300

# Add caching to expensive queries
@cache_result(ttl=300)
def expensive_query():
    pass
```

**Vector Search Slow:**
```python
# Reduce K for search
DEFAULT_SEARCH_K = 5  # Was 10

# Use approximate search
USE_HNSW = True
```

---

### 3. Database Issues

#### Connection Refused
```bash
# Check if PostgreSQL is running
kubectl get pods -l app=postgres -n rag-engine

# Check service
kubectl get svc postgres -n rag-engine

# Test from API pod
kubectl exec <api-pod> -n rag-engine -- \
  pg_isready -h postgres -p 5432

# Check network policy
kubectl get networkpolicies -n rag-engine
```

#### Data Corruption
```bash
# Check PostgreSQL logs
kubectl logs <postgres-pod> -n rag-engine | grep ERROR

# Verify data integrity
kubectl exec <postgres-pod> -n rag-engine -- psql -U postgres -c "SELECT pg_database.datname, pg_database_size(pg_database.datname) FROM pg_database WHERE datname='rag_engine';"

# Restore from backup (if needed)
# See backup/restore scripts
```

---

### 4. SSL/TLS Issues

#### Certificate Expired
```bash
# Check certificate expiry
echo | openssl s_client -servername your-domain.com -connect your-domain.com:443 2>/dev/null | openssl x509 -noout -dates

# Renew with certbot
kubectl exec <certbot-pod> -- certbot renew

# Or recreate certificate
kubectl delete secret rag-engine-tls -n rag-engine
# Trigger cert-manager to recreate
```

#### Certificate Mismatch
```bash
# Verify certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com </dev/null 2>/dev/null | openssl x509 -text | grep "Subject:"

# Should match domain name
```

---

### 5. Scaling Issues

#### HPA Not Scaling
```bash
# Check HPA status
kubectl get hpa -n rag-engine
kubectl describe hpa rag-engine-api-hpa -n rag-engine

# Common issues:
# 1. Metrics server not running
kubectl get pods -n kube-system | grep metrics

# 2. No resource requests set
kubectl get deployment rag-engine-api -n rag-engine -o yaml | grep resources -A 5

# 3. Current metrics
kubectl top pods -n rag-engine
```

**Fix Missing Metrics Server:**
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

---

### 6. Disk Space Issues

#### Node Disk Pressure
```bash
# Check node status
kubectl describe node <node-name> | grep -A 5 Conditions

# Check disk usage on node
kubectl debug node/<node-name> -it --image=busybox -- df -h

# Clean up
kubectl delete pods --all-namespaces --field-selector=status.phase=Failed

# Or clean Docker
docker system prune -af
```

#### PVC Full
```bash
# Check PVC usage
kubectl get pvc -n rag-engine

# Describe for details
kubectl describe pvc <pvc-name> -n rag-engine

# Expand PVC (if supported)
kubectl patch pvc <pvc-name> -n rag-engine -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'

# Or clean up data
kubectl exec <postgres-pod> -n rag-engine -- psql -U postgres -c "VACUUM FULL;"
```

---

### 7. Networking Issues

#### DNS Resolution Failures
```bash
# Test DNS from pod
kubectl run -it --rm debug --image=busybox:1.28 --restart=Never -- nslookup kubernetes.default

# Check CoreDNS
kubectl get pods -n kube-system | grep dns

# Check DNS config
kubectl get configmap coredns -n kube-system -o yaml
```

#### Service Not Accessible
```bash
# Check endpoints
kubectl get endpoints rag-engine-api -n rag-engine

# Should show pod IPs

# Test from another pod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://rag-engine-api:80/health

# Check if selector matches pods
kubectl get pods -l app=rag-engine-api -n rag-engine
```

---

### 8. Security Incidents

#### Suspicious Activity
```bash
# Check audit logs
kubectl logs <audit-pod> -n kube-system | grep -i "unauthorized"

# Check pod security context
kubectl get pod <pod-name> -n rag-engine -o yaml | grep securityContext -A 10

# Review RBAC
kubectl get rolebindings -n rag-engine
kubectl get roles -n rag-engine
```

#### DDoS Attack
```bash
# Enable rate limiting
kubectl annotate ingress rag-engine-ingress -n rag-engine \
  nginx.ingress.kubernetes.io/limit-rps=10

# Scale up
kubectl scale deployment rag-engine-api --replicas=20 -n rag-engine

# Enable WAF rules
kubectl annotate ingress rag-engine-ingress -n rag-engine \
  nginx.ingress.kubernetes.io/enable-modsecurity=true
```

---

## Diagnostic Scripts

### System Health Check Script
```bash
#!/bin/bash
# health-check.sh

echo "=== RAG Engine Health Check ==="
echo "Timestamp: $(date)"

# Check pods
echo -e "\n1. Pod Status:"
kubectl get pods -n rag-engine -o wide

# Check resources
echo -e "\n2. Resource Usage:"
kubectl top pods -n rag-engine 2>/dev/null || echo "Metrics server not available"

# Check services
echo -e "\n3. Services:"
kubectl get svc -n rag-engine

# Check ingress
echo -e "\n4. Ingress:"
kubectl get ingress -n rag-engine

# Test endpoint
echo -e "\n5. Health Check:"
curl -s http://your-domain.com/health | jq .

echo -e "\n=== Check Complete ==="
```

### Log Analysis Script
```bash
#!/bin/bash
# analyze-logs.sh

POD=$1
NAMESPACE="rag-engine"

if [ -z "$POD" ]; then
  echo "Usage: $0 <pod-name>"
  exit 1
fi

echo "Analyzing logs for pod: $POD"

# Error count
echo -e "\n1. Error Count:"
kubectl logs $POD -n $NAMESPACE | grep -i error | wc -l

# Top error messages
echo -e "\n2. Top Error Messages:"
kubectl logs $POD -n $NAMESPACE | grep -i error | sort | uniq -c | sort -rn | head -10

# Response times
echo -e "\n3. Slow Requests:"
kubectl logs $POD -n $NAMESPACE | grep -E "[0-9]+ms" | grep -v "[0-9]{1,2}ms" | tail -20

echo -e "\nAnalysis complete"
```

---

## Escalation Procedures

### Severity Levels

**SEV 1 (Critical):** Complete outage
- All users affected
- Revenue impact
- Data loss risk
- Action: Page on-call immediately, all hands

**SEV 2 (High):** Major functionality impaired
- Partial outage
- Workarounds available
- Action: Page on-call, begin incident response

**SEV 3 (Medium):** Minor impact
- Degraded performance
- Single feature affected
- Action: Create ticket, investigate during business hours

**SEV 4 (Low):** Cosmetic/minimal impact
- Minor bug
- Documentation issue
- Action: Create ticket, backlog

### Incident Response Flow

```
1. DETECT (via monitoring or report)
         ↓
2. ACKNOWLEDGE (within 5 minutes)
   - Post in #incidents channel
   - Assign severity
         ↓
3. ASSESS (within 15 minutes)
   - Determine impact
   - Identify scope
   - Begin timeline
         ↓
4. MITIGATE (immediate)
   - Apply quick fixes
   - Scale up/down
   - Enable/disable features
         ↓
5. RESOLVE (root cause fix)
   - Deploy fix
   - Verify resolution
   - Monitor
         ↓
6. POST-MORTEM (within 24-48 hours)
   - Document timeline
   - Identify root cause
   - Create action items
```

---

## Recovery Procedures

### Complete Cluster Failure

**1. Verify Backup Exists:**
```bash
# List available backups
velero backup get | grep rag-engine

# Check backup details
velero backup describe rag-engine-daily-20240115
```

**2. Restore:**
```bash
# Restore from backup
velero restore create --from-backup rag-engine-daily-20240115

# Verify restoration
kubectl get pods -n rag-engine
```

### Database Restore

```bash
# From S3 backup
aws s3 cp s3://backups/rag-engine/postgres-20240115.sql.gz - | gunzip | \
  kubectl exec -i <postgres-pod> -n rag-engine -- psql -U postgres rag_engine

# Verify
curl http://your-domain.com/health
```

---

## Prevention: Monitoring Checklist

### Set Up These Alerts

**Critical:**
- [ ] Pod crash loop
- [ ] High error rate (>1%)
- [ ] API down for >2 minutes
- [ ] Database connection failures
- [ ] Disk space >90%

**Warning:**
- [ ] High CPU (>80% for 5 min)
- [ ] High memory (>85%)
- [ ] Response time >2 seconds
- [ ] SSL expiring in 7 days
- [ ] Failed backup

**Info:**
- [ ] Deployment completed
- [ ] Scaling events
- [ ] Certificate renewed

---

## Quick Fixes

### Emergency Scale Up
```bash
kubectl scale deployment rag-engine-api --replicas=20 -n rag-engine
```

### Quick Rollback
```bash
# Check history
kubectl rollout history deployment/rag-engine-api -n rag-engine

# Rollback to previous
kubectl rollout undo deployment/rag-engine-api -n rag-engine

# Or rollback to specific revision
kubectl rollout undo deployment/rag-engine-api --to-revision=3 -n rag-engine
```

### Restart All Pods
```bash
kubectl rollout restart deployment/rag-engine-api -n rag-engine
```

### Clear Failed Pods
```bash
kubectl delete pods --field-selector=status.phase=Failed -n rag-engine
```

---

## Resources

**Runbooks:**
- Database failover: `docs/runbooks/db-failover.md`
- Cache rebuild: `docs/runbooks/cache-rebuild.md`
- Security incident: `docs/runbooks/security-incident.md`

**Contacts:**
- On-call: +1-XXX-XXX-XXXX
- Slack: #incidents
- PagerDuty: https://pagerduty.com/rag-engine

**Tools:**
- Kubectl cheatsheet: `kubectl cheatsheet`
- Cluster dashboard: `https://grafana.example.com`
- Logs: `https://loki.example.com`

---

**Remember:** When in doubt, scale up and investigate. It's better to over-provision than to have downtime.

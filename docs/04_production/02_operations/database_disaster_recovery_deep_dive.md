# Database Disaster Recovery Deep Dive

## Overview

This document provides comprehensive disaster recovery (DR) strategies and procedures for database systems in production environments. It covers architecture patterns, cross-cloud replication, testing procedures, and operational runbooks for SRE teams.

### Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| RTO (Recovery Time Objective) | Maximum acceptable time to restore service | 15 minutes - 4 hours |
| RPO (Recovery Point Objective) | Maximum acceptable data loss (time-based) | 0 - 1 hour |
| Recovery Confidence | Percentage of successful test restores | 99.9% |

---

## 1. DR Architecture Patterns

### 1.1 Backup and Restore (Cold)

The simplest DR pattern where periodic backups are stored in a separate location.

```yaml
# Kubernetes CronJob for PostgreSQL backup
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup-cron
  namespace: database
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  successfulJobsHistoryLimit: 7
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            env:
            - name: POSTGRES_HOST
              value: "postgres-primary.database.svc.cluster.local"
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-credentials
                  key: password
            - name: BACKUP_BUCKET
              value: "gs://prod-database-backups"
            command:
            - /bin/sh
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              pg_dump -h $POSTGRES_HOST -U postgres -Fc -f /backups/backup_$DATE.dump
              gsutil cp /backups/backup_$DATE.dump $BACKUP_BUCKET/
              # Clean up local backups older than 7 days
              find /backups -type f -mtime +7 -delete
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            emptyDir:
              sizeLimit: 50Gi
          restartPolicy: OnFailure
```

**Characteristics:**
- RTO: Hours to days
- RPO: Based on backup frequency (typically 24 hours)
- Cost: Lowest (storage only)
- Use cases: Non-critical dev/staging, archival

### 1.2 Warm Standby

A replica database is maintained in a different availability zone or region, running in read-only mode and promoted on failure.

```yaml
# PostgreSQL Warm Standby with Patroni
apiVersion: v1
kind: ConfigMap
metadata:
  name: patroni-config
  namespace: database
data:
  patroni.yml: |
    scope: postgres-cluster
    name: "{{ POD_NAME }}"
    
    restapi:
      listen: "0.0.0.0:8008"
      connect_address: "{{ POD_NAME }}.patroni.database.svc.cluster.local:8008"
    
    bootstrap:
      dcs:
        postgresql:
          parameters:
            max_connections: 200
            shared_buffers: 256MB
          pg_hba:
            - host all all 0.0.0.0/0 md5
            - host replication all 0.0.0.0/0 md5
      
      initdb:
        - encoding: UTF8
        - data-checksums
    
    postgresql:
      listen: "0.0.0.0:5432"
      connect_address: "{{ POD_NAME }}:5432"
      data_dir: /data/patroni
      
      parameters:
        wal_level: replica
        max_wal_senders: 10
        wal_keep_size: 16GB
        hot_standby: on
        
      pg_hba:
        - host all all 0.0.0.0/0 md5
        - host replication all 0.0.0.0/0 md5
      
      recovery_conf:
        restore_command: /usr/bin/pgbackrest restore --stanza=main --type=imm %r
        archive_cleanup_command: /usr/bin/pgbackrest archive-cleanup -stanza=main %r
        promote_trigger_file: /tmp/promote
    
    watchdog:
      mode: required
      device: /dev/watchdog
      safety_margin: 5
    
    tags:
      nofailover: "false"
      noloadbalance: "false"
      clonefrom: "false"
```

```python
# Python script for warm standby promotion
import psycopg2
import subprocess
import time
from datetime import datetime

class WarmStandbyFailover:
    def __init__(self, standby_conn_info, primary_conn_info):
        self.standby_conn_info = standby_conn_info
        self.primary_conn_info = primary_conn_info
    
    def check_standby_health(self) -> bool:
        """Verify standby is catched up and ready for promotion."""
        try:
            conn = psycopg2.connect(**self.standby_conn_info)
            cursor = conn.cursor()
            
            # Check if replication lag is acceptable (< 10 seconds)
            cursor.execute("""
                SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
            """)
            lag = cursor.fetchone()[0]
            
            # Check WAL receiver is running
            cursor.execute("""
                SELECT status FROM pg_stat_wal_receiver
            """)
            status = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return lag < 10 and status == 'streaming'
            
        except Exception as e:
            print(f"Standby health check failed: {e}")
            return False
    
    def promote_standby(self) -> bool:
        """Promote standby to primary."""
        try:
            # Create promotion trigger file
            subprocess.run([
                "kubectl", "exec", "-n", "database",
                "patroni-0", "--", "touch", "/tmp/promote"
            ], check=True)
            
            # Wait for promotion to complete
            time.sleep(30)
            
            # Verify promotion
            conn = psycopg2.connect(**self.standby_conn_info)
            cursor = conn.cursor()
            cursor.execute("SELECT pg_is_in_recovery()")
            is_recovery = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return not is_recovery
            
        except Exception as e:
            print(f"Promotion failed: {e}")
            return False
    
    def execute_failover(self, notification_callback=None) -> dict:
        """Execute complete failover procedure."""
        start_time = datetime.utcnow()
        result = {
            "success": False,
            "start_time": start_time,
            "steps": []
        }
        
        # Step 1: Verify standby health
        if not self.check_standby_health():
            result["steps"].append({
                "step": "health_check",
                "status": "failed",
                "message": "Standby not healthy for promotion"
            })
            return result
        
        result["steps"].append({
            "step": "health_check",
            "status": "success"
        })
        
        # Step 2: Notify stakeholders
        if notification_callback:
            notification_callback("FAILOVER_STARTED", {
                "timestamp": start_time.isoformat(),
                "standby": self.standby_conn_info["host"]
            })
        
        # Step 3: Promote standby
        if not self.promote_standby():
            result["steps"].append({
                "step": "promotion",
                "status": "failed"
            })
            return result
        
        result["steps"].append({
            "step": "promotion",
            "status": "success"
        })
        
        # Step 4: Update connection strings (DNS, service discovery)
        # Implementation depends on infrastructure
        
        end_time = datetime.utcnow()
        result["success"] = True
        result["end_time"] = end_time
        result["duration_seconds"] = (end_time - start_time).total_seconds()
        
        return result
```

**Characteristics:**
- RTO: Minutes (typically 2-5 minutes)
- RPO: Seconds to minutes (depends on replication lag)
- Cost: Moderate (additional infrastructure)
- Use cases: Production databases requiring moderate RTO

### 1.3 Hot Standby

Real-time replication with a fully synchronized standby that can handle read queries and quickly take over write operations.

```yaml
# MySQL Group Replication (Hot Standby) configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-hot-standby-config
  namespace: database
data:
  my.cnf: |
    [mysqld]
    # Group Replication settings
    plugin_load_add='group_replication.so'
    group_replication_group_name="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    group_replication_start_on_boot=OFF
    group_replication_local_address="mysql-0.mysql-headless.database.svc.cluster.local:33061"
    group_replication_group_seeds="mysql-0.mysql-headless.database.svc.cluster.local:33061,mysql-1.mysql-headless.database.svc.cluster.local:33061,mysql-2.mysql-headless.database.svc.cluster.local:33061"
    group_replication_bootstrap_group=OFF
    
    # Consistency settings
    group_replication_enforce_update_everywhere_checks=ON
    group_replication_single_primary_mode=OFF
    
    # Performance tuning
    binlog_transaction_dependency_tracking=WRITESET
    transaction_write_set_extraction=XXHASH64
    slave_parallel_type=LOGICAL_CLOCK
    slave_parallel_workers=4
    
    # Replication safety
    relay_log_recovery=ON
    relay_log_purge=ON
    max_allowed_packet=64MB
```

```python
# Application-level hot standby routing
import random
from contextlib import contextmanager
from typing import Optional, List
import psycopg2
from psycopg2 import pool

class HotStandbyRouter:
    def __init__(
        self,
        primary_host: str,
        replica_hosts: List[str],
        connection_pool_size: int = 10
    ):
        self.primary_host = primary_host
        self.replica_hosts = replica_hosts
        self.connection_pool_size = connection_pool_size
        
        # Create connection pools
        self.primary_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=connection_pool_size,
            host=primary_host,
            database="appdb",
            user="appuser",
            password="password"
        )
        
        self.replica_pools = [
            pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=connection_pool_size // 2,
                host=host,
                database="appdb",
                user="appuser",
                password="password"
            )
            for host in replica_hosts
        ]
    
    @contextmanager
    def get_read_connection(self):
        """Route read queries to a healthy replica."""
        # Check replica health and select best one
        healthy_replicas = []
        
        for i, pool in enumerate(self.replica_pools):
            try:
                conn = pool.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                pool.putconn(conn)
                healthy_replicas.append(i)
            except Exception:
                pool.putconn(conn, close=True)
        
        if not healthy_replicas:
            # Fallback to primary if no replicas available
            yield self.primary_pool.getconn()
            return
        
        # Load balance across healthy replicas
        replica_index = random.choice(healthy_replicas)
        conn = self.replica_pools[replica_index].getconn()
        
        try:
            yield conn
        finally:
            self.replica_pools[replica_index].putconn(conn)
    
    @contextmanager
    def get_write_connection(self):
        """Always route writes to primary."""
        conn = self.primary_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.primary_pool.putconn(conn)
    
    def health_check_all(self) -> dict:
        """Check health of all database nodes."""
        health = {
            "primary": {"healthy": False, "replication_lag": None},
            "replicas": []
        }
        
        # Check primary
        try:
            conn = self.primary_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT pg_is_in_recovery()")
            is_recovery = cursor.fetchone()[0]
            health["primary"]["healthy"] = not is_recovery
            cursor.close()
            self.primary_pool.putconn(conn)
        except Exception as e:
            health["primary"]["error"] = str(e)
        
        # Check replicas
        for i, replica_pool in enumerate(self.replica_pools):
            replica_health = {"healthy": False, "lag": None}
            try:
                conn = replica_pool.getconn()
                cursor = conn.cursor()
                
                # Check if replica is catching up
                cursor.execute("""
                    SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
                """)
                lag = cursor.fetchone()[0]
                replica_health["lag"] = lag
                replica_health["healthy"] = lag < 5  # 5 second threshold
                
                cursor.close()
                replica_pool.putconn(conn)
            except Exception as e:
                replica_health["error"] = str(e)
            
            health["replicas"].append(replica_health)
        
        return health
```

**Characteristics:**
- RTO: Seconds to sub-minute
- RPO: Near-zero (synchronous replication)
- Cost: Higher (requires real-time replication infrastructure)
- Use cases: Mission-critical systems requiring minimal downtime

### 1.4 Active-Active (Multi-Master)

Multiple database nodes accept both read and write operations, providing highest availability and geographic distribution.

```yaml
# CockroachDB Active-Active configuration
apiVersion: v1
kind: StatefulSet
metadata:
  name: cockroachdb
  namespace: database
spec:
  serviceName: cockroachdb
  replicas: 3
  template:
    spec:
      containers:
      - name: cockroachdb
        image: cockroachdb/cockroach:v23.1.0
        ports:
        - containerPort: 26257
          name: grpc
        - containerPort: 8080
          name: http
        env:
        - name: COCKROACH_CHANNEL
          value: kubernetes-secure
        command:
        - /bin/bash
        - -ecx
        - |
          # Start cockroach with cluster initialization
          exec /cockroach/cockroach start \
            --advertise_addr=${HOSTNAME}.cockroachdb.database.svc.cluster.local \
            --join=cockroachdb-0.cockroachdb,cockroachdb-1.cockroachdb,cockroachdb-2.cockroachdb \
            --insecure \
            --cache=25% \
            --max-sql-memory=25% \
            --locality=region=$(REGION),zone=$(ZONE)
        volumeMounts:
        - name: cockroach-data
          mountPath: /cockroach/cockroach-data
  volumeClaimTemplates:
  - metadata:
      name: cockroach-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

```python
# Active-Active conflict resolution strategies
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

class ConflictResolutionStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    BUSINESS_LOGIC = "business_logic"
    MERGE = "merge"

@dataclass
class ConflictRecord:
    record_id: str
    key: str
    conflicting_values: list
    timestamp: datetime
    source_region: str

class ConflictResolver:
    """Handle conflicts in active-active database setup."""
    
    def __init__(self, strategy: ConflictResolutionStrategy):
        self.strategy = strategy
    
    def resolve(
        self,
        conflict: ConflictRecord,
        business_logic: Optional[callable] = None
    ) -> Any:
        """Resolve conflict based on configured strategy."""
        
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return max(
                conflict.conflicting_values,
                key=lambda v: v.get("updated_at", datetime.min)
            )
        
        elif self.strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            return min(
                conflict.conflicting_values,
                key=lambda v: v.get("updated_at", datetime.min)
            )
        
        elif self.strategy == ConflictResolutionStrategy.BUSINESS_LOGIC:
            if business_logic:
                return business_logic(conflict)
            raise ValueError("Business logic function required")
        
        elif self.strategy == ConflictResolutionStrategy.MERGE:
            return self._merge_values(conflict.conflicting_values)
        
        raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _merge_values(self, values: list) -> dict:
        """Merge conflicting values intelligently."""
        merged = {}
        
        for value in values:
            for key, val in value.items():
                if key not in merged:
                    merged[key] = val
                elif isinstance(val, dict) and isinstance(merged[key], dict):
                    merged[key] = self._merge_values([merged[key], val])
                elif isinstance(val, list) and isinstance(merged[key], list):
                    merged[key] = list(set(merged[key] + val))
        
        return merged
```

**Characteristics:**
- RTO: Zero (continues operating during failure)
- RPO: Zero (synchronous multi-region)
- Cost: Highest (full multi-region deployment)
- Use cases: Global applications requiring lowest latency and highest availability

---

## 2. Cross-Cloud Database Replication

### 2.1 AWS to GCP PostgreSQL Replication

```python
import boto3
from google.cloud import storage, compute_v1
import psycopg2
from datetime import datetime, timedelta

class CrossCloudReplication:
    """Manage PostgreSQL replication between AWS and GCP."""
    
    def __init__(self, config: dict):
        self.aws_config = config["aws"]
        self.gcp_config = config["gcp"]
        self.replication_slot = config.get("replication_slot", "aws_to_gcp")
    
    def setup_aws_postgres_walshipping(self) -> dict:
        """Configure PostgreSQL for WAL shipping to GCP."""
        
        # Configure PostgreSQL for archiving
        archive_config = {
            "wal_level": "replica",
            "max_wal_senders": 5,
            "wal_keep_size": "1GB",
            "archive_mode": "on",
            "archive_command": (
                "aws s3 cp %p s3://{bucket}/wal/%f"
            ).format(bucket=self.aws_config["s3_bucket"]),
            "archive_timeout": "300"  # 5 minutes
        }
        
        return {
            "config_changes": [
                f"ALTER SYSTEM SET {k} = '{v}'"
                for k, v in archive_config.items()
            ],
            "required_extensions": ["aws_s3", "aws_commons"],
            "iam_permissions": [
                "s3:PutObject",
                "s3:GetObject"
            ]
        }
    
    def create_replication_slot(self, conn) -> str:
        """Create logical replication slot."""
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT pg_create_logical_replication_slot(
                '{self.replication_slot}',
                'pgoutput'
            )
        """)
        slot_name = cursor.fetchone()[0]
        cursor.close()
        return slot_name
    
    def setup_gcp_receiver(self) -> dict:
        """Configure GCP PostgreSQL as replica."""
        
        return {
            "primary_conninfo": (
                f"host={self.aws_config['primary_host']} "
                f"port=5432 "
                f"user={self.aws_config['replication_user']} "
                f"password={self.aws_config['replication_password']} "
                f"sslmode=require"
            ),
            "recovery_config": {
                "restore_command": (
                    "aws s3 cp s3://{bucket}/wal/%f %p"
                ).format(bucket=self.aws_config["s3_bucket"]),
                "primary_slot_name": self.replication_slot,
                "recovery_target_timeline": "latest"
            }
        }

# Configuration for cross-cloud streaming replication
CROSS_CLOUD_CONFIG = {
    "aws": {
        "primary_host": "postgres.aws.production.internal",
        "region": "us-east-1",
        "s3_bucket": "prod-postgres-wal-archive",
        "replication_user": "replication",
        "kms_key_arn": "arn:aws:kms:us-east-1:123456789012:key/xxx"
    },
    "gcp": {
        "replica_host": "postgres-gcp.production.internal",
        "region": "us-central1",
        "storage_bucket": "gcp-postgres-wal"
    },
    "replication": {
        "slot_name": "aws_to_gcp",
        "max_wal_senders": 3,
        "wal_keep_segments": 100
    }
}
```

### 2.2 Azure to AWS MySQL Replication

```python
from azure.storage.blob import BlockBlobService
import mysql.connector
from replication_utils import GTIDBasedReplication

class AzureToAWSMySQLReplication:
    """Cross-cloud MySQL replication using GTID."""
    
    def __init__(self, config: dict):
        self.azure_config = config["azure"]
        self.aws_config = config["aws"]
    
    def setup_gtid_replication(self) -> dict:
        """Configure GTID-based replication."""
        
        # Azure MySQL configuration
        azure_mysql_config = """
            # Enable GTID
            gtid_mode = ON
            enforce_gtid_consistency = ON
            
            # Binary logging
            log_bin = mysql-bin
            binlog_format = ROW
            binlog_row_image = FULL
            expire_logs_days = 7
            
            # Replication
            log_slave_updates = ON
            read_only = OFF  # Primary is writable
        """
        
        # AWS RDS MySQL configuration  
        aws_rds_config = """
            # Enable GTID
            gtid_mode = ON
            enforce_gtid_consistency = ON
            
            # Binary logging (RDS managed)
            # Note: RDS handles binlog retention automatically
            
            # Replication from external source
            server_id = 2
            relay_log = mysql-relay-bin
            log_slave_updates = ON
            read_only = ON
        """
        
        return {
            "azure_primary": {
                "connection": self.azure_config["endpoint"],
                "required_settings": azure_mysql_config,
                "dump_command": (
                    "mysqldump -h {host} -u {user} -p "
                    "--single-transaction "
                    "--set-gtid-purged=ON "
                    "--master-data=2 "
                    "--all-databases > backup.sql"
                ).format(
                    host=self.azure_config["host"],
                    user=self.azure_config["admin_user"]
                )
            },
            "aws_replica": {
                "connection": self.aws_config["endpoint"],
                "required_settings": aws_rds_config,
                "replication_command": """
                    CHANGE MASTER TO
                        MASTER_HOST = '{azure_host}',
                        MASTER_USER = '{repl_user}',
                        MASTER_PASSWORD = '{repl_password}',
                        MASTER_AUTO_POSITION = 1;
                    
                    START SLAVE;
                """.format(
                    azure_host=self.azure_config["private_ip"],
                    repl_user=self.azure_config["replication_user"],
                    repl_password=self.azure_config["replication_password"]
                )
            }
        }
```

---

## 3. Backup Verification and Testing Procedures

### 3.1 Automated Backup Verification

```python
import hashlib
import psycopg2
import boto3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class BackupVerificationResult:
    backup_id: str
    timestamp: datetime
    checks_passed: List[str]
    checks_failed: List[str]
    data_integrity_hash: str
    row_count: int
    table_count: int
    
    def is_successful(self) -> bool:
        return len(self.checks_failed) == 0
    
    def to_dict(self) -> dict:
        return {
            "backup_id": self.backup_id,
            "timestamp": self.timestamp.isoformat(),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "data_integrity_hash": self.data_integrity_hash,
            "row_count": self.row_count,
            "table_count": self.table_count,
            "successful": self.is_successful()
        }

class BackupVerificationSystem:
    """Automated backup verification and testing."""
    
    def __init__(self, db_config: dict, backup_config: dict):
        self.db_config = db_config
        self.backup_config = backup_config
        self.s3_client = boto3.client('s3')
    
    def verify_backup(self, backup_id: str) -> BackupVerificationResult:
        """Run comprehensive backup verification."""
        
        checks_passed = []
        checks_failed = []
        
        # Check 1: Backup file exists and is not empty
        try:
            backup_file = self._get_backup_file(backup_id)
            if backup_file and backup_file["Size"] > 0:
                checks_passed.append("backup_file_exists")
            else:
                checks_failed.append("backup_file_missing_or_empty")
        except Exception as e:
            checks_failed.append(f"backup_file_check: {str(e)}")
        
        # Check 2: Restore to temporary database
        temp_db_name = f"restore_test_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        try:
            self._restore_to_temp_database(backup_id, temp_db_name)
            checks_passed.append("restore_successful")
        except Exception as e:
            checks_failed.append(f"restore_failed: {str(e)}")
            return BackupVerificationResult(
                backup_id=backup_id,
                timestamp=datetime.utcnow(),
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                data_integrity_hash="",
                row_count=0,
                table_count=0
            )
        
        # Check 3: Data integrity validation
        try:
            integrity_hash = self._validate_data_integrity(temp_db_name)
            checks_passed.append("data_integrity_valid")
        except Exception as e:
            integrity_hash = ""
            checks_failed.append(f"data_integrity_check: {str(e)}")
        
        # Check 4: Row count verification
        try:
            row_count, table_count = self._verify_row_counts(temp_db_name)
            checks_passed.append("row_counts_verified")
        except Exception as e:
            row_count = 0
            table_count = 0
            checks_failed.append(f"row_count_check: {str(e)}")
        
        # Check 5: Schema validation
        try:
            self._validate_schema(temp_db_name)
            checks_passed.append("schema_valid")
        except Exception as e:
            checks_failed.append(f"schema_validation: {str(e)}")
        
        # Cleanup temporary database
        try:
            self._drop_temp_database(temp_db_name)
            checks_passed.append("cleanup_successful")
        except Exception as e:
            checks_failed.append(f"cleanup_failed: {str(e)}")
        
        return BackupVerificationResult(
            backup_id=backup_id,
            timestamp=datetime.utcnow(),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            data_integrity_hash=integrity_hash,
            row_count=row_count,
            table_count=table_count
        )
    
    def _validate_data_integrity(self, db_name: str) -> str:
        """Calculate data integrity hash across all tables."""
        
        conn = psycopg2.connect(
            host=self.db_config["host"],
            database=db_name,
            user=self.db_config["user"],
            password=self.db_config["password"]
        )
        
        hash_builder = hashlib.sha256()
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            # Get row counts per table
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            
            # Include in hash
            hash_builder.update(f"{table_name}:{count}".encode())
        
        cursor.close()
        conn.close()
        
        return hash_builder.hexdigest()
    
    def schedule_daily_verification(self):
        """Schedule daily backup verification."""
        
        verification_script = """
        #!/bin/bash
        # Daily backup verification cron job
        
        DATE=$(date -d 'yesterday' +%Y%m%d)
        BACKUP_ID="backup_${DATE}_02"
        
        python3 /opt/scripts/verify_backup.py \\
            --backup-id ${BACKUP_ID} \\
            --output /var/log/backup_verification/${DATE}.json
        
        # Send results to monitoring
        python3 /opt/scripts/send_verification_metrics.py \\
            --file /var/log/backup_verification/${DATE}.json
        """
        
        return {
            "schedule": "0 6 * * *",  # 6 AM daily
            "script": verification_script,
            "notification_on_failure": True
        }
```

### 3.2 Backup Testing Playbook

```yaml
# Kubernetes job for backup testing
apiVersion: batch/v1
kind: Job
metadata:
  name: backup-test-weekly
  namespace: database
  labels:
    type: backup-verification
spec:
  ttlSecondsAfterFinished: 604800  # Keep for 1 week
  template:
    spec:
      serviceAccountName: database-backup-verifier
      containers:
      - name: backup-tester
        image: database-backup-tester:v1.0.0
        env:
        - name: BACKUP_BUCKET
          value: "s3://prod-database-backups"
        - name: TEST_DB_HOST
          value: "postgres-test.database.svc.cluster.local"
        - name: SLACK_WEBHOOK
          valueFrom:
            secretKeyRef:
              name: notification-secrets
              key: slack-webhook
        command:
        - /bin/sh
        - -c
        - |
          # Run backup verification
          /app/verify-backup.sh
          
          # Run restore test
          /app/test-restore.sh
          
          # Run data integrity checks
          /app/check-integrity.sh
          
          # Report results
          /app/report-results.sh
        volumeMounts:
        - name: test-results
          mountPath: /results
      volumes:
      - name: test-results
        emptyDir: {}
      restartPolicy: Never
```

---

## 4. RTO/RPO Optimization Strategies

### 4.1 Achieving Sub-Minute RTO

```python
from dataclasses import dataclass
from typing import Optional, List
import time
import threading

@dataclass
class RTOConfiguration:
    target_rto_seconds: int
    target_rpo_seconds: int
    current_rto_seconds: Optional[int] = None
    current_rpo_seconds: Optional[int] = None

class RTOOptimizer:
    """Optimize RTO/RPO metrics."""
    
    def __init__(self, config: RTOConfiguration):
        self.config = config
        self.optimization_strategies = []
    
    def analyze_current_state(self) -> dict:
        """Analyze current RTO/RPO performance."""
        
        return {
            "current_rto": {
                "value": self.config.current_rto_seconds,
                "breakdown": {
                    "detection_time": 30,  # Time to detect failure
                    "alert_escalation": 30,  # Time for human response
                    "failover_execution": 60,  # Actual failover time
                    "application_reconnect": 30  # App reconnection
                }
            },
            "current_rpo": {
                "value": self.config.current_rpo_seconds,
                "breakdown": {
                    "backup_frequency": 3600,  # Current backup every hour
                    "replication_lag": 5,  # Average replication lag
                    "backup_transfer_time": 120  # Time to copy backup
                }
            },
            "gap_analysis": {
                "rto_gap": self.config.target_rto_seconds - (self.config.current_rto_seconds or 999),
                "rpo_gap": self.config.target_rpo_seconds - (self.config.current_rpo_seconds or 999)
            }
        }
    
    def recommend_optimizations(self) -> List[dict]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Detection optimization
        recommendations.append({
            "area": "Failure Detection",
            "current": "30 seconds (health checks every 10s)",
            "target": "Sub-second",
            "strategy": "Implement distributed health checking with heartbeat",
            "implementation": """
                # Use Prometheus blackbox exporter for external monitoring
                # Implement database-specific health checks
                # Add synthetic transactions for monitoring
            """
        })
        
        # Failover automation
        recommendations.append({
            "area": "Failover Automation",
            "current": "Manual or 60 seconds",
            "target": "Automatic with <5 seconds",
            "strategy": "Implement automatic failover with leader election",
            "implementation": """
                # Use Patroni for PostgreSQL automatic failover
                # Configure proper watchdog settings
                # Implement application-side connection retry logic
            """
        })
        
        # Application reconnection
        recommendations.append({
            "area": "Application Reconnection",
            "current": "30 seconds (default connection timeout)",
            "target": "Immediate",
            "strategy": "Implement aggressive retry with connection pooling",
            "implementation": """
                # Configure connection pool with fast failover
                # Implement connection string rotation
                # Add DNS-based service discovery
            """
        })
        
        return recommendations

# Configuration for sub-minute RTO
SUB_MINUTE_RTO_CONFIG = {
    "detection": {
        "method": "multi-layered",
        "components": [
            "database_internal_health",
            "load_balancer_health_checks", 
            "external_monitoring",
            "application_heartbeats"
        ],
        "interval": "1 second"
    },
    "failover": {
        "type": "automatic",
        "mechanism": "leader_election",
        "expected_time": "3-5 seconds",
        "verification": "synthetic_transaction"
    },
    "replication": {
        "type": "synchronous",
        "target_lag": "0 seconds",
        "quorum": "majority"
    },
    "connection": {
        "pool_size": 20,
        "min_connections": 5,
        "connection_timeout": "2 seconds",
        "idle_timeout": "60 seconds",
        "retry_policy": {
            "max_attempts": 10,
            "backoff_multiplier": 1.5,
            "initial_delay": "100ms",
            "max_delay": "2 seconds"
        }
    }
}
```

### 4.2 RPO Optimization Through Continuous Backup

```yaml
# PostgreSQL continuous backup with pgbackrest
apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbackrest-config
  namespace: database
data:
  pgbackrest.conf: |
    [global]
    repo1-path=/var/lib/pgbackrest
    repo1-type=s3
    repo1-s3-bucket=${BACKUP_BUCKET}
    repo1-s3-endpoint=s3.amazonaws.com
    repo1-s3-region=us-east-1
    repo1-s3-key=${AWS_ACCESS_KEY_ID}
    repo1-s3-key-secret=${AWS_SECRET_ACCESS_KEY}
    repo1-s3-verify-tls=n
    log-level-console=info
    log-level-file=debug
    process-max=4
    delta=y
    
    [main]
    pg1-path=/var/lib/postgresql/data
    pg1-port=5432
    pg1-user=postgres
    
    [repo_backup]
    prune=true
    retention-archive=2
    retention-diff=3
    retention-full=7
    
    [archive]
    archive-async=y
    archive-check=y
    archive-copy=y
    spool-path=/var/spool/pgbackrest
```

```python
# Calculate RPO based on WAL generation and shipping
class RPOCalculator:
    """Calculate actual RPO based on backup configuration."""
    
    def __init__(self, wal_stats: dict):
        self.wal_stats = wal_stats
    
    def calculate_current_rpo(self) -> dict:
        """Calculate current RPO based on WAL shipping configuration."""
        
        # Factors affecting RPO
        wal_segment_size = 16  # MB, typical for PostgreSQL
        wal_generation_rate = self.wal_stats["mb_per_second"]
        
        # Archive timeout (how often WAL is shipped)
        archive_timeout = 60  # seconds
        
        # Network latency for WAL shipping
        network_latency = self.wal_stats.get("network_latency_ms", 100) / 1000
        
        # Calculate components
        max_data_loss_during_normal = (
            wal_generation_rate * archive_timeout
        ) / (1024 * 1024)  # Convert to MB
        
        # If using synchronous replication, RPO ~= network latency
        # If using asynchronous, RPO depends on lag
        replication_lag = self.wal_stats.get("replication_lag_seconds", 0)
        
        total_rpo_seconds = archive_timeout + replication_lag + network_latency
        
        return {
            "rpo_seconds": total_rpo_seconds,
            "breakdown": {
                "archive_timeout": archive_timeout,
                "replication_lag": replication_lag,
                "network_latency": network_latency
            },
            "recommendations": [
                "Reduce archive_timeout to <30 seconds" if archive_timeout > 30 else None,
                "Consider synchronous replication" if replication_lag > 1 else None,
                "Optimize network path" if network_latency > 0.1 else None
            ]
        }
```

---

## 5. Chaos Engineering for Databases

### 5.1 Database Chaos Experiments

```python
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List
import subprocess
import json

@dataclass
class ChaosExperiment:
    name: str
    description: str
    impact_level: "low" | "medium" | "high" | "critical"
    duration_seconds: int
    affected_systems: List[str]

class DatabaseChaosEngine:
    """Run chaos engineering experiments on databases."""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.experiments = []
        self.metrics_collector = []
    
    def register_experiment(self, experiment: ChaosExperiment):
        """Register a new chaos experiment."""
        self.experiments.append(experiment)
    
    def run_experiment(
        self,
        experiment: ChaosExperiment,
        rollback_callback: Callable
    ) -> dict:
        """Execute a chaos experiment with rollback capability."""
        
        result = {
            "experiment": experiment.name,
            "start_time": time.time(),
            "success": False,
            "metrics": {}
        }
        
        try:
            # Pre-experiment health check
            pre_health = self._check_database_health()
            result["pre_health"] = pre_health
            
            # Execute chaos action
            self._execute_chaos_action(experiment)
            
            # Wait for experiment duration
            time.sleep(experiment.duration_seconds)
            
            # Measure impact
            result["metrics"] = self._collect_metrics(experiment)
            
            # Execute rollback
            rollback_callback()
            
            # Verify recovery
            post_health = self._check_database_health()
            result["post_health"] = post_health
            
            result["success"] = post_health["healthy"]
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            # Always attempt rollback
            rollback_callback()
        
        result["end_time"] = time.time()
        result["duration"] = result["end_time"] - result["start_time"]
        
        return result
    
    def _execute_chaos_action(self, experiment: ChaosExperiment):
        """Execute the chaos action."""
        
        if experiment.name == "network_partition":
            self._simulate_network_partition(
                experiment.affected_systems,
                experiment.duration_seconds
            )
        elif experiment.name == "database_crash":
            self._simulate_database_crash(
                experiment.affected_systems[0]
            )
        elif experiment.name == "replication_lag":
            self._introduce_replication_lag(
                experiment.affected_systems,
                experiment.duration_seconds
            )
        elif experiment.name == "disk_saturation":
            self._simulate_disk_saturation(
                experiment.affected_systems[0],
                experiment.duration_seconds
            )
        elif experiment.name == "connection_exhaustion":
            self._exhaust_connections(
                experiment.affected_systems[0],
                experiment.duration_seconds
            )
    
    def _simulate_network_partition(self, systems: List[str], duration: int):
        """Simulate network partition using firewall rules."""
        
        for system in systems:
            # Block all traffic to/from the database
            subprocess.run([
                "iptables", "-A", "INPUT", "-s", system, "-j", "DROP"
            ], check=False)
            
            subprocess.run([
                "iptables", "-A", "OUTPUT", "-d", system, "-j", "DROP"
            ], check=False)
        
        # Schedule cleanup
        time.sleep(duration)
        
        for system in systems:
            subprocess.run([
                "iptables", "-D", "INPUT", "-s", system, "-j", "DROP"
            ], check=False)
            
            subprocess.run([
                "iptables", "-D", "OUTPUT", "-d", system, "-j", "DROP"
            ], check=False)
    
    def _simulate_database_crash(self, db_host: str):
        """Simulate database crash by killing the process."""
        
        subprocess.run([
            "kubectl", "exec", "-n", "database",
            "postgres-0", "--", "kill", "-9", "1"
        ], check=True)
    
    def _introduce_replication_lag(self, replicas: List[str], duration: int):
        """Introduce artificial replication lag."""
        
        for replica in replicas:
            # Use tc (traffic control) to add delay
            subprocess.run([
                "kubectl", "exec", "-n", "database", replica, "--",
                "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
                "delay", "30000ms"
            ], check=True)
        
        time.sleep(duration)
        
        for replica in replicas:
            subprocess.run([
                "kubectl", "exec", "-n", "database", replica, "--",
                "tc", "qdisc", "del", "dev", "eth0", "root"
            ], check=True)
    
    def _check_database_health(self) -> dict:
        """Check database health metrics."""
        
        import psycopg2
        
        try:
            conn = psycopg2.connect(
                host=self.db_config["host"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
            
            cursor = conn.cursor()
            
            # Check basic connectivity
            cursor.execute("SELECT 1")
            
            # Check replication status
            cursor.execute("""
                SELECT status, 
                       EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
                FROM pg_stat_wal_receiver
            """)
            repl_status = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                "healthy": True,
                "replication_status": repl_status[0] if repl_status else "N/A",
                "replication_lag": repl_status[1] if repl_status else None
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
```

### 5.2 Chaos Experiment Library

```python
# Predefined chaos experiments for databases
CHAOS_EXPERIMENTS = {
    "minor": [
        ChaosExperiment(
            name="single_query_timeout",
            description="Simulate a single query timeout",
            impact_level="low",
            duration_seconds=30,
            affected_systems=["app-service"]
        ),
        ChaosExperiment(
            name="temporary_connection_failure",
            description="Brief connection failures",
            impact_level="low",
            duration_seconds=10,
            affected_systems=["app-service"]
        )
    ],
    "medium": [
        ChaosExperiment(
            name="replica_failure",
            description="Fail one read replica",
            impact_level="medium",
            duration_seconds=300,
            affected_systems=["read-replica-1"]
        ),
        ChaosExperiment(
            name="replication_lag_spike",
            description="Introduce 30-second replication lag",
            impact_level="medium",
            duration_seconds=120,
            affected_systems=["postgres-replica"]
        ),
        ChaosExperiment(
            name="disk_io_degradation",
            description="Degrade disk I/O performance",
            impact_level="medium",
            duration_seconds=180,
            affected_systems=["postgres-primary"]
        )
    ],
    "major": [
        ChaosExperiment(
            name="primary_failure",
            description="Simulate primary database failure",
            impact_level="high",
            duration_seconds=600,
            affected_systems=["postgres-primary"]
        ),
        ChaosExperiment(
            name="network_partition",
            description="Complete network partition",
            impact_level="high",
            duration_seconds=60,
            affected_systems=["postgres-primary", "postgres-replica"]
        ),
        ChaosExperiment(
            name="data_corruption",
            description="Simulate partial data corruption",
            impact_level="critical",
            duration_seconds=300,
            affected_systems=["postgres-primary"]
        )
    ]
}

# Runbook for chaos experiment execution
CHAOS_RUNBOOK = """
# Chaos Experiment Execution Runbook

## Pre-requisites
1. All experiments must be approved by the team lead
2. Schedule experiments during maintenance windows for major tests
3. Ensure on-call SRE is aware and monitoring

## Execution Steps

### 1. Minor Experiments (can run anytime)
```bash
python chaos_engine.py run --experiment single_query_timeout
```

### 2. Medium Experiments (notify team, schedule)
```bash
# Notify on Slack #incidents
python chaos_engine.py run --experiment replica_failure --schedule "2024-01-15 14:00"
```

### 3. Major Experiments (full procedures)
```bash
# Create incident channel
# Notify all stakeholders
# Execute with automated rollback
python chaos_engine.py run --experiment primary_failure \
    --auto-rollback --monitor
```

## Rollback Procedures
- All chaos actions have automatic rollback after experiment duration
- Manual intervention available via:
  kubectl exec -it chaos-controller -n database-system -- /app/rollback
  
## Post-Experiment
1. Document findings in chaos-log repository
2. Update monitoring/alerting based on observations
3. Share learnings in team review
"""
```

---

## 6. Runbook Templates for Common Failure Scenarios

### 6.1 Primary Database Failure Runbook

```yaml
# Runbook: Primary Database Failure
apiVersion: v1
kind: ConfigMap
metadata:
  name: runbook-primary-failure
  namespace: database
data:
  runbook.md: |
    # Primary Database Failure Runbook
    
    ## Severity: SEV-1
    
    ## Detection
    - Alert: `DatabasePrimaryDown` triggered
    - Symptoms: Application cannot connect to primary database
    
    ## Immediate Actions (First 2 Minutes)
    
    ### 1. Verify the Alert (30 seconds)
    ```bash
    # Check database status
    kubectl exec -it postgres-0 -n database -- patroni ctl list
    
    # Check recent events
    kubectl logs -n database postgres-0 --tail=50 | grep -i error
    ```
    
    ### 2. Identify Failure Type (1 minute)
    - Is the pod crashlooping?
    - Is the node down?
    - Is etcd failing?
    
    ### 3. Attempt Automatic Failover (30 seconds)
    ```bash
    # Patroni should handle this automatically
    # Verify failover status
    kubectl exec -it postgres-0 -n database -- patroni ctl list
    ```
    
    ## Manual Intervention (If Auto-Failover Fails)
    
    ### Force Failover (if needed)
    ```bash
    # Only if automatic failover didn't trigger
    kubectl exec -it postgres-0 -n database -- \
        patroni ctl switchover --force --andidate postgres-1
    ```
    
    ### Check Application Connectivity
    ```bash
    # Test application database connections
    kubectl exec -it app-deployment-xxx -n production -- \
        pg_isready -h postgres.database.svc.cluster.local
    ```
    
    ## Recovery Verification
    
    - [ ] New primary elected and healthy
    - [ ] All replicas synchronized
    - [ ] Application connections restored
    - [ ] No data loss (verify last transaction)
    - [ ] Monitoring shows green status
    
    ## Post-Incident Actions
    
    1. Document incident timeline
    2. Analyze root cause
    3. Update runbook if improvements found
    4. Schedule follow-up review meeting
```

### 6.2 Replication Lag Runbook

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: runbook-replication-lag
  namespace: database
data:
  runbook.md: |
    # Replication Lag Runbook
    
    ## Severity: SEV-2 (if > 60s), SEV-1 (if > 300s)
    
    ## Detection
    - Alert: `DatabaseReplicationLagHigh`
    - Threshold: Warning at 30s, Critical at 60s
    
    ## Diagnostic Steps
    
    ### 1. Check Current Lag
    ```bash
    # On primary
    SELECT slot_name, pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) as lag_bytes
    FROM pg_replication_slots;
    
    # On replica
    SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds;
    ```
    
    ### 2. Identify Cause
    
    #### Check WAL Generation Rate
    ```bash
    # Is the primary producing WAL faster than replicas can apply?
    SELECT pg_current_wal_lsn() - pg_last_xact_replay_timestamp();
    ```
    
    #### Check I/O on Replica
    ```bash
    # Check disk I/O on replica
    iostat -x 5
    ```
    
    #### Check Network Throughput
    ```bash
    # Monitor network between primary and replica
    ifstat -i eth0 5
    ```
    
    ## Resolution Steps
    
    ### If Caused by Heavy Write Load
    - Wait for load to subside
    - Consider temporary write throttling
    
    ### If Caused by Slow Replica I/O
    - Check replica disk health
    - Consider upgrading replica storage
    
    ### If Caused by Network Issues
    - Check network connectivity
    - Review network metrics
    
    ### Emergency: Catch-up Clone
    ```bash
    # If lag is excessive and cannot be recovered
    # WARNING: This will require application reconnection
    
    kubectl exec -it postgres-replica-0 -n database -- \
        pg_ctl promote -D /var/lib/postgresql/data
    ```
    
    ## Verification
    - [ ] Replication lag < 10 seconds
    - [ ] No active alerts
    - [ ] Write throughput normal
```

### 6.3 Database Connection Pool Exhaustion Runbook

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: runbook-connection-exhaustion
  namespace: database
data:
  runbook.md: |
    # Connection Pool Exhaustion Runbook
    
    ## Severity: SEV-1
    
    ## Symptoms
    - Applications reporting "too many connections"
    - Database CPU low but queries queuing
    - Connection count at max
    
    ## Diagnostic Steps
    
    ### 1. Check Current Connections
    ```sql
    -- Active connections by state
    SELECT state, count(*) 
    FROM pg_stat_activity 
    WHERE datname = 'appdb'
    GROUP BY state;
    
    -- Top connection holders
    SELECT usename, application_name, count(*)
    FROM pg_stat_activity
    WHERE datname = 'appdb'
    GROUP BY usename, application_name
    ORDER BY count(*) DESC
    LIMIT 10;
    
    -- Long running queries
    SELECT pid, now() - pg_stat_activity.query_start AS duration, query
    FROM pg_stat_activity
    WHERE state = 'active' AND query NOT ILIKE '%pg_stat_activity%'
    ORDER BY duration DESC
    LIMIT 5;
    ```
    
    ### 2. Check Connection Pool Status
    ```bash
    # If using PgBouncer
    kubectl exec -it pgbouncer-0 -n database -- \
        pgb_admin -c "SHOW DATABASES"
    
    kubectl exec -it pgbouncer-0 -n database -- \
        pgb_admin -c "SHOW CLIENTS"
    
    kubectl exec -it pgbouncer-0 -n database -- \
        pgb_admin -c "SHOW POOLS"
    ```
    
    ## Resolution
    
    ### Immediate Actions
    
    1. **Kill Long-Running Queries** (if malicious/rogue)
    ```sql
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE state = 'active' 
    AND query_start < now() - interval '5 minutes'
    AND query NOT ILIKE '%pg_stat_activity%';
    ```
    
    2. **Restart Application** (if connections stuck)
    ```bash
    kubectl rollout restart deployment/app -n production
    ```
    
    3. **Temporarily Increase Max Connections** (if safe)
    ```sql
    ALTER SYSTEM SET max_connections = 500;
    SELECT pg_reload_conf();
    ```
    
    ### Root Cause Investigation
    
    - Application connection leak?
    - Connection pool misconfigured?
    - Sudden traffic spike?
    - Long-running transaction?
    
    ## Prevention
    
    - Implement connection pool with proper limits
    - Add connection timeout
    - Implement query timeout
    - Monitor connection pool metrics
```

---

## Summary

This document covers the critical aspects of database disaster recovery:

1. **DR Architecture Patterns**: From simple backup/restore to complex active-active configurations, each with specific RTO/RPO characteristics
2. **Cross-Cloud Replication**: Techniques for maintaining database redundancy across cloud providers
3. **Backup Verification**: Automated testing procedures to ensure backups are valid and restorable
4. **RTO/RPO Optimization**: Strategies to achieve sub-minute recovery objectives
5. **Chaos Engineering**: Proactive testing of failure scenarios
6. **Runbooks**: Step-by-step procedures for common database failure scenarios

All configurations follow production-ready patterns with Kubernetes manifests, Python code for automation, and comprehensive monitoring integration.

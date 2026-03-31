# Backup and Recovery

Reliable backup and recovery strategies are essential for production database systems, especially for AI/ML applications where data loss can impact model training, inference, and business operations.

## Overview

Backup and recovery ensure data durability and availability in the face of hardware failures, software bugs, human errors, and disasters. For senior AI/ML engineers, understanding these strategies is critical for building resilient data platforms.

## Backup Types

### Full Backups
- **Definition**: Complete copy of the entire database
- **Frequency**: Daily or weekly
- **Advantages**: Simple restore process, complete data set
- **Disadvantages**: Large size, longer backup time

### Incremental Backups
- **Definition**: Backup of changes since last full backup
- **Frequency**: Hourly or daily
- **Advantages**: Smaller size, faster backup time
- **Disadvantages**: Complex restore (need full + all incrementals)

### Differential Backups
- **Definition**: Backup of changes since last full backup (not last incremental)
- **Frequency**: Daily
- **Advantages**: Faster restore than incremental (only full + latest differential)
- **Disadvantages**: Larger than incremental backups

### Logical Backups
- **Definition**: SQL dump format (pg_dump, mysqldump)
- **Advantages**: Human-readable, portable across versions
- **Disadvantages**: Slower restore, larger size

### Physical Backups
- **Definition**: Raw file copies (WAL logs, data files)
- **Advantages**: Fast restore, point-in-time recovery
- **Disadvantages**: Database-specific, less portable

## Implementation Examples

### PostgreSQL Backup Strategies
```bash
# Full logical backup
pg_dump -U username -h localhost dbname > backup.sql

# Compressed backup
pg_dump -Fc -U username -h localhost dbname > backup.dump

# WAL-based physical backup
pg_basebackup -D /path/to/backup -Ft -z -P -R -X stream -c fast

# Point-in-time recovery setup
# 1. Enable WAL archiving
ALTER SYSTEM SET archive_mode = 'on';
ALTER SYSTEM SET archive_command = 'cp %p /path/to/wal_archive/%f';

# 2. Take base backup
pg_basebackup -D /path/to/backup -Ft -z -P -R

# 3. Restore with PITR
# In recovery.conf:
restore_command = 'cp /path/to/wal_archive/%f %p'
recovery_target_time = '2024-01-15 10:30:00'
```

### MySQL Backup Strategies
```bash
# Logical backup
mysqldump -u username -p database_name > backup.sql

# Physical backup with Percona XtraBackup
xtrabackup --backup --target-dir=/path/to/backup --user=username --password=password

# Binary log backup for PITR
mysqlbinlog --start-datetime="2024-01-15 09:00:00" \
            --stop-datetime="2024-01-15 10:30:00" \
            /var/log/mysql/binlog.000001 > binlog_replay.sql
```

## Recovery Strategies

### Point-in-Time Recovery (PITR)
Restore database to any point in time using transaction logs.

```sql
-- PostgreSQL PITR example
-- recovery.conf configuration
standby_mode = 'off'
recovery_target_time = '2024-01-15 10:30:00'
restore_command = 'cp /wal_archive/%f %p'
```

### Hot Standby Recovery
Maintain a standby replica that can take over immediately.

```sql
-- PostgreSQL streaming replication for hot standby
-- Primary: wal_level = 'replica', max_wal_senders = 10
-- Standby: primary_conninfo = 'host=primary port=5432 user=replicator'

-- Promote standby to primary
pg_ctl promote -D /path/to/standby
```

### Multi-Region Disaster Recovery
Geographically distributed backups for maximum resilience.

```
Primary Region → Async Replication → Secondary Region
                     ↓
               Backup Storage (S3/GCS)
                     ↓
               Tertiary Region (Cold standby)
```

## AI/ML Specific Considerations

### Model Registry Backup
- **Critical data**: Model metadata, version history, metrics
- **Recovery priority**: High (impacts ML operations)
- **Strategy**: Frequent logical backups + WAL archiving

```sql
-- Model registry backup strategy
CREATE TABLE model_backups (
    backup_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    version_number INT NOT NULL,
    backup_time TIMESTAMPTZ DEFAULT NOW(),
    backup_type VARCHAR(20) NOT NULL,  -- 'full', 'incremental'
    status VARCHAR(20) NOT NULL,       -- 'completed', 'failed'
    size_bytes BIGINT
);

-- Automated backup job
CREATE OR REPLACE FUNCTION backup_model_registry()
RETURNS VOID AS $$
DECLARE
    backup_file TEXT;
BEGIN
    backup_file := '/backups/model_registry_' || NOW()::TEXT || '.dump';
    
    -- Execute pg_dump command
    EXECUTE format('pg_dump -Fc -U %s -h %s %s > %s',
        current_setting('role'),
        current_setting('server_hostname'),
        current_database(),
        backup_file);
    
    -- Log backup completion
    INSERT INTO model_backups (model_id, version_number, backup_type, status, size_bytes)
    VALUES (NULL, 0, 'full', 'completed', pg_total_relation_size('models'));
END;
$$ LANGUAGE plpgsql;
```

### Feature Store Backup
- **Data characteristics**: Time-series, high volume
- **Recovery requirements**: Point-in-time for specific time windows
- **Strategy**: Partitioned backups + WAL archiving

### Training Data Backup
- **Volume**: Very large datasets
- **Recovery needs**: Complete dataset integrity
- **Strategy**: Incremental backups + checksum verification

## Backup Verification and Testing

### Regular Restore Testing
- **Frequency**: Monthly for critical systems
- **Process**: Restore to test environment, verify data integrity
- **Automation**: Scripted restore validation

```bash
#!/bin/bash
# Backup verification script
echo "Testing backup restore..."
pg_restore -d test_db backup.dump
psql -d test_db -c "SELECT COUNT(*) FROM users;" | grep -v "COUNT"
echo "Backup restore successful!"
```

### Data Integrity Checks
- **Checksums**: Verify backup integrity
- **Consistency checks**: Validate referential integrity
- **Application validation**: Test critical queries

```sql
-- Data consistency verification
SELECT 
    'users' as table_name,
    COUNT(*) as row_count,
    SUM(CHECKSUM(*)) as checksum
FROM users;

-- Referential integrity check
SELECT COUNT(*) 
FROM orders o
LEFT JOIN users u ON o.user_id = u.id
WHERE u.id IS NULL;
```

## Best Practices

1. **3-2-1 Rule**: 3 copies, 2 different media, 1 offsite
2. **Test restores regularly**: Backup is only useful if restorable
3. **Document recovery procedures**: Step-by-step runbooks
4. **Monitor backup success**: Alert on failures
5. **Consider RPO/RTO**: Recovery Point Objective and Recovery Time Objective
6. **Automate everything**: Reduce human error

## Related Resources

- [Disaster Recovery] - Comprehensive DR planning
- [High Availability] - HA strategies complementing backup
- [Database Security] - Secure backup storage
- [Operational Excellence] - Production database operations
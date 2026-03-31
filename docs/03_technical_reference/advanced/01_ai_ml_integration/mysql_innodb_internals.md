# MySQL InnoDB Internals Deep Dive

InnoDB is MySQL's default storage engine and one of the most widely used transactional storage engines in the world. Developed by Innobase (now part of Oracle), InnoDB provides ACID-compliant transaction support, row-level locking, and multi-version concurrency control. This document provides an in-depth exploration of InnoDB's internal architecture, covering storage management, concurrency control, and recovery mechanisms.

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Redo Logging and Log Groups](#2-redo-logging-and-log-groups)
3. [Undo Tablespaces and Purge](#3-undo-tablespaces-and-purge)
4. [Lock Management System](#4-lock-management-system)
5. [Adaptive Hash Indexing](#5-adaptive-hash-indexing)
6. [Buffer Pool Management](#6-buffer-pool-management)
7. [Checkpoint Management](#7-checkpoint-management)
8. [Index Structures](#8-index-structures)
9. [Configuration Best Practices](#9-configuration-best-practices)

---

## 1. Architecture Overview

### 1.1 InnoDB Process Architecture

InnoDB uses a multi-threaded architecture with dedicated threads for various background tasks:

```
┌─────────────────────────────────────────────────────────────────┐
│                    InnoDB Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    mysqld Process                           ││
│  │                                                              ││
│  │  ┌──────────────────────────────────────────────────────┐   ││
│  │  │              InnoDB Storage Engine                    │   ││
│  │  │                                                        │   ││
│  │  │  ┌───────────────────────────────────────────────┐   │   ││
│  │  │  │           Master Thread                         │   │   ││
│  │  │  │  - Orchestrates background operations          │   │   ││
│  │  │  │  - Purge, flush, checkpoint coordination       │   │   ││
│  │  │  └───────────────────────────────────────────────┘   │   ││
│  │  │                                                        │   ││
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       │   ││
│  │  │  │ IO Thread  │ │ IO Thread  │ │ IO Thread  │       │   ││
│  │  │  │ (read)     │ │ (write)    │ │ (redo)     │       │   ││
│  │  │  └────────────┘ └────────────┘ └────────────┘       │   ││
│  │  │                                                        │   ││
│  │  │  ┌───────────────────────────────────────────────┐   │   ││
│  │  │  │           Buffer Pool (In-Memory)              │   │   ││
│  │  │  │  - Page hash table                             │   │   ││
│  │  │  │  - LRU list                                    │   │   ││
│  │  │  │  - Flush list                                  │   │   ││
│  │  │  │  - Adaptive hash index                         │   │   ││
│  │  │  └───────────────────────────────────────────────┘   │   ││
│  │  │                                                        │   ││
│  │  │  ┌───────────────────────────────────────────────┐   │   ││
│  │  │  │           Lock Manager                         │   │   ││
│  │  │  │  - Transaction locks                          │   │   ││
│  │  │  │  - Lock wait queue                            │   │   ││
│  │  │  └───────────────────────────────────────────────┘   │   ││
│  │  │                                                        │   ││
│  │  └──────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Disk Structures                           ││
│  │                                                              ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   ││
│  │  │ System      │ │ Undo        │ │ Redo Log Files      │   ││
│  │  │ Tablespace  │ │ Tablespaces │ │ (ib_logfile0/1)     │   ││
│  │  │ (ibdata1)   │ │ (*.ibd)     │ │                     │   ││
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   ││
│  │                                                              ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   ││
│  │  │ Data Files  │ │ Doublewrite │ │ Temp Tablespace     │   ││
│  │  │ (*.ibd)     │ │ Buffer      │ │ (ibtmp1)            │   ││
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 InnoDB Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 InnoDB Memory Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Buffer Pool                             │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ LRU List (Most Recently Used)                      │  │  │
│  │  │ ┌──────────────────────────────────────────────┐    │  │  │
│  │  │ │  New Sublist (5/8 of pool)                  │    │  │  │
│  │  │ │  - Young pages                               │    │  │  │
│  │  │ │  - Recently accessed                         │    │  │  │
│  │  │ ├──────────────────────────────────────────────┤    │  │  │
│  │  │ │  Old Sublist (3/8 of pool)                  │    │  │  │
│  │  │ │  - Less frequently accessed                  │    │  │  │
│  │  │ │  - Candidates for eviction                   │    │  │  │
│  │  │ └──────────────────────────────────────────────┘    │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Flush List                                          │  │  │
│  │  │ - Dirty pages ordered by oldest_modification LSN   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Adaptive Hash Index                                 │  │  │
│  │  │ - Built on frequently accessed buffer pool pages   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Log Buffer                              │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Redo log records waiting to be written to disk      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Other Memory                            │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │  │
│  │  │ Lock Info   │ │ Dictionary │ │ Transaction         │  │  │
│  │  │             │ │ Cache      │ │ System              │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Redo Logging and Log Groups

### 2.1 Redo Log Fundamentals

InnoDB uses write-ahead logging (WAL) to ensure durability. All changes are written to the redo log before being applied to data pages in the buffer pool. This allows for crash recovery by replaying the log.

#### Key Concepts

- **LSN (Log Sequence Number)**: A 64-bit counter representing the position in the redo log
- **Log Group**: Collection of redo log files that store the log records
- **Redo Log Record**: Contains information to redo an operation (space ID, page number, data)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Redo Log Structure                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Redo Log Files (ib_logfile0, ib_logfile1)   │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Log Record 1: UPDATE page 5, offset 100             │ │  │
│  │  │ - Space ID, Page Number                              │ │  │
│  │  │ - Offset, Length                                     │ │  │
│  │  │ - Before image (optional)                            │ │  │
│  │  │ - After image                                        │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ Log Record 2: INSERT into page 7                    │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ ...                                                  │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Checkpoint LSN: 1000000                             │ │  │
│  │  │ - Latest checkpoint position                         │ │  │
│  │  │ - Written periodically                              │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  LSN Progression:                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 0 ─────────────────────────► Current LSN                 ││
│  │   ▲                             │                         ││
│  │   │                             │                         ││
│  │   └─────────────────────────────┘                         ││
│  │   Checkpoint LSN                                           ││
│  │   (Earliest log needed for recovery)                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Redo Log Write Process

```
┌─────────────────────────────────────────────────────────────────┐
│                  Redo Log Write Process                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Transaction: UPDATE orders SET status = 'shipped'             │
│               WHERE order_id = 12345;                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 1: Generate Redo Log Record                            ││
│  │                                                          ││
│  │  ┌─────────────────────────────────────────────────────┐  ││
│  │  │ redo_record = {                                      │  ││
│  │  │   type: MLOG_UPDATE,                                 │  ││
│  │  │   space_id: 5,                                       │  ││
│  │  │   page_no: 100,                                      │  ││
│  │  │   offset: 50,                                        │  ││
│  │  │   data: "shipped"                                    │  ││
│  │  │ }                                                    │  ││
│  │  └─────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 2: Write to Log Buffer (innodb_log_buffer_size)       ││
│  │                                                          ││
│  │  log_buffer.append(redo_record)                          ││
│  │  new_lsn = calculate_lsn(log_buffer)                    ││
│  │                                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 3: Write to Log Files (circular)                     ││
│  │                                                          ││
│  │  if (log_buffer full or commit):                        ││
│  │      write_to_log_file(log_buffer)                       ││
│  │      fsync() (based on innodb_flush_log_at_trx_commit)  ││
│  │                                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Step 4: Modify Buffer Pool Page                             ││
│  │                                                          ││
│  │  page = buffer_pool.get_page(space_id, page_no)         ││
│  │  page.modify("status", "shipped")                        ││
│  │  page.set_dirty()                                        ││
│  │                                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Note: Data page modification happens AFTER redo log write     │
│        (Write-Ahead Logging principle)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Log Group Configuration

```sql
-- Redo log file size (total across all files)
-- Important: total size = innodb_log_file_size * innodb_log_files_in_group
SET GLOBAL innodb_log_file_size = 1073741824;  -- 1GB per file

-- Number of redo log files (default: 2)
SET GLOBAL innodb_log_files_in_group = 2;

-- Log buffer size (default: 16MB)
SET GLOBAL innodb_log_buffer_size = 67108864;  -- 64MB

-- Flush method for redo log
-- 1: Full durability (default) - fsync on commit
-- 2: Group commit - flush on commit, don't wait for disk
-- 0: Async - flush every second, may lose up to 1 second of transactions
SET GLOBAL innodb_flush_log_at_trx_commit = 1;

-- Redo log compression (MySQL 8.0.21+)
SET GLOBAL innodb_log_compression = ON;

-- Checkpoint interval
SET GLOBAL innodb_checkpoint_interval = 1;  -- Checkpoint every 1 second

-- Monitoring redo log
SHOW ENGINE INNODB STATUS;
-- Look for:
-- - Log sequence number
-- - Log flushed up to
-- - Last checkpoint at
-- - Pending log writes

-- Information schema
SELECT * FROM information_schema.INNODB_REDO_LOG_FILES;
```

### 2.4 Redo Log Recovery

During startup, InnoDB performs recovery:

```
┌─────────────────────────────────────────────────────────────────┐
│                    InnoDB Recovery Process                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Phase 1: Read Checkpoint                                    ││
│  │                                                          ││
│  │  1. Find last checkpoint in log files                    ││
│  │  2. Read checkpoint LSN                                    ││
│  │  3. Determine which transactions need recovery            ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Phase 2: Redo Pass                                         ││
│  │                                                          ││
│  │  FOR each log record from checkpoint LSN to end:         ││
│  │      IF record modifies page:                            ││
│  │          READ page from disk (if not in buffer pool)      ││
│  │          APPLY redo record to page                        ││
│  │                                                          ││
│  │  This ensures all committed changes are persisted         ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Phase 3: Undo Pass (if needed)                             ││
│  │                                                          ││
│  │  - Roll back uncommitted transactions                     ││
│  │  - Use undo logs to revert changes                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Recovery Time Depends on:                                      │
│  - Size of redo log to process                                  │
│  - Number of transactions to roll back                          │
│  - innodb_flush_log_at_trx_commit setting                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Undo Tablespaces and Purge

### 3.1 Undo Log Fundamentals

Undo logs store the previous version of modified rows, enabling:

- **MVCC**: Reading consistent snapshots without locks
- **Transaction Rollback**: Reverting uncommitted changes
- **Purge**: Cleaning up old row versions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Undo Log Structure                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Undo Tablespaces                           │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ undo_001.ibd                                         │ │  │
│  │  │ ┌─────────────────────────────────────────────────┐ │ │  │
│  │  │ │ Rollback Segment 0                              │ │ │  │
│  │  │ │ ┌─────────────────────────────────────────────┐ │ │ │  │
│  │  │ │ │ Undo Slot 0  → Undo Log [trx1 UPDATE]      │ │ │ │  │
│  │  │ │ │ Undo Slot 1  → Undo Log [trx2 INSERT]      │ │ │ │  │
│  │  │ │ │ Undo Slot 2  → Undo Log [trx3 DELETE]      │ │ │ │  │
│  │  │ │ │ ...                                          │ │ │ │  │
│  │  │ │ └─────────────────────────────────────────────┘ │ │ │  │
│  │  │ └─────────────────────────────────────────────────┘ │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ undo_002.ibd                                         │ │  │
│  │  │ └────────────────────────────────────────────────────┘ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Undo Log Record Structure:                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ - Next Undo Log Pointer                                   │  │
│  │ - Transaction ID                                           │  │
│  │ - Undo Log Type (INSERT, UPDATE, DELETE)                  │  │
│  │ - Previous Value (before image)                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 MVCC with Undo Logs

```
┌─────────────────────────────────────────────────────────────────┐
│                   MVCC Using Undo Logs                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Initial Table: orders                                          │
│  ┌─────┬─────────────┬────────┐                               │
│  │ id  │ status      │ total  │                               │
│  ├─────┼─────────────┼────────┤                               │
│  │ 1   │ pending     │ 100.00 │                               │
│  └─────┴─────────────┴────────┘                               │
│                                                                  │
│  Transaction A (XID 100): UPDATE orders SET status = 'shipped'│
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 1: Create undo log entry                             │ │
│  │   undo_log = { old_value: "pending", trx_id: 100 }       │ │
│  │   Insert into rollback segment                            │ │
│  │                                                            │ │
│  │ Step 2: Modify the row (in-memory)                        │ │
│  │   row.status = "shipped"                                  │ │
│  │   row.roll_ptr = pointer to undo_log                     │ │
│  │                                                            │ │
│  │ Step 3: Write redo log                                    │ │
│  │   redo_log = undo_log + row_change                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Row Structure After Update:                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ ┌────────────┬──────────────┬────────────┬─────────────┐ │ │
│  │ │ id: 1      │ status:      │ total:     │ roll_ptr:   │ │ │
│  │ │            │ 'shipped'    │ 100.00     │ →undo_log   │ │ │
│  │ └────────────┴──────────────┴────────────┴─────────────┘ │ │
│  │                              │                             │ │
│  │                              ▼                             │ │
│  │   Undo Log (in rollback segment)                         │ │
│  │   ┌───────────────────────────────────────────────┐       │ │
│  │   │ Old Value: status = 'pending'                 │       │ │
│  │   │ Transaction ID: 100                           │       │ │
│  │   │ Next: null                                     │       │ │
│  │   └───────────────────────────────────────────────┘       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Reader with Snapshot:                                           │
│  - Transaction B starts with snapshot at XID 99                │
│  - Reads row: sees status = 'pending' (from undo log)          │
│  - Because: update transaction 100 is not yet committed        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Purge System

Purge removes obsolete row versions that are no longer needed by any transaction:

```sql
-- Configuration
SET GLOBAL innodb_purge_threads = 4;         -- Dedicated purge threads
SET GLOBAL innodb_purge_batch_size = 300;    -- Records per purge batch
SET GLOBAL innodb_max_purge_lag = 0;         -- Delay DML if purge lags

-- Monitor purge
SHOW ENGINE INNODB STATUS;
-- Look for:
-- - Purge status
-- - Undo log entries to purge
-- - Purge history length

SELECT * FROM information_schema.INNODB_TRX;
-- trx_history_list_length: number of undo slots used
```

### 3.4 Undo Tablespace Management

```sql
-- Create dedicated undo tablespace (MySQL 8.0+)
CREATE UNDO TABLESPACE undo_2 ADD DATAFILE 'undo_002.ibd';
CREATE UNDO TABLESPACE undo_3 ADD DATAFILE 'undo_003.ibd';
CREATE UNDO TABLESPACE undo_4 ADD DATAFILE 'undo_004.ibd';

-- Set number of undo tablespaces
SET GLOBAL innodb_undo_tablespaces = 4;

-- Configure undo retention
SET GLOBAL innodb_undo_retention = 1000;  -- Seconds to retain undo

-- Check undo tablespaces
SELECT * FROM information_schema.INNODB_TABLESPACES
WHERE SPACE_TYPE = 'Undo';

-- Truncate undo tablespace (when not in use)
ALTER UNDO TABLESPACE undo_2 SET INACTIVE;
-- After all transactions using it complete, file can be truncated

-- Monitor undo log usage
SELECT
    trx_id,
    trx_state,
    trx_history_list_length,
    trx_tables_in_use,
    trx_tables_locked
FROM information_schema.INNODB_TRX;
```

---

## 4. Lock Management System

### 4.1 Lock Types

InnoDB implements row-level locking with several lock types:

```
┌─────────────────────────────────────────────────────────────────┐
│                    InnoDB Lock Types                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Shared Lock (S)                          │  │
│  │   - Allows reading a row                                  │  │
│  │   - Multiple transactions can hold S lock on same row    │  │
│  │   - Blocks X lock                                         │  │
│  │                                                            │  │
│  │   SELECT * FROM orders WHERE id = 1 LOCK IN SHARE MODE;  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                Exclusive Lock (X)                          │  │
│  │   - Allows reading and modifying a row                    │  │
│  │   - Only one transaction can hold X lock                  │  │
│  │   - Blocks both S and X locks                             │  │
│  │                                                            │  │
│  │   SELECT * FROM orders WHERE id = 1 FOR UPDATE;          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Intention Locks (Table-Level)                 │  │
│  │                                                            │  │
│  │   IS - Intention Shared: Will acquire S lock on rows     │  │
│  │   IX - Intention Exclusive: Will acquire X lock on rows  │  │
│  │                                                            │  │
│  │   Purpose: Quickly check if table has row-level locks    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Gap Locks                               │  │
│  │   - Locks the gap between index records                   │  │
│  │   - Prevents insertion into the gap                        │  │
│  │                                                            │  │
│  │   SELECT * FROM orders WHERE id > 10 AND id < 20          │  │
│  │   LOCK IN SHARE MODE;                                     │  │
│  │   -- Locks gap (10, 20), no insertion allowed            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               Next-Key Locks                               │  │
│  │   - Combination of record lock + gap lock                 │  │
│  │   - Locks the row and prevents insertion before it       │  │
│  │                                                            │  │
│  │   Default lock mode in REPEATABLE READ                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Insert Intention Locks                        │  │
│  │   - Signal that insertion is about to happen              │  │
│  │   - Waits for gap lock holders to release                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Lock Wait and Deadlock Detection

```
┌─────────────────────────────────────────────────────────────────┐
│              Lock Wait and Deadlock Detection                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Lock Wait Timeout                           │  │
│  │                                                            │  │
│  │  Transaction A: SELECT * FROM orders FOR UPDATE;         │  │
│  │                 (acquires X lock on row 1)               │  │
│  │                                                            │  │
│  │  Transaction B: SELECT * FROM orders FOR UPDATE;         │  │
│  │                 (waits for X lock on row 1)              │  │
│  │                                                            │  │
│  │  After innodb_lock_wait_timeout seconds (default 50):     │  │
│  │  ERROR 1205 (HY000): Lock wait timeout exceeded          │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Deadlock Detection                         │  │
│  │                                                            │  │
│  │  T1: SELECT * FROM orders WHERE id = 1 FOR UPDATE;      │  │
│  │      (X lock on row 1)                                    │  │
│  │                                                            │  │
│  │  T2: SELECT * FROM products WHERE id = 1 FOR UPDATE;     │  │
│  │      (X lock on row 1 in products)                      │  │
│  │                                                            │  │
│  │  T1: SELECT * FROM products WHERE id = 1 FOR UPDATE;     │  │
│  │      (waits for T2's lock on product row 1)             │  │
│  │                                                            │  │
│  │  T2: SELECT * FROM orders WHERE id = 1 FOR UPDATE;      │  │
│  │      (waits for T1's lock on order row 1)               │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐│ │
│  │  │ Deadlock detected!                                  ││ │
│  │  │                                                      ││ │
│  │  │ T1 → holds lock on orders.id=1                     ││ │
│  │  │    → waiting for products.id=1                     ││ │
│  │  │                                                      ││ │
│  │  │ T2 → holds lock on products.id=1                   ││ │
│  │  │    → waiting for orders.id=1                        ││ │
│  │  │                                                      ││ │
│  │  │ Solution: Roll back transaction with least undo    ││ │
│  │  │          log (usually the one that started later)  ││ │
│  │  └─────────────────────────────────────────────────────┘│ │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Configuration:                                                  │
│  SET GLOBAL innodb_lock_wait_timeout = 10;                     │
│  SET GLOBAL innodb_deadlock_detect = ON;                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Lock Monitoring

```sql
-- View current locks
SELECT
    t.trx_id,
    t.trx_state,
    t.trx_started,
    l.lock_id,
    l.lock_mode,
    l.lock_type,
    l.lock_table,
    l.lock_index,
    l.lock_space,
    l.lock_page,
    l.lock_rec
FROM information_schema.INNODB_LOCKS l
JOIN information_schema.INNODB_TRX t ON l.lock_trx_id = t.trx_id;

-- View lock waits
SELECT
    r.trx_id waiting_trx,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query
FROM information_schema.INNODB_LOCK_WAITS w
JOIN information_schema.INNODB_TRX b ON w.blocking_trx_id = b.trx_id
JOIN information_schema.INNODB_TRX r ON w.requesting_trx_id = r.trx_id;

-- View transaction details
SELECT * FROM information_schema.INNODB_TRX;

-- Processlist with lock info
SELECT
    p.ID,
    p.USER,
    p.HOST,
    p.DB,
    p.COMMAND,
    p.TIME,
    p.STATE,
    p.INFO,
    t.trx_id,
    t.trx_state,
    t.trx_started
FROM information_schema.PROCESSLIST p
LEFT JOIN information_schema.INNODB_TRX t ON p.ID = t.trx_mysql_thread_id
WHERE t.trx_id IS NOT NULL;

-- Enable lock instrumentation
SET GLOBAL innodb_status_output = 'ON';
SET GLOBAL innodb_status_output_locks = 'ON';

-- Check InnoDB status for locks
SHOW ENGINE INNODB STATUS;
```

---

## 5. Adaptive Hash Indexing

### 5.1 Adaptive Hash Index Overview

The Adaptive Hash Index (AHI) optimizes index lookups by building hash indexes for frequently accessed B-tree pages. This can significantly speed up lookups for equalities on indexed columns.

```
┌─────────────────────────────────────────────────────────────────┐
│              Adaptive Hash Index Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              B-tree Index (orders.customer_id)            │  │
│  │                                                            │  │
│  │         Root ──┬── Page 5 (customer_id: 100-500)          │  │
│  │                │                                          │  │
│  │         ┌─────┴─────┐                                    │  │
│  │        Page 10    Page 15                                 │  │
│  │     (customer   (customer                                │  │
│  │      100-300)    301-500)                                │  │
│  │                                                            │  │
│  │  When customer_id=250 is accessed frequently:            │  │
│  │  - InnoDB monitors access pattern                        │  │
│  │  - Creates hash entry for page 10                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Adaptive Hash Index                          │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Hash Table                                            │ │  │
│  │  │                                                       │ │  │
│  │  │ hash(customer_id=250) → pointer to B-tree page 10  │ │  │
│  │  │ hash(customer_id=350) → pointer to B-tree page 15  │ │  │
│  │  │                                                       │ │  │
│  │  │ Instead of B-tree traversal:                         │ │  │
│  │  │ - Compute hash(key)                                 │ │  │
│  │  │ - Direct pointer to page                             │ │  │
│  │  │ - O(1) lookup instead of O(log n)                   │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 AHI Operation

```
┌─────────────────────────────────────────────────────────────────┐
│                AHI Lookup Process                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: SELECT * FROM orders WHERE customer_id = 250;          │
│                                                                  │
│  Without AHI:                                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Start at root of B-tree                                │  │
│  │ 2. Navigate through internal pages to find target       │  │
│  │ 3. Read leaf page                                        │  │
│  │ 4. Find exact record                                     │  │
│  │                                                           │  │
│  │ Cost: ~3-4 page reads for deep B-tree                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  With AHI:                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Compute hash = hash(250)                              │  │
│  │ 2. Look up hash table                                    │  │
│  │ 3. Direct pointer to leaf page                            │  │
│  │ 4. Find exact record                                     │  │
│  │                                                           │  │
│  │ Cost: 1 page read (the leaf page itself)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  AHI Conditions:                                                 │
│  - Page accessed > 100 times in buffer pool                     │  │
│  - Pattern is equality search (not range)                      │  │
│  - B-tree root-to-leaf path has > 2 pages                      │  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 AHI Configuration and Monitoring

```sql
-- Enable/disable AHI (default: ON)
SET GLOBAL innodb_adaptive_hash_index = ON;

-- Partition AHI for better concurrency (MySQL 8.0.18+)
-- Splits hash index into multiple partitions
SET GLOBAL innodb_adaptive_hash_index_parts = 8;

-- Monitor AHI usage
SHOW ENGINE INNODB STATUS\G
-- Look for:
-- -------------------------------------
-- INSERT BUFFER AND ADAPTIVE HASH INDEX
-- -------------------------------------
-- Hash table size 3175291, node heap has 0 buffer(s)
-- Hash table size 3175291, node heap has 0 buffer(s)
-- 0.00 hash searches/s, 0.00 non-hash searches/s

-- Check AHI statistics
SELECT
    SUBSTRING_INDEX(NAME, ' ', 2) AS metric,
    COUNT
FROM information_schema.INNODB_METRICS
WHERE NAME LIKE 'adaptive_hash%';

-- Performance schema
SELECT * FROM performance_schema.events_statements_history
WHERE SQL_TEXT LIKE '%customer_id%';
```

---

## 6. Buffer Pool Management

### 6.1 Buffer Pool Architecture

InnoDB's buffer pool is the heart of its caching system, storing data pages and index pages in memory:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Buffer Pool Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Buffer Pool (InnoDB)                   │  │
│  │  Default: 128MB (or 50-80% of available RAM)              │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                   Page Hash Table                    │  │  │
│  │  │  Maps: (space_id, page_id) → buffer frame           │  │  │
│  │  │  O(1) lookup to find page in memory                 │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              LRU (Least Recently Used) List         │  │  │
│  │  │                                                      │  │  │
│  │  │  New Sublist (5/8) ──► Frequently accessed          │  │  │
│  │  │         │                                            │  │  │
│  │  │         │                                            │  │  │
│  │  │  Old Sublist (3/8) ──► Less frequent, eviction    │  │  │
│  │  │                    candidates                       │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              Flush List (Dirty Pages)              │  │  │
│  │  │  Pages modified but not yet written to disk        │  │  │
│  │  │  Ordered by oldest_modification LSN                  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Buffer Pool Instance (if partitioned)       │  │
│  │  Multiple instances can reduce contention                │  │
│  │  innodb_buffer_pool_instances = 4 (or more)            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Page Operations

```
┌─────────────────────────────────────────────────────────────────┐
│                Buffer Pool Page Operations                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Page Read                                 │  │
│  │                                                            │  │
│  │  SELECT * FROM orders WHERE id = 12345;                  │  │
│  │                                                            │  │
│  │  1. Check if page is in buffer pool (hash table)         │  │
│  │     IF found:                                             │  │
│  │        Move to new sublist (if in old)                   │  │
│  │        Return page                                        │  │
│  │     ELSE:                                                 │  │
│  │        Allocate buffer frame                              │  │
│  │        Read page from disk                               │  │
│  │        Insert into buffer pool                           │  │
│  │        Return page                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Page Write                                │  │
│  │                                                            │  │
│  │  UPDATE orders SET status = 'shipped' WHERE id = 12345; │  │
│  │                                                            │  │
│  │  1. Find page in buffer pool                             │  │
│  │  2. Modify page data                                      │  │
│  │  3. Add to flush list (mark as dirty)                    │  │
│  │  4. Write redo log                                       │  │
│  │  5. Background thread will eventually flush to disk     │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               Page Replacement (LRU)                       │  │
│  │                                                            │  │
│  │  When buffer pool needs space for new page:              │  │
│  │                                                            │  │
│  │  1. Start at tail of old sublist                         │  │
│  │  2. If page is dirty: flush to disk                     │  │
│  │  3. Evict page                                            │  │
│  │  4. Insert new page at head of new sublist               │  │
│  │                                                            │  │
│  │  Configuration:                                            │  │
│  │  innodb_old_blocks_pct = 37 (3/8 = 37%)                 │  │
│  │  innodb_old_blocks_time = 1000 (ms)                      │  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Buffer Pool Configuration

```sql
-- Buffer pool size (typically 50-80% of available RAM)
SET GLOBAL innodb_buffer_pool_size = 17179869184;  -- 16GB

-- Buffer pool instances (reduce contention on multi-core)
SET GLOBAL innodb_buffer_pool_instances = 8;

-- LRU tuning
SET GLOBAL innodb_old_blocks_pct = 37;    -- Old sublist percentage
SET GLOBAL innodb_old_blocks_time = 1000;  -- Time before moving to new

-- Flush behavior
SET GLOBAL innodb_max_dirty_pages_pct = 90;    -- Max dirty pages %
SET GLOBAL innodb_max_dirty_pages_pct_lwm = 10; % Low water mark
SET GLOBAL innodb_flush_neighbors = 1;         -- Flush adjacent pages

-- Prefetching (read-ahead)
SET GLOBAL innodb_read_ahead_threshold = 56;     -- Pages to trigger read-ahead
SET GLOBAL innodb_random_read_ahead = 0;        -- Random read-ahead

-- Monitoring buffer pool
SELECT
    POOL_ID,
    POOL_SIZE,
    FREE_BUFFERS,
    DATABASE_PAGES,
    OLD_DATABASE_PAGES,
    MODIFIED_DATABASE_PAGES,
    PENDING_DECOMPRESS,
    PENDING_READ
FROM information_schema.INNODB_BUFFER_POOL_STATS;

-- Check pages in buffer pool
SELECT
    SPACE,
    PAGE_NUMBER,
    PAGE_TYPE,
    TABLE_NAME,
    INDEX_NAME,
    IS_NEW,
    IS_OLD,
    IS_DIRTY
FROM information_schema.INNODB_BUFFER_PAGE
WHERE SPACE = (SELECT SPACE FROM information_schema.INNODB_TABLES
               WHERE NAME = 'test/orders')
LIMIT 20;
```

---

## 7. Checkpoint Management

### 7.1 Checkpoint Fundamentals

Checkpoints ensure that dirty pages are periodically written to disk, enabling:

- Recovery point reduction
- Clean shutdown
- Log file reuse

```
┌─────────────────────────────────────────────────────────────────┐
│                   Checkpoint Mechanism                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Checkpoint Triggers                           │  │
│  │                                                            │  │
│  │  1. Sharp checkpoint (shutdown)                           │  │
│  │     - All dirty pages flushed on clean shutdown           │  │
│  │                                                            │  │
│  │  2. Fuzzy checkpoint (runtime)                            │  │
│  │     - innodb_flush_log_at_trx_commit = 1                │  │
│  │     - Adaptive checkpoint based on dirty page threshold   │  │
│  │                                                            │  │
│  │  3. Timeout checkpoint                                     │  │
│  │     - innodb_flush_log_at_trx_commit = 1                 │  │
│  │     - Every second                                         │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Checkpoint Process                            │  │
│  │                                                            │  │
│  │  1. Write checkpoint marker to redo log                  │  │
│  │     - LSN of checkpoint                                   │  │
│  │     - Transaction state                                    │  │
│  │                                                            │  │
│  │  2. Flush dirty pages from buffer pool                    │  │
│  │     - Prioritize oldest (lowest LSN) first               │  │
│  │     - Based on innodb_flush_neighbors                     │  │
│  │                                                            │  │
│  │  3. Update control file                                   │  │
│  │     - Latest checkpoint LSN                               │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         LSN and Checkpoint Relationship                     │  │
│  │                                                            │  │
│  │  Log: ┌───────────────────────────────────────────────►   │  │
│  │       │                                               │   │  │
│  │       │  ┌────────┐                                  │   │  │
│  │       │  │Checkpt1│  ──── Only redo after this      │   │  │
│  │       │  │ LSN:1M │       can be discarded          │   │  │
│  │       │  └────────┘                                  │   │  │
│  │       │            ┌────────┐                        │   │  │
│  │       │            │Checkpt2│  ──── Current         │   │  │
│  │       │            │ LSN:2M │                        │   │  │
│  │       │            └────────┘                        │   │  │
│  │       │                                               │   │  │
│  │       └──────────────────────────────────────────────►   │  │
│  │                   Current LSN                              │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Adaptive Checkpointing

MySQL 8.0 introduces adaptive checkpointing:

```sql
-- Adaptive checkpoint (MySQL 8.0+)
SET GLOBAL innodb_adaptive_flushing = ON;
SET GLOBAL innodb_adaptive_flushing_lwm = 10;

-- Configure checkpoint age target
SET GLOBAL innodb_flush_sync = ON;
SET GLOBAL innodb_flush_neighbors = 1;

-- Control checkpoint frequency
SET GLOBAL innodb_flush_log_at_timeout = 1;  -- Flush every 1 second

-- Monitoring checkpoint
SHOW ENGINE INNODB STATUS\G
-- Look for:
-- ---
-- LOG
-- ---
-- Log sequence number 12345678
-- Log flushed up to 12345678
-- Last checkpoint at 12345678

SELECT * FROM information_schema.INNODB_METRICS
WHERE NAME LIKE 'checkpoint%';
```

---

## 8. Index Structures

### 8.1 B-tree Index

InnoDB uses B+tree indexes with all data in leaf nodes:

```
┌─────────────────────────────────────────────────────────────────┐
│                    InnoDB B+tree Index                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Root Page (Internal)                   │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Page Header                                          │ │  │
│  │  │ - Index ID                                           │ │  │
│  │  │ - Level: 2                                           │ │  │
│  │  │ - Number of records                                  │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ Directory Slots (pointers to records)               │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ Records:                                             │ │  │
│  │  │ - key: (100), child_pointer → Page 5               │ │  │
│  │  │ - key: (200), child_pointer → Page 10              │ │  │
│  │  │ - key: (300), child_pointer → Page 15              │ │  │
│  │  │ - ...                                                │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│          ┌─────────────────┼─────────────────┐                   │
│          ▼                 ▼                 ▼                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Page 5    │   │  Page 10    │   │  Page 15    │           │
│  │  (100-199)  │   │ (200-299)   │   │ (300-399)   │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│          │                 │                 │                   │
│          └─────────────────┼─────────────────┘                   │
│                            ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Leaf Page (Level 0)                        │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Records (key, primary_key, transaction_id, rollback│ │  │
│  │  │          pointer, payload)                           │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ (100, row_data1, 123, ptr, ...)  → row 1           │ │  │
│  │  │ (150, row_data2, 124, ptr, ...)  → row 2           │ │  │
│  │  │ (180, row_data3, 125, ptr, ...)  → row 3           │ │  │
│  │  │ (200, row_data4, 126, ptr, ...)  → row 4           │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ Next Page Pointer: → Page 10                       │ │  │
│  │  │ Previous Page Pointer: ← Page 5                    │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  Key Properties:                                          │  │
│  │  - All leaf pages linked in doubly-linked list           │  │
│  │  - Index includes primary key (clustered)                │  │
│  │  - Contains actual row data in clustered index           │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Clustered vs Secondary Indexes

```sql
-- Primary key is the clustered index
-- Row data is stored in the leaf pages of the clustered index

-- Secondary index: B-tree with primary key in leaf pages
CREATE INDEX idx_customer ON orders (customer_id);

-- Index leaf page structure:
-- (customer_id value, primary_key_value)

-- When using secondary index:
-- 1. Search secondary index B-tree
-- 2. Get primary key value
-- 3. Search clustered index using primary key
-- 4. Retrieve row data

-- Index-only scans possible when all needed columns are in index
CREATE INDEX idx_status_total ON orders (status, total);
SELECT status, total FROM orders WHERE status = 'pending';
-- Can be satisfied entirely from index
```

---

## 9. Configuration Best Practices

### 9.1 Memory Configuration

```sql
-- Configuration for 32GB RAM server

-- Buffer pool: 70% of RAM
SET GLOBAL innodb_buffer_pool_size = 22548578304;  -- ~21GB

-- Buffer pool instances: one per CPU core (up to 8)
SET GLOBAL innodb_buffer_pool_instances = 8;

-- Log buffer
SET GLOBAL innodb_log_buffer_size = 67108864;  -- 64MB

-- Redo log: typically 1GB to match buffer pool
SET GLOBAL innodb_log_file_size = 1073741824;  -- 1GB
SET GLOBAL innodb_log_files_in_group = 2;
```

### 9.2 Performance Tuning

```sql
-- Flush behavior for performance
SET GLOBAL innodb_flush_log_at_trx_commit = 1;  -- Full durability
-- For better performance (less durability):
-- SET GLOBAL innodb_flush_log_at_trx_commit = 2;

-- Disable doublewrite for instant crash recovery (MySQL 8.0.20+)
SET GLOBAL innodb_doublewrite = 'DETECT_ONLY';

-- Optimize for SSD
SET GLOBAL innodb_flush_neighbors = 0;          -- Don't flush neighbors
SET GLOBAL innodb_io_capacity = 2000;         -- IOPS capacity
SET GLOBAL innodb_io_capacity_max = 4000;     -- Max IOPS
SET GLOBAL innodb_read_io_threads = 4;
SET GLOBAL innodb_write_io_threads = 4;

-- Adaptive hash index tuning
SET GLOBAL innodb_adaptive_hash_index = ON;
SET GLOBAL innodb_adaptive_hash_index_parts = 8;
```

### 9.3 Monitoring and Diagnostics

```sql
-- Overall InnoDB status
SHOW ENGINE INNODB STATUS\G

-- Buffer pool statistics
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool%';

-- Log statistics
SHOW GLOBAL STATUS LIKE 'Innodb_log%';

-- Lock statistics
SHOW GLOBAL STATUS LIKE 'Innodb_lock%';

-- Performance schema
SELECT * FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC LIMIT 10;
```

---

## Related Documentation

- [Database Fundamentals](../01_foundations/01_database_fundamentals.md)
- [Indexing Fundamentals](../01_foundations/04_indexing_fundamentals.md)
- [Query Optimization Deep Dive](../02_core_concepts/query_optimization_deep_dive.md)
- [Replication Patterns](../02_intermediate/01_scaling_strategies/02_replication_patterns.md)

---

## References

- MySQL 8.0 Documentation: https://dev.mysql.com/doc/refman/8.0/en/
- "High Performance MySQL" by Baron Schwartz et al.
- InnoDB Source Code: https://github.com/mysql/mysql-server
- "MySQL Internals" by Sasha Zaychenko

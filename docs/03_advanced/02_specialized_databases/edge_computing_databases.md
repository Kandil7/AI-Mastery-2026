# Edge Computing and Distributed Edge Databases

## Overview

Edge computing moves data processing closer to where data is generated—devices, sensors, and local edge nodes—rather than sending everything to centralized cloud data centers. This architectural shift creates unique database requirements that differ significantly from traditional cloud or on-premises deployments.

The proliferation of Internet of Things (IoT) devices, mobile applications requiring offline capability, and applications demanding ultra-low latency has driven the emergence of edge database technologies. Understanding these technologies is essential for building modern applications that can operate effectively in distributed, intermittently connected environments.

This guide explores edge database architectures, synchronization strategies, and practical implementation patterns. We cover the unique challenges of edge computing, the databases designed to address them, and patterns for building resilient edge applications.

The examples assume familiarity with basic database concepts and distributed systems. While specific implementation details use particular technologies, the patterns apply broadly across different edge database solutions.

## Understanding Edge Computing

### The Edge Computing Paradigm

Traditional cloud architecture sends all data to centralized data centers for processing. This approach works well for many applications but creates problems for scenarios requiring real-time responses, operation in areas with poor connectivity, or processing of massive data volumes at the source.

Edge computing addresses these challenges by placing compute and storage resources closer to data sources:

```
Traditional Architecture:
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   IoT      │─────▶│   Cloud     │─────▶│  Database  │
│   Device   │      │   Server    │      │   Cluster  │
└─────────────┘      └─────────────┘      └─────────────┘
      500ms                200ms                 50ms
       latency             latency              query

Edge Architecture:
┌─────────────┐      ┌─────────────┐
│   IoT      │─────▶│   Edge      │─────▶ Sync ──▶ Cloud
│   Device   │      │   Database  │
└─────────────┘      └─────────────┘
      500ms                5ms
                           local query
```

### Edge Database Characteristics

Edge databases must address several unique requirements:

**Offline Operation**: Edge devices may lose connectivity frequently. Databases must operate without constant network access and synchronize when connectivity returns.

**Limited Resources**: Edge devices typically have constrained CPU, memory, and storage. Databases must be lightweight while still providing ACID guarantees when possible.

**Low Latency**: Many edge applications require immediate responses. Database operations must complete in milliseconds, not seconds.

**Data Synchronization**: When connectivity is available, databases must efficiently synchronize with central systems, handling conflicts intelligently.

**Security**: Edge devices may be physically accessible or operate in untrusted environments. Data must be protected at rest and in transit.

## Edge Database Options

### Lightweight Databases for Edge

Several databases are designed specifically for edge deployment:

**SQLite**: The most ubiquitous embedded database, SQLite runs on virtually any platform with minimal resources. Its single-file design makes it ideal for edge devices.

**RocksDB**: An embedded key-value store optimized for fast storage, commonly used in edge scenarios with high write throughput.

**Realm**: A mobile-first database with built-in synchronization, popular for mobile and edge applications.

**CouchDB**: A document database with built-in sync capabilities, designed for distributed edge scenarios.

**DuckDB**: An analytical database that can run embedded, useful for edge analytics scenarios.

### SQLite for Edge

SQLite's minimal footprint makes it an excellent choice for edge scenarios:

```python
import sqlite3
from dataclasses import dataclass
from typing import Optional
import json

class EdgeSensorDatabase:
    """
    SQLite-based database for edge sensor data collection
    Optimized for embedded/edge deployment
    """
    
    def __init__(self, db_path: str, max_memory_mb: int = 64):
        self.db_path = db_path
        self._init_database(max_memory_mb)
    
    def _init_database(self, max_memory_mb: int):
        """Initialize database with edge-optimized settings"""
        conn = sqlite3.connect(self.db_path)
        
        # Configure for memory efficiency
        conn.execute(f"PRAGMA cache_size = {-max_memory_mb * 256}")  # In pages
        conn.execute("PRAGMA page_size = 4096")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_id TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                quality TEXT DEFAULT 'good'
            );
            
            CREATE TABLE IF NOT EXISTS device_config (
                device_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS pending_sync (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                sync_attempts INTEGER DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_readings_sensor_time 
            ON sensor_readings(sensor_id, timestamp);
        """)
        
        conn.close()
    
    def insert_reading(self, sensor_id: str, value: float, timestamp: int):
        """Insert sensor reading with local tracking"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """INSERT INTO sensor_readings (sensor_id, value, timestamp)
                   VALUES (?, ?, ?)""",
                (sensor_id, value, timestamp)
            )
            
            # Track for later synchronization
            conn.execute(
                """INSERT INTO pending_sync 
                   (table_name, record_id, operation, data_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    "sensor_readings",
                    cursor.lastrowid,
                    "INSERT",
                    json.dumps({"sensor_id": sensor_id, "value": value, "timestamp": timestamp}),
                    timestamp
                )
            )
            
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_recent_readings(self, sensor_id: str, limit: int = 100):
        """Get recent readings for a sensor"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """SELECT * FROM sensor_readings 
                   WHERE sensor_id = ? 
                   ORDER BY timestamp DESC 
                   LIMIT ?""",
                (sensor_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_pending_sync(self, limit: int = 1000):
        """Get records pending synchronization"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """SELECT * FROM pending_sync 
                   ORDER BY created_at ASC 
                   LIMIT ?""",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def mark_synced(self, sync_ids: list):
        """Mark records as synced"""
        if not sync_ids:
            return
        
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join("?" * len(sync_ids))
            conn.execute(
                f"DELETE FROM pending_sync WHERE id IN ({placeholders})",
                sync_ids
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_storage_stats(self):
        """Get storage statistics"""
        import os
        conn = sqlite3.connect(self.db_path)
        try:
            reading_count = conn.execute(
                "SELECT COUNT(*) FROM sensor_readings"
            ).fetchone()[0]
            
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM pending_sync"
            ).fetchone()[0]
            
            return {
                "reading_count": reading_count,
                "pending_sync_count": pending_count,
                "file_size_bytes": os.path.getsize(self.db_path),
                "wal_size_bytes": os.path.getsize(self.db_path + "-wal") 
                                    if os.path.exists(self.db_path + "-wal") else 0
            }
        finally:
            conn.close()
```

### CouchDB and Mobile Synchronization

CouchDB provides built-in synchronization with conflict resolution:

```python
import couchdb
from typing import Dict, Any, List

class CouchDBEdgeSync:
    """
    CouchDB with sync for edge devices
    """
    
    def __init__(self, local_url: str, remote_url: str, db_name: str):
        # Connect to local (edge) database
        self.local_db = couchdb.Database(local_url + "/" + db_name)
        
        # Set up remote sync
        self.remote_url = remote_url
        self.db_name = db_name
    
    def initialize(self):
        """Create design documents for the database"""
        design = {
            "_id": "_design/sensors",
            "views": {
                "by_timestamp": {
                    "map": """function(doc) {
                        if (doc.type === 'reading') {
                            emit(doc.timestamp, doc);
                        }
                    }"""
                },
                "by_sensor": {
                    "map": """function(doc) {
                        if (doc.type === 'reading') {
                            emit([doc.sensor_id, doc.timestamp], doc);
                        }
                    }"""
                },
                "conflicts": {
                    "map": """function(doc) {
                        if (doc._conflicts) {
                            emit(doc._id, doc._conflicts);
                        }
                    }"""
                }
            }
        }
        
        try:
            self.local_db.save(design)
        except couchdb.ResourceConflict:
            pass  # Already exists
    
    def save_reading(self, sensor_id: str, value: float, timestamp: int):
        """Save a sensor reading"""
        doc = {
            "type": "reading",
            "sensor_id": sensor_id,
            "value": value,
            "timestamp": timestamp
        }
        
        self.local_db.save(doc)
        return doc["_id"]
    
    def get_recent_readings(self, sensor_id: str, limit: int = 100) -> List[Dict]:
        """Get recent readings"""
        view = self.local_db.view(
            "sensors/by_sensor",
            startkey=[sensor_id, {}],
            endkey=[sensor_id],
            descending=True,
            limit=limit,
            include_docs=True
        )
        
        return [row.doc for row in view]
    
    def sync_with_remote(self) -> Dict[str, Any]:
        """Synchronize with remote database"""
        from couchdb.replication import Replication
        
        # Pull from remote
        pull_result = self.local_db.pull(
            self.remote_url + "/" + self.db_name,
            self.db_name
        )
        
        # Push to remote
        push_result = self.local_db.push(
            self.remote_url + "/" + self.db_name,
            self.db_name
        )
        
        return {
            "pulled": pull_result.get("docs_read", 0),
            "pushed": push_result.get("docs_written", 0),
            "conflicts": self._resolve_conflicts()
        }
    
    def _resolve_conflicts(self) -> int:
        """Resolve conflicts using last-write-wins"""
        conflicts = list(self.local_db.view("sensors/conflicts"))
        resolved = 0
        
        for row in conflicts:
            doc = self.local_db[row.id]
            
            # Get all conflicts
            conflict_versions = doc.get("_conflicts", [])
            
            # Keep the version with the latest timestamp
            versions = [doc]
            for conflict_id in conflict_versions:
                conflict_doc = self.local_db[conflict_id]
                versions.append(conflict_doc)
            
            # Sort by timestamp and keep latest
            versions.sort(key=lambda d: d.get("timestamp", 0), reverse=True)
            winner = versions[0]
            
            # Remove conflicts
            winner["_conflicts"] = []
            
            # Save winning version
            try:
                self.local_db.save(winner)
                resolved += 1
                
                # Delete losing versions
                for v in versions[1:]:
                    try:
                        self.local_db.delete(v)
                    except:
                        pass
            except:
                pass
        
        return resolved
```

## Synchronization Strategies

### Conflict Resolution Strategies

When edge databases synchronize, conflicts inevitably arise. Understanding resolution strategies is crucial:

**Last-Write-Wins (LWW)**: The simplest approach—most recent timestamp wins. Easy to implement but can lose data.

**Application-Specific Resolution**: Custom logic determines the correct version based on business rules.

**Three-Way Merge**: Compare base version with both changes; attempt automatic merge.

**Manual Resolution**: Present conflicts to users for decision.

```python
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass

class ConflictResolutionStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    APPLICATION_LOGIC = "application_logic"
    THREE_WAY_MERGE = "three_way_merge"
    MANUAL = "manual"


@dataclass
class Conflict:
    record_id: str
    local_version: dict
    remote_version: dict
    base_version: dict = None


class ConflictResolver:
    """Handles sync conflicts with configurable strategy"""
    
    def __init__(self, strategy: ConflictResolutionStrategy):
        self.strategy = strategy
    
    def resolve(self, conflicts: List[Conflict]) -> List[dict]:
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return self._resolve_lww(conflicts)
        elif self.strategy == ConflictResolutionStrategy.APPLICATION_LOGIC:
            return self._resolve_application(conflicts)
        elif self.strategy == ConflictResolutionStrategy.THREE_WAY_MERGE:
            return self._resolve_three_way(conflicts)
        else:
            return [c.local_version for c in conflicts]  # Keep local
    
    def _resolve_lww(self, conflicts: List[Conflict]) -> List[dict]:
        """Last-write-wins resolution"""
        resolved = []
        
        for conflict in conflicts:
            local_time = conflict.local_version.get("updated_at", 0)
            remote_time = conflict.remote_version.get("updated_at", 0)
            
            if remote_time > local_time:
                resolved.append(conflict.remote_version)
            else:
                resolved.append(conflict.local_version)
        
        return resolved
    
    def _resolve_application(self, conflicts: List[Conflict]) -> List[dict]:
        """Application-specific resolution"""
        resolved = []
        
        for conflict in conflicts:
            # Example: for orders, remote wins
            # For sensor readings, keep both as separate records
            if conflict.local_version.get("type") == "order":
                resolved.append(conflict.remote_version)
            else:
                # Keep local but note conflict
                conflict.local_version["_conflict_resolved"] = True
                resolved.append(conflict.local_version)
        
        return resolved
    
    def _resolve_three_way(self, conflicts: List[Conflict]) -> List[dict]:
        """Three-way merge when base version available"""
        resolved = []
        
        for conflict in conflicts:
            if not conflict.base_version:
                # Can't do three-way without base
                resolved.append(conflict.remote_version)
                continue
            
            merged = self._merge_versions(
                conflict.base_version,
                conflict.local_version,
                conflict.remote_version
            )
            resolved.append(merged)
        
        return resolved
    
    def _merge_versions(self, base: dict, local: dict, remote: dict) -> dict:
        """Simple field-level merge"""
        result = base.copy()
        
        # Apply remote changes
        for key, value in remote.items():
            if key not in local or local[key] == base.get(key):
                result[key] = value
        
        # Apply local changes
        for key, value in local.items():
            if key not in remote or remote[key] == base.get(key):
                result[key] = value
        
        # Both changed differently - remote wins for simplicity
        return result
```

### Sync Protocol Design

Design efficient synchronization protocols:

```python
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class SyncProtocol:
    """
    Efficient sync protocol for edge databases
    """
    
    def __init__(self, local_db, remote_db):
        self.local_db = local_db
        self.remote_db = remote_db
    
    async def sync(self, last_sync_timestamp: int) -> Dict[str, Any]:
        """Perform bidirectional synchronization"""
        
        # Step 1: Push local changes to remote
        pushed = await self._push_changes(last_sync_timestamp)
        
        # Step 2: Pull remote changes
        pulled = await self._pull_changes(last_sync_timestamp)
        
        # Step 3: Detect and resolve conflicts
        conflicts = await self._detect_conflicts(pulled)
        
        # Step 4: Resolve conflicts
        resolved = self._resolve_conflicts(conflicts)
        
        # Step 5: Apply resolved changes locally
        await self._apply_resolved(resolved)
        
        return {
            "pushed_count": pushed,
            "pulled_count": pulled,
            "conflicts_found": len(conflicts),
            "conflicts_resolved": len(resolved),
            "sync_timestamp": int(datetime.utcnow().timestamp())
        }
    
    async def _push_changes(self, since: int) -> int:
        """Push local changes to remote"""
        # Get local changes since last sync
        changes = self.local_db.get_changes(since)
        
        pushed = 0
        for change in changes:
            try:
                self.remote_db.put(change)
                pushed += 1
            except Exception as e:
                print(f"Push failed for {change['_id']}: {e}")
        
        return pushed
    
    async def _pull_changes(self, since: int) -> List[dict]:
        """Pull remote changes"""
        return self.remote_db.get_changes(since)
    
    async def _detect_conflicts(self, remote_changes: List[dict]) -> List[Conflict]:
        """Detect conflicts between local and remote"""
        conflicts = []
        
        for remote_change in remote_changes:
            local = self.local_db.get(remote_change["_id"])
            
            if local:
                # Check if both changed since base
                if (local.get("_rev") != remote_change.get("_rev") and
                    local.get("updated_at") > remote_change.get("updated_at", 0)):
                    # This is a conflict - both changed
                    # In practice, get base version for three-way merge
                    conflicts.append(Conflict(
                        record_id=local["_id"],
                        local_version=local,
                        remote_version=remote_change
                    ))
        
        return conflicts
    
    def _resolve_conflicts(self, conflicts: List[Conflict]) -> List[dict]:
        """Resolve conflicts using configured strategy"""
        resolver = ConflictResolver(ConflictResolutionStrategy.LAST_WRITE_WINS)
        return resolver.resolve(conflicts)
    
    async def _apply_resolved(self, resolved: List[dict]):
        """Apply resolved changes to local database"""
        for doc in resolved:
            self.local_db.put(doc)
```

## Offline-First Application Patterns

### Building Offline-First Applications

Offline-first architecture assumes connectivity is intermittent and designs for it:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import queue
import threading

@dataclass
class OfflineOperation:
    """Represents an operation to be performed when online"""
    operation_id: str
    operation_type: str  # CREATE, UPDATE, DELETE
    table_name: str
    record_id: str
    data: Dict[str, Any]
    created_at: datetime
    retry_count: int = 0


class OfflineFirstManager:
    """
    Manages offline-first operations with automatic sync
    """
    
    def __init__(self, local_db, sync_service):
        self.local_db = local_db
        self.sync_service = sync_service
        self.operation_queue = queue.Queue()
        self.sync_in_progress = False
    
    def create_offline_operation(self, operation: OfflineOperation):
        """Queue an operation for later sync"""
        # Save to local database
        self.local_db.save_pending_operation(operation)
        
        # Add to queue
        self.operation_queue.put(operation)
        
        # Try immediate sync if online
        self._attempt_sync()
    
    def _attempt_sync(self):
        """Attempt to sync if not already in progress"""
        if self.sync_in_progress:
            return
        
        if not self._is_online():
            return
        
        self.sync_in_progress = True
        
        try:
            while not self.operation_queue.empty():
                operation = self.operation_queue.get()
                self._process_operation(operation)
        finally:
            self.sync_in_progress = False
    
    def _process_operation(self, operation: OfflineOperation):
        """Process a single operation"""
        try:
            # Send to remote
            self.sync_service.execute(operation)
            
            # Mark as synced
            self.local_db.mark_operation_synced(operation.operation_id)
            
        except Exception as e:
            operation.retry_count += 1
            
            if operation.retry_count < 3:
                # Retry later
                self.operation_queue.put(operation)
            else:
                # Mark as failed, require manual intervention
                self.local_db.mark_operation_failed(
                    operation.operation_id, 
                    str(e)
                )
    
    def _is_online(self) -> bool:
        """Check if we have network connectivity"""
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False


# Example: Offline-first task management
class TaskManager:
    """Task manager with offline-first capabilities"""
    
    def __init__(self, db, sync_manager: OfflineFirstManager):
        self.db = db
        self.sync_manager = sync_manager
    
    def create_task(self, title: str, description: str = "") -> str:
        """Create a task (works offline)"""
        import uuid
        
        task_id = str(uuid.uuid4())
        task = {
            "_id": task_id,
            "type": "task",
            "title": title,
            "description": description,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "_sync_status": "pending"
        }
        
        # Save locally first
        self.db.save(task)
        
        # Queue for sync
        self.sync_manager.create_offline_operation(OfflineOperation(
            operation_id=str(uuid.uuid4()),
            operation_type="CREATE",
            table_name="tasks",
            record_id=task_id,
            data=task,
            created_at=datetime.utcnow()
        ))
        
        return task_id
    
    def get_tasks(self, status: str = None) -> List[dict]:
        """Get tasks (from local database)"""
        if status:
            return self.db.query(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC",
                (status,)
            )
        return self.db.query(
            "SELECT * FROM tasks ORDER BY created_at DESC"
        )
    
    def complete_task(self, task_id: str):
        """Mark task as complete"""
        task = self.db.get(task_id)
        task["status"] = "completed"
        task["updated_at"] = datetime.utcnow().isoformat()
        task["_sync_status"] = "pending"
        
        self.db.save(task)
        
        # Queue for sync
        self.sync_manager.create_offline_operation(OfflineOperation(
            operation_id=str(uuid.uuid4()),
            operation_type="UPDATE",
            table_name="tasks",
            record_id=task_id,
            data=task,
            created_at=datetime.utcnow()
        ))
```

### Real-Time Sync with Change Tracking

Implement real-time sync when connectivity is available:

```python
import asyncio
import json
from typing import Callable, Dict, Any

class ChangeTracker:
    """
    Tracks local changes for efficient synchronization
    """
    
    def __init__(self, db):
        self.db = db
        self._setup_tracking()
    
    def _setup_tracking(self):
        """Set up change tracking infrastructure"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS _change_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                old_data TEXT,
                new_data TEXT,
                timestamp INTEGER NOT NULL,
                synced INTEGER DEFAULT 0
            )
        """)
    
    def record_change(
        self, 
        table_name: str, 
        record_id: str, 
        operation: str,
        old_data: Dict = None,
        new_data: Dict = None
    ):
        """Record a change"""
        import time
        
        self.db.execute(
            """INSERT INTO _change_log 
               (table_name, record_id, operation, old_data, new_data, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                table_name,
                record_id,
                operation,
                json.dumps(old_data) if old_data else None,
                json.dumps(new_data) if new_data else None,
                int(time.time())
            )
        )
    
    def get_unsynced_changes(self, limit: int = 1000) -> list:
        """Get changes that need to be synced"""
        return self.db.query(
            """SELECT * FROM _change_log 
               WHERE synced = 0 
               ORDER BY timestamp ASC 
               LIMIT ?""",
            (limit,)
        )
    
    def mark_synced(self, change_ids: list):
        """Mark changes as synced"""
        if not change_ids:
            return
        
        placeholders = ",".join("?" * len(change_ids))
        self.db.execute(
            f"UPDATE _change_log SET synced = 1 WHERE id IN ({placeholders})",
            change_ids
        )


class RealTimeSync:
    """
    Real-time synchronization with continuous change tracking
    """
    
    def __init__(
        self, 
        change_tracker: ChangeTracker,
        remote_endpoint: str,
        batch_size: int = 100
    ):
        self.change_tracker = change_tracker
        self.remote_endpoint = remote_endpoint
        self.batch_size = batch_size
        self.running = False
    
    async def start_sync_loop(self, interval_seconds: int = 5):
        """Start continuous sync loop"""
        self.running = True
        
        while self.running:
            try:
                await self._sync_once()
            except Exception as e:
                print(f"Sync error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def _sync_once(self):
        """Perform one sync iteration"""
        # Get unsynced changes
        changes = self.change_tracker.get_unsynced_changes(self.batch_size)
        
        if not changes:
            return
        
        # Batch changes for remote
        batches = self._create_batches(changes)
        
        for batch in batches:
            success = await self._send_batch(batch)
            
            if success:
                # Mark as synced
                self.change_tracker.mark_synced([c["id"] for c in batch])
    
    def _create_batches(self, changes: list) -> list:
        """Create batches of changes"""
        batches = []
        
        for i in range(0, len(changes), self.batch_size):
            batches.append(changes[i:i + self.batch_size])
        
        return batches
    
    async def _send_batch(self, batch: list) -> bool:
        """Send batch to remote (simplified)"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.remote_endpoint}/sync",
                    json={"changes": batch}
                ) as response:
                    return response.status == 200
            except Exception as e:
                print(f"Batch send failed: {e}")
                return False
    
    def stop(self):
        """Stop sync loop"""
        self.running = False
```

## IoT Database Considerations

### Time-Series Data from IoT Devices

IoT applications generate continuous streams of time-series data:

```python
from datetime import datetime, timedelta
from typing import List, Dict, Any
import statistics

class IoTDataManager:
    """
    Manages IoT sensor data with efficient storage and queries
    """
    
    def __init__(self, db):
        self.db = db
        self._setup_schema()
    
    def _setup_schema(self):
        """Set up optimized schema for IoT data"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS iot_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL NOT NULL,
                quality TEXT DEFAULT 'good',
                timestamp INTEGER NOT NULL
            );
            
            -- Partition by day for efficient queries and maintenance
            CREATE INDEX idx_iot_device_time 
            ON iot_readings(device_id, timestamp);
            
            CREATE INDEX idx_iot_type_time 
            ON iot_readings(sensor_type, timestamp);
            
            -- Metadata table for devices
            CREATE TABLE IF NOT EXISTS iot_devices (
                device_id TEXT PRIMARY KEY,
                device_type TEXT,
                location TEXT,
                last_seen INTEGER,
                metadata_json TEXT
            );
        """)
    
    def insert_reading(
        self, 
        device_id: str, 
        sensor_type: str, 
        value: float,
        timestamp: int = None,
        quality: str = "good"
    ):
        """Insert sensor reading"""
        if timestamp is None:
            timestamp = int(datetime.utcnow().timestamp())
        
        self.db.execute(
            """INSERT INTO iot_readings 
               (device_id, sensor_type, value, quality, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (device_id, sensor_type, value, quality, timestamp)
        )
        
        # Update device last_seen
        self.db.execute(
            """INSERT OR REPLACE INTO iot_devices (device_id, last_seen)
               VALUES (?, ?)""",
            (device_id, timestamp)
        )
    
    def insert_batch(self, readings: List[Dict[str, Any]]):
        """Insert multiple readings efficiently"""
        self.db.executemany(
            """INSERT INTO iot_readings 
               (device_id, sensor_type, value, quality, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    r["device_id"],
                    r["sensor_type"],
                    r["value"],
                    r.get("quality", "good"),
                    r.get("timestamp", int(datetime.utcnow().timestamp()))
                )
                for r in readings
            ]
        )
    
    def get_device_readings(
        self, 
        device_id: str, 
        start_time: int, 
        end_time: int
    ) -> List[dict]:
        """Get readings for a device in time range"""
        return self.db.query(
            """SELECT * FROM iot_readings 
               WHERE device_id = ? AND timestamp BETWEEN ? AND ?
               ORDER BY timestamp ASC""",
            (device_id, start_time, end_time)
        )
    
    def get_aggregated_readings(
        self,
        device_id: str,
        sensor_type: str,
        start_time: int,
        end_time: int,
        interval_seconds: int = 3600
    ) -> List[dict]:
        """Get aggregated readings over time intervals"""
        return self.db.query(f"""
            SELECT 
                (timestamp / {interval_seconds}) * {interval_seconds} as interval_start,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as sample_count
            FROM iot_readings
            WHERE device_id = ?
              AND sensor_type = ?
              AND timestamp BETWEEN ? AND ?
            GROUP BY interval_start
            ORDER BY interval_start
        """, (device_id, sensor_type, start_time, end_time))
    
    def detect_anomalies(
        self, 
        device_id: str, 
        sensor_type: str, 
        window_minutes: int = 60,
        std_threshold: float = 3.0
    ) -> List[dict]:
        """Detect anomalous readings using standard deviation"""
        end_time = int(datetime.utcnow().timestamp())
        start_time = int((datetime.utcnow() - timedelta(minutes=window_minutes)).timestamp())
        
        # Get recent readings
        readings = self.get_device_readings(device_id, start_time, end_time)
        
        if len(readings) < 10:
            return []
        
        # Calculate statistics
        values = [r["value"] for r in readings]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        # Find anomalies
        anomalies = []
        for reading in readings:
            z_score = abs((reading["value"] - mean) / std) if std > 0 else 0
            
            if z_score > std_threshold:
                anomalies.append({
                    "reading": reading,
                    "z_score": z_score,
                    "mean": mean,
                    "std": std
                })
        
        return anomalies
    
    def get_device_statistics(self) -> List[dict]:
        """Get summary statistics for all devices"""
        return self.db.query("""
            SELECT 
                device_id,
                MIN(timestamp) as first_reading,
                MAX(timestamp) as last_reading,
                COUNT(*) as total_readings,
                AVG(value) as avg_value
            FROM iot_readings
            GROUP BY device_id
        """)
```

### Data Retention Strategies

Implement appropriate data retention for IoT scenarios:

```python
class IoTDataRetention:
    """
    Manages data retention policies for IoT data
    """
    
    def __init__(self, db):
        self.db = db
    
    def apply_retention_policy(
        self,
        retention_policies: Dict[str, int]
    ):
        """
        Apply retention policies to tables
        
        Args:
            retention_policies: Dict mapping table to retention period in days
        """
        for table, retention_days in retention_policies.items():
            cutoff = int(
                (datetime.utcnow() - timedelta(days=retention_days)).timestamp()
            )
            
            deleted = self.db.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (cutoff,)
            )
            
            print(f"Deleted {deleted.rowcount} records from {table}")
    
    def create_aggregates_before_delete(
        self,
        sensor_type: str,
        retention_days: int = 90
    ):
        """Create hourly aggregates before deleting raw data"""
        cutoff = int(
            (datetime.utcnow() - timedelta(days=retention_days)).timestamp()
        )
        
        # Create aggregates table if not exists
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS iot_aggregates (
                device_id TEXT,
                sensor_type TEXT,
                hour INTEGER,
                avg_value REAL,
                min_value REAL,
                max_value REAL,
                sample_count INTEGER,
                PRIMARY KEY (device_id, sensor_type, hour)
            )
        """)
        
        # Create aggregates from raw data
        self.db.execute(f"""
            INSERT OR REPLACE INTO iot_aggregates
            SELECT 
                device_id,
                sensor_type,
                (timestamp / 3600) * 3600 as hour,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as sample_count
            FROM iot_readings
            WHERE sensor_type = ? AND timestamp < ?
            GROUP BY device_id, sensor_type, hour
        """, (sensor_type, cutoff))
    
    def run_maintenance(self):
        """Run all maintenance tasks"""
        # Apply retention to raw data, keep 30 days
        self.apply_retention_policy({
            "iot_readings": 30,
            "iot_devices": 365
        })
        
        # Create aggregates before deleting old data
        self.create_aggregates_before_delete("temperature", 30)
        self.create_aggregates_before_delete("humidity", 30)
        
        # Vacuum to reclaim space
        self.db.execute("VACUUM")
```

## Production Considerations

### Monitoring Edge Deployments

Monitor edge databases effectively:

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

@dataclass
class EdgeMetrics:
    """Metrics for edge database monitoring"""
    db_path: str
    total_readings: int
    pending_sync: int
    storage_bytes: int
    last_sync: datetime
    device_count: int
    error_count: int


class EdgeMonitoring:
    """Monitor edge database health and sync status"""
    
    def __init__(self, db, sync_endpoint: str = None):
        self.db = db
        self.sync_endpoint = sync_endpoint
    
    def collect_metrics(self) -> EdgeMetrics:
        """Collect current metrics"""
        return EdgeMetrics(
            db_path=str(self.db.db_path),
            total_readings=self._get_reading_count(),
            pending_sync=self._get_pending_sync_count(),
            storage_bytes=self._get_storage_size(),
            last_sync=self._get_last_sync_time(),
            device_count=self._get_device_count(),
            error_count=self._get_error_count()
        )
    
    def check_health(self) -> Dict:
        """Check overall health status"""
        metrics = self.collect_metrics()
        
        issues = []
        
        # Check sync lag
        if metrics.pending_sync > 1000:
            issues.append({
                "severity": "high",
                "issue": "Large sync queue",
                "detail": f"{metrics.pending_sync} records pending sync"
            })
        
        # Check storage
        if metrics.storage_bytes > 10 * 1024 * 1024 * 1024:  # 10GB
            issues.append({
                "severity": "medium",
                "issue": "High storage usage",
                "detail": f"Database size: {metrics.storage_bytes / 1024**3:.1f} GB"
            })
        
        # Check last sync
        if metrics.last_sync:
            sync_age = (datetime.utcnow() - metrics.last_sync).total_seconds()
            if sync_age > 3600:  # 1 hour
                issues.append({
                    "severity": "high",
                    "issue": "Sync not recent",
                    "detail": f"Last sync: {sync_age/60:.0f} minutes ago"
                })
        
        return {
            "healthy": len(issues) == 0,
            "metrics": metrics,
            "issues": issues
        }
    
    def _get_reading_count(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM iot_readings")[0][0]
    
    def _get_pending_sync_count(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM pending_sync")[0][0]
    
    def _get_storage_size(self) -> int:
        import os
        return os.path.getsize(self.db.db_path)
    
    def _get_last_sync_time(self) -> datetime:
        # Implementation depends on sync tracking
        return None
    
    def _get_device_count(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM iot_devices")[0][0]
    
    def _get_error_count(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM sync_errors")[0][0]
```

### Security Considerations

Secure edge deployments:

```python
class EdgeSecurity:
    """Security for edge databases"""
    
    @staticmethod
    def encrypt_database(db_path: str, key: str):
        """Enable encryption for SQLite database"""
        import subprocess
        # Use SQLCipher or similar for encryption
        # This is conceptual - actual implementation varies
        pass
    
    @staticmethod
    def secure_connection(endpoint: str) -> bool:
        """Verify secure connection to remote"""
        import ssl
        context = ssl.create_default_context()
        return True  # Verify certificate
    
    @staticmethod
    def audit_access(db):
        """Set up access auditing"""
        db.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT,
                table_name TEXT,
                timestamp INTEGER,
                details TEXT
            )
        """)
```

## Best Practices Summary

1. **Design for Offline-First**: Assume connectivity is intermittent. All operations should work offline with later synchronization.

2. **Choose Appropriate Database**: SQLite for simple cases, specialized databases for complex requirements. Match database to workload.

3. **Implement Robust Sync**: Handle conflicts with clear resolution strategies. Monitor sync health and alert on failures.

4. **Optimize for Resources**: Edge devices have limited resources. Use appropriate data types, compression, and retention policies.

5. **Secure Edge Data**: Encrypt at rest, use secure connections, validate data from untrusted sources.

6. **Monitor Comprehensively**: Track local operations, sync status, storage usage, and error rates.

7. **Plan for Scale**: Design data models and sync strategies that work when the device fleet grows.

Edge computing represents a fundamental shift in how we build applications. By understanding these patterns and techniques, you can build applications that operate effectively in distributed, intermittently connected environments while maintaining data consistency and application reliability.

## See Also

- [SQLite Deep Dive](../02_core_concepts/sqlite_deep_dive.md) - SQLite optimization for edge
- [Real-time Streaming Patterns](../02_intermediate/05_realtime_streaming_database_patterns.md) - Synchronization strategies
- [Database Selection Framework](../01_foundations/database_selection_framework.md) - Choosing databases for edge
- [Database Troubleshooting](../04_tutorials/troubleshooting/02_database_troubleshooting_debugging.md) - Debugging edge deployments

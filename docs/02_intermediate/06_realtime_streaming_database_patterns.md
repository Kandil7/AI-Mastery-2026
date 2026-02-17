# Real-Time and Streaming Database Patterns

## Overview

Modern applications increasingly require real-time data processing capabilities. From monitoring dashboards showing live metrics to notification systems alerting users of important events, the ability to process and serve data as it arrives has become a fundamental requirement. This document explores the patterns, technologies, and implementation strategies for building real-time and streaming database systems.

The shift from batch to streaming represents a fundamental change in how applications process data. Rather than periodically running queries to refresh data, streaming patterns process events as they occur, enabling immediate responses and always-current views. This approach introduces new challenges around handling continuous data flows, managing state, and ensuring exactly-once processing semantics.

This guide covers the architectural patterns for building streaming systems, implementation techniques using modern databases and frameworks, and production considerations for deploying real-time data pipelines. The examples demonstrate practical implementations using PostgreSQL, Kafka, and related technologies, with explanations that apply broadly across different technology stacks.

## Change Data Capture (CDC)

### Understanding CDC

Change Data Capture is a pattern for identifying and capturing changes made to data in a database, then delivering those changes to downstream systems in real-time. CDC enables architectures where multiple systems stay synchronized without tight coupling between them.

Traditional approaches to data synchronization involved periodic exports, bulk loads, or polling for changes. These approaches suffer from latency (data is always somewhat stale), load (full scans impact performance), and complexity (handling incremental updates is difficult). CDC solves these problems by capturing changes at the source as they occur.

CDC works by reading the database's transaction log, which records all modifications. Modern databases including PostgreSQL, MySQL, and MongoDB support logical replication that exposes these changes in a structured format. Applications can consume these changes and propagate them to other systems.

### PostgreSQL CDC Implementation

PostgreSQL's logical decoding feature provides a robust CDC foundation:

```python
import asyncio
import json
import psycopg2
from psycopg2 import sql
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Callable
import threading

@dataclass
class ChangeEvent:
    """Represents a database change"""
    operation: str  # INSERT, UPDATE, DELETE
    table_name: str
    old_data: dict
    new_data: dict
    commit_timestamp: datetime
    transaction_id: int
    lsn: str  # Log Sequence Number for exactly-once processing

class PostgreSQLCDC:
    """
    PostgreSQL Change Data Capture using logical replication
    """
    
    def __init__(
        self,
        dsn: str,
        publication: str = "cdc_publication",
        slot_name: str = "cdc_slot"
    ):
        self.dsn = dsn
        self.publication = publication
        self.slot_name = slot_name
        self.replication_conn = None
        self.consumer_thread = None
        self.running = False
        self.handlers: List[Callable] = []
    
    def initialize(self):
        """Set up logical replication slot and publication"""
        conn = psycopg2.connect(self.dsn)
        try:
            # Create publication for all tables (or specify specific tables)
            conn.autocommit = True
            cur = conn.cursor()
            
            # Create publication
            cur.execute(f"""
                CREATE PUBLICATION {self.publication} 
                FOR ALL TABLES
            """)
            
            # Create replication slot with output plugin
            cur.execute(f"""
                CREATE_REPLICATION_SLOT {self.slot_name} 
                LOGICAL pgoutput
            """)
            
            print(f"CDC initialized: publication={self.publication}, slot={self.slot_name}")
        
        except psycopg2.errors.DuplicateObject:
            print("Publication and slot already exist")
        finally:
            conn.close()
    
    def add_handler(self, handler: Callable[[ChangeEvent], None]):
        """Add a handler for change events"""
        self.handlers.append(handler)
    
    def start(self):
        """Start consuming changes"""
        self.running = True
        
        # Connect using replication protocol
        self.replication_conn = psycopg2.connect(
            self.dsn,
            connection_factory=psycopg2.extras.LogicalReplicationConnection
        )
        
        self.consumer_thread = threading.Thread(
            target=self._consume_changes,
            daemon=True
        )
        self.consumer_thread.start()
    
    def _consume_changes(self):
        """Consume replication stream"""
        cursor = self.replication_conn.cursor()
        
        try:
            # Start streaming from the slot
            cursor.start_replication(
                slot_name=self.slot_name,
                start_pos=None,  # Start from beginning of slot
                options={
                    "format": "json",
                    "include_transaction": False
                }
            )
            
            while self.running:
                # Consume next message
                msg = cursor.read_message()
                if msg:
                    self._process_message(msg)
                else:
                    # No message available, sleep briefly
                    cursor.consume_stream()
        
        except Exception as e:
            print(f"CDC error: {e}")
            self.running = False
    
    def _process_message(self, msg):
        """Process a replication message"""
        if msg.data:
            data = json.loads(msg.data)
            
            # Handle different message types
            if data.get("action") == "I":  # INSERT
                event = ChangeEvent(
                    operation="INSERT",
                    table_name=data.get("table"),
                    old_data={},
                    new_data=data.get("data", {}),
                    commit_timestamp=datetime.utcnow(),
                    transaction_id=0,
                    lsn=str(msg.data_start)
                )
            elif data.get("action") == "U":  # UPDATE
                event = ChangeEvent(
                    operation="UPDATE",
                    table_name=data.get("table"),
                    old_data=data.get("old", {}),
                    new_data=data.get("data", {}),
                    commit_timestamp=datetime.utcnow(),
                    transaction_id=0,
                    lsn=str(msg.data_start)
                )
            elif data.get("action") == "D":  # DELETE
                event = ChangeEvent(
                    operation="DELETE",
                    table_name=data.get("table"),
                    old_data=data.get("old", {}),
                    new_data={},
                    commit_timestamp=datetime.utcnow(),
                    transaction_id=0,
                    lsn=str(msg.data_start)
                )
            else:
                return
            
            # Call all handlers
            for handler in self.handlers:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Handler error: {e}")
    
    def stop(self):
        """Stop consuming changes"""
        self.running = False
        if self.replication_conn:
            self.replication_conn.close()

# Usage example
def handle_change(event: ChangeEvent):
    """Example event handler"""
    print(f"{event.operation} on {event.table_name}: {event.new_data}")

# cdc = PostgreSQLCDC("postgresql://user:pass@localhost/mydb")
# cdc.initialize()
# cdc.add_handler(handle_change)
# cdc.start()
```

### MySQL CDC with Debezium

For MySQL, Debezium provides a mature CDC solution that integrates with Kafka:

```yaml
# Docker Compose configuration for Debezium with MySQL
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: inventory
      MYSQL_USER: mysqluser
      MYSQL_PASSWORD: mysqlpass
    command: 
      - --server-id=1
      - --log-bin=mysql-bin
      - --binlog-format=ROW
      - --binlog-row-image=FULL
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

  debezium:
    image: debezium/connect:2.4
    ports:
      - "8083:8083"
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: debezium-group
      CONFIG_STORAGE_TOPIC: debezium_configs
      OFFSET_STORAGE_TOPIC: debezium_offsets
      STATUS_STORAGE_TOPIC: debezium_status
      CONFIG_STORAGE_REPLICATION_FACTOR: 1
      OFFSET_STORAGE_REPLICATION_FACTOR: 1
      STATUS_STORAGE_REPLICATION_FACTOR: 1
    depends_on:
      - kafka

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk

volumes:
  mysql_data:
```

After setting up the infrastructure, configure Debezium to capture MySQL changes:

```json
{
  "name": "mysql-source-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql",
    "database.port": "3306",
    "database.user": "mysqluser",
    "database.password": "mysqlpass",
    "database.server.id": "1",
    "database.server.name": "inventory",
    "database.include.list": "inventory",
    "table.include.list": "inventory.customers,inventory.orders,inventory.products",
    "database.history.kafka.bootstrap.servers": "kafka:9092",
    "database.history.kafka.topic": "schema-changes.inventory",
    "include.schema.changes": "true",
    "transforms": "unwrap",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter"
  }
}
```

## Database Streaming and Event-Driven Architecture

### Event-Driven Patterns

Event-driven architecture (EDA) uses events to trigger actions and communicate between services. When combined with database capabilities, EDA enables powerful real-time applications that respond to data changes immediately.

The core pattern involves three components:

**Event Producers**: Services or database triggers that generate events when meaningful actions occur.

**Event Transport**: Infrastructure that reliably delivers events to consumers. Kafka, RabbitMQ, or database-native features can serve this role.

**Event Consumers**: Services that react to events, updating their state or triggering further actions.

```python
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Callable
from dataclasses import dataclass, asdict

@dataclass
class DomainEvent:
    """Base class for domain events"""
    event_type: str
    aggregate_id: str
    timestamp: datetime
    payload: dict
    metadata: dict = None
    
    def to_json(self) -> str:
        return json.dumps({
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata or {}
        })

class EventStore:
    """
    Persistent event store for event sourcing and event-driven architecture
    """
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def initialize(self):
        """Create event store tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id BIGSERIAL PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    aggregate_id VARCHAR(255) NOT NULL,
                    aggregate_type VARCHAR(100),
                    payload JSONB NOT NULL,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                )
            """)
            
            await conn.execute("""
                CREATE INDEX idx_events_aggregate 
                ON events (aggregate_id, version)
            """)
            
            await conn.execute("""
                CREATE INDEX idx_events_type 
                ON events (event_type, timestamp)
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id BIGSERIAL PRIMARY KEY,
                    subscriber_id VARCHAR(255) NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    handler JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
    
    async def append(
        self, 
        event: DomainEvent, 
        expected_version: int = None
    ) -> int:
        """Append event to the store"""
        async with self.db_pool.acquire() as conn:
            # Check version if provided (optimistic concurrency)
            if expected_version:
                existing = await conn.fetchval("""
                    SELECT MAX(version) FROM events 
                    WHERE aggregate_id = $1
                """, event.aggregate_id)
                
                if existing and existing != expected_version:
                    raise ConcurrencyError(
                        f"Version mismatch: expected {expected_version}, found {existing}"
                    )
            
            # Get next version
            next_version = await conn.fetchval("""
                SELECT COALESCE(MAX(version), 0) + 1 
                FROM events 
                WHERE aggregate_id = $1
            """, event.aggregate_id)
            
            # Insert event
            event_id = await conn.fetchval("""
                INSERT INTO events (event_type, aggregate_id, aggregate_type, 
                                   payload, metadata, timestamp, version)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """,
                event.event_type,
                event.aggregate_id,
                event.payload.get("_type"),
                json.dumps(event.payload),
                json.dumps(event.metadata) if event.metadata else None,
                event.timestamp,
                next_version
            )
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            return event_id
    
    async def _notify_subscribers(self, event: DomainEvent):
        """Notify all subscribers for this event type"""
        async with self.db_pool.acquire() as conn:
            subscriptions = await conn.fetch("""
                SELECT * FROM subscriptions WHERE event_type = $1
            """, event.event_type)
            
            for sub in subscriptions:
                # In production, this would queue or publish to a message system
                await self._deliver_event(sub, event)
    
    async def _deliver_event(self, subscription: dict, event: DomainEvent):
        """Deliver event to a specific subscriber"""
        # Implementation depends on delivery mechanism
        print(f"Delivering {event.event_type} to {subscription['subscriber_id']}")

class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails"""
    pass
```

### Materialized Views for Real-Time Data

Materialized views provide pre-computed query results that can be refreshed on demand or continuously. This pattern is essential for real-time applications that need fast query responses:

```python
class MaterializedViewManager:
    """
    Manages materialized views with automatic refresh
    """
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def create_order_summary_view(self):
        """Create materialized view for order analytics"""
        async with self.db_pool.acquire() as conn:
            # Create materialized view
            await conn.execute("""
                CREATE MATERIALIZED VIEW order_summary AS
                SELECT 
                    DATE_TRUNC('day', o.created_at) AS day,
                    o.customer_id,
                    c.name AS customer_name,
                    COUNT(o.id) AS order_count,
                    SUM(o.total_amount) AS total_revenue,
                    AVG(o.total_amount) AS avg_order_value
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                GROUP BY DATE_TRUNC('day', o.created_at), o.customer_id, c.name
                WITH DATA
            """)
            
            # Create index for common queries
            await conn.execute("""
                CREATE INDEX idx_order_summary_day 
                ON order_summary (day)
            """)
            
            await conn.execute("""
                CREATE INDEX idx_order_summary_customer 
                ON order_summary (customer_id)
            """)
    
    async def refresh_view(self, view_name: str, concurrently: bool = True):
        """Refresh materialized view"""
        async with self.db_pool.acquire() as conn:
            if concurrently:
                # Concurrent refresh allows reads during refresh
                await conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view_name}")
            else:
                await conn.execute(f"REFRESH MATERIALIZED VIEW {view_name}")
    
    async def incremental_refresh_order_summary(self):
        """Incrementally refresh order summary based on recent changes"""
        async with self.db_pool.acquire() as conn:
            # Get the last refresh time
            last_refresh = await conn.fetchval("""
                SELECT MAX(refresh_time) 
                FROM materialized_view_refreshes 
                WHERE view_name = 'order_summary'
            """)
            
            if not last_refresh:
                # Full refresh if no previous refresh
                await self.refresh_view("order_summary", concurrently=True)
            else:
                # Incremental refresh using last refresh timestamp
                await conn.execute("""
                    INSERT INTO order_summary
                    SELECT 
                        DATE_TRUNC('day', o.created_at) AS day,
                        o.customer_id,
                        c.name AS customer_name,
                        COUNT(o.id) AS order_count,
                        SUM(o.total_amount) AS total_revenue,
                        AVG(o.total_amount) AS avg_order_value
                    FROM orders o
                    JOIN customers c ON o.customer_id = c.id
                    WHERE o.updated_at > $1
                    GROUP BY DATE_TRUNC('day', o.created_at), o.customer_id, c.name
                    ON CONFLICT (day, customer_id) DO UPDATE SET
                        order_count = EXCLUDED.order_count,
                        total_revenue = EXCLUDED.total_revenue,
                        avg_order_value = EXCLUDED.avg_order_value
                """, last_refresh)
                
                # Record refresh time
                await conn.execute("""
                    INSERT INTO materialized_view_refreshes (view_name, refresh_time)
                    VALUES ('order_summary', NOW())
                """)
```

## Kafka Connect and Database Integration

### Kafka Connect Fundamentals

Kafka Connect provides a scalable framework for integrating databases with Kafka. It provides pre-built connectors for common databases and a framework for building custom connectors. The architecture consists of:

**Connectors**: Plugins that define how to integrate with specific systems. Source connectors read from external systems and produce to Kafka topics. Sink connectors consume from Kafka topics and write to external systems.

**Workers**: The processes that execute connectors. Standalone workers run in a single process. Distributed workers run in a cluster for scalability and fault tolerance.

**Converters**: Handle serialization and deserialization of data. Common converters include JSON, Avro, and Protocol Buffers.

### JDBC Source Connector for Database CDC

The JDBC source connector can poll for changes, though for true CDC, logical replication (as shown earlier) is preferred:

```json
{
  "name": "jdbc-source-connector",
  "config": {
    "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector",
    "connection.url": "jdbc:postgresql://localhost:5432/mydb",
    "connection.user": "postgres",
    "connection.password": "password",
    "topic.prefix": "db-",
    "mode": "timestamp",
    "timestamp.column.name": "updated_at",
    "validate.non.null": false,
    "query": "SELECT id, name, email, created_at, updated_at FROM users WHERE updated_at > ${offset}",
    "poll.interval.ms": "1000",
    "batch.max.rows": "1000",
    "topics": "db-users"
  }
}
```

For more sophisticated CDC, use Debezium as shown earlier, which reads the transaction log directly.

### Kafka Connect Sink Connectors

Sink connectors write from Kafka topics to databases or other systems:

```json
{
  "name": "jdbc-sink-connector",
  "config": {
    "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
    "connection.url": "jdbc:postgresql://analytics-db:5432/analytics",
    "connection.user": "analytics",
    "connection.password": "password",
    "topics": "orders,customers,products",
    "table.name.format": "${topic}",
    "pk.mode": "record_value",
    "pk.fields": "id",
    "auto.create": true,
    "auto.evolve": true,
    "insert.mode": "upsert",
    "fields.whitelist": "id,name,price,category,updated_at"
  }
}
```

### Custom Kafka Connector for Advanced Scenarios

For complex CDC scenarios, implement a custom Kafka connector:

```python
from typing import Dict, Any, Optional
import json
import threading
import time

class DatabaseSourceConnector:
    """
    Custom Kafka Connect source connector for database CDC
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.worker_thread = None
        self.offset = None
    
    def version(self) -> str:
        return "1.0.0"
    
    def start(self, props: Dict[str, Any]):
        """Initialize the connector"""
        self.config = props
        self.running = True
        self.offset = self._load_offset()
        
        self.worker_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True
        )
        self.worker_thread.start()
    
    def _poll_loop(self):
        """Main polling loop"""
        db_config = self.config["database"]
        
        while self.running:
            try:
                # Poll for changes
                changes = self._fetch_changes()
                
                for change in changes:
                    # Create Kafka record
                    record = self._create_record(change)
                    
                    # Send to Kafka (simplified)
                    self._send_to_kafka(record)
                    
                    # Update offset
                    self.offset = change["lsn"]
                    self._save_offset()
            
            except Exception as e:
                print(f"Poll error: {e}")
                time.sleep(5)  # Back off on error
    
    def stop(self):
        """Stop the connector"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
    
    def _fetch_changes(self) -> list:
        """Fetch changes from database"""
        # Implementation depends on database
        # For PostgreSQL, use pg_logical_slot_get_changes
        pass
    
    def _create_record(self, change: dict) -> dict:
        """Create Kafka record from database change"""
        return {
            "topic": f"cdc-{change['table']}",
            "key": str(change.get("id")),
            "value": json.dumps(change),
            "timestamp": int(time.time() * 1000)
        }
    
    def _send_to_kafka(self, record: dict):
        """Send record to Kafka"""
        pass
    
    def _load_offset(self) -> Optional[Any]:
        """Load offset from offset storage"""
        pass
    
    def _save_offset(self):
        """Save current offset"""
        pass

# Register the connector with Kafka Connect
# This requires implementing the full Connect API
```

## Real-Time Analytics with Databases

### Streaming Aggregations

Real-time analytics require continuous computation over streaming data:

```python
import asyncio
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class StreamingAggregation:
    """
    In-memory streaming aggregation for real-time analytics
    """
    window_size_seconds: int
    max_events: int = 10000
    aggregations: dict = field(default_factory=dict)
    _events: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    def add_event(self, event: dict):
        """Add an event and update aggregations"""
        self._events.append({
            "data": event,
            "timestamp": datetime.utcnow()
        })
        
        self._evict_old_events()
        self._update_aggregations(event)
    
    def _evict_old_events(self):
        """Remove events outside the window"""
        cutoff = datetime.utcnow() - timedelta(seconds=self.window_size_seconds)
        
        while self._events and self._events[0]["timestamp"] < cutoff:
            old_event = self._events.popleft()
            self._remove_from_aggregations(old_event["data"])
    
    def _update_aggregations(self, event: dict):
        """Update running aggregations with new event"""
        # Count aggregations
        for key in ["category", "status", "region"]:
            if key in event:
                value = event[key]
                self.aggregations.setdefault(f"count_{key}", {})
                self.aggregations[f"count_{key}"][value] = \
                    self.aggregations[f"count_{key}"].get(value, 0) + 1
        
        # Sum aggregations
        if "amount" in event:
            self.aggregations["sum_amount"] = \
                self.aggregations.get("sum_amount", 0) + event["amount"]
        
        if "quantity" in event:
            self.aggregations["sum_quantity"] = \
                self.aggregations.get("sum_quantity", 0) + event["quantity"]
    
    def _remove_from_aggregations(self, event: dict):
        """Remove event from running aggregations"""
        # Reverse the _update_aggregations operations
        pass
    
    def get_results(self) -> dict:
        """Get current aggregation results"""
        return {
            "event_count": len(self._events),
            "window_seconds": self.window_size_seconds,
            "aggregations": self.aggregations,
            "min_timestamp": self._events[0]["timestamp"].isoformat() if self._events else None,
            "max_timestamp": self._events[-1]["timestamp"].isoformat() if self._events else None
        }


class TimeSeriesAggregator:
    """
    Time-bucketed aggregation for metrics
    """
    
    def __init__(self, bucket_size_seconds: int = 60):
        self.bucket_size = timedelta(seconds=bucket_size_seconds)
        self.buckets: dict[datetime, dict] = {}
    
    def record(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Round down to bucket
        bucket_time = timestamp.replace(
            second=(timestamp.second // 60) * 60,
            microsecond=0
        )
        
        if bucket_time not in self.buckets:
            self.buckets[bucket_time] = {
                "count": 0,
                "sum": 0,
                "min": float('inf'),
                "max": float('-inf')
            }
        
        bucket = self.buckets[bucket_time]
        bucket["count"] += 1
        bucket["sum"] += value
        bucket["min"] = min(bucket["min"], value)
        bucket["max"] = max(bucket["max"], value)
    
    def get_buckets(
        self, 
        start: datetime = None, 
        end: datetime = None
    ) -> dict:
        """Get aggregated buckets within time range"""
        result = {}
        
        for bucket_time, data in self.buckets.items():
            if start and bucket_time < start:
                continue
            if end and bucket_time > end:
                continue
            
            result[bucket_time.isoformat()] = {
                "count": data["count"],
                "avg": data["sum"] / data["count"] if data["count"] > 0 else 0,
                "min": data["min"],
                "max": data["max"]
            }
        
        return result
```

### Real-Time Dashboard Backend

Putting together streaming, CDC, and materialized views for a real-time dashboard:

```python
class RealTimeDashboardBackend:
    """
    Backend for real-time dashboard using streaming + cached data
    """
    
    def __init__(
        self,
        db_pool,
        cdc_consumer,
        aggregation: StreamingAggregation
    ):
        self.db = db_pool
        self.cdc = cdc_consumer
        self.aggregation = aggregation
        self.cache = {}
        self.cache_ttl = 5  # seconds
    
    async def initialize(self):
        """Initialize dashboard backend"""
        # Set up CDC to update cache
        self.cdc.add_handler(self._on_data_change)
        
        # Pre-load initial data
        await self._load_initial_data()
        
        # Start background refresh
        asyncio.create_task(self._periodic_cache_refresh())
    
    async def _load_initial_data(self):
        """Load initial data into cache"""
        async with self.db.acquire() as conn:
            # Load summary metrics
            self.cache["summary"] = await conn.fetchrow("""
                SELECT 
                    COUNT(*) AS total_orders,
                    SUM(total_amount) AS total_revenue,
                    AVG(total_amount) AS avg_order_value
                FROM orders
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            # Load category breakdown
            self.cache["categories"] = await conn.fetch("""
                SELECT category, COUNT(*) AS count, SUM(amount) AS revenue
                FROM products p
                JOIN order_items oi ON p.id = oi.product_id
                JOIN orders o ON oi.order_id = o.id
                WHERE o.created_at > NOW() - INTERVAL '24 hours'
                GROUP BY category
            """)
            
            # Load timestamps
            self.cache["loaded_at"] = datetime.utcnow()
    
    async def _on_data_change(self, event: ChangeEvent):
        """Handle data change from CDC"""
        if event.table_name == "orders":
            if event.operation == "INSERT":
                self.aggregation.add_event({
                    "category": "order",
                    "amount": event.new_data.get("total_amount", 0)
                })
            
            # Invalidate cache
            self.cache.pop("summary", None)
            self.cache.pop("categories", None)
    
    async def _periodic_cache_refresh(self):
        """Periodically refresh stale cache"""
        while True:
            await asyncio.sleep(self.cache_ttl)
            
            now = datetime.utcnow()
            if "loaded_at" not in self.cache:
                await self._load_initial_data()
                continue
            
            cache_age = (now - self.cache["loaded_at"]).total_seconds()
            if cache_age > self.cache_ttl:
                await self._load_initial_data()
    
    async def get_dashboard_data(self) -> dict:
        """Get current dashboard data"""
        return {
            "summary": dict(self.cache.get("summary", {})) if "summary" in self.cache else None,
            "categories": [dict(r) for r in self.cache.get("categories", [])],
            "realtime": self.aggregation.get_results(),
            "timestamp": datetime.utcnow().isoformat()
        }
```

## Production Considerations

### Handling Backpressure

Streaming systems must handle scenarios where data arrives faster than it can be processed:

```python
import asyncio
from collections import deque
from datetime import datetime

class BackpressureHandler:
    """
    Handle backpressure in streaming systems
    """
    
    def __init__(self, max_queue_size: int = 10000, drop_policy: str = "tail"):
        self.max_queue_size = max_queue_size
        self.drop_policy = drop_policy  # "tail", "head", "error"
        self.queue = deque()
        self.dropped_count = 0
        self.total_processed = 0
    
    async def add(self, item):
        """Add item to the queue with backpressure handling"""
        if len(self.queue) >= self.max_queue_size:
            if self.drop_policy == "error":
                raise Exception("Backpressure: queue full")
            elif self.drop_policy == "tail":
                # Drop newest
                self.dropped_count += 1
                return
            elif self.drop_policy == "head":
                # Drop oldest
                self.queue.popleft()
                self.dropped_count += 1
        
        self.queue.append(item)
    
    async def process(self, processor, batch_size: int = 100):
        """Process items from queue"""
        batch = []
        
        for _ in range(min(batch_size, len(self.queue))):
            if self.queue:
                batch.append(self.queue.popleft())
        
        if batch:
            await processor(batch)
            self.total_processed += len(batch)
        
        return {
            "processed": len(batch),
            "queued": len(self.queue),
            "dropped": self.dropped_count,
            "total_processed": self.total_processed
        }
```

### Exactly-Once Semantics

Ensuring each event is processed exactly once is challenging but essential for critical applications:

```python
class ExactlyOnceProcessor:
    """
    Process events with exactly-once semantics
    """
    
    def __init__(self, db_pool, consumer_group: str):
        self.db = db_pool
        self.consumer_group = consumer_group
        self.processed_offsets = set()
    
    async def process_batch(self, events: list) -> list:
        """Process a batch of events exactly once"""
        results = []
        
        async with self.db.acquire() as conn:
            for event in events:
                offset = event.get("offset")
                
                # Skip if already processed
                if offset in self.processed_offsets:
                    results.append({
                        "event": event,
                        "status": "skipped_already_processed"
                    })
                    continue
                
                try:
                    # Process the event
                    await self._process_single_event(conn, event)
                    
                    # Mark as processed
                    self.processed_offsets.add(offset)
                    
                    # Persist offset
                    await self._save_offset(conn, offset)
                    
                    results.append({
                        "event": event,
                        "status": "processed"
                    })
                
                except Exception as e:
                    results.append({
                        "event": event,
                        "status": "error",
                        "error": str(e)
                    })
        
        return results
    
    async def _process_single_event(self, conn, event: dict):
        """Process a single event"""
        # Implementation depends on event type
        pass
    
    async def _save_offset(self, conn, offset: str):
        """Persist processed offset"""
        await conn.execute("""
            INSERT INTO processed_offsets (consumer_group, offset, processed_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (consumer_group, offset) DO NOTHING
        """, self.consumer_group, offset)
    
    async def initialize(self):
        """Load previously processed offsets"""
        async with self.db.acquire() as conn:
            offsets = await conn.fetch("""
                SELECT offset FROM processed_offsets 
                WHERE consumer_group = $1
            """, self.consumer_group)
            
            self.processed_offsets = {o["offset"] for o in offsets}
```

## Best Practices Summary

1. **Choose the right CDC approach**: Use logical replication for low-latency CDC; JDBC polling for simpler setups.

2. **Design for eventual consistency**: Real-time systems typically provide eventual consistency. Design UI and APIs accordingly.

3. **Implement proper monitoring**: Track event processing latency, queue depths, and error rates.

4. **Handle failures gracefully**: Implement retry logic with backoff, dead letter queues, and alerting.

5. **Consider exactly-once semantics**: For critical data, implement idempotent processing and offset tracking.

6. **Use materialized views wisely**: Pre-compute expensive aggregations but understand refresh costs.

7. **Test failure scenarios**: Regularly simulate failures to ensure recovery procedures work.

8. **Plan capacity**: Streaming systems can generate large volumes of data. Plan storage and processing capacity accordingly.

Real-time and streaming patterns enable powerful applications that respond to data immediately. By understanding these patterns and their trade-offs, you can build systems that deliver fresh data to users while managing the complexity of continuous data processing.

## See Also

- [Database Architecture Patterns](../03_system_design/database_architecture_patterns.md) - Event Sourcing and CQRS
- [Database Troubleshooting](../04_tutorials/troubleshooting/02_database_troubleshooting_debugging.md) - Debugging streaming systems
- [Database API Design](../02_intermediate/04_database_api_design.md) - API design for real-time data
- [Edge Computing Databases](../03_advanced/edge_computing_databases.md) - Edge synchronization patterns

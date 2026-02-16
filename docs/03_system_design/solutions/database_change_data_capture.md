# Database Change Data Capture

This document provides comprehensive guidance on implementing Change Data Capture (CDC) systems for databases. It covers architectural patterns, implementation guides using Debezium, event-driven database patterns, stream processing with database events, and building reactive data pipelines that respond to database changes in real-time.

## Table of Contents

1. [CDC Architecture Patterns](#1-cdc-architecture-patterns)
2. [Debezium Implementation Guide](#2-debezium-implementation-guide)
3. [Event-Driven Database Patterns](#3-event-driven-database-patterns)
4. [Stream Processing with Database Events](#4-stream-processing-with-database-events)
5. [Building Reactive Data Pipelines](#5-building-reactive-data-pipelines)

---

## 1. CDC Architecture Patterns

### 1.1 Understanding CDC Fundamentals

Change Data Capture is a design pattern that identifies and tracks changes to data in a database so that downstream systems can respond to those changes in real-time. CDC captures insert, update, and delete operations from the database transaction log, providing a reliable stream of data changes without impacting the performance of the source database. This approach is fundamentally different from polling-based change detection, which introduces latency and places load on the source database.

The primary advantages of CDC include near real-time data propagation with latency measured in milliseconds, zero impact on source database performance since changes are read from the transaction log, complete capture of all changes including deletes which polling might miss, preservation of change ordering at the transaction level, and ability to reconstruct current state by replaying the change history. These characteristics make CDC essential for modern event-driven architectures, data replication scenarios, and maintaining materialized views.

### 1.2 CDC Architecture Components

A complete CDC architecture consists of several integrated components that work together to capture, transform, and deliver database changes reliably. The transaction log serves as the source of truth, containing a record of all database modifications. The CDC connector reads this log and converts changes into a standardized format. A message broker provides durable storage and delivery semantics, while consumer applications process the change events for various purposes.

The connector component is responsible for reading the database transaction log, tracking the current position in the log to enable resume after failures, converting log entries into structured events, and managing schema changes in the source database. The message broker component provides persistent storage of change events, ensures at-least-once or exactly-once delivery guarantees, enables horizontal scaling through partitioning, and maintains event ordering within partitions. The consumer component processes change events, may transform or filter events, can maintain derived state based on changes, and must handle failures and reprocessing appropriately.

```
CDC Architecture Components:

┌─────────────────────────────────────────────────────────────────────────────┐
│                         CDC Architecture Overview                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────┐
                              │    Source Database     │
                              │  ┌─────────────────┐   │
                              │  │ Transaction Log │   │
                              │  │   (WAL/Binlog) │   │
                              │  └────────┬────────┘   │
                              └──────────┼────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │   CDC Connector         │
                              │ ┌─────────────────────┐ │
                              │ │ Log Reader          │ │
                              │ │ Schema History      │ │
                              │ │ Event Transforms    │ │
                              │ └──────────┬──────────┘ │
                              └───────────┼────────────┘
                                          │
                                          ▼
                              ┌─────────────────────────┐
                              │   Message Broker       │
                              │ ┌─────────────────────┐ │
                              │ │ Topics              │ │
                              │ │ Partitions          │ │
                              │ │ Retention           │ │
                              │ └──────────┬──────────┘ │
                              └───────────┼────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │ Consumer Group A  │  │ Consumer Group B │  │ Consumer Group C │
         │ (Data Warehouse)  │  │ (Search Index)   │  │ (Cache Update)   │
         └──────────────────┘  └──────────────────┘  └──────────────────┘

Component Responsibilities:

1. Transaction Log (Source)
   - Write-Ahead Log (PostgreSQL)
   - Binary Log (MySQL)
   - Transaction Log (Oracle)
   - CDC Redo Log (DB2)

2. CDC Connector
   - Log mining
   - Schema extraction
   - Event serialization
   - Offset management

3. Message Broker
   - Event persistence
   - Delivery guarantees
   - Partitioning
   - Ordering guarantees

4. Consumers
   - Event processing
   - State management
   - Failure handling
   - Exactly-once semantics
```

### 1.3 CDC Deployment Patterns

Different deployment patterns address various use cases and requirements. The single database to single consumer pattern represents the simplest CDC deployment, suitable for basic replication or triggering downstream actions. The single database to multiple consumers pattern enables broadcasting changes to multiple independent systems, each processing the same change stream for different purposes. The distributed CDC pattern scales CDC infrastructure across multiple source databases, aggregating changes into a unified event stream.

For high availability scenarios, the active-passive CDC pattern maintains redundant CDC instances with automatic failover, ensuring continuous change capture even during maintenance or failures. The fan-in pattern consolidates changes from multiple source databases into a single topic, simplifying consumption for applications that need a unified view. The fan-out pattern distributes different types of changes to separate topics, enabling specialized processing for different event types.

```yaml
# CDC Deployment Configuration Examples

# Single Database to Multiple Consumers
cdc_single_multi:
  source:
    database:
      type: postgresql
      host: db-primary.example.com
      port: 5432
      database: orders
      schema: public
      
  connector:
    type: debezium
    config:
      connector.class: io.debezium.connector.postgresql.PostgresConnector
      plugin.name: pgoutput
      publication.name: dbz_publication
      slot.name: debezium_slot
      
  topics:
    - name: orders.public.users
      partitions: 6
      retention: 7days
    - name: orders.public.orders
      partitions: 6
      retention: 7days
    - name: orders.public.order_items
      partitions: 6
      retention: 7days
      
  consumers:
    - name: analytics_consumer
      topic: orders.public.orders
      processor: analytics_pipeline
    - name: search_consumer
      topic: orders.public.*
      processor: elasticsearch_sync
    - name: cache_consumer
      topic: orders.public.users
      processor: cache_invalidation

# High Availability CDC
cdc_ha:
  source:
    database:
      type: mysql
      host: mysql-cluster.example.com
      port: 3306
      database: commerce
      
  connector:
    replicas:
      - name: debezium-1
        priority: 1
      - name: debezium-2
        priority: 2
        
  failover:
    enabled: true
    healthcheck_interval: 10s
    failover_timeout: 30s
    
  message_broker:
    type: kafka
    bootstrap_servers:
      - kafka-1:9092
      - kafka-2:9092
      - kafka-3:9092
      
  topic_config:
    replication_factor: 3
    min_insync_replicas: 2
    
# Distributed CDC with Multiple Sources
cdc_distributed:
  sources:
    - name: users_db
      database:
        type: postgresql
        host: users-db.example.com
      topic_prefix: users
    - name: orders_db
      database:
        type: postgresql
        host: orders-db.example.com
      topic_prefix: orders
    - name: products_db
      database:
        type: mysql
        host: products-db.example.com
      topic_prefix: products
        
  aggregation:
    enabled: true
    unified_topic: all_database_changes
    include_source_metadata: true
```

---

## 2. Debezium Implementation Guide

### 2.1 Debezium Connector Configuration

Debezium is an open-source CDC platform that provides connectors for various databases including PostgreSQL, MySQL, MongoDB, Oracle, and SQL Server. The connector configuration defines how to connect to the source database, which tables to monitor, how to handle schema changes, and where to send the change events. Proper configuration ensures reliable change capture while minimizing impact on the source database.

The following configuration demonstrates a comprehensive Debezium setup for PostgreSQL with detailed options for snapshot behavior, topic routing, event formatting, and transformation chains. This configuration enables include list filtering to capture only specific tables, configures appropriate snapshot modes, and sets up the Debezium Connect framework for distributed processing.

```json
{
  "name": "orders-postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres.example.com",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "${secrets:DATABASE_PASSWORD}",
    "database.dbname": "orders",
    "database.server.name": "orders",
    "database.include.list": "public.users,public.orders,public.order_items,public.products",
    "database.exclude.list": "public.audit_log,public.sessions",
    
    "table.include.list": "public.users,public.orders,public.order_items,public.products",
    "table.exclude.list": "public.temp_.*,public.staging_.*",
    
    "plugin.name": "pgoutput",
    "publication.name": "debezium_publication",
    "slot.name": "debezium_slot",
    "publication.autocreate.mode": "filtered",
    
    "snapshot.mode": "initial",
    "snapshot.lock.timeout.ms": "10000",
    "snapshot.fetch.size": "10240",
    "snapshot.delay.ms": "0",
    
    "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
    "schema.history.internal.kafka.topic": "orders.schema-changes",
    "schema.history.internal.kafka.recovery.poll.interval.ms": "10000",
    
    "topic.prefix": "orders",
    "topic.delimiter": ".",
    "topic.creation.default.replication.factor": "3",
    "topic.creation.default.partitions": "6",
    "topic.creation.default.cleanup.policy": "compact",
    
    "transforms": "unwrap,route",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite",
    "transforms.unwrap.flatten": "true",
    "transforms.unwrap.flatten.structContaining": "true",
    
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.(.+)",
    "transforms.route.replacement": "$2",
    
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": "false",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": "true",
    
    "decimal.handling.mode": "double",
    "time.precision.mode": "adaptive",
    "bigint.handling.mode": "long",
    
    "heartbeat.interval.ms": "10000",
    "heartbeat.topics.prefix": "heartbeat",
    
    "poll.interval.ms": "1000",
    "max.batch.size": "2048",
    "max.queue.size": "8192",
    "max.queue.size.in.bytes": "67108864",
    
    "offset.storage.topic": "orders-offset-storage",
    "offset.storage.replication.factor": "3",
    "offset.storage.partitions": "6",
    
    "status.storage.topic": "orders-status-storage",
    "status.storage.replication.factor": "3",
    "status.storage.partitions": "6",
    
    "enable.time.adjuster": "true",
    "transforms.extract.state.enabled": "true",
    "transforms.extract.state.type": "io.debezium.transforms.ExtractChangedState",
    "header.key": "id"
  }
}
```

### 2.2 MySQL CDC with Binary Log Reading

MySQL CDC uses the binary log (binlog) as the source of change data. The following configuration demonstrates a complete Debezium setup for MySQL with GTID-based positioning, row-level logging for complete change capture, and appropriate settings for different MySQL server configurations. This implementation includes support for row image options, timezone handling, and quote characters for database identifiers.

```json
{
  "name": "commerce-mysql-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql.example.com",
    "database.port": "3306",
    "database.user": "debezium",
    "database.password": "${secrets:MYSQL_PASSWORD}",
    "database.include.list": "commerce",
    "database.exclude.list": "mysql,information_schema,performance_schema,sys",
    
    "table.include.list": "commerce.users,commerce.orders,commerce.order_items,commerce.products,commerce.categories",
    "table.exclude.list": "commerce.order_history.*,commerce.audit.*",
    
    "database.server.name": "commerce",
    "database.server.id": "184054",
    
    "include.schema.changes": "true",
    "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
    "schema.history.internal.kafka.topic": "commerce.schema-history",
    
    "snapshot.mode": "when_needed",
    "snapshot.locking.mode": "minimal",
    "snapshot.fetch.size": "10000",
    
    "binlog.buffer.size": "65536",
    "binlog.concurrent.readers": "4",
    "binlog.connect.timeout.ms": "20000",
    "binlog.heartbeat.interval.ms": "10000",
    "binlog.max.wal.size": "67108864",
    "binlog.rotation.interval.ms": "3600000",
    
    "gtid.source.filter.dml": "exclude",
    "gtid.channel.name": "mysql-binlog-source",
    
    "decimal.handling.mode": "double",
    "time.precision.mode": "adaptive",
    "bigint.unsigned.handling.mode": "long",
    
    "row.image": "FULL",
    "connect.timeout.ms": "30000",
    "database.ssl.mode": "preferred",
    "database.tcp.keepalive": "true",
    "database.tcp.keepalive.interval.ms": "30000",
    
    "transforms": "unwrap,route",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite",
    "transforms.unwrap.add.fields": "op,table,source.ts_ms,source.file,source.pos",
    "transforms.unwrap.add.fields.op": "op",
    "transforms.unwrap.add.fields.table": "table",
    "transforms.unwrap.add.fields.ts_ms": "source.ts_ms",
    
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.([^.]+)\\.(.+)",
    "transforms.route.replacement": "$3",
    
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": "false",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": "true",
    
    "heartbeat.interval.ms": "5000",
    "heartbeat.topics.prefix": "heartbeat",
    
    "poll.interval.ms": "1000",
    "max.batch.size": "10240",
    "max.queue.size": "20240"
  }
}
```

### 2.3 Debezium SMT Transformations

Single Message Transformations (SMTs) in Debezium enable event routing, filtering, content transformation, and enrichment. These transformations are applied as events flow through the Kafka Connect pipeline and can modify event structure, route events to different topics, filter out unwanted events, or add additional fields. Understanding and implementing appropriate transformations is crucial for building efficient CDC pipelines.

The following examples demonstrate common SMT configurations for various use cases, including new record state extraction for simplified downstream processing, event time extraction for windowing operations, content-based routing for directing events to specific topics, and field filtering for removing sensitive data before downstream consumption.

```java
// Debezium SMT Implementation Examples

import org.apache.kafka.connect.transforms.*;
import org.apache.kafka.connect.transforms.util.SimpleConfig;
import java.util.*;

// Example 1: Custom SMT for PII Redaction
public class PiiRedaction<S> implements Transformation<S> {
    
    private static final String FIELDS_CONFIG = "fields.to.redact";
    private Set<String> fieldsToRedact;
    
    @Override
    public void configure(Map<String, ?> configs) {
        String fields = (String) configs.get(FIELDS_CONFIG);
        fieldsToRedact = new HashSet<>(Arrays.asList(fields.split(",")));
    }
    
    @Override
    public S apply(S record) {
        if (record instanceof org.apache.kafka.connect.sink.SinkRecord) {
            org.apache.kafka.connect.sink.SinkRecord sinkRecord = 
                (org.apache.kafka.connect.sink.SinkRecord) record;
            
            org.apache.kafka.connect.Struct value = 
                (org.apache.kafka.connect.Struct) sinkRecord.value();
            
            if (value != null) {
                org.apache.kafka.connect.Struct newValue = 
                    new org.apache.kafka.connect.Struct(value.schema());
                
                for (org.apache.kafka.connect.data.Field field : value.schema().fields()) {
                    String fieldName = field.name();
                    if (fieldsToRedact.contains(fieldName)) {
                        newValue.put(fieldName, "REDACTED");
                    } else {
                        newValue.put(fieldName, value.get(fieldName));
                    }
                }
                
                return (S) new org.apache.kafka.connect.sink.SinkRecord(
                    sinkRecord.topic(),
                    sinkRecord.kafkaPartition(),
                    sinkRecord.key(),
                    sinkRecord.keySchema(),
                    newValue,
                    sinkRecord.valueSchema(),
                    sinkRecord.timestamp(),
                    sinkRecord.headers()
                );
            }
        }
        return record;
    }
    
    @Override
    public Schema schema() {
        return null;
    }
    
    @Override
    public void close() {}
    
    public static class PiiRedactionConfig {
        public static final String FIELDS_TO_REDACT = "fields.to.redact";
    }
}

// Example 2: Event Enrichment SMT
public class EventEnrichment<S> implements Transformation<S> {
    
    private static final String ENRICHMENT_FIELDS_CONFIG = "enrichment.fields";
    private Map<String, String> enrichmentFields;
    
    @Override
    public void configure(Map<String, ?> configs) {
        String fields = (String) configs.get(ENRICHMENT_FIELDS_CONFIG);
        enrichmentFields = new HashMap<>();
        if (fields != null) {
            for (String field : fields.split(",")) {
                String[] parts = field.split(":");
                if (parts.length == 2) {
                    enrichmentFields.put(parts[0], parts[1]);
                }
            }
        }
    }
    
    @Override
    public S apply(S record) {
        if (record instanceof org.apache.kafka.connect.sink.SinkRecord) {
            org.apache.kafka.connect.sink.SinkRecord sinkRecord = 
                (org.apache.kafka.connect.sink.SinkRecord) record;
            
            // Add enrichment fields to headers
            org.apache.kafka.connect.header.Headers headers = sinkRecord.headers();
            for (Map.Entry<String, String> entry : enrichmentFields.entrySet()) {
                headers.addString(entry.getKey(), entry.getValue());
            }
            
            return (S) new org.apache.kafka.connect.sink.SinkRecord(
                sinkRecord.topic(),
                sinkRecord.kafkaPartition(),
                sinkRecord.key(),
                sinkRecord.keySchema(),
                sinkRecord.value(),
                sinkRecord.valueSchema(),
                sinkRecord.timestamp(),
                headers
            );
        }
        return record;
    }
    
    @Override
    public Schema schema() {
        return null;
    }
    
    @Override
    public void close() {}
}

// Example 3: Timezone Conversion SMT
public class TimezoneConverter<S> implements Transformation<S> {
    
    private static final String SOURCE_TZ_CONFIG = "source.timezone";
    private static final String TARGET_TZ_CONFIG = "target.timezone";
    private java.time.ZoneId sourceTimezone;
    private java.time.ZoneId targetTimezone;
    
    @Override
    public void configure(Map<String, ?> configs) {
        String sourceTz = (String) configs.get(SOURCE_TZ_CONFIG);
        String targetTz = (String) configs.get(TARGET_TZ_CONFIG);
        
        sourceTimezone = java.time.ZoneId.of(sourceTz != null ? sourceTz : "UTC");
        targetTimezone = java.time.ZoneId.of(targetTz != null ? targetTz : "UTC");
    }
    
    @Override
    public S apply(S record) {
        // Implementation for timezone conversion of timestamp fields
        return record;
    }
    
    @Override
    public Schema schema() {
        return null;
    }
    
    @Override
    public void close() {}
}
```

---

## 3. Event-Driven Database Patterns

### 3.1 Event Sourcing with Database Changes

Event sourcing is a pattern where application state is derived from a sequence of immutable events rather than stored directly. When combined with CDC, this pattern becomes powerful because the database transaction log itself serves as the event store. This approach provides complete auditability, enables temporal queries, supports event replay for debugging, and simplifies implementing complex features like materialized views.

Implementing event sourcing with CDC involves capturing domain events that represent meaningful business actions rather than raw database changes. These events are then stored in an event store and used to reconstruct application state. The CDC stream provides a reliable foundation for building event-sourced systems, as it captures all changes without requiring modifications to the application code that writes to the database.

```typescript
// Event Sourcing with CDC Events

interface DomainEvent {
  eventId: string;
  eventType: string;
  aggregateId: string;
  aggregateType: string;
  version: number;
  timestamp: Date;
  payload: Record<string, any>;
  metadata: EventMetadata;
}

interface EventMetadata {
  correlationId?: string;
  causationId?: string;
  userId?: string;
  source: string;
  schemaVersion: string;
}

class CdcEventProcessor {
  private eventStore: EventStore;
  private projectionEngine: ProjectionEngine;
  
  async processCdcEvent(cdcEvent: CdcEvent): Promise<void> {
    // Extract domain event from CDC event
    const domainEvent = this.transformToDomainEvent(cdcEvent);
    
    // Store event in event store
    await this.eventStore.append(domainEvent);
    
    // Update projections
    await this.projectionEngine.project(domainEvent);
    
    // Publish to event subscribers
    await this.publishToSubscribers(domainEvent);
  }
  
  private transformToDomainEvent(cdcEvent: CdcEvent): DomainEvent {
    const { before, after, op, source } = cdcEvent;
    
    switch (op) {
      case 'c': // Create
        return this.handleCreate(after, source);
      case 'u': // Update
        return this.handleUpdate(before, after, source);
      case 'd': // Delete
        return this.handleDelete(before, source);
      default:
        throw new UnknownOperationError(op);
    }
  }
  
  private handleCreate(after: Record<string, any>, source: CdcSource): DomainEvent {
    const eventType = `UserCreated`; // Domain-specific event type
    
    return {
      eventId: uuidv4(),
      eventType,
      aggregateId: after.id,
      aggregateType: 'User',
      version: 1,
      timestamp: new Date(),
      payload: {
        email: after.email,
        username: after.username,
        firstName: after.first_name,
        lastName: after.last_name,
        status: after.status
      },
      metadata: {
        source: `${source.db}.${source.table}`,
        schemaVersion: '1.0'
      }
    };
  }
  
  private handleUpdate(before: Record<string, any>, after: Record<string, any>, source: CdcSource): DomainEvent {
    const changes = this.calculateChanges(before, after);
    const eventType = this.determineEventType(changes);
    
    return {
      eventId: uuidv4(),
      eventType,
      aggregateId: after.id,
      aggregateType: 'User',
      version: (before.version || 0) + 1,
      timestamp: new Date(),
      payload: changes,
      metadata: {
        source: `${source.db}.${source.table}`,
        schemaVersion: '1.0'
      }
    };
  }
  
  private calculateChanges(before: Record<string, any>, after: Record<string, any>): Record<string, any> {
    const changes: Record<string, any> = {};
    
    for (const key of Object.keys(after)) {
      if (before[key] !== after[key]) {
        changes[key] = {
          from: before[key],
          to: after[key]
        };
      }
    }
    
    return changes;
  }
  
  private determineEventType(changes: Record<string, any>): string {
    if (changes.status?.to === 'suspended') {
      return 'UserSuspended';
    }
    if (changes.status?.to === 'active' && changes.status?.from === 'inactive') {
      return 'UserReactivated';
    }
    return 'UserUpdated';
  }
  
  private handleDelete(before: Record<string, any>, source: CdcSource): DomainEvent {
    return {
      eventId: uuidv4(),
      eventType: 'UserDeleted',
      aggregateId: before.id,
      aggregateType: 'User',
      version: (before.version || 0) + 1,
      timestamp: new Date(),
      payload: {
        deletedEmail: before.email,
        deletedAt: new Date()
      },
      metadata: {
        source: `${source.db}.${source.table}`,
        schemaVersion: '1.0'
      }
    };
  }
}

// Event Store Implementation
class EventStore {
  private pool: Pool;
  
  async append(event: DomainEvent): Promise<void> {
    const client = await this.pool.connect();
    
    try {
      await client.query('BEGIN');
      
      // Append event to event stream
      await client.query(
        `INSERT INTO events (id, event_type, aggregate_id, aggregate_type, version, timestamp, payload, metadata)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
        [
          event.eventId,
          event.eventType,
          event.aggregateId,
          event.aggregateType,
          event.version,
          event.timestamp,
          JSON.stringify(event.payload),
          JSON.stringify(event.metadata)
        ]
      );
      
      // Update aggregate state
      await client.query(
        `INSERT INTO aggregates (id, type, current_version, updated_at)
         VALUES ($1, $2, $3, NOW())
         ON CONFLICT (id) DO UPDATE SET 
           current_version = $3,
           updated_at = NOW()`,
        [event.aggregateId, event.aggregateType, event.version]
      );
      
      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
  
  async getEventsForAggregate(aggregateId: string): Promise<DomainEvent[]> {
    const result = await this.pool.query(
      `SELECT * FROM events WHERE aggregate_id = $1 ORDER BY version ASC`,
      [aggregateId]
    );
    
    return result.rows.map(row => ({
      eventId: row.id,
      eventType: row.event_type,
      aggregateId: row.aggregate_id,
      aggregateType: row.aggregate_type,
      version: row.version,
      timestamp: row.timestamp,
      payload: JSON.parse(row.payload),
      metadata: JSON.parse(row.metadata)
    }));
  }
}

// Projection Engine
class ProjectionEngine {
  private projections: Map<string, Projection>;
  
  constructor() {
    this.projections = new Map();
    this.registerProjections();
  }
  
  private registerProjections() {
    this.projections.set('UserSummaryProjection', new UserSummaryProjection());
    this.projections.set('UserActivityProjection', new UserActivityProjection());
    this.projections.set('OrderAnalyticsProjection', new OrderAnalyticsProjection());
  }
  
  async project(event: DomainEvent): Promise<void> {
    for (const projection of this.projections.values()) {
      if (projection.canHandle(event)) {
        await projection.project(event);
      }
    }
  }
}

interface Projection {
  canHandle(event: DomainEvent): boolean;
  project(event: DomainEvent): Promise<void>;
}

class UserSummaryProjection implements Projection {
  private pool: Pool;
  
  canHandle(event: DomainEvent): boolean {
    return event.aggregateType === 'User' && 
           ['UserCreated', 'UserUpdated', 'UserSuspended', 'UserReactivated'].includes(event.eventType);
  }
  
  async project(event: DomainEvent): Promise<void> {
    const { aggregateId, eventType, payload, version } = event;
    
    switch (eventType) {
      case 'UserCreated':
        await this.pool.query(
          `INSERT INTO user_summaries (id, email, username, full_name, status, created_at, version)
           VALUES ($1, $2, $3, $4, $5, $6, $7)`,
          [aggregateId, payload.email, payload.username, 
           `${payload.firstName} ${payload.lastName}`, payload.status, new Date(), version]
        );
        break;
        
      case 'UserUpdated':
        await this.pool.query(
          `UPDATE user_summaries SET 
             email = COALESCE($2, email),
             username = COALESCE($3, username),
             full_name = COALESCE($4, full_name),
             status = COALESCE($5, status),
             version = $6,
             updated_at = NOW()
           WHERE id = $1`,
          [aggregateId, payload.email?.to, payload.username?.to, 
           payload.fullName?.to, payload.status?.to, version]
        );
        break;
    }
  }
}
```

### 3.2 CQRS with CDC Events

Command Query Responsibility Segregation (CQRS) separates read and write operations into distinct models. CDC provides the perfect foundation for implementing CQRS by delivering domain events that can be used to maintain multiple optimized read models. This pattern enables different representations of the same underlying data, specialized read optimizations, and independent scaling of read and write workloads.

The implementation demonstrates how CDC events feed a CQRS architecture with separate read models optimized for different query patterns. The write model maintains authoritative state and captures all changes as events, while read models are continuously updated through event processing.

```typescript
// CQRS Implementation with CDC Events

// Command Side - Write Model
class UserCommandHandler {
  private eventStore: EventStore;
  private commandBus: CommandBus;
  
  async handleCommand(command: Command): Promise<CommandResult> {
    switch (command.type) {
      case 'CreateUser':
        return this.handleCreateUser(command as CreateUserCommand);
      case 'UpdateUser':
        return this.handleUpdateUser(command as UpdateUserCommand);
      case 'DeleteUser':
        return this.handleDeleteUser(command as DeleteUserCommand);
      case 'ActivateUser':
        return this.handleActivateUser(command as ActivateUserCommand);
      default:
        throw new UnknownCommandError(command.type);
    }
  }
  
  private async handleCreateUser(command: CreateUserCommand): Promise<CommandResult> {
    // Validate command
    const validation = this.validateCreateUser(command);
    if (!validation.valid) {
      return { success: false, errors: validation.errors };
    }
    
    // Create domain event
    const event = new UserCreatedEvent({
      userId: uuidv4(),
      email: command.email,
      username: command.username,
      firstName: command.firstName,
      lastName: command.lastName,
      createdAt: new Date()
    });
    
    // Persist event
    await this.eventStore.append(event);
    
    // Publish for CDC capture
    await this.commandBus.publish(event);
    
    return { success: true, aggregateId: event.userId, version: 1 };
  }
  
  private async handleUpdateUser(command: UpdateUserCommand): Promise<CommandResult> {
    // Load current state
    const events = await this.eventStore.getEventsForAggregate(command.userId);
    const userState = this.reconstructState(events);
    
    if (!userState) {
      return { success: false, errors: ['User not found'] };
    }
    
    // Apply changes
    const changes = this.calculateChanges(userState, command.updates);
    
    // Create update event
    const event = new UserUpdatedEvent({
      userId: command.userId,
      changes,
      previousVersion: userState.version,
      newVersion: userState.version + 1,
      updatedAt: new Date()
    });
    
    // Persist and publish
    await this.eventStore.append(event);
    await this.commandBus.publish(event);
    
    return { success: true, aggregateId: command.userId, version: event.newVersion };
  }
  
  private reconstructState(events: DomainEvent[]): UserState | null {
    if (events.length === 0) return null;
    
    let state: UserState = {
      userId: '',
      email: '',
      username: '',
      firstName: '',
      lastName: '',
      status: 'pending',
      version: 0
    };
    
    for (const event of events) {
      state = this.applyEvent(state, event);
    }
    
    return state;
  }
  
  private applyEvent(state: UserState, event: DomainEvent): UserState {
    switch (event.eventType) {
      case 'UserCreated':
        return { ...state, ...event.payload, version: event.version };
      case 'UserUpdated':
        const newState = { ...state };
        for (const [key, change] of Object.entries(event.payload)) {
          newState[key] = change.to;
        }
        return { ...newState, version: event.version };
      case 'UserDeleted':
        return { ...state, status: 'deleted', version: event.version };
      default:
        return state;
    }
  }
}

// Query Side - Read Models
interface UserReadModel {
  userId: string;
  email: string;
  username: string;
  displayName: string;
  status: string;
  createdAt: Date;
  lastActivityAt: Date;
  orderCount: number;
  totalSpent: number;
}

interface UserListItem {
  userId: string;
  displayName: string;
  email: string;
  status: string;
}

// Read Model Repository - Maintains user read model
class UserReadModelRepository {
  private pool: Pool;
  private cache: Redis;
  
  async onUserCreated(event: UserCreatedEvent): Promise<void> {
    await this.pool.query(
      `INSERT INTO user_read_models (user_id, email, username, first_name, last_name, status, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)`,
      [event.userId, event.email, event.username, event.firstName, event.lastName, 'active', event.createdAt]
    );
    
    await this.cache.del(`user:${event.userId}`);
  }
  
  async onUserUpdated(event: UserUpdatedEvent): Promise<void> {
    const updates: string[] = [];
    const values: any[] = [];
    let paramIndex = 1;
    
    for (const [field, change] of Object.entries(event.changes)) {
      updates.push(`${this.toSnakeCase(field)} = $${paramIndex++}`);
      values.push(change.to);
    }
    
    updates.push(`updated_at = NOW()`);
    values.push(event.userId);
    
    await this.pool.query(
      `UPDATE user_read_models SET ${updates.join(', ')} WHERE user_id = $${paramIndex}`,
      values
    );
    
    await this.cache.del(`user:${event.userId}`);
  }
  
  async getUserById(userId: string): Promise<UserReadModel | null> {
    // Check cache first
    const cached = await this.cache.get(`user:${userId}`);
    if (cached) {
      return JSON.parse(cached);
    }
    
    const result = await this.pool.query(
      `SELECT * FROM user_read_models WHERE user_id = $1`,
      [userId]
    );
    
    if (result.rows.length === 0) return null;
    
    const row = result.rows[0];
    const model: UserReadModel = {
      userId: row.user_id,
      email: row.email,
      username: row.username,
      displayName: `${row.first_name} ${row.last_name}`,
      status: row.status,
      createdAt: row.created_at,
      lastActivityAt: row.last_activity_at,
      orderCount: row.order_count || 0,
      totalSpent: parseFloat(row.total_spent) || 0
    };
    
    // Cache for 5 minutes
    await this.cache.setex(`user:${userId}`, 300, JSON.stringify(model));
    
    return model;
  }
  
  async getUserList(filter: UserListFilter, pagination: Pagination): Promise<UserListItem[]> {
    const conditions: string[] = [];
    const values: any[] = [];
    let paramIndex = 1;
    
    if (filter.status) {
      conditions.push(`status = $${paramIndex++}`);
      values.push(filter.status);
    }
    
    if (filter.search) {
      conditions.push(`(email ILIKE $${paramIndex} OR username ILIKE $${paramIndex} OR first_name ILIKE $${paramIndex} OR last_name ILIKE $${paramIndex})`);
      values.push(`%${filter.search}%`);
      paramIndex++;
    }
    
    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
    
    const result = await this.pool.query(
      `SELECT user_id, username, first_name, last_name, email, status 
       FROM user_read_models ${whereClause}
       ORDER BY created_at DESC
       LIMIT $${paramIndex++} OFFSET $${paramIndex}`,
      [...values, pagination.limit, pagination.offset]
    );
    
    return result.rows.map(row => ({
      userId: row.user_id,
      displayName: `${row.first_name} ${row.last_name}`,
      email: row.email,
      status: row.status
    }));
  }
  
  private toSnakeCase(str: string): string {
    return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
  }
}

// Optimized Read Model - User with Orders
class UserOrdersReadModel {
  private pool: Pool;
  
  async onUserCreated(event: UserCreatedEvent): Promise<void> {
    // Initialize user orders summary
    await this.pool.query(
      `INSERT INTO user_orders_read_model (user_id, orders, total_spent, last_order_at)
       VALUES ($1, 0, 0, NULL)`,
      [event.userId]
    );
  }
  
  async onOrderCreated(event: OrderCreatedEvent): Promise<void> {
    await this.pool.query(
      `UPDATE user_orders_read_model SET 
         orders = orders + 1,
         total_spent = total_spent + $2,
         last_order_at = $3
       WHERE user_id = $1`,
      [event.userId, event.total, event.createdAt]
    );
  }
  
  async onOrderCancelled(event: OrderCancelledEvent): Promise<void> {
    await this.pool.query(
      `UPDATE user_orders_read_model SET 
         orders = orders - 1,
         total_spent = total_spent - $2
       WHERE user_id = $1`,
      [event.userId, event.total]
    );
  }
  
  async getUserWithOrders(userId: string): Promise<UserWithOrders | null> {
    const result = await this.pool.query(
      `SELECT u.*, uorm.orders, uorm.total_spent, uorm.last_order_at
       FROM user_read_models u
       JOIN user_orders_read_model uorm ON u.user_id = uorm.user_id
       WHERE u.user_id = $1`,
      [userId]
    );
    
    if (result.rows.length === 0) return null;
    
    const row = result.rows[0];
    return {
      userId: row.user_id,
      displayName: `${row.first_name} ${row.last_name}`,
      email: row.email,
      orderCount: row.orders,
      totalSpent: parseFloat(row.total_spent),
      lastOrderAt: row.last_order_at
    };
  }
}
```

---

## 4. Stream Processing with Database Events

### 4.1 Kafka Streams for CDC Events

Kafka Streams provides a powerful framework for processing CDC events in real-time, enabling filtering, transformation, aggregation, and enrichment of change data streams. The distributed stream processing capabilities allow complex event processing logic to be implemented with exactly-once semantics, stateful processing, and windowing operations.

The implementation demonstrates a complete Kafka Streams application that processes CDC events from multiple tables, performs complex transformations and aggregations, maintains stateful computations, and produces enriched output for downstream consumption.

```java
// Kafka Streams Application for CDC Event Processing

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.*;
import org.apache.kafka.streams.state.*;
import java.time.Duration;
import java.util.*;

public class CdcEventProcessorApplication {
    
    public static void main(String[] args) {
        Properties config = new Properties();
        config.put(StreamsConfig.APPLICATION_ID_CONFIG, "cdc-event-processor");
        config.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka:9092");
        config.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        config.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        config.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 1000);
        config.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, StreamsConfig.EXACTLY_ONCE_V2);
        config.put(StreamsConfig.STATE_DIR_CONFIG, "/var/lib/kafka-streams");
        
        StreamsBuilder builder = new StreamsBuilder();
        
        // Process user CDC events
        processUserEvents(builder);
        
        // Process order events with enrichment
        processOrderEvents(builder);
        
        // Generate aggregated metrics
        generateMetrics(builder);
        
        // Detect anomalies
        detectAnomalies(builder);
        
        KafkaStreams streams = new KafkaStreams(builder.build(), config);
        streams.start();
        
        Runtime.getRuntime().addShutdownHook(new Thread(streams::close));
    }
    
    private static void processUserEvents(StreamsBuilder builder) {
        // Source: User CDC events
        KStream<String, String> userEvents = builder.stream(
            "orders.public.users",
            Consumed.with(Serdes.String(), Serdes.String())
        );
        
        // Filter active user events
        KStream<String, String> activeUserEvents = userEvents
            .filter((key, value) -> {
                CdcEvent event = parseCdcEvent(value);
                return event != null && "active".equals(event.getAfter().get("status"));
            })
            .mapValues(CdcEventProcessorApplication::enrichUserEvent);
        
        // Branch by operation type
        KStream<String, String>[] branches = activeUserEvents
            .branch(
                (key, value) -> parseCdcEvent(value).getOp().equals("c"),
                (key, value) -> parseCdcEvent(value).getOp().equals("u"),
                (key, value) -> parseCdcEvent(value).getOp().equals("d")
            );
        
        // Process creates
        branches[0]
            .to("user-creates", Produced.with(Serdes.String(), Serdes.String()));
        
        // Process updates
        branches[1]
            .to("user-updates", Produced.with(Serdes.String(), Serdes.String()));
        
        // Process deletes
        branches[2]
            .to("user-deletes", Produced.with(Serdes.String(), Serdes.String()));
    }
    
    private static void processOrderEvents(StreamsBuilder builder) {
        // Source: Order CDC events
        KStream<String, String> orderEvents = builder.stream(
            "orders.public.orders",
            Consumed.with(Serdes.String(), Serdes.String())
        );
        
        // GlobalKTable for user data lookup
        GlobalKTable<String, String> userTable = builder.globalTable(
            "user-enriched-topic",
            Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as("user-store")
                .withKeySerde(Serdes.String())
                .withValueSerde(Serdes.String())
        );
        
        // Join orders with user information
        KStream<String, String> enrichedOrders = orderEvents
            .filter((key, value) -> {
                CdcEvent event = parseCdcEvent(value);
                return event != null && !event.getOp().equals("d");
            })
            .selectKey((key, value) -> {
                CdcEvent event = parseCdcEvent(value);
                return event.getAfter().get("user_id");
            })
            .leftJoin(
                userTable,
                (orderKey, orderValue) -> orderKey,
                (orderEvent, userEvent) -> mergeOrderAndUser(orderEvent, userEvent)
            );
        
        // Windowed aggregation: Orders per time window
        TimeWindows window = TimeWindows.of(Duration.ofMinutes(5));
        
        KTable<Windowed<String>, Long> orderCountByWindow = enrichedOrders
            .groupByKey()
            .windowedBy(window)
            .count(Materialized.as("order-count-store"));
        
        orderCountByWindow
            .toStream()
            .to("order-count-by-window", Produced.with(WindowedSerdes.String(), Serdes.Long()));
        
        // Running totals per user
        KTable<String, Long> runningOrderCount = enrichedEvents
            .groupBy((key, value) -> KeyValue.pair(key, value))
            .count(Materialized.as("user-order-counts"));
        
        runningOrderCount
            .toStream()
            .to("user-order-counts", Produced.with(Serdes.String(), Serdes.Long()));
    }
    
    private static void generateMetrics(StreamsBuilder builder) {
        // Aggregate all events for metrics calculation
        KStream<String, String> allEvents = builder.stream(
            Arrays.asList("orders.public.users", "orders.public.orders", "orders.public.order_items"),
            Consumed.with(Serdes.String(), Serdes.String())
        );
        
        // Count operations by type
        KTable<String, Long> operationCounts = allEvents
            .mapValues(CdcEventProcessorApplication::extractEventType)
            .groupBy((key, eventType) -> eventType)
            .count(Materialized.as("operation-counts"));
        
        operationCounts
            .toStream()
            .to("metrics-operation-counts", Produced.with(Serdes.String(), Serdes.Long()));
        
        // Calculate throughput metrics
        TimeWindows window = TimeWindows.of(Duration.ofMinutes(1));
        
        KTable<Windowed<String>, Long> throughputByTable = allEvents
            .mapValues(value -> {
                CdcEvent event = parseCdcEvent(value);
                return event != null ? event.getSource().get("table") : "unknown";
            })
            .groupBy((key, table) -> table)
            .windowedBy(window)
            .count(Materialized.as("throughput-by-table"));
        
        throughputByTable
            .toStream()
            .mapValues((windowedKey, count) -> 
                String.format("{\"table\":\"%s\",\"window_start\":%d,\"count\":%d}",
                    windowedKey.key(), windowedKey.window().start(), count))
            .to("metrics-throughput", Produced.with(WindowedSerdes.String(), Serdes.String()));
    }
    
    private static void detectAnomalies(StreamsBuilder builder) {
        // High frequency order detection
        KStream<String, String> orderEvents = builder.stream(
            "orders.public.orders",
            Consumed.with(Serdes.String(), Serdes.String())
        );
        
        TimeWindows window = TimeWindows.of(Duration.ofMinutes(1));
        
        KTable<Windowed<String>, Long> ordersPerUser = orderEvents
            .filter((key, value) -> {
                CdcEvent event = parseCdcEvent(value);
                return event != null && event.getOp().equals("c");
            })
            .map((key, value) -> {
                CdcEvent event = parseCdcEvent(value);
                return KeyValue.pair(event.getAfter().get("user_id"), value);
            })
            .groupByKey()
            .windowedBy(window)
            .count(Materialized.as("orders-per-user-window"));
        
        // Detect anomaly: More than 10 orders per minute
        KStream<String, String> anomalies = ordersPerUser
            .toStream()
            .filter((windowedKey, count) -> count > 10)
            .map((windowedKey, count) -> 
                KeyValue.pair(
                    windowedKey.key(),
                    String.format("{\"user_id\":\"%s\",\"order_count\":%d,\"window_start\":%d,\"anomaly_type\":\"HIGH_ORDER_FREQUENCY\"}",
                        windowedKey.key(), count, windowedKey.window().start())
                ));
        
        anomalies.to("anomaly-detections", Produced.with(Serdes.String(), Serdes.String()));
        
        // Large order detection
        KStream<String, String> largeOrders = orderEvents
            .filter((key, value) -> {
                CdcEvent event = parseCdcEvent(value);
                if (event == null || event.getOp().equals("d")) return false;
                double total = Double.parseDouble(event.getAfter().get("total").toString());
                return total > 10000; // Orders over $10,000
            })
            .mapValues(value -> {
                CdcEvent event = parseCdcEvent(value);
                return String.format("{\"order_id\":\"%s\",\"user_id\":\"%s\",\"total\":%s,\"anomaly_type\":\"LARGE_ORDER\"}",
                    event.getAfter().get("id"), event.getAfter().get("user_id"), event.getAfter().get("total"));
            });
        
        largeOrders.to("anomaly-detections", Produced.with(Serdes.String(), Serdes.String()));
    }
    
    private static CdcEvent parseCdcEvent(String json) {
        // JSON parsing implementation
        return new CdcEvent(json);
    }
    
    private static String enrichUserEvent(String eventJson) {
        // Event enrichment implementation
        return eventJson;
    }
    
    private static String mergeOrderAndUser(String orderEvent, String userEvent) {
        // Merge order and user data
        return orderEvent;
    }
    
    private static String extractEventType(String eventJson) {
        CdcEvent event = parseCdcEvent(eventJson);
        return event != null ? event.getSource().get("table") + "-" + event.getOp() : "unknown";
    }
}
```

### 4.2 Event Processing with Apache Flink

For more sophisticated stream processing requirements, Apache Flink provides advanced capabilities including complex event processing, event time handling with watermarks, and flexible windowing operations. This implementation demonstrates how to use Flink CDC connector to process database changes with sophisticated streaming analytics.

```java
// Apache Flink CDC Processing Application

import org.apache.flink.api.common.eventtime.*;
import org.apache.flink.api.common.functions.*;
import org.apache.flink.api.common.state.*;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.*;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.*;
import org.apache.flink.streaming.api.windowing.assigners.*;
import org.apache.flink.streaming.api.windowing.windows.*;
import org.apache.flink.table.api.*;
import org.apache.flink.connector.kafka.source.*;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import com.ververica.cdc.debezium.json.*;

public class FlinkCdcProcessor {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(5000);
        env.getCheckpointConfig().setMinPauseBetweenCheckpoints(1000);
        
        // Create CDC source from Debezium
        DataStream<String> cdcStream = env.fromSource(
            KafkaSource.<String>builder()
                .setBootstrapServers("kafka:9092")
                .setGroupId("flink-cdc-consumer")
                .setTopics("orders.public.*")
                .setStartingOffsets(OffsetsInitializer.latest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build(),
            WatermarkStrategy.noWatermarks(),
            "CDC Source"
        );
        
        // Parse CDC events
        DataStream<CdcEvent> events = cdcStream
            .map(new CdcEventParser())
            .name("Parse CDC Events");
        
        // Process different event types
        DataStream<OrderEvent> orderEvents = events
            .filter(event -> "orders".equals(event.getTable()))
            .filter(event -> "orders".equals(event.getTable()))
            .map(event -> new OrderEvent(
                event.getAfter().get("id"),
                event.getAfter().get("user_id"),
                event.getOp(),
                Double.parseDouble(event.getAfter().get("total").toString()),
                event.getTimestamp()
            ))
            .name("Extract Order Events");
        
        // Tumbling window aggregation
        DataStream<OrderMetrics> orderMetrics = orderEvents
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<OrderEvent>forBoundedOutOfOrderness(Duration.ofMinutes(5))
                    .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
            )
            .keyBy(OrderEvent::getUserId)
            .window(TumblingEventTimeWindows.of(Time.minutes(5)))
            .process(new OrderWindowFunction())
            .name("Calculate Order Metrics");
        
        // Sink to metrics system
        orderMetrics.addSink(
            KafkaSink.<String>builder()
                .setBootstrapServers("kafka:9092")
                .setRecordSerializer(new MetricsRecordSerializer("order-metrics"))
                .build()
        ).name("Sink to Kafka");
        
        // Complex Event Processing: Detect order pattern anomalies
        DataStream<AnomalyAlert> anomalies = orderEvents
            .keyBy(OrderEvent::getUserId)
            .process(new AnomalyDetectionFunction())
            .name("Detect Anomalies");
        
        anomalies.print();
        
        env.execute("Flink CDC Processor");
    }
    
    // Custom window function for order metrics
    public static class OrderWindowFunction 
            extends ProcessWindowFunction<OrderEvent, OrderMetrics, String, TimeWindow> {
        
        private ValueState<OrderState> orderState;
        
        @Override
        public void open(Configuration parameters) {
            orderState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("order-state", OrderState.class)
            );
        }
        
        @Override
        public void process(
                String key,
                Context context,
                Iterable<OrderEvent> elements,
                Collector<OrderMetrics> out) {
            
            OrderState state = orderState.value();
            if (state == null) {
                state = new OrderState();
            }
            
            double totalAmount = 0;
            int orderCount = 0;
            double maxOrderAmount = 0;
            
            for (OrderEvent event : elements) {
                if ("c".equals(event.getOp())) {
                    state.orderCount++;
                    state.totalAmount += event.getTotal();
                    state.maxOrderAmount = Math.max(state.maxOrderAmount, event.getTotal());
                    orderCount++;
                    totalAmount += event.getTotal();
                    maxOrderAmount = Math.max(maxOrderAmount, event.getTotal());
                } else if ("d".equals(event.getOp())) {
                    // Handle delete - approximate adjustment
                    state.orderCount = Math.max(0, state.orderCount - 1);
                }
            }
            
            orderState.update(state);
            
            out.collect(new OrderMetrics(
                key,
                orderCount,
                totalAmount,
                maxOrderAmount,
                state.totalAmount,
                state.orderCount,
                context.window().getStart(),
                context.window().getEnd()
            ));
        }
    }
    
    // Anomaly detection function using CEP-like pattern
    public static class AnomalyDetectionFunction 
            extends KeyedProcessFunction<String, OrderEvent, AnomalyAlert> {
        
        private ValueState<List<OrderEvent>> recentOrders;
        
        @Override
        public void open(Configuration parameters) {
            recentOrders = getRuntimeContext().getState(
                new ValueStateDescriptor<>("recent-orders", 
                    TypeInformation.of(new TypeHint<List<OrderEvent>>() {}))
            );
        }
        
        @Override
        public void processElement(OrderEvent event, Context ctx, 
                Collector<AnomalyAlert> out) throws Exception {
            
            if (!"c".equals(event.getOp())) return;
            
            List<OrderEvent> orders = recentOrders.value();
            if (orders == null) {
                orders = new ArrayList<>();
            }
            
            // Clean old orders (older than 1 hour)
            long oneHourAgo = System.currentTimeMillis() - 3600000;
            orders.removeIf(o -> o.getTimestamp() < oneHourAgo);
            
            // Check for anomalies
            // 1. High frequency: more than 5 orders in 10 minutes
            long tenMinutesAgo = System.currentTimeMillis() - 600000;
            long recentCount = orders.stream().filter(o -> o.getTimestamp() > tenMinutesAgo).count();
            
            if (recentCount > 5) {
                out.collect(new AnomalyAlert(
                    event.getUserId(),
                    "HIGH_ORDER_FREQUENCY",
                    String.format("User placed %d orders in the last 10 minutes", recentCount + 1),
                    event.getTimestamp()
                ));
            }
            
            // 2. Unusual order pattern: many small orders followed by large order
            double smallOrderThreshold = 50;
            double largeOrderThreshold = 5000;
            
            double recentSmallOrders = orders.stream()
                .filter(o -> o.getTimestamp() > tenMinutesAgo)
                .filter(o -> o.getTotal() < smallOrderThreshold)
                .count();
            
            if (recentSmallOrders > 10 && event.getTotal() > largeOrderThreshold) {
                out.collect(new AnomalyAlert(
                    event.getUserId(),
                    "UNUSUAL_ORDER_PATTERN",
                    "Pattern detected: Many small orders followed by large order",
                    event.getTimestamp()
                ));
            }
            
            orders.add(event);
            recentOrders.update(orders);
        }
    }
    
    // Data classes
    public static class OrderEvent {
        private String orderId;
        private String userId;
        private String op;
        private double total;
        private long timestamp;
        
        public OrderEvent(String orderId, String userId, String op, double total, long timestamp) {
            this.orderId = orderId;
            this.userId = userId;
            this.op = op;
            this.total = total;
            this.timestamp = timestamp;
        }
        
        // Getters and setters
    }
    
    public static class OrderMetrics {
        private String userId;
        private int windowOrderCount;
        private double windowTotalAmount;
        private double windowMaxOrderAmount;
        private double runningTotalAmount;
        private int runningOrderCount;
        private long windowStart;
        private long windowEnd;
        
        // Constructor and getters
    }
    
    public static class OrderState {
        public int orderCount = 0;
        public double totalAmount = 0;
        public double maxOrderAmount = 0;
    }
    
    public static class AnomalyAlert {
        private String userId;
        private String anomalyType;
        private String description;
        private long timestamp;
        
        public AnomalyAlert(String userId, String anomalyType, String description, long timestamp) {
            this.userId = userId;
            this.anomalyType = anomalyType;
            this.description = description;
            this.timestamp = timestamp;
        }
    }
}
```

---

## 5. Building Reactive Data Pipelines

### 5.1 Reactive CDC Consumer Architecture

Reactive data pipelines process CDC events asynchronously and non-blocking, enabling high throughput and efficient resource utilization. The reactive architecture uses event-driven components that respond to database changes in real-time, transforming and routing events through a series of processing stages. This approach provides backpressure handling, graceful degradation under load, and horizontal scalability.

The implementation demonstrates a complete reactive CDC consumer using Project Reactor, which provides the reactive streams implementation. This consumer handles CDC events from Kafka, processes them through multiple reactive stages, and outputs to various sinks while maintaining backpressure and exactly-once processing guarantees.

```java
// Reactive CDC Consumer with Project Reactor

import reactor.core.publisher.*;
import reactor.kafka.receiver.*;
import reactor.kafka.receiver.ReceiverOptions;
import reactor.core.scheduler.Schedulers;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.serialization.StringSerializer;
import java.time.Duration;
import java.util.*;

public class ReactiveCdcConsumer {
    
    private final KafkaReceiver<String, String> receiver;
    private final FluxSink<CdcEvent> eventSink;
    private final Disposable subscription;
    
    public ReactiveCdcConsumer(String bootstrapServers, String groupId, String topic) {
        // Configure Kafka receiver
        Map<String, Object> receiverProps = new HashMap<>();
        receiverProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        receiverProps.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        receiverProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        receiverProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        receiverProps.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, "100");
        
        ReceiverOptions<String, String> options = ReceiverOptions.<String, String>create(receiverProps)
            .subscription(Collections.singletonList(topic))
            .addAssignListener(partitions -> 
                System.out.println("Assigned partitions: " + partitions))
            .addRevokeListener(partitions -> 
                System.out.println("Revoked partitions: " + partitions));
        
        receiver = KafkaReceiver.create(options);
        
        // Create sink for event processing
        FluxProcessor<CdcEvent, CdcEvent> processor = 
            DirectProcessor.<CdcEvent>create().serialize();
        eventSink = processor.sink();
        
        // Build processing pipeline
        subscription = buildProcessingPipeline(processor);
    }
    
    private Disposable buildProcessingPipeline(FluxProcessor<CdcEvent, CdcEvent> processor) {
        return receiver.receive()
            .flatMap(record -> 
                Mono.fromCallable(() -> parseRecord(record))
                    .subscribeOn(Schedulers.boundedElastic())
                    .doOnNext(event -> eventSink.next(event))
                    .doOnSuccess(v -> commitOffset(record))
                    .onErrorResume(e -> 
                        handleError(record, e).then(Mono.empty())
                    )
            )
            .subscribe();
    }
    
    private Mono<Void> commitOffset(ReceiverRecord<String, String> record) {
        return Mono.fromRunnable(() -> 
            ((KafkaReceiver<String, String>) receiver).commitOffset()
        ).subscribeOn(Schedulers.boundedElastic()).then();
    }
    
    private Mono<Void> handleError(ReceiverRecord<String, String> record, Throwable error) {
        System.err.println("Error processing record: " + error.getMessage());
        
        // Send to dead letter topic
        return sendToDeadLetter(record, error);
    }
    
    private Mono<Void> sendToDeadLetter(ReceiverRecord<String, String> record, Throwable error) {
        // Implementation for dead letter handling
        return Mono.empty();
    }
    
    // Process events through reactive pipeline
    public Flux<CdcEvent> processEvents() {
        return eventSink.asFlux()
            .groupBy(CdcEvent::getTable)
            .flatMap(group -> 
                group.publishOn(Schedulers.parallel())
                    .window(Duration.ofSeconds(10), 100)
                    .flatMap(window -> 
                        window.collectList()
                            .filter(list -> !list.isEmpty())
                            .flatMap(this::processEventBatch)
                    )
            );
    }
    
    private Mono<Void> processEventBatch(List<CdcEvent> events) {
        // Process batch of events
        return Flux.fromIterable(events)
            .flatMap(this::processSingleEvent)
            .then();
    }
    
    private Mono<Void> processSingleEvent(CdcEvent event) {
        switch (event.getTable()) {
            case "users":
                return processUserEvent(event);
            case "orders":
                return processOrderEvent(event);
            case "products":
                return processProductEvent(event);
            default:
                return Mono.empty();
        }
    }
    
    // Event-specific processors
    private Mono<Void> processUserEvent(CdcEvent event) {
        return Mono.fromRunnable(() -> {
            switch (event.getOp()) {
                case "c":
                    handleUserCreate(event.getAfter());
                    break;
                case "u":
                    handleUserUpdate(event.getBefore(), event.getAfter());
                    break;
                case "d":
                    handleUserDelete(event.getBefore());
                    break;
            }
        }).subscribeOn(Schedulers.boundedElastic()).then();
    }
    
    private Mono<Void> processOrderEvent(CdcEvent event) {
        return Mono.fromRunnable(() -> {
            switch (event.getOp()) {
                case "c":
                    handleOrderCreate(event.getAfter());
                    // Emit to order created stream
                    break;
                case "u":
                    handleOrderUpdate(event.getBefore(), event.getAfter());
                    break;
                case "d":
                    handleOrderCancel(event.getBefore());
                    break;
            }
        })
        .subscribeOn(Schedulers.boundedElastic())
        .then();
    }
    
    // Handle user operations
    private void handleUserCreate(Map<String, Object> userData) {
        // Create user in downstream systems
        System.out.println("Creating user: " + userData.get("id"));
    }
    
    private void handleUserUpdate(Map<String, Object> before, Map<String, Object> after) {
        // Update user in downstream systems
        System.out.println("Updating user: " + after.get("id"));
    }
    
    private void handleUserDelete(Map<String, Object> userData) {
        // Delete user from downstream systems
        System.out.println("Deleting user: " + userData.get("id"));
    }
    
    private void handleOrderCreate(Map<String, Object> orderData) {
        System.out.println("Creating order: " + orderData.get("id"));
    }
    
    private void handleOrderUpdate(Map<String, Object> before, Map<String, Object> after) {
        System.out.println("Updating order: " + after.get("id"));
    }
    
    private void handleOrderCancel(Map<String, Object> orderData) {
        System.out.println("Cancelling order: " + orderData.get("id"));
    }
    
    private void handleProductEvent(CdcEvent event) {
        // Handle product events
    }
    
    private CdcEvent parseRecord(ReceiverRecord<String, String> record) {
        // Parse CDC event from record value
        return CdcEvent.fromJson(record.value());
    }
    
    public void shutdown() {
        subscription.dispose();
    }
}

// Backpressure-aware batch processor
class BackpressureAwareProcessor {
    
    private final int maxBatchSize;
    private final Duration maxBatchTime;
    private final FluxSink<BatchContext> batchSink;
    
    public BackpressureAwareProcessor(int maxBatchSize, Duration maxBatchTime) {
        this.maxBatchSize = maxBatchSize;
        this.maxBatchTime = maxBatchTime;
        
        // Create processor with both size and time-based triggers
        FluxProcessor<BatchContext, BatchContext> processor = 
            Flux.<BatchContext>create(sink -> {
                this.batchSink = sink;
            })
            .scanWith(
                () -> new BatchAccumulator(maxBatchSize, maxBatchTime),
                BatchAccumulator::add
            )
            .filter(BatchAccumulator::isComplete)
            .flatMap(this::processBatch);
        
        processor.subscribe();
    }
    
    public void addEvent(CdcEvent event) {
        batchSink.next(new BatchContext(event, System.currentTimeMillis()));
    }
    
    private Mono<Void> processBatch(BatchAccumulator accumulator) {
        List<CdcEvent> events = accumulator.getEvents();
        
        return Flux.fromIterable(events)
            .flatMap(this::processEvent, maxBatchSize)
            .then()
            .doOnSuccess(v -> System.out.println("Processed batch of " + events.size()))
            .doOnError(e -> System.err.println("Batch processing error: " + e.getMessage()));
    }
    
    private Mono<Void> processEvent(CdcEvent event) {
        // Process individual event
        return Mono.empty();
    }
    
    static class BatchAccumulator {
        private final int maxSize;
        private final Duration maxTime;
        private final List<CdcEvent> events;
        private final long startTime;
        
        BatchAccumulator(int maxSize, Duration maxTime) {
            this.maxSize = maxSize;
            this.maxTime = maxTime;
            this.events = new ArrayList<>();
            this.startTime = System.currentTimeMillis();
        }
        
        BatchAccumulator add(BatchContext context) {
            if (!isComplete()) {
                events.add(context.getEvent());
            }
            return this;
        }
        
        boolean isComplete() {
            return events.size() >= maxSize || 
                   (System.currentTimeMillis() - startTime) >= maxTime.toMillis();
        }
        
        List<CdcEvent> getEvents() {
            return new ArrayList<>(events);
        }
    }
    
    static class BatchContext {
        private final CdcEvent event;
        private final long timestamp;
        
        BatchContext(CdcEvent event, long timestamp) {
            this.event = event;
            this.timestamp = timestamp;
        }
        
        CdcEvent getEvent() { return event; }
        long getTimestamp() { return timestamp; }
    }
}
```

### 5.2 Multi-Sink CDC Pipeline

A robust CDC pipeline typically feeds multiple downstream systems simultaneously. This implementation demonstrates a multi-sink reactive pipeline that distributes CDC events to various consumers including search indexes, data warehouses, caching systems, and analytics platforms, each with potentially different transformation requirements.

```typescript
// Multi-Sink CDC Pipeline Implementation

import { Kafka, Consumer, Producer, EachMessagePayload } from 'kafkajs';
import { Pool } from 'pg';
import { Client } from '@elastic/elasticsearch';
import Redis from 'ioredis';

class MultiSinkCdcPipeline {
  private kafka: Kafka;
  private consumer: Consumer;
  private producer: Producer;
  private dbPool: Pool;
  private elasticsearch: Client;
  private redis: Redis;
  private running: boolean = false;
  
  constructor(config: PipelineConfig) {
    this.kafka = new Kafka({
      clientId: 'cdc-pipeline',
      brokers: config.kafkaBrokers,
      retry: {
        initialRetryTime: 100,
        retries: 8
      }
    });
    
    this.consumer = this.kafka.consumer({ 
      groupId: 'cdc-multi-sink',
      sessionTimeout: 30000,
      heartbeatInterval: 3000
    });
    
    this.producer = this.kafka.producer();
    this.dbPool = new Pool({ connectionString: config.databaseUrl });
    this.elasticsearch = new Client({ 
      node: config.elasticsearchUrl 
    });
    this.redis = new Redis(config.redisUrl);
  }
  
  async start() {
    await this.producer.connect();
    await this.consumer.connect();
    
    await this.consumer.subscribe({ 
      topic: 'orders.public.*', 
      fromBeginning: false 
    });
    
    this.running = true;
    
    await this.consumer.run({
      eachMessage: async (payload: EachMessagePayload) => {
        await this.processMessage(payload);
      }
    });
    
    console.log('CDC pipeline started');
  }
  
  async processMessage(payload: EachMessagePayload) {
    const { topic, message } = payload;
    
    if (!message.value) return;
    
    const cdcEvent = JSON.parse(message.value.toString());
    const table = this.extractTableFromTopic(topic);
    
    try {
      // Process in parallel to multiple sinks
      await Promise.all([
        this.sinkToSearchIndex(table, cdcEvent),
        this.sinkToDataWarehouse(table, cdcEvent),
        this.sinkToCache(table, cdcEvent),
        this.sinkToAnalytics(table, cdcEvent)
      ]);
      
      // Commit offset only after all sinks processed
      await this.consumer.commitOffsets([{
        topic: message.topic,
        partition: message.partition,
        offset: message.offset
      }]);
      
    } catch (error) {
      console.error('Error processing CDC event:', error);
      await this.handleProcessingError(cdcEvent, error);
    }
  }
  
  private extractTableFromTopic(topic: string): string {
    const parts = topic.split('.');
    return parts[parts.length - 1];
  }
  
  // Sink 1: Search Index (Elasticsearch)
  private async sinkToSearchIndex(table: string, event: CdcEvent): Promise<void> {
    if (!this.shouldIndex(table)) return;
    
    const indexName = this.getSearchIndexName(table);
    const documentId = event.after?.id || event.before?.id;
    
    if (!documentId) return;
    
    try {
      switch (event.op) {
        case 'c': // Create
        case 'u': // Update
          const searchDocument = this.transformToSearchDocument(table, event);
          await this.elasticsearch.index({
            index: indexName,
            id: documentId,
            document: searchDocument,
            refresh: 'wait_for'
          });
          break;
          
        case 'd': // Delete
          await this.elasticsearch.delete({
            index: indexName,
            id: documentId
          }).catch(err => {
            if (err.meta?.statusCode !== 404) throw err;
          });
          break;
      }
    } catch (error) {
      console.error(`Search index sink error for ${table}:`, error);
      throw error;
    }
  }
  
  private transformToSearchDocument(table: string, event: CdcEvent): any {
    const data = event.op === 'd' ? event.before : event.after;
    
    switch (table) {
      case 'users':
        return {
          id: data.id,
          email: data.email,
          username: data.username,
          fullName: `${data.first_name} ${data.last_name}`,
          status: data.status,
          createdAt: data.created_at,
          updatedAt: data.updated_at,
          searchText: `${data.email} ${data.username} ${data.first_name} ${data.last_name}`
        };
        
      case 'orders':
        return {
          id: data.id,
          userId: data.user_id,
          status: data.status,
          total: parseFloat(data.total),
          currency: data.currency,
          createdAt: data.created_at,
          searchText: `${data.id} ${data.user_id}`
        };
        
      case 'products':
        return {
          id: data.id,
          name: data.name,
          description: data.description,
          price: parseFloat(data.price),
          category: data.category,
          createdAt: data.created_at,
          searchText: `${data.name} ${data.description} ${data.category}`
        };
        
      default:
        return data;
    }
  }
  
  // Sink 2: Data Warehouse (PostgreSQL)
  private async sinkToDataWarehouse(table: string, event: CdcEvent): Promise<void> {
    if (!this.shouldWarehouse(table)) return;
    
    const client = await this.dbPool.connect();
    
    try {
      await client.query('BEGIN');
      
      const warehouseTable = `${table}_warehouse`;
      
      switch (event.op) {
        case 'c':
          await this.insertToWarehouse(client, warehouseTable, event.after);
          break;
          
        case 'u':
          await this.updateToWarehouse(client, warehouseTable, event.after);
          break;
          
        case 'd':
          await this.deleteFromWarehouse(client, warehouseTable, event.before.id);
          break;
      }
      
      // Record change for audit
      await client.query(
        `INSERT INTO cdc_audit_log (source_table, operation, record_id, processed_at)
         VALUES ($1, $2, $3, NOW())`,
        [table, event.op, event.after?.id || event.before?.id]
      );
      
      await client.query('COMMIT');
      
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
  
  private async insertToWarehouse(client: any, table: string, data: any): Promise<void> {
    const columns = Object.keys(data);
    const values = Object.values(data);
    const placeholders = values.map((_, i) => `$${i + 1}`).join(', ');
    
    await client.query(
      `INSERT INTO ${table} (${columns.join(', ')})
       VALUES (${placeholders})
       ON CONFLICT (id) DO UPDATE SET
         ${columns.map((c, i) => `${c} = $${i + 1}`).join(', ')}`,
      values
    );
  }
  
  private async updateToWarehouse(client: any, table: string, data: any): Promise<void> {
    const columns = Object.keys(data);
    const values = Object.values(data);
    
    await client.query(
      `UPDATE ${table} SET 
         ${columns.map((c, i) => `${c} = $${i + 1}`).join(', ')}
       WHERE id = $${values.length}`,
      [...values, data.id]
    );
  }
  
  private async deleteFromWarehouse(client: any, table: string, id: string): Promise<void> {
    await client.query(`DELETE FROM ${table} WHERE id = $1`, [id]);
  }
  
  // Sink 3: Cache (Redis)
  private async sinkToCache(table: string, event: CdcEvent): Promise<void> {
    if (!this.shouldCache(table)) return;
    
    const cacheKey = this.getCacheKey(table, event);
    
    try {
      switch (event.op) {
        case 'c':
        case 'u':
          const cacheValue = JSON.stringify(event.after);
          await this.redis.setex(cacheKey, 3600, cacheValue); // 1 hour TTL
          break;
          
        case 'd':
          await this.redis.del(cacheKey);
          break;
      }
      
      // Invalidate list caches
      await this.invalidateListCaches(table, event);
      
    } catch (error) {
      console.error(`Cache sink error for ${table}:`, error);
      // Cache errors are non-critical, don't throw
    }
  }
  
  private async invalidateListCaches(table: string, event: CdcEvent): Promise<void> {
    // Invalidate related list caches
    const listPatterns: Record<string, string> = {
      users: 'users:list:*',
      orders: 'orders:user:*',
      products: 'products:list:*'
    };
    
    const pattern = listPatterns[table];
    if (pattern) {
      const keys = await this.redis.keys(pattern);
      if (keys.length > 0) {
        await this.redis.del(...keys);
      }
    }
  }
  
  // Sink 4: Analytics (Kafka to Analytics Topic)
  private async sinkToAnalytics(table: string, event: CdcEvent): Promise<void> {
    if (!this.shouldAnalytics(table)) return;
    
    const analyticsEvent = {
      table: table,
      operation: event.op,
      timestamp: Date.now(),
      data: {
        id: event.after?.id || event.before?.id,
        ...this.extractAnalyticsFields(table, event)
      }
    };
    
    await this.producer.send({
      topic: `analytics.${table}.events`,
      messages: [{
        key: analyticsEvent.data.id,
        value: JSON.stringify(analyticsEvent)
      }]
    });
  }
  
  private extractAnalyticsFields(table: string, event: CdcEvent): any {
    const data = event.after || event.before;
    
    switch (table) {
      case 'orders':
        return {
          userId: data.user_id,
          total: parseFloat(data.total),
          status: data.status
        };
        
      case 'products':
        return {
          price: parseFloat(data.price),
          inventory: data.inventory
        };
        
      default:
        return {};
    }
  }
  
  // Configuration methods
  private shouldIndex(table: string): boolean {
    return ['users', 'orders', 'products'].includes(table);
  }
  
  private shouldWarehouse(table: string): boolean {
    return ['users', 'orders', 'order_items', 'products'].includes(table);
  }
  
  private shouldCache(table: string): boolean {
    return ['users', 'products'].includes(table);
  }
  
  private shouldAnalytics(table: string): boolean {
    return true; // All tables to analytics
  }
  
  private getSearchIndexName(table: string): string {
    return `${table}-search-index`;
  }
  
  private getCacheKey(table: string, event: CdcEvent): string {
    const id = event.after?.id || event.before?.id;
    return `${table}:${id}`;
  }
  
  private async handleProcessingError(event: CdcEvent, error: Error): Promise<void> {
    // Send to dead letter topic
    await this.producer.send({
      topic: 'cdc.dead-letter',
      messages: [{
        key: event.after?.id || event.before?.id,
        value: JSON.stringify({
          event: event,
          error: error.message,
          retryCount: 0,
          failedAt: new Date().toISOString()
        })
      }]
    );
  }
  
  async shutdown() {
    this.running = false;
    await this.consumer.disconnect();
    await this.producer.disconnect();
    await this.dbPool.end();
    await this.elasticsearch.close();
    this.redis.disconnect();
  }
}

// Configuration interface
interface PipelineConfig {
  kafkaBrokers: string[];
  databaseUrl: string;
  elasticsearchUrl: string;
  redisUrl: string;
}

// CDC Event interface
interface CdcEvent {
  op: 'c' | 'u' | 'd' | 'r';
  before?: Record<string, any>;
  after?: Record<string, any>;
  source: {
    table: string;
    db: string;
  };
}
```

---

## Conclusion

This documentation has provided comprehensive coverage of Database Change Data Capture patterns and implementations essential for building modern, event-driven data systems. The patterns and implementations presented address the most critical aspects of CDC from architectural design through practical implementation.

The key takeaways from this documentation include the understanding that CDC serves as the foundation for event-driven architectures, providing reliable, low-latency change propagation from databases to downstream systems. Debezium provides a robust, well-tested CDC implementation that supports multiple databases and integrates seamlessly with Kafka. Event-driven patterns like event sourcing and CQRS, when combined with CDC, enable powerful architectures that maintain complete change history while supporting diverse query patterns. Stream processing frameworks like Kafka Streams and Apache Flink provide sophisticated capabilities for transforming, aggregating, and analyzing CDC events in real-time. Reactive pipelines enable efficient, scalable processing of high-volume change streams with proper backpressure handling and error management.

These patterns form the essential toolkit for building modern data platforms that can respond to database changes in real-time, maintain multiple data representations, and scale to handle enterprise workloads.

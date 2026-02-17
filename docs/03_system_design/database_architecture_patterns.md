# Database Architecture Patterns

## Overview

Modern applications require sophisticated architectural patterns to handle complex business requirements, scale efficiently, and maintain data consistency across distributed systems. This document explores three powerful database architecture patterns—CQRS, Event Sourcing, and the Saga pattern—that enable building systems with high availability, event-driven capabilities, and distributed transaction management.

These patterns emerged from real-world challenges in enterprise software development. As applications grew in complexity and moved to distributed architectures, traditional monolithic approaches to data management proved insufficient. CQRS, Event Sourcing, and Saga patterns provide solutions to these challenges by fundamentally rethinking how data flows through a system.

Understanding when and how to apply these patterns is crucial for building robust, scalable systems. Each pattern addresses specific challenges but also introduces complexity. The key is knowing when the benefits outweigh the costs and how to implement the pattern correctly to avoid common pitfalls.

This guide assumes familiarity with relational databases, basic system design concepts, and application development patterns. The examples use Python and TypeScript to demonstrate implementation details, but the concepts apply across programming languages and database systems.

## CQRS (Command Query Responsibility Segregation)

### Understanding CQRS

CQRS is an architectural pattern that separates read and write operations for a data store. Rather than using a single model for both reading and writing data, CQRS splits the application into two distinct parts: the command side (writes) and the query side (reads). This separation allows each side to be optimized independently for its specific workload.

In traditional CRUD applications, the same data model handles both creating, updating, and deleting records (commands) as well as retrieving them (queries). This approach works well for simple applications but becomes problematic as systems grow in complexity. Read and write workloads often have different performance requirements, scalability needs, and optimization strategies.

CQRS addresses these challenges by allowing the command and query models to evolve independently. The write model can be normalized for data integrity while the read model can be denormalized for query performance. This flexibility enables building systems that handle high read throughput while maintaining complex business rules on the write side.

### CQRS Architecture

The CQRS pattern consists of several key components working together:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
├─────────────────────────┬───────────────────────────────────────┤
│   Command Side         │         Query Side                     │
│  ┌─────────────────┐   │   ┌─────────────────────────────────┐   │
│  │ Command Handler │   │   │ Query Handler                  │   │
│  └────────┬────────┘   │   └───────────────┬────────────────┘   │
│           │            │                   │                    │
│           ▼            │                   ▼                    │
│  ┌─────────────────┐   │   ┌─────────────────────────────────┐   │
│  │  Write Model     │   │   │ Read Model (Materialized)     │   │
│  │  (Normalized)   │   │   │ (Denormalized)                 │   │
│  └────────┬────────┘   │   └───────────────┬────────────────┘   │
│           │            │                   │                    │
└───────────┼────────────┴───────────────────┼────────────────────┘
            │                                │
            ▼                                ▼
┌───────────────────────┐        ┌───────────────────────────────┐
│   Write Database      │        │     Read Database            │
│   (Primary/Leader)    │        │     (Replica/Denormalized)   │
└───────────────────────┘        └───────────────────────────────┘
```

The command side processes business operations and maintains the authoritative data store. When a command succeeds, it modifies the write database. These changes are then propagated to the read database through a synchronization mechanism, which can be synchronous or asynchronous depending on consistency requirements.

### Implementing CQRS

Let's implement a CQRS-based order management system:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum
import asyncio

# ============== COMMANDS ==============

@dataclass
class CreateOrderCommand:
    customer_id: int
    items: List[dict]
    shipping_address: str

@dataclass  
class UpdateOrderStatusCommand:
    order_id: int
    new_status: str

@dataclass
class CancelOrderCommand:
    order_id: int
    reason: str

# ============== QUERIES ==============

@dataclass
class OrderSummary:
    order_id: int
    customer_name: str
    status: str
    total_amount: float
    item_count: int
    created_at: datetime

@dataclass
class OrderDetails:
    order_id: int
    customer_id: int
    customer_name: str
    items: List[dict]
    shipping_address: str
    status: str
    total_amount: float
    created_at: datetime
    updated_at: datetime

# ============== COMMAND HANDLERS ==============

class OrderCommandHandler:
    def __init__(self, write_db_pool):
        self.write_db = write_db_pool
    
    async def handle_create_order(self, command: CreateOrderCommand) -> int:
        """Process order creation with business logic validation"""
        
        # Validate business rules
        if not command.items:
            raise ValueError("Order must have at least one item")
        
        # Calculate total
        total_amount = sum(item["price"] * item["quantity"] for item in command.items)
        
        async with self.write_db.acquire() as conn:
            # Create order in normalized write model
            order_id = await conn.fetchval("""
                INSERT INTO orders (customer_id, status, total_amount, shipping_address, created_at)
                VALUES ($1, 'PENDING', $2, $3, NOW())
                RETURNING id
            """, command.customer_id, total_amount, command.shipping_address)
            
            # Insert order items
            for item in command.items:
                await conn.execute("""
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                    VALUES ($1, $2, $3, $4)
                """, order_id, item["product_id"], item["quantity"], item["price"])
            
            # Publish event for read model sync
            await self._publish_order_created_event(order_id, command)
            
            return order_id
    
    async def handle_update_status(self, command: UpdateOrderStatusCommand) -> bool:
        """Update order status with state machine validation"""
        
        valid_transitions = {
            "PENDING": ["CONFIRMED", "CANCELLED"],
            "CONFIRMED": ["SHIPPED", "CANCELLED"],
            "SHIPPED": ["DELIVERED"],
            "DELIVERED": [],
            "CANCELLED": []
        }
        
        async with self.write_db.acquire() as conn:
            # Get current status
            current_status = await conn.fetchval(
                "SELECT status FROM orders WHERE id = $1",
                command.order_id
            )
            
            if not current_status:
                raise ValueError(f"Order {command.order_id} not found")
            
            # Validate state transition
            if command.new_status not in valid_transitions.get(current_status, []):
                raise ValueError(
                    f"Invalid status transition from {current_status} to {command.new_status}"
                )
            
            # Update status
            await conn.execute("""
                UPDATE orders SET status = $1, updated_at = NOW()
                WHERE id = $2
            """, command.new_status, command.order_id)
            
            await self._publish_status_changed_event(command.order_id, current_status, command.new_status)
            
            return True
    
    async def _publish_order_created_event(self, order_id: int, command: CreateOrderCommand):
        """Publish event for read model synchronization"""
        # In production, this would publish to a message queue
        event = {
            "type": "ORDER_CREATED",
            "order_id": order_id,
            "customer_id": command.customer_id,
            "items": command.items,
            "timestamp": datetime.utcnow().isoformat()
        }
        await event_store.append("orders", event)

# ============== QUERY HANDLERS ==============

class OrderQueryHandler:
    def __init__(self, read_db_pool):
        self.read_db = read_db_pool
    
    async def get_order_summary(self, order_id: int) -> Optional[OrderSummary]:
        """Query denormalized order summary"""
        row = await self.read_db.fetchrow("""
            SELECT 
                o.id AS order_id,
                c.name AS customer_name,
                o.status,
                o.total_amount,
                COUNT(oi.id) AS item_count,
                o.created_at
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            LEFT JOIN order_items oi ON o.id = oi.order_id
            WHERE o.id = $1
            GROUP BY o.id, c.name, o.status, o.total_amount, o.created_at
        """, order_id)
        
        if not row:
            return None
        
        return OrderSummary(
            order_id=row["order_id"],
            customer_name=row["customer_name"],
            status=row["status"],
            total_amount=float(row["total_amount"]),
            item_count=row["item_count"],
            created_at=row["created_at"]
        )
    
    async def list_recent_orders(self, customer_id: int, limit: int = 50) -> List[OrderSummary]:
        """List recent orders for a customer"""
        rows = await self.read_db.fetch("""
            SELECT 
                o.id AS order_id,
                c.name AS customer_name,
                o.status,
                o.total_amount,
                COUNT(oi.id) AS item_count,
                o.created_at
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            LEFT JOIN order_items oi ON o.id = oi.order_id
            WHERE o.customer_id = $1
            GROUP BY o.id, c.name, o.status, o.total_amount, o.created_at
            ORDER BY o.created_at DESC
            LIMIT $2
        """, customer_id, limit)
        
        return [
            OrderSummary(
                order_id=row["order_id"],
                customer_name=row["customer_name"],
                status=row["status"],
                total_amount=float(row["total_amount"]),
                item_count=row["item_count"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
    
    async def get_order_statistics(self, start_date: datetime, end_date: datetime) -> dict:
        """Get aggregated order statistics"""
        return await self.read_db.fetchrow("""
            SELECT 
                COUNT(*) AS total_orders,
                SUM(total_amount) AS total_revenue,
                AVG(total_amount) AS avg_order_value,
                COUNT(DISTINCT customer_id) AS unique_customers
            FROM orders
            WHERE created_at BETWEEN $1 AND $2
        """, start_date, end_date)
```

### Read Model Synchronization

The synchronization between write and read models is a critical component of CQRS. There are several approaches:

```python
class ReadModelSynchronizer:
    """Synchronizes write database to read database"""
    
    def __init__(self, write_db_pool, read_db_pool, event_store):
        self.write_db = write_db_pool
        self.read_db = read_db_pool
        self.event_store = event_store
    
    async def sync_order_created(self, event: dict):
        """Handle ORDER_CREATED event - populate read model"""
        order_id = event["order_id"]
        
        async with self.read_db.acquire() as conn:
            # Build denormalized order record for fast reads
            await conn.execute("""
                INSERT INTO order_read_model (
                    order_id, customer_id, customer_name, customer_email,
                    status, total_amount, item_count, shipping_address,
                    created_at, updated_at
                )
                SELECT 
                    o.id, o.customer_id, c.name, c.email,
                    o.status, o.total_amount, 
                    (SELECT COUNT(*) FROM order_items WHERE order_id = o.id),
                    o.shipping_address,
                    o.created_at, o.updated_at
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                WHERE o.id = $1
                ON CONFLICT (order_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    total_amount = EXCLUDED.total_amount,
                    item_count = EXCLUDED.item_count,
                    updated_at = EXCLUDED.updated_at
            """, order_id)
    
    async def sync_order_items(self, event: dict):
        """Sync order items for detailed queries"""
        order_id = event["order_id"]
        
        async with self.read_db.acquire() as conn:
            # Denormalize items into the read model for fast access
            await conn.execute("""
                INSERT INTO order_items_read (order_id, items_json)
                SELECT 
                    o.id,
                    json_agg(json_build_object(
                        'product_id', oi.product_id,
                        'product_name', p.name,
                        'quantity', oi.quantity,
                        'unit_price', oi.unit_price
                    ))
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                JOIN products p ON oi.product_id = p.id
                WHERE o.id = $1
                GROUP BY o.id
                ON CONFLICT (order_id) DO UPDATE SET
                    items_json = EXCLUDED.items_json
            """, order_id)
    
    async def sync_order_status_changed(self, event: dict):
        """Handle ORDER_STATUS_CHANGED event"""
        order_id = event["order_id"]
        new_status = event["new_status"]
        
        async with self.read_db.acquire() as conn:
            await conn.execute("""
                UPDATE order_read_model 
                SET status = $1, updated_at = NOW()
                WHERE order_id = $2
            """, new_status, order_id)
```

### CQRS Trade-offs and When to Use It

CQRS provides significant benefits for certain types of applications:

**Benefits**:
- Independent scaling of read and write workloads
- Optimized read models for specific query patterns
- Flexibility to use different databases for commands and queries
- Clear separation of concerns in complex domains

**Costs and Complexity**:
- Eventual consistency between read and write models
- Increased system complexity and maintenance burden
- Potential for divergence if synchronization fails
- More difficult debugging due to asynchronous processing

**When to Use CQRS**:
- Applications with significantly different read and write patterns
- Systems requiring high read scalability
- Complex domains with different read and write models
- Event-driven architectures
- Systems that benefit from denormalized read models

**When NOT to Use CQRS**:
- Simple CRUD applications with balanced read/write loads
- Systems requiring strong consistency
- Teams without experience with event-driven patterns
- Applications where the complexity outweighs the benefits

## Event Sourcing with Databases

### Understanding Event Sourcing

Event Sourcing is a pattern where the state of the application is stored as a sequence of events rather than as a current state snapshot. Instead of storing just the current balance of an account, you store every transaction that affected that balance. The current state can be reconstructed by replaying all events.

This pattern provides complete auditability, as every change to the system is captured as an immutable event. It also enables powerful capabilities like temporal queries (what was the state at a specific point in time), event replay for debugging, and building new read models without modifying the event log.

Event sourcing naturally complements CQRS. The event log serves as the write model, while materialized views built from events serve as read models. Together, they form a powerful foundation for building event-driven systems with complete audit trails.

> **Related Pattern**: For capturing database changes in real-time for event sourcing, see [Change Data Capture (CDC)](../02_intermediate/05_realtime_streaming_database_patterns.md) which provides the infrastructure for feeding database changes into event streams.

### Implementing Event Sourcing

Let's build an event-sourced bank account system:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Callable, Any
import json
import asyncio
from abc import ABC, abstractmethod

# ============== EVENTS ==============

@dataclass
class Event(ABC):
    event_id: str
    aggregate_id: str
    timestamp: datetime
    version: int
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "event_type": self.__class__.__name__,
            "data": self._serialize()
        }
    
    @abstractmethod
    def _serialize(self) -> dict:
        pass

@dataclass
class AccountCreated(Event):
    initial_balance: float
    owner_name: str
    
    def _serialize(self) -> dict:
        return {
            "initial_balance": self.initial_balance,
            "owner_name": self.owner_name
        }

@dataclass
class MoneyDeposited(Event):
    amount: float
    deposit_id: str
    
    def _serialize(self) -> dict:
        return {
            "amount": self.amount,
            "deposit_id": self.deposit_id
        }

@dataclass
class MoneyWithdrawn(Event):
    amount: float
    withdrawal_id: str
    
    def _serialize(self) -> dict:
        return {
            "amount": self.amount,
            "withdrawal_id": self.withdrawal_id
        }

@dataclass
class MoneyTransferred(Event):
    amount: float
    from_account_id: str
    to_account_id: str
    transfer_id: str
    
    def _serialize(self) -> dict:
        return {
            "amount": self.amount,
            "from_account_id": self.from_account_id,
            "to_account_id": self.to_account_id,
            "transfer_id": self.transfer_id
        }

# ============== AGGREGATE ==============

@dataclass
class BankAccount:
    account_id: str
    owner_name: str
    balance: float = 0.0
    version: int = 0
    pending_events: List[Event] = field(default_factory=list)
    
    def create_account(self, account_id: str, owner_name: str, initial_balance: float):
        """Factory method for creating new accounts"""
        event = AccountCreated(
            event_id=f"evt_{account_id}_{datetime.utcnow().timestamp()}",
            aggregate_id=account_id,
            timestamp=datetime.utcnow(),
            version=1,
            initial_balance=initial_balance,
            owner_name=owner_name
        )
        self._apply_event(event)
        self.pending_events.append(event)
    
    def deposit(self, amount: float) -> str:
        """Deposit money into account"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        deposit_id = f"dep_{datetime.utcnow().timestamp()}"
        event = MoneyDeposited(
            event_id=deposit_id,
            aggregate_id=self.account_id,
            timestamp=datetime.utcnow(),
            version=self.version + 1,
            amount=amount,
            deposit_id=deposit_id
        )
        self._apply_event(event)
        self.pending_events.append(event)
        return deposit_id
    
    def withdraw(self, amount: float) -> str:
        """Withdraw money from account"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if self.balance < amount:
            raise ValueError(f"Insufficient funds: balance={self.balance}, requested={amount}")
        
        withdrawal_id = f"wd_{datetime.utcnow().timestamp()}"
        event = MoneyWithdrawn(
            event_id=withdrawal_id,
            aggregate_id=self.account_id,
            timestamp=datetime.utcnow(),
            version=self.version + 1,
            amount=amount,
            withdrawal_id=withdrawal_id
        )
        self._apply_event(event)
        self.pending_events.append(event)
        return withdrawal_id
    
    def transfer(self, amount: float, to_account_id: str) -> str:
        """Transfer money to another account"""
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")
        
        if self.balance < amount:
            raise ValueError(f"Insufficient funds for transfer: balance={self.balance}")
        
        transfer_id = f"trf_{datetime.utcnow().timestamp()}"
        event = MoneyTransferred(
            event_id=transfer_id,
            aggregate_id=self.account_id,
            timestamp=datetime.utcnow(),
            version=self.version + 1,
            amount=amount,
            from_account_id=self.account_id,
            to_account_id=to_account_id,
            transfer_id=transfer_id
        )
        self._apply_event(event)
        self.pending_events.append(event)
        return transfer_id
    
    def _apply_event(self, event: Event):
        """Apply event to aggregate state"""
        if isinstance(event, AccountCreated):
            self.account_id = event.aggregate_id
            self.owner_name = event.owner_name
            self.balance = event.initial_balance
            self.version = event.version
        
        elif isinstance(event, MoneyDeposited):
            self.balance += event.amount
            self.version = event.version
        
        elif isinstance(event, MoneyWithdrawn):
            self.balance -= event.amount
            self.version = event.version
        
        elif isinstance(event, MoneyTransferred):
            self.balance -= event.amount
            self.version = event.version
    
    def clear_pending_events(self):
        """Clear pending events after persistence"""
        self.pending_events.clear()
    
    @staticmethod
    def from_events(events: List[Event]) -> 'BankAccount':
        """Reconstruct account from event history"""
        account = BankAccount(account_id="", owner_name="")
        for event in sorted(events, key=lambda e: e.version):
            account._apply_event(event)
        return account
```

### Event Store Implementation

The event store is the persistence layer for events:

```python
import uuid
from typing import List, Optional, Callable
from datetime import datetime

class EventStore:
    """Persistent event store for event sourcing"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def initialize(self):
        """Create event store tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id VARCHAR(255) PRIMARY KEY,
                    aggregate_id VARCHAR(255) NOT NULL,
                    aggregate_type VARCHAR(100) NOT NULL,
                    version INTEGER NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata JSONB,
                    UNIQUE (aggregate_id, version)
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_aggregate_id
                ON events (aggregate_id, version)
            """)
    
    async def append(self, aggregate_id: str, event: Event):
        """Append event to the event stream"""
        async with self.db_pool.acquire() as conn:
            # Check for duplicate version (optimistic concurrency)
            existing = await conn.fetchval("""
                SELECT version FROM events 
                WHERE aggregate_id = $1 
                ORDER BY version DESC LIMIT 1
            """, aggregate_id)
            
            expected_version = (existing or 0) + 1
            if event.version != expected_version:
                raise ConcurrencyException(
                    f"Version mismatch: expected {expected_version}, got {event.version}"
                )
            
            await conn.execute("""
                INSERT INTO events (
                    event_id, aggregate_id, aggregate_type, version,
                    event_type, event_data, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                event.event_id,
                aggregate_id,
                "BankAccount",
                event.version,
                event.__class__.__name__,
                json.dumps(event.to_dict()["data"]),
                event.timestamp
            )
    
    async def get_events_for_aggregate(self, aggregate_id: str) -> List[dict]:
        """Retrieve all events for an aggregate"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM events 
                WHERE aggregate_id = $1 
                ORDER BY version
            """, aggregate_id)
            
            return [dict(row) for row in rows]
    
    async def get_all_events(
        self, 
        start_version: int = 0, 
        limit: int = 100
    ) -> List[dict]:
        """Get events across all aggregates"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM events 
                WHERE version > $1 
                ORDER BY timestamp, version
                LIMIT $2
            """, start_version, limit)
            
            return [dict(row) for row in rows]

class ConcurrencyException(Exception):
    """Raised when optimistic concurrency check fails"""
    pass
```

### Projections and Materialized Views

Build materialized views from events:

```python
class AccountProjection:
    """Materialized view of account balances"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def initialize(self):
        """Create projection table"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS account_balances (
                    account_id VARCHAR(255) PRIMARY KEY,
                    owner_name VARCHAR(255) NOT NULL,
                    balance DECIMAL(19,4) NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
    
    async def project(self, events: List[dict]):
        """Project events to materialized view"""
        for event in events:
            event_type = event["event_type"]
            event_data = event["event_data"]
            
            async with self.db_pool.acquire() as conn:
                if event_type == "AccountCreated":
                    await conn.execute("""
                        INSERT INTO account_balances (account_id, owner_name, balance, version, updated_at)
                        VALUES ($1, $2, $3, $4, $5)
                    """,
                        event["aggregate_id"],
                        event_data["owner_name"],
                        event_data["initial_balance"],
                        event["version"],
                        event["timestamp"]
                    )
                
                elif event_type == "MoneyDeposited":
                    await conn.execute("""
                        UPDATE account_balances 
                        SET balance = balance + $1, version = $2, updated_at = $3
                        WHERE account_id = $4
                    """,
                        event_data["amount"],
                        event["version"],
                        event["timestamp"],
                        event["aggregate_id"]
                    )
                
                elif event_type == "MoneyWithdrawn":
                    await conn.execute("""
                        UPDATE account_balances 
                        SET balance = balance - $1, version = $2, updated_at = $3
                        WHERE account_id = $4
                    """,
                        event_data["amount"],
                        event["version"],
                        event["timestamp"],
                        event["aggregate_id"]
                    )
```

## Saga Patterns for Distributed Transactions

### Understanding the Saga Pattern

In distributed systems, traditional ACID transactions don't work across multiple services or databases. The Saga pattern provides a way to manage distributed transactions by breaking them into a sequence of local transactions. Each local transaction updates the database and publishes an event or message that triggers the next step in the saga.

If a step fails, the saga executes compensating transactions to undo the changes made by previous steps. For example, if an order creation saga fails at the payment step, it compensates by canceling the order and refunding the payment.

Sagas differ from ACID transactions in their consistency guarantees. Instead of atomic commitment, sagas provide eventual consistency—the system will eventually reach a consistent state, either by completing all steps or by undoing all changes through compensation.

### Implementing Choreography-Based Sagas

In choreography-based sagas, each service publishes events that other services react to:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Callable
import asyncio
import uuid

# ============== SAGA DEFINITIONS ==============

@dataclass
class OrderSagaState:
    saga_id: str
    order_id: Optional[int]
    customer_id: int
    items: List[dict]
    total_amount: float
    current_step: str
    completed_steps: List[str]
    failed_step: Optional[str]
    compensation_data: dict

class OrderSaga:
    """Orchestrates the order creation saga"""
    
    def __init__(
        self,
        order_service,
        payment_service,
        inventory_service,
        shipping_service,
        event_bus
    ):
        self.order_service = order_service
        self.payment_service = payment_service
        self.inventory_service = inventory_service
        self.shipping_service = shipping_service
        self.event_bus = event_bus
    
    async def start_saga(self, customer_id: int, items: List[dict]) -> str:
        """Initiate the order creation saga"""
        saga_id = f"saga_{uuid.uuid4().hex[:12]}"
        
        # Initialize saga state
        state = OrderSagaState(
            saga_id=saga_id,
            order_id=None,
            customer_id=customer_id,
            items=items,
            total_amount=sum(item["price"] * item["quantity"] for item in items),
            current_step="CREATE_ORDER",
            completed_steps=[],
            failed_step=None,
            compensation_data={}
        )
        
        # Store initial state
        await self._save_state(state)
        
        # Start the saga
        await self._execute_step(state, "CREATE_ORDER")
        
        return saga_id
    
    async def _execute_step(self, state: OrderSagaState, step_name: str):
        """Execute a saga step with compensation support"""
        try:
            state.current_step = step_name
            
            if step_name == "CREATE_ORDER":
                order_id = await self.order_service.create_order(
                    state.customer_id, state.items
                )
                state.order_id = order_id
                state.compensation_data["order_id"] = order_id
                await self._save_state(state)
                
                # Publish event to trigger next step
                await self.event_bus.publish("ORDER_CREATED", {
                    "saga_id": state.saga_id,
                    "order_id": order_id,
                    "customer_id": state.customer_id,
                    "total_amount": state.total_amount
                })
                
                await self._execute_step(state, "RESERVE_INVENTORY")
            
            elif step_name == "RESERVE_INVENTORY":
                reservation = await self.inventory_service.reserve_items(
                    state.items
                )
                state.compensation_data["inventory_reservation"] = reservation
                state.completed_steps.append(step_name)
                await self._save_state(state)
                
                await self.event_bus.publish("INVENTORY_RESERVED", {
                    "saga_id": state.saga_id,
                    "order_id": state.order_id,
                    "reservation_id": reservation["id"]
                })
                
                await self._execute_step(state, "PROCESS_PAYMENT")
            
            elif step_name == "PROCESS_PAYMENT":
                payment = await self.payment_service.process_payment(
                    state.customer_id,
                    state.total_amount,
                    f"ORDER_{state.order_id}"
                )
                state.compensation_data["payment_id"] = payment["id"]
                state.completed_steps.append(step_name)
                await self._save_state(state)
                
                await self.event_bus.publish("PAYMENT_PROCESSED", {
                    "saga_id": state.saga_id,
                    "order_id": state.order_id,
                    "payment_id": payment["id"]
                })
                
                await self._execute_step(state, "SCHEDULE_SHIPPING")
            
            elif step_name == "SCHEDULE_SHIPPING":
                shipping = await self.shipping_service.schedule_shipping(
                    state.order_id,
                    state.items
                )
                state.compensation_data["shipping_id"] = shipping["id"]
                state.completed_steps.append(step_name)
                await self._save_state(state)
                
                # Saga completed successfully
                await self.event_bus.publish("ORDER_SAGA_COMPLETED", {
                    "saga_id": state.saga_id,
                    "order_id": state.order_id
                })
        
        except Exception as e:
            state.failed_step = step_name
            await self._save_state(state)
            await self._compensate(state)
            raise SagaFailedException(f"Saga {state.saga_id} failed at {step_name}: {e}")
    
    async def _compensate(self, state: OrderSagaState):
        """Execute compensating transactions for completed steps"""
        # Compensate in reverse order
        for step in reversed(state.completed_steps):
            try:
                if step == "SCHEDULE_SHIPPING":
                    await self.shipping_service.cancel_shipping(
                        state.compensation_data["shipping_id"]
                    )
                
                if step == "PROCESS_PAYMENT":
                    await self.payment_service.refund_payment(
                        state.compensation_data["payment_id"]
                    )
                
                if step == "RESERVE_INVENTORY":
                    await self.inventory_service.release_reservation(
                        state.compensation_data["inventory_reservation"]["id"]
                    )
                
                if step == "CREATE_ORDER":
                    await self.order_service.cancel_order(
                        state.compensation_data["order_id"]
                    )
            
            except CompensationFailedException as ce:
                # Log for manual intervention
                await self._create_compensation_alert(state, step, ce)

class SagaFailedException(Exception):
    pass

class CompensationFailedException(Exception):
    pass
```

### Implementing Orchestration-Based Sagas

Orchestration-based sagas use a central coordinator that tells each service what to do:

```python
class OrderSagaOrchestrator:
    """Central orchestrator for order saga"""
    
    def __init__(self, services: dict):
        self.services = services
    
    async def execute(self, order_request: dict) -> dict:
        """Execute saga with full orchestration"""
        saga_id = f"saga_{uuid.uuid4().hex[:12]}"
        context = {
            "saga_id": saga_id,
            "order_request": order_request,
            "completed_steps": [],
            "compensation_handlers": []
        }
        
        try:
            # Step 1: Create order
            order = await self._execute_step(
                context,
                "create_order",
                self.services["order"].create,
                lambda ctx: self.services["order"].cancel(ctx["order"]["id"])
            )
            
            # Step 2: Reserve inventory
            reservation = await self._execute_step(
                context,
                "reserve_inventory",
                lambda: self.services["inventory"].reserve(order_request["items"]),
                lambda ctx: self.services["inventory"].release(ctx["reservation"]["id"])
            )
            
            # Step 3: Process payment
            payment = await self._execute_step(
                context,
                "process_payment",
                lambda: self.services["payment"].charge(
                    order_request["customer_id"],
                    order_request["total_amount"]
                ),
                lambda ctx: self.services["payment"].refund(ctx["payment"]["id"])
            )
            
            # Step 4: Schedule shipping
            shipping = await self._execute_step(
                context,
                "schedule_shipping",
                lambda: self.services["shipping"].schedule(order),
                lambda ctx: self.services["shipping"].cancel(ctx["shipping"]["id"])
            )
            
            return {"status": "COMPLETED", "saga_id": saga_id, "order": order}
        
        except Exception as e:
            # Compensate all completed steps
            await self._compensate(context)
            return {"status": "FAILED", "saga_id": saga_id, "error": str(e)}
    
    async def _execute_step(self, context: dict, step_name: str, action, compensation):
        """Execute a single saga step"""
        result = await action()
        
        context["completed_steps"].append({
            "name": step_name,
            "result": result,
            "compensate": compensation
        })
        
        return result
    
    async def _compensate(self, context: dict):
        """Execute compensation in reverse order"""
        for step in reversed(context["completed_steps"]):
            try:
                compensation_fn = step["compensate"]
                await compensation_fn(context)
            except Exception as e:
                # Log for manual intervention
                print(f"Compensation failed for {step['name']}: {e}")
```

### Saga Failure Recovery

Implement robust saga failure handling:

```python
class SagaRecoveryManager:
    """Handles saga failures and recovery"""
    
    def __init__(self, saga_store, orchestrator):
        self.saga_store = saga_store
        self.orchestrator = orchestrator
    
    async def recover_pending_sagas(self):
        """Find and recover pending sagas"""
        pending = await self.saga_store.find_pending(max_age_minutes=30)
        
        for saga in pending:
            if saga["failed_step"]:
                await self._retry_compensation(saga)
            else:
                await self._resume_saga(saga)
    
    async def _resume_saga(self, saga: dict):
        """Resume a saga that was interrupted"""
        # Load saga state and continue from last step
        await self.orchestrator.resume(saga["saga_id"])
    
    async def _retry_compensation(self, saga: dict):
        """Retry compensation for failed saga"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                await self.orchestrator.compensate(saga["saga_id"])
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    await self._escalate_for_manual_intervention(saga, e)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _escalate_for_manual_intervention(self, saga: dict, error: Exception):
        """Escalate failed saga for manual resolution"""
        await self.saga_store.mark_requiring_manual_intervention(saga["saga_id"])
        # Send alert to operations team
        await self._send_alert(saga, error)
```

## Best Practices Summary

### CQRS Best Practices

1. **Start simple**: Don't implement CQRS unless you have clear evidence that read and write workloads differ significantly.

2. **Choose synchronization strategy carefully**: Synchronous sync provides strong consistency but negates some benefits. Asynchronous sync scales better but introduces eventual consistency.

3. **Handle divergence**: Monitor and alert on synchronization lag between write and read models.

4. **Design read models for specific queries**: Don't try to make a single read model serve all query needs.

### Event Sourcing Best Practices

1. **Design events carefully**: Events are permanent. Include all information needed to reconstruct state and understand the change.

2. **Version your events**: Events will evolve. Include version information and handle migration.

3. **Keep events small**: Large events impact performance and storage. Store references to external data when possible.

4. **Implement idempotent handlers**: Event handlers may run multiple times. Design for idempotency.

### Saga Best Practices

1. **Design compensating actions**: Always design compensation before implementing forward actions.

2. **Set timeouts appropriately**: Long-running sagas hold resources. Set appropriate timeouts for each step.

3. **Monitor saga health**: Track saga completion times and failure rates.

4. **Plan for manual intervention**: Some sagas will fail in ways that cannot be automatically compensated. Build alerting and manual intervention capabilities.

5. **Test failure scenarios**: Regularly test saga failure and compensation paths.

These architecture patterns provide powerful capabilities for building sophisticated, scalable systems. However, they introduce significant complexity. Apply them selectively where their benefits justify the costs, and invest in proper tooling and monitoring to manage that complexity effectively.

## See Also

- [Real-time Streaming Patterns](../02_intermediate/05_realtime_streaming_database_patterns.md) - CDC and event-driven architecture
- [Database API Design](../02_intermediate/04_database_api_design.md) - Repository and Unit of Work patterns
- [Database Troubleshooting](../04_tutorials/troubleshooting/02_database_troubleshooting_debugging.md) - Monitoring distributed systems
- [Edge Computing Databases](../03_advanced/edge_computing_databases.md) - Event sourcing at the edge

# Database API Design and Integration Patterns

## Overview

Building robust applications requires careful consideration of how your application interacts with databases. This document covers the essential patterns, best practices, and considerations for designing database APIs that are performant, secure, and maintainable. Whether you're building a REST API, GraphQL endpoint, or internal service layer, understanding these patterns will help you create systems that scale and perform well under production workloads.

The relationship between your application layer and database is foundational to system architecture. Poorly designed database integrations can lead to connection exhaustion, security vulnerabilities, performance bottlenecks, and maintenance nightmares. Conversely, well-designed integrations provide a clean abstraction layer that enables your application to evolve without constant database schema changes affecting business logic.

This guide assumes familiarity with basic database concepts and SQL. We'll explore practical implementation patterns that work across different database systems while highlighting important trade-offs and considerations for production systems.

## REST API Design for Database Operations

### Resource-Oriented Design

When designing REST APIs that interact with databases, adopt a resource-oriented approach that maps cleanly to your data model. Each resource should represent a noun (user, order, product) rather than a verb (getUser, createOrder). This philosophy extends naturally to database operations where the HTTP methods map to CRUD actions: POST for creation, GET for reading, PUT/PATCH for updates, and DELETE for removal.

Consider an e-commerce system where you need to manage products. Instead of creating endpoints like `/api/createProduct` or `/api/getProductById`, design your API around the resource:

```
GET    /api/products          # List all products
GET    /api/products/{id}    # Get single product
POST   /api/products          # Create new product
PUT    /api/products/{id}    # Update product (full replacement)
PATCH  /api/products/{id}    # Partial update
DELETE /api/products/{id}    # Delete product
```

This approach provides a consistent interface that developers find intuitive and that maps naturally to database operations under the hood.

### Query Parameter Design for Database Operations

REST APIs must handle complex database queries through URL parameters. Implement standardized query parameter patterns that translate to database operations:

**Pagination Parameters**: Use `limit` and `offset` or cursor-based pagination with `cursor` and `limit`. Cursor-based pagination performs better for large datasets because it avoids offset calculations on large tables:

```python
# Python FastAPI example for cursor-based pagination
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/api/products")
async def list_products(
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = None,
    category: Optional[str] = None
):
    query = "SELECT * FROM products WHERE 1=1"
    params = []
    
    if cursor:
        query += " AND id > %s"
        params.append(cursor)
    
    if category:
        query += " AND category = %s"
        params.append(category)
    
    query += " ORDER BY id LIMIT %s"
    params.append(limit + 1)  # Fetch one extra to check for more results
    
    # Execute query and check if there are more results
    results = await db.fetch_all(query, params)
    has_more = len(results) > limit
    products = results[:limit]
    
    next_cursor = products[-1]["id"] if has_more and products else None
    
    return {
        "data": products,
        "pagination": {
            "next_cursor": next_cursor,
            "has_more": has_more
        }
    }
```

**Filtering and Sorting**: Provide filter parameters that map to database columns and sorting parameters that control order:

```
GET /api/products?price_min=10&price_max=100&sort=price:asc&category=electronics
```

### Error Handling and Status Codes

Database operations require careful error handling that provides meaningful feedback while maintaining security. Use appropriate HTTP status codes and include detailed error information:

| Status Code | Use Case |
|-------------|----------|
| 200 | Successful GET, PUT, or PATCH |
| 201 | Successful POST creating a new resource |
| 204 | Successful DELETE with no content to return |
| 400 | Invalid input data or malformed request |
| 401 | Authentication required |
| 403 | Authenticated but not authorized |
| 404 | Resource not found |
| 409 | Conflict (e.g., duplicate key violation) |
| 422 | Validation errors on business logic |
| 429 | Rate limit exceeded |
| 500 | Internal server error (database failures) |
| 503 | Service unavailable (database connection pool exhausted) |

For database errors, log the detailed error internally but return a generic message to the client:

```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

@app.post("/api/products")
async def create_product(product_data: dict):
    try:
        product_id = await db.execute(
            "INSERT INTO products (name, price, category) VALUES (%s, %s, %s)",
            [product_data["name"], product_data["price"], product_data["category"]]
        )
        return {"id": product_id, "status": "created"}
    except psycopg2.IntegrityError as e:
        # Log full error internally
        logger.error(f"Database integrity error: {e}")
        # Return generic error to client
        raise HTTPException(status_code=409, detail="Product with this name already exists")
    except Exception as e:
        logger.error(f"Database error creating product: {e}")
        raise HTTPException(status_code=500, detail="Failed to create product")
```

## GraphQL with Databases

### Schema Design Principles

GraphQL provides a flexible alternative to REST for database operations, allowing clients to specify exactly what data they need. Design your GraphQL schema to mirror your database schema while providing appropriate abstractions:

```graphql
type Product {
  id: ID!
  name: String!
  price: Decimal!
  category: Category!
  inventory: Int!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Category {
  id: ID!
  name: String!
  products: [Product!]!
}

type Query {
  products(
    first: Int = 10
    after: String
    filter: ProductFilter
    sortBy: ProductSort
  ): ProductConnection!
  
  product(id: ID!): Product
  categories: [Category!]!
}

type Mutation {
  createProduct(input: CreateProductInput!): Product!
  updateProduct(id: ID!, input: UpdateProductInput!): Product
  deleteProduct(id: ID!): Boolean!
}

input ProductFilter {
  categoryId: ID
  priceMin: Decimal
  priceMax: Decimal
  search: String
}

input ProductSort {
  field: ProductSortField!
  direction: SortDirection!
}

enum ProductSortField {
  CREATED_AT
  PRICE
  NAME
}
```

### N+1 Query Problem and Data Loaders

One of the most common performance issues in GraphQL with databases is the N+1 query problem, where fetching a list of items with related data results in one query for the list plus N queries for each related item. Solve this using DataLoader pattern:

```python
from aiodataloader import DataLoader
from typing import List

class CategoryLoader(DataLoader):
    def __init__(self, db_pool):
        super().__init__()
        self.db_pool = db_pool
    
    async def batch_load_fn(self, product_ids: List[str]) -> List[Category]:
        # Fetch all categories for the requested products in one query
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT p.id as product_id, c.*
                FROM products p
                JOIN categories c ON p.category_id = c.id
                WHERE p.id = ANY($1)
            """, product_ids)
        
        # Build mapping of product_id to category
        product_to_category = {row["product_id"]: row for row in rows}
        
        # Return categories in same order as product_ids
        return [product_to_category.get(pid) for pid in product_ids]

# In resolver
class ProductResolver:
    def __init__(self, db_pool):
        self.category_loader = CategoryLoader(db_pool)
    
    async def resolve_category(self, product):
        return await self.category_loader.load(product["id"])
```

### Connection Pattern for Pagination

Implement the GraphQL connection pattern for efficient cursor-based pagination that performs well with large databases:

```python
import hashlib
from dataclasses import dataclass

@dataclass
class Edge:
    node: dict
    cursor: str

@dataclass 
class PageInfo:
    has_next_page: bool
    has_previous_page: bool
    start_cursor: str
    end_cursor: str

def encode_cursor(row_id: int) -> str:
    """Encode row ID into a cursor string"""
    return hashlib.b64encode(str(row_id).encode()).decode()

def decode_cursor(cursor: str) -> int:
    """Decode cursor back to row ID"""
    return int(hashlib.b64decode(cursor.encode()).decode())

async def resolve_products_connection(
    db, 
    first: int = 10, 
    after: str = None,
    filter_args: dict = None
):
    # Build base query
    query = "SELECT * FROM products WHERE 1=1"
    params = []
    param_index = 1
    
    if after:
        cursor_id = decode_cursor(after)
        query += f" AND id > ${param_index}"
        params.append(cursor_id)
        param_index += 1
    
    if filter_args:
        if "categoryId" in filter_args:
            query += f" AND category_id = ${param_index}"
            params.append(filter_args["categoryId"])
            param_index += 1
    
    # Fetch one extra to determine hasNextPage
    query += f" ORDER BY id LIMIT ${param_index}"
    params.append(first + 1)
    
    rows = await db.fetch_all(query, params)
    
    has_next = len(rows) > first
    rows = rows[:first]
    
    edges = [
        Edge(node=row, cursor=encode_cursor(row["id"]))
        for row in rows
    ]
    
    page_info = PageInfo(
        has_next_page=has_next,
        has_previous_page=after is not None,
        start_cursor=edges[0].cursor if edges else None,
        end_cursor=edges[-1].cursor if edges else None
    )
    
    return {"edges": edges, "pageInfo": page_info}
```

## Database Connection Pooling Strategies

### Pool Sizing and Configuration

Connection pooling is essential for database performance in production applications. Without pooling, each request creates a new database connection, adding latency and potentially exhausting database limits. However, improperly sized pools can cause resource contention or leave connections idle.

**Calculate optimal pool size** using the formula:

```
Optimal Pool Size = (Number of CPUs * 2) + Effective Spinning Disk Count
```

For most web applications, a pool size between 10-30 connections handles reasonable load. For applications with long-running analytical queries, smaller pools work better to avoid connection contention:

```python
# SQLAlchemy connection pool configuration
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    poolclass=QueuePool,
    pool_size=20,              # Base number of connections
    max_overflow=10,          # Additional connections under load
    pool_timeout=30,          # Seconds to wait for available connection
    pool_recycle=1800,        # Recycle connections after 30 minutes
    pool_pre_ping=True,      # Test connection before using
)
```

### Pool Monitoring and Health Checks

Monitor your connection pool metrics to detect issues before they become problems:

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PoolMetrics:
    size: int
    checked_in: int
    checked_out: int
    overflow: int
    invalid: int

async def get_pool_metrics(engine) -> PoolMetrics:
    pool = engine.pool
    
    return PoolMetrics(
        size=pool.size(),
        checked_in=pool.checkedin(),
        checked_out=pool.checkedout(),
        overflow=pool.overflow(),
        invalid=pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0
    )

async def health_check_loop(engine, alert_threshold_seconds=60):
    """Monitor pool health and alert on issues"""
    while True:
        metrics = await get_pool_metrics(engine)
        
        # Alert if too many connections checked out
        if metrics.checked_out > metrics.size * 0.8:
            print(f"WARNING: High connection usage: {metrics.checked_out}/{metrics.size}")
        
        # Alert if overflow is maxed
        if metrics.overflow >= 10:  # Assuming max_overflow=10
            print(f"WARNING: Connection pool overflow exhausted")
        
        await asyncio.sleep(30)
```

### Connection Pool Patterns for Different Workloads

Different application workloads require different pooling strategies:

**Web Applications with Short Requests**: Use moderate pool sizes with quick timeout and connection recycling:

```python
# For web apps with request-response cycles under 1 second
engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    pool_size=25,
    max_overflow=15,
    pool_timeout=10,
    pool_recycle=300,  # Recycle every 5 minutes
    pool_pre_ping=True,
)
```

**Long-Running Batch Processing**: Use smaller pools or disable pooling entirely:

```python
# For batch jobs - each worker gets its own connection
engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    poolclass=NullPool,  # No pooling - new connection each time
)
```

**Serverless/Function as a Service**: Implement connection pooling at the instance level:

```python
# For serverless - manage pool in global scope
db_pool = None

def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = create_engine(
            "postgresql://user:pass@localhost/mydb",
            pool_size=5,
            max_overflow=5,
        )
    return db_pool
```

## ORM vs Raw SQL Trade-offs

### When to Use ORM

Object-Relational Mapping (ORM) frameworks like SQLAlchemy, Django ORM, or Entity Framework provide productivity benefits for application development. Use ORM when:

- Rapid development and maintainability are priorities
- Your queries are relatively straightforward CRUD operations
- Team has Python/Java/C# developers more comfortable with OOP than SQL
- Database schema is likely to change during development
- You need database portability across different database systems

```python
# SQLAlchemy ORM approach - productive for standard CRUD
from sqlalchemy.orm import Session

class ProductRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_category(self, category: str, limit: int = 10):
        return (
            self.session.query(Product)
            .filter(Product.category == category)
            .limit(limit)
            .all()
        )
    
    def create(self, product_data: dict):
        product = Product(**product_data)
        self.session.add(product)
        self.session.commit()
        return product
    
    def update_price(self, product_id: int, new_price: Decimal):
        self.session.query(Product).filter(
            Product.id == product_id
        ).update({"price": new_price})
        self.session.commit()
```

### When to Use Raw SQL

Raw SQL provides maximum control and is often necessary for:

- Complex queries that ORM cannot express efficiently
- Performance-critical operations where you need exact control
- Database-specific features not abstracted by ORM
- Bulk operations where ORM overhead matters
- When you need to use specific query hints or execution plans

```python
# Raw SQL for complex analytical query
async def get_sales_analytics(db, start_date: date, end_date: date):
    query = """
        SELECT 
            DATE_TRUNC('day', o.created_at) as day,
            COUNT(DISTINCT o.id) as order_count,
            COUNT(DISTINCT o.user_id) as unique_customers,
            SUM(oi.quantity * oi.unit_price) as revenue,
            AVG(oi.quantity * oi.unit_price) as avg_order_value
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        WHERE o.created_at BETWEEN %s AND %s
        GROUP BY DATE_TRUNC('day', o.created_at)
        ORDER BY day
    """
    return await db.fetch_all(query, [start_date, end_date])

# Raw SQL for efficient bulk operations
async def bulk_update_inventory(db, updates: list[dict]):
    """
    Efficiently update inventory for multiple products.
    Uses batched updates instead of individual queries.
    """
    query = """
        INSERT INTO inventory (product_id, quantity, updated_at)
        VALUES %s
        ON CONFLICT (product_id) 
        DO UPDATE SET 
            quantity = EXCLUDED.quantity,
            updated_at = EXCLUDED.updated_at
    """
    # Prepare values as list of tuples
    values = [
        (u["product_id"], u["quantity"], datetime.utcnow())
        for u in updates
    ]
    
    # Use execute_batch for efficiency
    await db.execute_batch(query, values)
```

### Hybrid Approach

Most production systems benefit from a hybrid approach, using ORM for standard CRUD while dropping to raw SQL for complex queries:

```python
from sqlalchemy import text
from sqlalchemy.orm import Session

class HybridRepository:
    def __init__(self, session: Session):
        self.session = session
    
    # Standard CRUD with ORM
    def get_product(self, product_id: int):
        return self.session.query(Product).get(product_id)
    
    def list_products(self, limit: int = 10):
        return self.session.query(Product).limit(limit).all()
    
    # Complex analytics with raw SQL
    def get_popular_products(self, days: int = 30, limit: int = 100):
        query = text("""
            SELECT p.*, COUNT(oi.id) as order_count
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.created_at > NOW() - INTERVAL ':days days'
            GROUP BY p.id
            ORDER BY order_count DESC
            LIMIT :limit
        """)
        result = self.session.execute(
            query, 
            {"days": days, "limit": limit}
        )
        return result.fetchall()
```

## Database Client Design Patterns

### Repository Pattern

The Repository pattern provides an abstraction layer between your business logic and data access, making it easier to test and maintain:

```python
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic

T = TypeVar('T')

class Repository(ABC, Generic[T]):
    @abstractmethod
    async def get_by_id(self, id: int) -> Optional[T]:
        pass
    
    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        pass
    
    @abstractmethod
    async def update(self, id: int, entity: T) -> Optional[T]:
        pass
    
    @abstractmethod
    async def delete(self, id: int) -> bool:
        pass

class SQLProductRepository(Repository[Product]):
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.table_name = "products"
    
    async def get_by_id(self, id: int) -> Optional[Product]:
        row = await self.db_pool.fetchrow(
            f"SELECT * FROM {self.table_name} WHERE id = $1", id
        )
        return Product(**row) if row else None
    
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Product]:
        rows = await self.db_pool.fetch(
            f"SELECT * FROM {self.table_name} ORDER BY id LIMIT $1 OFFSET $2",
            limit, offset
        )
        return [Product(**row) for row in rows]
    
    async def create(self, product: Product) -> Product:
        row = await self.db_pool.fetchrow(
            f"""INSERT INTO {self.table_name} 
                (name, price, category, inventory) 
                VALUES ($1, $2, $3, $4) 
                RETURNING *""",
            product.name, product.price, product.category, product.inventory
        )
        return Product(**row)
    
    async def update(self, id: int, product: Product) -> Optional[Product]:
        row = await self.db_pool.fetchrow(
            f"""UPDATE {self.table_name} 
                SET name=$1, price=$2, category=$3, inventory=$4 
                WHERE id=$5 
                RETURNING *""",
            product.name, product.price, product.category, product.inventory, id
        )
        return Product(**row) if row else None
    
    async def delete(self, id: int) -> bool:
        result = await self.db_pool.execute(
            f"DELETE FROM {self.table_name} WHERE id = $1", id
        )
        return "DELETE 1" in result
```

### Unit of Work Pattern

The Unit of Work pattern tracks changes and commits them atomically:

```python
from dataclasses import dataclass, field
from typing import Dict, List, TypeVar, Generic, Callable
from enum import Enum

T = TypeVar('T')

class OperationType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

@dataclass
class Change:
    entity_type: str
    entity_id: int
    operation: OperationType
    data: dict

class UnitOfWork:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.changes: List[Change] = []
        self._connection = None
    
    async def __aenter__(self):
        self._connection = await self.db_pool.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
        await self.db_pool.release(self._connection)
    
    async def commit(self):
        """Commit all tracked changes in a transaction"""
        async with self._connection.transaction():
            for change in self.changes:
                await self._apply_change(change)
        self.changes.clear()
    
    async def rollback(self):
        """Rollback all tracked changes"""
        self.changes.clear()
    
    async def register_new(self, entity_type: str, entity_id: int, data: dict):
        self.changes.append(Change(entity_type, entity_id, OperationType.CREATE, data))
    
    async def register_dirty(self, entity_type: str, entity_id: int, data: dict):
        self.changes.append(Change(entity_type, entity_id, OperationType.UPDATE, data))
    
    async def register_deleted(self, entity_type: str, entity_id: int):
        self.changes.append(Change(entity_type, entity_id, OperationType.DELETE, {}))
    
    async def _apply_change(self, change: Change):
        if change.operation == OperationType.CREATE:
            await self._connection.execute(
                f"INSERT INTO {change.entity_type} (id, data) VALUES ($1, $2)",
                change.entity_id, json.dumps(change.data)
            )
        elif change.operation == OperationType.UPDATE:
            await self._connection.execute(
                f"UPDATE {change.entity_type} SET data = $1 WHERE id = $2",
                json.dumps(change.data), change.entity_id
            )
        elif change.operation == OperationType.DELETE:
            await self._connection.execute(
                f"DELETE FROM {change.entity_type} WHERE id = $1",
                change.entity_id
            )

# Usage
async def process_order(order_data: dict, items: list):
    async with UnitOfWork(db_pool) as uow:
        order_id = await create_order(uow, order_data)
        for item in items:
            await create_order_item(uow, order_id, item)
        await update_inventory(uow, items)
    # All changes committed or rolled back together
```

## API Rate Limiting and Database Protection

### Rate Limiting Strategies

Protect your database from overload with multiple layers of rate limiting:

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

class RateLimiter:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.memory_store = defaultdict(list)
        self.window_seconds = 60
        self.max_requests = 100
    
    async def check_rate_limit(self, client_id: str) -> bool:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old entries from memory
        self.memory_store[client_id] = [
            ts for ts in self.memory_store[client_id]
            if ts > window_start
        ]
        
        if len(self.memory_store[client_id]) >= self.max_requests:
            return False
        
        self.memory_store[client_id].append(now)
        return True
    
    async def check_rate_limit_database(self, client_id: str) -> bool:
        """Database-backed rate limiting for distributed systems"""
        async with self.db_pool.acquire() as conn:
            # Clean old entries
            await conn.execute("""
                DELETE FROM rate_limits 
                WHERE client_id = $1 AND window_start < NOW() - INTERVAL '1 minute'
            """, client_id)
            
            # Count requests in current window
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM rate_limits 
                WHERE client_id = $1 AND window_start >= NOW() - INTERVAL '1 minute'
            """, client_id)
            
            if count >= self.max_requests:
                return False
            
            # Record this request
            await conn.execute("""
                INSERT INTO rate_limits (client_id, window_start) 
                VALUES ($1, NOW())
            """, client_id)
            
            return True

app = FastAPI()
rate_limiter = RateLimiter(db_pool)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.headers.get("X-API-Key", request.client.host)
    
    if not await rate_limiter.check_rate_limit_database(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await call_next(request)
    return response
```

### Query Cost Limits

Implement query cost limits to prevent expensive queries from overwhelming your database:

```python
class QueryCostLimiter:
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def estimate_query_cost(self, query: str) -> dict:
        """Use EXPLAIN to estimate query cost"""
        async with self.db_pool.acquire() as conn:
            plan = await conn.fetchrow(f"EXPLAIN (FORMAT JSON) {query}")
            return plan["QUERY PLAN"][0]["Plan"]
    
    async def validate_query(self, query: str, max_cost: float = 1000) -> bool:
        """Validate query doesn't exceed cost threshold"""
        cost_info = await self.estimate_query_cost(query)
        total_cost = cost_info.get("Total Cost", float('inf'))
        
        if total_cost > max_cost:
            raise HTTPException(
                status_code=400,
                detail=f"Query cost {total_cost} exceeds limit {max_cost}"
            )
        return True

# Usage in endpoint
@app.get("/api/analytics")
async def run_analytics(query: str):
    await query_limiter.validate_query(query)
    return await db.fetch_all(query)
```

## Best Practices Summary

When designing database APIs and integration patterns, follow these core principles:

1. **Use connection pooling** with appropriately sized pools based on your workload characteristics. Monitor pool metrics in production.

2. **Choose REST or GraphQL** based on your client needs. REST works well for simple CRUD operations; GraphQL excels when clients need flexible data shapes.

3. **Implement proper error handling** that logs details internally but returns safe messages to clients.

4. **Use cursor-based pagination** for large datasets instead of offset-based pagination.

5. **Solve N+1 problems** in GraphQL using DataLoader patterns.

6. **Consider hybrid ORM/raw SQL** approaches, using ORM for CRUD and raw SQL for complex analytics.

7. **Implement rate limiting** at multiple layers to protect your database from overload.

8. **Design for failure** by implementing circuit breakers and fallback behaviors.

These patterns provide a foundation for building robust database integrations that scale with your application requirements.

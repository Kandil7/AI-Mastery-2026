# Database Design

This document covers the essential concepts of database design, including Entity-Relationship (ER) modeling, schema design patterns, normalization forms, and denormalization strategies. Proper database design is crucial for building scalable, maintainable AI applications.

---

## Table of Contents

1. [Entity-Relationship Modeling](#1-entity-relationship-modeling)
2. [Schema Design Patterns](#2-schema-design-patterns)
3. [Normalization Forms](#3-normalization-forms)
4. [Denormalization Strategies](#4-denormalization-strategies)
5. [Domain-Driven Design with Databases](#5-domain-driven-design-with-databases)
6. [Data Modeling for Different Use Cases](#6-data-modeling-for-different-use-cases)

---

## 1. Entity-Relationship Modeling

Entity-Relationship (ER) modeling is a conceptual way to describe data and its relationships. It provides a visual and mathematical foundation for database design.

### Entities and Attributes

An **entity** is a real-world object that can be distinctly identified (e.g., Customer, Product, Order). An **attribute** is a property of an entity (e.g., name, price, date).

**Entity Types**:
- **Strong Entity**: Has its own primary key (e.g., Customer)
- **Weak Entity**: Depends on another entity for identification (e.g., OrderItem)
- ** Associative Entity**: Represents a many-to-many relationship (e.g., Enrollment)

**Attributes Types**:
- **Simple**: Single atomic value (e.g., email)
- **Composite**: Multiple simple values (e.g., address = street + city + zip)
- **Multi-valued**: Multiple values (e.g., phone_numbers)
- **Derived**: Calculated from other attributes (e.g., age from birth_date)

**Example - Creating Tables from Entities**:

```sql
-- Strong entity: Customers
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weak entity: Orders (depends on customers)
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date DATE DEFAULT CURRENT_DATE,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

### Relationships

Relationships describe how entities are connected. They can be classified by cardinality (one-to-one, one-to-many, many-to-many).

**Types of Relationships**:

| Type | Description | Example |
|------|-------------|---------|
| One-to-One (1:1) | Each record in Table A relates to one record in Table B | User ↔ UserProfile |
| One-to-Many (1:N) | Each record in Table A relates to multiple records in Table B | Customer ↔ Orders |
| Many-to-Many (M:N) | Records in Table A can relate to multiple records in Table B | Students ↔ Courses |

**Example - Implementing Relationships**:

```sql
-- One-to-Many: Customer has many Orders
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Many-to-Many: Products can be in many Orders via OrderItems
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Many-to-Many: Users can have many Roles
CREATE TABLE user_roles (
    user_id INT NOT NULL,
    role_id INT NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (role_id) REFERENCES roles(role_id)
);
```

### Cardinality Notation

In ER diagrams, cardinality is often represented using crow's foot notation:

```
|-------<   : One-to-Many
|>-------|  : Many-to-One
|-----<|> : Many-to-Many
|-----||  : One-to-One
```

---

## 2. Schema Design Patterns

Different schema patterns suit different use cases. Understanding these patterns helps you choose the right approach for your application.

### Star Schema

The star schema organizes data into fact and dimension tables, with the fact table at the center connected to dimension tables.

**Components**:
- **Fact Table**: Contains quantitative data (metrics) to be analyzed
- **Dimension Tables**: Contain descriptive attributes for filtering and labeling

**Best For**: Data warehousing, business intelligence, analytics

**Example - E-Commerce Star Schema**:

```sql
-- Fact Table: Sales transactions
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    date_key INT NOT NULL,
    product_key INT NOT NULL,
    customer_key INT NOT NULL,
    store_key INT NOT NULL,
    quantity_sold INT NOT NULL,
    sale_amount DECIMAL(10, 2) NOT NULL,
    cost_amount DECIMAL(10, 2),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (store_key) REFERENCES dim_store(store_key)
);

-- Dimension Table: Products
CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,
    product_id INT NOT NULL,
    product_name VARCHAR(100),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    brand VARCHAR(50),
    unit_price DECIMAL(10, 2)
);

-- Dimension Table: Customers
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id INT NOT NULL,
    customer_name VARCHAR(100),
    email VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50),
    segment VARCHAR(20)
);

-- Dimension Table: Dates
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE,
    day_of_week INT,
    day_name VARCHAR(10),
    month INT,
    month_name VARCHAR(10),
    quarter INT,
    year INT
);
```

**Advantages**:
- Simple queries with fewer joins
- Fast aggregation queries
- Easy to understand and maintain

### Snowflake Schema

The snowflake schema is a normalized version of the star schema where dimension tables are further normalized into multiple related tables.

**Example - Snowflake Schema**:

```sql
-- Normalized dimension: Products with categories
CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,
    product_id INT NOT NULL,
    product_name VARCHAR(100),
    category_key INT,
    brand_key INT,
    unit_price DECIMAL(10, 2)
);

CREATE TABLE dim_category (
    category_key INT PRIMARY KEY,
    category_id INT NOT NULL,
    category_name VARCHAR(50),
    subcategory_key INT
);

CREATE TABLE dim_brand (
    brand_key INT PRIMARY KEY,
    brand_id INT NOT NULL,
    brand_name VARCHAR(50),
    manufacturer VARCHAR(50)
);
```

**Advantages**:
- Reduced data redundancy
- Easier data maintenance
- Better storage efficiency

**Disadvantages**:
- More complex queries (more joins)
- Slower query performance

### Galaxy Schema (Fact Constellation)

Multiple fact tables share dimension tables, useful for complex analytical requirements.

### Entity-Attribute-Value (EAV)

Used for highly variable schemas where entities can have different attributes.

```sql
CREATE TABLE product_attributes (
    entity_id INT NOT NULL,
    attribute_name VARCHAR(50) NOT NULL,
    value VARCHAR(255),
    PRIMARY KEY (entity_id, attribute_name)
);
```

**Best For**: Product catalogs with varying attributes, CMS systems

**Considerations**: Complex queries, performance overhead

### Document Model (NoSQL)

For flexible schemas, document databases allow varying structures:

```javascript
// Product with variable attributes
{
  "_id": "prod123",
  "name": "T-Shirt",
  "base_price": 19.99,
  "attributes": {
    "size": ["S", "M", "L", "XL"],
    "color": ["red", "blue", "black"],
    "material": "cotton"
  }
}

// Different product with different attributes
{
  "_id": "prod456", 
  "name": "Laptop",
  "base_price": 999.99,
  "attributes": {
    "processor": "Intel i7",
    "ram": "16GB",
    "storage": "512GB SSD",
    "screen_size": "15.6 inch"
  }
}
```

---

## 3. Normalization Forms

Normalization is the process of organizing data to minimize redundancy and prevent update anomalies. Each normal form represents a specific level of organization.

### First Normal Form (1NF)

Each column contains only atomic (indivisible) values. No repeating groups or arrays.

**Requirements**:
- Each column contains only single values
- Each row is unique
- No repeating groups

**Example - Violation and Fix**:

```sql
-- NOT 1NF: Multiple values in single column
CREATE TABLE users_bad (
    user_id INT PRIMARY KEY,
    name VARCHAR(50),
    phone_numbers VARCHAR(100)  -- Contains multiple phones like "555-1234,555-5678"
);

-- 1NF: Single atomic values
CREATE TABLE users_good (
    user_id INT PRIMARY KEY,
    name VARCHAR(50),
    phone_number VARCHAR(20)
);

-- Or use separate table for multi-valued attribute
CREATE TABLE user_phones (
    user_id INT NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    PRIMARY KEY (user_id, phone_number),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

### Second Normal Form (2NF)

Must be in 1NF, and all non-key attributes must fully functionally depend on the entire primary key (no partial dependencies).

**Requirements**:
- Must be in 1NF
- No partial dependencies on part of a composite primary key

**Example - Violation and Fix**:

```sql
-- NOT 2NF: Partial dependency
CREATE TABLE order_items_bad (
    order_id INT,
    product_id INT,
    product_name VARCHAR(100),  -- Depends only on product_id, not on composite key
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);

-- 2NF: Separate tables
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    price DECIMAL(10, 2)
);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

### Third Normal Form (3NF)

Must be in 2NF, and no transitive dependencies (non-key attributes should depend only on the primary key).

**Requirements**:
- Must be in 2NF
- No transitive dependencies

**Example - Violation and Fix**:

```sql
-- NOT 3NF: Transitive dependency
CREATE TABLE employees_bad (
    employee_id INT PRIMARY KEY,
    name VARCHAR(100),
    department_id INT,
    department_name VARCHAR(50)  -- Depends on department_id, not on employee_id
);

-- 3NF: Separate departments table
CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50)
);

CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    name VARCHAR(100),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

### Boyce-Codd Normal Form (BCNF)

A stronger version of 3NF where every determinant must be a candidate key.

**Requirements**:
- Must be in 3NF
- For every functional dependency X → Y, X must be a superkey

**Example - BCNF Violation**:

```sql
-- NOT BCNF: Multiple candidate keys with overlapping dependencies
CREATE TABLE course_students_bad (
    course_id INT,
    professor_id INT,
    student_id INT,
    grade VARCHAR(2),
    PRIMARY KEY (course_id, student_id),
    -- Also has: professor_id → course_id (professor teaches one course)
    -- professor_id is not a superkey
);

-- BCNF: Decompose into two tables
CREATE TABLE course_professors (
    course_id INT PRIMARY KEY,
    professor_id INT
);

CREATE TABLE course_students (
    course_id INT,
    student_id INT,
    grade VARCHAR(2),
    PRIMARY KEY (course_id, student_id),
    FOREIGN KEY (course_id) REFERENCES course_professors(course_id)
);
```

### Normalization Summary

| Normal Form | Goal | Key Rule |
|-------------|------|----------|
| 1NF | Eliminate repeating groups | Atomic values |
| 2NF | Eliminate partial dependencies | Full functional dependency on PK |
| 3NF | Eliminate transitive dependencies | Non-key attributes depend only on PK |
| BCNF | Strengthen 3NF | Every determinant is a candidate key |

---

## 4. Denormalization Strategies

Denormalization intentionally adds redundancy to improve read performance. It's typically used after normalization when performance requirements demand it.

### When to Denormalize

- **Read-heavy workloads** where query speed is critical
- **Reporting/analytics** requiring complex aggregations
- **Caching frequently accessed data** from multiple tables
- **Reducing join complexity** for common queries

### Pre-Joining Tables

Store joined data to avoid expensive joins at query time.

```sql
-- Denormalized: Pre-joined order with customer info
CREATE TABLE orders_denormalized (
    order_id INT PRIMARY KEY,
    customer_id INT,
    customer_name VARCHAR(100),
    customer_email VARCHAR(100),
    customer_city VARCHAR(50),
    order_date DATE,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20)
);

-- Update denormalized table via trigger or application logic
CREATE OR REPLACE FUNCTION update_order_denormalized()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO orders_denormalized
        SELECT NEW.*, c.name, c.email, c.city
        FROM customers c WHERE c.customer_id = NEW.customer_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Storing Derived Data

Pre-compute and store calculated values.

```sql
-- Store aggregate data for fast reads
CREATE TABLE customer_stats (
    customer_id INT PRIMARY KEY,
    total_orders INT DEFAULT 0,
    total_spent DECIMAL(12, 2) DEFAULT 0,
    avg_order_value DECIMAL(10, 2),
    last_order_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Materialized view alternative
CREATE MATERIALIZED VIEW customer_stats_mv AS
SELECT 
    c.customer_id,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent,
    AVG(o.total_amount) AS avg_order_value,
    MAX(o.order_date) AS last_order_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id;

-- Refresh on schedule
REFRESH MATERIALIZED VIEW CONCURRENTLY customer_stats_mv;
```

### Storing Duplicate Data

Copy frequently accessed columns to reduce lookups.

```sql
-- Store product info in order items for historical accuracy
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    product_name VARCHAR(100),  -- Denormalized: copy from products
    category_name VARCHAR(50),  -- Denormalized: copy from categories
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

### Denormalization Trade-offs

| Pros | Cons |
|------|------|
| Faster reads | Data inconsistency risk |
| Simpler queries | Increased storage |
| Reduced joins | Update complexity |
| Better performance | More maintenance |

### Best Practices

1. **Normalize first, denormalize later**: Start with a normalized design
2. **Document denormalization**: Make it explicit in code and docs
3. **Use triggers/materialized views**: Automate consistency
4. **Monitor for drift**: Check for data inconsistencies
5. **Consider alternatives**: Indexes, caching, query optimization first

---

## 5. Domain-Driven Design with Databases

Domain-Driven Design (DDD) provides patterns for aligning database design with business domain logic.

### Aggregates and Aggregate Roots

An aggregate is a cluster of related objects treated as a single unit. The aggregate root is the entity that controls access to the entire cluster.

```python
# Aggregate Root: Order
class Order:
    def __init__(self, order_id, customer_id):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = []
        self.status = 'pending'
    
    def add_item(self, product_id, quantity, price):
        if self.status != 'pending':
            raise ValueError("Cannot modify shipped order")
        
        # Business rule: Check inventory
        if not self._check_inventory(product_id, quantity):
            raise ValueError("Insufficient inventory")
            
        item = OrderItem(product_id, quantity, price)
        self.items.append(item)
    
    def calculate_total(self):
        return sum(item.quantity * item.price for item in self.items)
    
    def _check_inventory(self, product_id, quantity):
        # Check inventory logic
        return True

class OrderItem:
    def __init__(self, product_id, quantity, price):
        self.product_id = product_id
        self.quantity = quantity
        self.price = price
```

### Repository Pattern

The repository pattern abstracts data access, providing a collection-like interface.

```python
from abc import ABC, abstractmethod
from typing import Optional, List
import sqlite3

class OrderRepository(ABC):
    @abstractmethod
    def find_by_id(self, order_id: int) -> Optional[Order]:
        pass
    
    @abstractmethod
    def save(self, order: Order) -> None:
        pass
    
    @abstractmethod
    def find_by_customer(self, customer_id: int) -> List[Order]:
        pass

class SQLiteOrderRepository(OrderRepository):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def find_by_id(self, order_id: int) -> Optional[Order]:
        conn = sqlite3.connect(self.connection_string)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        order = Order(row[0], row[1])
        # Load items...
        
        conn.close()
        return order
    
    def save(self, order: Order) -> None:
        conn = sqlite3.connect(self.connection_string)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO orders (order_id, customer_id, status) VALUES (?, ?, ?)",
            (order.order_id, order.customer_id, order.status)
        )
        
        conn.commit()
        conn.close()
```

### Value Objects

Value objects are immutable objects defined by their attributes rather than a unique identity.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Address:
    street: str
    city: str
    state: str
    zip_code: str
    
    def __str__(self):
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"

# Usage
address = Address("123 Main St", "Boston", "MA", "02101")
# address.street = "456 Oak Ave"  # Error: objects are immutable
```

---

## 6. Data Modeling for Different Use Cases

Different applications require different data models. Here are common patterns for AI/ML applications.

### E-Commerce Data Model

```sql
-- Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Products table
CREATE TABLE products (
    product_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    inventory_count INT DEFAULT 0,
    category_id UUID REFERENCES categories(category_id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders table
CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id),
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    shipping_address JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Order items
CREATE TABLE order_items (
    order_item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(order_id),
    product_id UUID NOT NULL REFERENCES products(product_id),
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
```

### IoT Time-Series Data Model

```sql
-- Sensor readings with automatic partitioning
CREATE TABLE sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    device_id UUID NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    battery_level DOUBLE PRECISION
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sensor_readings', 'time');

-- Add compression for older data
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id'
);

-- Add compression policy
SELECT add_compression_policy('sensor_readments', INTERVAL '7 days');
```

### User Behavior/Event Tracking

```sql
-- Events table for tracking user actions
CREATE TABLE events (
    event_id BIGSERIAL PRIMARY KEY,
    user_id UUID,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    session_id UUID,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for common queries
CREATE INDEX idx_events_user_time ON events(user_id, created_at DESC);
CREATE INDEX idx_events_type_time ON events(event_type, created_at DESC);

-- Partition by time for large datasets
CREATE TABLE events_y2024m01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Machine Learning Model Registry

```sql
-- ML Models table
CREATE TABLE ml_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    framework VARCHAR(50),
    metrics JSONB,
    hyperparameters JSONB,
    artifact_path TEXT,
    status VARCHAR(20) DEFAULT 'staging',
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model predictions log
CREATE TABLE predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_id UUID REFERENCES ml_models(model_id),
    input_features JSONB NOT NULL,
    output_predictions JSONB NOT NULL,
    prediction_proba JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for model lookups
CREATE INDEX idx_predictions_model_time 
ON predictions(model_id, created_at DESC);
```

---

## Related Resources

- For practical database tutorials, see [04. Tutorials](../04_tutorials/README.md)
- For database fundamentals, see [Database Fundamentals](./database_fundamentals.md)
- For advanced query optimization, see [Query Optimization Deep Dive](./query_optimization_deep_dive.md)
- For time-series databases, see [Time-Series Fundamentals](./time_series_fundamentals.md)
- For vector databases in AI applications, see [Vector Search Basics](./vector_search_basics.md)

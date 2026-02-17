---
title: "Data Types Fundamentals"
category: "foundations"
subcategory: "database_basics"
tags: ["data types", "numeric", "string", "datetime", "json", "boolean"]
related: ["01_acid_properties.md", "02_database_types.md", "03_storage_architectures.md"]
difficulty: "beginner"
estimated_reading_time: 30
---

# Data Types Fundamentals

Understanding data types is crucial for designing efficient, reliable databases. The right data type choice affects storage efficiency, query performance, data integrity, and application correctness.

This guide covers the most important data types across PostgreSQL and MySQL, with practical guidance on when to use each type.

---

## Table of Contents

1. [Numeric Types](#1-numeric-types)
2. [String Types](#2-string-types)
3. [Date/Time Types](#3-datetime-types)
4. [Boolean and NULL Handling](#4-boolean-and-null-handling)
5. [JSON and Document Types](#5-json-and-document-types)
6. [Specialized Types](#6-specialized-types)
7. [Type Conversion and Casting](#7-type-conversion-and-casting)
8. [Best Practices Summary](#8-best-practices-summary)

---

## 1. Numeric Types

### INTEGER vs BIGINT vs SMALLINT

| Type | Range (PostgreSQL) | Range (MySQL) | Storage | When to Use |
|------|-------------------|---------------|---------|-------------|
| `SMALLINT` | -32,768 to 32,767 | -32,768 to 32,767 | 2 bytes | Small counters, status codes, flags |
| `INTEGER`/`INT` | -2,147,483,648 to 2,147,483,647 | Same | 4 bytes | Most common integer usage (IDs, counts, ages) |
| `BIGINT` | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 | Same | 8 bytes | Large IDs, timestamps in microseconds, financial calculations |

**Example - Choosing the Right Integer Type:**
```sql
-- Good: Use SMALLINT for status codes (0-100)
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,        -- Large ID space needed
    status SMALLINT NOT NULL,           -- Only 0-5 possible values
    quantity INTEGER NOT NULL           -- Typical quantities (1-1000)
);

-- Bad: Using BIGINT for everything (wastes space)
CREATE TABLE users (
    id BIGINT PRIMARY KEY,              -- Overkill if < 2B users
    age BIGINT NOT NULL,                -- Age never > 150
    score BIGINT NOT NULL               -- Scores typically < 1M
);
```

### DECIMAL vs NUMERIC vs FLOAT

| Type | Precision | Scale | Storage | When to Use |
|------|-----------|-------|---------|-------------|
| `DECIMAL(p,s)`/`NUMERIC(p,s)` | Exact, up to p digits | s digits after decimal | Variable (up to 13 bytes) | Money, exact calculations, scientific data |
| `REAL`/`FLOAT` | Approximate, ~6 decimal digits | N/A | 4 bytes | Scientific calculations where approximation is acceptable |
| `DOUBLE PRECISION`/`DOUBLE` | Approximate, ~15 decimal digits | N/A | 8 bytes | Higher precision floating point |

**Critical Rule for Financial Data**: Always use `DECIMAL` or `NUMERIC`, never `FLOAT` or `DOUBLE`.

**Example - Financial Calculations:**
```sql
-- Good: Exact decimal arithmetic
CREATE TABLE transactions (
    transaction_id BIGINT PRIMARY KEY,
    amount DECIMAL(10, 2) NOT NULL,     -- $99999999.99 max
    fee DECIMAL(6, 2) NOT NULL,         -- Up to $9999.99 fee
    total DECIMAL(10, 2) NOT NULL       -- Exact calculation
);

-- Bad: Floating point errors
CREATE TABLE bad_transactions (
    amount FLOAT NOT NULL,              -- 0.1 + 0.2 ≠ 0.3!
    fee FLOAT NOT NULL
);

-- Demonstration of floating point error:
SELECT 0.1 + 0.2 AS result;             -- Returns 0.30000000000000004 in most systems
SELECT CAST(0.1 AS DECIMAL(10,2)) + CAST(0.2 AS DECIMAL(10,2)) AS result; -- Returns 0.30
```

### SERIAL vs GENERATED ALWAYS AS IDENTITY

| Feature | `SERIAL` | `GENERATED ALWAYS AS IDENTITY` |
|---------|----------|-------------------------------|
| Standard | PostgreSQL-specific | SQL:2016 standard |
| Storage | Uses sequence | Uses identity column |
| Portability | Low | High |
| Control | Less flexible | More control options |

**Recommendation**: Use `GENERATED ALWAYS AS IDENTITY` for new projects, `SERIAL` for legacy PostgreSQL compatibility.

```sql
-- Modern approach (SQL standard)
CREATE TABLE products (
    product_id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Legacy PostgreSQL approach
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
```

---

## 2. String Types

### VARCHAR vs TEXT vs CHAR

| Type | Max Length | Storage | When to Use |
|------|------------|---------|-------------|
| `CHAR(n)` | Fixed n characters | Always n bytes | Fixed-length codes (ISO country codes, gender) |
| `VARCHAR(n)` | Up to n characters | Actual length + overhead | Variable-length text with known maximum |
| `TEXT` | Unlimited (theoretical) | Variable | Long text, descriptions, content |

**Storage Details:**
- `CHAR(n)`: Always uses n bytes, padded with spaces
- `VARCHAR(n)`: Uses actual length + 1-4 bytes overhead
- `TEXT`: Same as VARCHAR but no length limit

**Performance Considerations:**
- `CHAR` is faster for fixed-length data but wastes space
- `VARCHAR` is optimal for most variable-length strings
- `TEXT` has no performance penalty over `VARCHAR` in PostgreSQL

**Example - String Type Selection:**
```sql
-- Good choices
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,      -- Username max 50 chars
    email VARCHAR(255) NOT NULL,        -- Email max 255 chars (RFC standard)
    bio TEXT,                           -- User biography - unlimited length
    country_code CHAR(2) NOT NULL       -- ISO 3166-1 alpha-2 country code
);

-- Bad choices
CREATE TABLE bad_users (
    username TEXT NOT NULL,             -- Overkill for usernames
    country_code VARCHAR(10) NOT NULL   -- Wastes space for 2-char codes
);
```

### String Functions and Indexing

**Common String Operations:**
```sql
-- Case-insensitive search
SELECT * FROM users WHERE LOWER(username) = 'alice';

-- Pattern matching
SELECT * FROM products WHERE name ILIKE '%laptop%';

-- String manipulation
UPDATE users 
SET email = LOWER(email)
WHERE email != LOWER(email);
```

**Indexing Strings:**
```sql
-- Regular B-tree index (good for equality, prefix searches)
CREATE INDEX idx_users_username ON users(username);

-- Partial index for common prefixes
CREATE INDEX idx_active_users ON users(username) WHERE is_active = TRUE;

-- Trigram index for fuzzy matching (PostgreSQL)
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_users_username_gin ON users USING GIN (username gin_trgm_ops);
```

---

## 3. Date/Time Types

### TIMESTAMP vs DATE vs TIME vs INTERVAL

| Type | Format | Range | When to Use |
|------|--------|-------|-------------|
| `DATE` | YYYY-MM-DD | 1000-01-01 to 9999-12-31 | Birth dates, event dates, without time |
| `TIME` | HH:MM:SS[.nnnnnn] | 00:00:00 to 24:00:00 | Time of day, without date |
| `TIMESTAMP` | YYYY-MM-DD HH:MM:SS[.nnnnnn] | Same as DATE | Events with specific date/time |
| `TIMESTAMPTZ` | TIMESTAMP with timezone | Same as TIMESTAMP | Events that need timezone awareness |
| `INTERVAL` | [years-]months-days hours:minutes:seconds | Varies | Durations, time differences |

**Critical Distinction**: `TIMESTAMP` vs `TIMESTAMPTZ`
- `TIMESTAMP`: Stores time without timezone information
- `TIMESTAMPTZ`: Stores UTC time, converts to local timezone when displayed

**Example - Timezone Handling:**
```sql
-- Store in UTC (recommended)
CREATE TABLE events (
    event_id BIGINT PRIMARY KEY,
    event_time TIMESTAMPTZ NOT NULL,    -- Store UTC time
    location VARCHAR(100)
);

-- Insert with timezone
INSERT INTO events (event_time, location)
VALUES ('2024-01-15 14:30:00+00', 'New York');  -- UTC time

-- Query in local timezone
SELECT 
    event_time AT TIME ZONE 'America/New_York' AS ny_time,
    event_time AT TIME ZONE 'Asia/Tokyo' AS tokyo_time
FROM events;
```

### Date Arithmetic and Functions

```sql
-- Add/subtract time
SELECT 
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP + INTERVAL '7 days' AS next_week,
    CURRENT_TIMESTAMP - INTERVAL '30 minutes' AS 30_minutes_ago;

-- Extract components
SELECT 
    EXTRACT(YEAR FROM event_time) AS year,
    EXTRACT(MONTH FROM event_time) AS month,
    EXTRACT(DAY FROM event_time) AS day,
    EXTRACT(HOUR FROM event_time) AS hour;

-- Age calculation
SELECT 
    first_name,
    age(CURRENT_DATE, birth_date) AS age_interval
FROM users;

-- Date truncation (for grouping)
SELECT 
    DATE_TRUNC('month', event_time) AS month_start,
    COUNT(*) AS events
FROM events
GROUP BY DATE_TRUNC('month', event_time);
```

---

## 4. Boolean and NULL Handling

### BOOLEAN Type

| Value | Description |
|-------|-------------|
| `TRUE` | Logical true |
| `FALSE` | Logical false |
| `NULL` | Unknown/missing value |

**Important**: In SQL, `NULL` is not equal to `TRUE` or `FALSE`. It represents unknown.

```sql
-- Correct boolean handling
SELECT * FROM users WHERE is_active = TRUE;      -- Active users
SELECT * FROM users WHERE is_active IS FALSE;    -- Inactive users  
SELECT * FROM users WHERE is_active IS NULL;     -- Users with unknown status

-- Wrong: This will NOT return NULL values
SELECT * FROM users WHERE is_active = NULL;      -- Returns empty set!
```

### NULL Best Practices

1. **Use `IS NULL`/`IS NOT NULL`**, never `= NULL`
2. **Consider DEFAULT values** for non-nullable columns
3. **Use COALESCE()** to handle NULLs gracefully
4. **Avoid NULL in primary keys** (they're automatically NOT NULL)

```sql
-- Good NULL handling
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    discount DECIMAL(5,2) DEFAULT 0.00,  -- Default 0% discount
    notes TEXT DEFAULT NULL              -- Optional notes
);

-- Handle NULL in queries
SELECT 
    order_id,
    COALESCE(discount, 0.00) AS effective_discount,
    COALESCE(notes, 'No notes') AS display_notes
FROM orders;

-- NULL-safe comparison
SELECT * FROM products 
WHERE COALESCE(category, '') = 'electronics';
```

### Three-Valued Logic

SQL uses three-valued logic: `TRUE`, `FALSE`, `UNKNOWN` (represented by `NULL`).

```sql
-- Understanding three-valued logic
SELECT 
    1 = 1 AS true_example,           -- TRUE
    1 = 2 AS false_example,          -- FALSE
    1 = NULL AS null_comparison,     -- UNKNOWN (NULL)
    NULL = NULL AS null_equals_null  -- UNKNOWN (NULL)
;

-- Use IS DISTINCT FROM for NULL-safe comparisons
SELECT * FROM users 
WHERE email IS DISTINCT FROM 'unknown@example.com';
```

---

## 5. JSON and Document Types

### JSON vs JSONB

| Feature | `JSON` | `JSONB` |
|---------|--------|---------|
| Storage | Text format | Binary format |
| Indexing | No | Yes (GIN/GiST) |
| Performance | Slower parsing | Faster parsing and operations |
| Duplicate keys | Preserved | Removed (last key wins) |
| Ordering | Preserved | Not preserved |

**Recommendation**: Use `JSONB` for almost all cases—it's more powerful and performs better.

**Example - JSONB Usage:**
```sql
CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Insert with JSONB
INSERT INTO products (name, metadata)
VALUES (
    'Laptop',
    '{
        "brand": "Dell",
        "model": "XPS 13",
        "specs": {
            "cpu": "Intel i7",
            "ram": "16GB",
            "storage": "512GB SSD"
        },
        "tags": ["laptop", "business", "premium"],
        "price": 1299.99
    }'::jsonb
);

-- Query JSONB data
SELECT 
    name,
    metadata->>'brand' AS brand,
    (metadata->'specs'->>'ram')::INTEGER AS ram_gb,
    metadata->'tags' AS tags
FROM products
WHERE metadata->>'brand' = 'Dell';

-- Index JSONB for fast lookups
CREATE INDEX idx_products_brand ON products ((metadata->>'brand'));
CREATE INDEX idx_products_tags ON products USING GIN (metadata->'tags');
```

### JSONB Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `->>` | Extract as text | `metadata->>'brand'` |
| `->` | Extract as JSON | `metadata->'specs'` |
| `@>` | Contains | `metadata @> '{"brand":"Dell"}'` |
| `<@` | Contained by | `'{"brand":"Dell"}' <@ metadata` |
| `?` | Key exists | `metadata ? 'tags'` |
| `?|` | Any key exists | `metadata ?| ARRAY['brand','model']` |

**Advanced JSONB Queries:**
```sql
-- Find products with RAM >= 16GB
SELECT name, metadata->>'brand' AS brand
FROM products
WHERE (metadata->'specs'->>'ram')::INTEGER >= 16;

-- Find products with 'premium' tag
SELECT name
FROM products
WHERE metadata->'tags' ? 'premium';

-- Update JSONB field
UPDATE products
SET metadata = jsonb_set(metadata, '{specs, ram}', '"32GB"'::jsonb)
WHERE product_id = 1;
```

---

## 6. Specialized Types

### UUID Type

Universally Unique Identifier - 128-bit value, excellent for distributed systems.

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Generate UUIDs
SELECT uuid_generate_v4();  -- Random UUID
SELECT uuid_generate_v1();  -- Time-based UUID
```

### ARRAY Type

Store multiple values in a single column.

```sql
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    interests TEXT[] NOT NULL DEFAULT '{}',
    scores INTEGER[] NOT NULL DEFAULT '{}'
);

-- Insert arrays
INSERT INTO users (interests, scores)
VALUES 
    (ARRAY['reading', 'hiking', 'cooking'], ARRAY[85, 92, 78]),
    (ARRAY['gaming', 'movies'], ARRAY[95, 88]);

-- Query array elements
SELECT 
    user_id,
    interests[1] AS first_interest,
    scores[2] AS second_score
FROM users;

-- Array operations
SELECT * FROM users WHERE 'hiking' = ANY(interests);
SELECT * FROM users WHERE ARRAY_LENGTH(interests, 1) > 2;
```

### ENUM Type

Create custom enumerated types for constrained values.

```sql
-- Create enum type
CREATE TYPE status_type AS ENUM ('pending', 'processing', 'completed', 'failed');

-- Use in table
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    status status_type NOT NULL DEFAULT 'pending'
);

-- Insert with enum values
INSERT INTO orders (status) VALUES ('processing');

-- Enum operations
SELECT * FROM orders WHERE status = 'completed';
```

---

## 7. Type Conversion and Casting

### Explicit Casting

```sql
-- Using CAST()
SELECT CAST('123' AS INTEGER) AS number;

-- Using double colon syntax (PostgreSQL)
SELECT '123'::INTEGER AS number;

-- Converting between types
SELECT 
    '2024-01-15'::DATE AS date_value,
    123::TEXT AS text_value,
    123.45::INTEGER AS integer_value,
    'true'::BOOLEAN AS boolean_value;
```

### Implicit Casting

Some conversions happen automatically:

```sql
-- String to integer (when context requires it)
SELECT 123 + '456';  -- Works: '456' converted to 456

-- But be careful with ambiguous cases
SELECT '123' + '456';  -- Error in PostgreSQL, concatenation in MySQL
```

### Safe Casting with TRY_CAST (PostgreSQL 16+)

```sql
-- Try to cast, return NULL on failure
SELECT TRY_CAST('123' AS INTEGER);    -- Returns 123
SELECT TRY_CAST('abc' AS INTEGER);    -- Returns NULL
```

---

## 8. Best Practices Summary

### Golden Rules for Data Types

1. **Choose the smallest appropriate type** - saves space and improves performance
2. **Use DECIMAL for money** - never FLOAT/DOUBLE
3. **Prefer JSONB over JSON** - better performance and indexing
4. **Use TIMESTAMPTZ for timestamps** - handles timezones correctly
5. **Be explicit about NULL handling** - use DEFAULTs and COALESCE
6. **Index appropriately** - consider which queries you'll run
7. **Validate data at the database level** - use CHECK constraints

### Type Selection Decision Tree

```
What kind of data?
├── Numbers
│   ├── Whole numbers only? → INTEGER (or SMALLINT/BIGINT as needed)
│   └── Decimal points? → DECIMAL(p,s) for exact, FLOAT for approximate
├── Text
│   ├── Fixed length? → CHAR(n)
│   ├── Variable length, known max? → VARCHAR(n)
│   └── Variable length, unknown max? → TEXT
├── Dates/Times
│   ├── Date only? → DATE
│   ├── Time only? → TIME
│   ├── Date + time? → TIMESTAMPTZ (preferred) or TIMESTAMP
│   └── Duration? → INTERVAL
├── True/False? → BOOLEAN
├── Structured data? → JSONB (preferred) or JSON
└── Special cases? → UUID, ARRAY, ENUM, etc.
```

### Common Pitfalls to Avoid

- Using `VARCHAR(255)` for everything (wastes space, limits flexibility)
- Storing dates as strings (breaks sorting, queries, and validation)
- Using FLOAT for monetary values (causes rounding errors)
- Ignoring NULL handling (leads to unexpected query results)
- Not considering timezone implications for timestamps

---

## 🧠 Knowledge Check: Data Types Quiz

Test your understanding with these multiple-choice questions:

1. **Which data type should you use for storing monetary values?**
   - A) `FLOAT`
   - B) `DOUBLE`
   - C) `DECIMAL` ✅
   - D) `INTEGER`

2. **What's the main advantage of `JSONB` over `JSON`?**
   - A) Smaller storage size
   - B) Better indexing support ✅
   - C) Preserves duplicate keys
   - D) Faster text parsing

3. **When should you use `TIMESTAMPTZ` instead of `TIMESTAMP`?**
   - A) When you need timezone awareness ✅
   - B) When you want smaller storage
   - C) When you're storing dates only
   - D) When you need faster queries

4. **How do you check for NULL values in SQL?**
   - A) `column = NULL`
   - B) `column == NULL`
   - C) `column IS NULL` ✅
   - D) `NULL(column)`

5. **Which operator extracts JSON data as text in PostgreSQL?**
   - A) `->`
   - B) `->>`
   - C) `@>` ✅
   - D) `?`

**Answers**: 1-C, 2-B, 3-A, 4-C, 5-B

---

## Next Steps

You now have a solid foundation in database data types. In the next document, we'll explore visual database concepts including ER diagrams, schema evolution, and query execution plans.

**Recommended next reading**: [`02_visual_database_concepts.md`](02_visual_database_concepts.md)

> 💡 **Pro Tip**: When designing your first database schema, create a data dictionary documenting each column's purpose, data type, constraints, and business rules. This becomes invaluable as your application grows.
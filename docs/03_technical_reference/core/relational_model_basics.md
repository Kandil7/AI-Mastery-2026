# Relational Model Basics for AI/ML Engineers

This document covers the fundamentals of the relational model, normalization theory, and ER modeling—essential knowledge for AI/ML engineers who need to design robust data systems.

## The Relational Model: Core Principles

The relational model, introduced by E.F. Codd in 1970, is based on mathematical set theory and predicate logic. Its key components:

### Relations (Tables)
- A relation is a set of tuples (rows) with attributes (columns)
- Each relation has a schema defining attribute names and domains
- Properties:
  - **No duplicate rows** (sets, not multisets)
  - **Order doesn't matter** (no inherent row ordering)
  - **Attributes have unique names** within a relation

### Keys and Constraints
- **Primary Key**: Minimal set of attributes that uniquely identifies each tuple
- **Foreign Key**: Attribute(s) referencing primary key of another relation
- **Candidate Keys**: All minimal superkeys (primary key is chosen from these)
- **Constraints**: NOT NULL, UNIQUE, CHECK, FOREIGN KEY

### Relational Algebra Operations
Fundamental operations that form the basis of SQL:
- **Selection (σ)**: Filter rows (WHERE clause)
- **Projection (π)**: Select columns (SELECT clause)
- **Cartesian Product (×)**: Combine all rows from two relations
- **Join (⨝)**: Combine related rows (JOIN clauses)
- **Union (∪)**: Combine sets of tuples
- **Difference (−)**: Remove tuples present in second relation
- **Rename (ρ)**: Change attribute names

## Normalization Theory

Normalization eliminates redundancy and prevents update anomalies through progressive normal forms:

### First Normal Form (1NF)
- Each attribute contains only atomic (indivisible) values
- No repeating groups or arrays in columns
- Example violation: `tags VARCHAR[]` instead of separate table

### Second Normal Form (2NF)
- Must be in 1NF
- All non-key attributes fully functionally dependent on the entire primary key
- Eliminates partial dependencies
- Example: In `(order_id, product_id, customer_name, quantity)`, `customer_name` depends only on `order_id`, not the composite key

### Third Normal Form (3NF)
- Must be in 2NF
- No transitive dependencies: non-key attributes don't depend on other non-key attributes
- Example violation: `(employee_id, department_id, department_name)` where `department_name` depends on `department_id`

### Boyce-Codd Normal Form (BCNF)
- Stronger than 3NF
- For every functional dependency X → Y, X must be a superkey
- Addresses cases where 3NF isn't sufficient (multi-valued dependencies)

### Fourth Normal Form (4NF) and Fifth Normal Form (5NF)
- Address multi-valued dependencies and join dependencies
- Less commonly needed in practice

## Entity-Relationship (ER) Modeling

ER modeling provides a conceptual view before implementing relational schemas:

### Core Components
- **Entities**: Real-world objects (e.g., User, Product, Model)
- **Attributes**: Properties of entities (e.g., user_id, name, created_at)
- **Relationships**: Associations between entities (e.g., User ↔ owns → Model)
- **Cardinalities**: 1:1, 1:N, N:M relationships

### ER Diagram Notation
```
┌─────────────┐       ┌─────────────┐
│    User     │───────│    Model    │
└─────────────┘ 1   N └─────────────┘
   ▲  │user_id      ▲  │model_id
   │  │             │  │
   │  └─────────────┴──┘
   │        owns
   │
┌─────────────┐
│  Training   │
└─────────────┘
   ▲  │training_id
   │  │
   └──┴─────────────┐
        trained_on   │
                     ▼
              ┌─────────────┐
              │   Dataset   │
              └─────────────┘
```

### From ER to Relational Schema
1. **Strong entities** → Tables with primary keys
2. **Weak entities** → Tables with composite keys including owner's PK
3. **1:N relationships** → Foreign key in "many" side
4. **N:M relationships** → Junction table with foreign keys
5. **ISA hierarchies** → Single table, multiple tables, or class table inheritance

## Practical Considerations for ML Workflows

### Feature Store Design
- **Dimension tables**: Static metadata (users, products, models)
- **Fact tables**: Time-series measurements (predictions, features, metrics)
- **Slowly changing dimensions (SCD)**: Handle evolving entity attributes

### Model Metadata Schema
```sql
-- Core model registry
CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    status VARCHAR NOT NULL CHECK (status IN ('draft', 'staging', 'production')),
    owner_id UUID REFERENCES users(user_id)
);

-- Model versions with lineage
CREATE TABLE model_versions (
    version_id UUID PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES models(model_id),
    git_commit_hash VARCHAR,
    training_data_version UUID REFERENCES datasets(dataset_id),
    hyperparameters JSONB,
    metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

-- Model deployments
CREATE TABLE model_deployments (
    deployment_id UUID PRIMARY KEY,
    model_version_id UUID NOT NULL REFERENCES model_versions(version_id),
    environment VARCHAR NOT NULL CHECK (environment IN ('dev', 'staging', 'prod')),
    endpoint_url VARCHAR,
    deployed_at TIMESTAMPTZ NOT NULL,
    status VARCHAR NOT NULL
);
```

## Common Pitfalls and Anti-patterns

### 1. Over-normalization
- Excessive joins hurting performance
- Complex queries for simple operations
- Solution: Strategic denormalization for read-heavy workloads

### 2. Under-normalization
- Data redundancy causing update anomalies
- Inconsistent data across records
- Solution: Apply appropriate normalization for write-heavy workloads

### 3. Ignoring Temporal Aspects
- Not tracking when data changed
- Losing historical context for ML reproducibility
- Solution: Use temporal tables or append-only designs

### 4. Poor Primary Key Design
- Using natural keys that change (emails, names)
- Large composite keys hurting index performance
- Solution: Use surrogate keys (UUID, auto-increment) for stability

## Performance Optimization for ML Workloads

### Indexing Strategies
- **Composite indexes**: For common query patterns (e.g., `(model_id, created_at DESC)`)
- **Partial indexes**: For filtered subsets (e.g., `WHERE status = 'production'`)
- **Expression indexes**: For computed values (e.g., `LOWER(name)`)

### Partitioning
- **Range partitioning**: By time (ideal for time-series ML data)
- **List partitioning**: By categorical values (e.g., model types)
- **Hash partitioning**: For even distribution across nodes

### Materialized Views
- Pre-computed aggregations for dashboard queries
- Refresh strategies: manual, on commit, scheduled
- Ideal for ML monitoring dashboards and experiment tracking

## Visual Diagrams

### Normalization Process
```
Unnormalized Form (UNF)
  ├── Repeating groups
  ├── Multi-valued attributes
  └── Non-atomic values
        ↓
First Normal Form (1NF)
  ├── Atomic values only
  ├── No repeating groups
  └── Each cell contains single value
        ↓
Second Normal Form (2NF)
  ├── All non-key attributes depend on ENTIRE PK
  └── Eliminates partial dependencies
        ↓
Third Normal Form (3NF)
  ├── No transitive dependencies
  └── Non-key attributes depend ONLY on PK
        ↓
Boyce-Codd Normal Form (BCNF)
  ├── Every determinant is a candidate key
  └── Strongest form for most practical cases
```

### Typical ML Data Schema
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Users     │    │   Models    │    │  Datasets   │
└─────────────┘    └─────────────┘    └─────────────┘
│ user_id (PK) │    │ model_id(PK)│    │ dataset_id(PK)│
│ name         │    │ name         │    │ name          │
│ email        │    │ description  │    │ version       │
│ created_at   │    │ created_at  │    │ size_bytes    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                    │
       │                   │                    │
       └─────────┬─────────┴───────────┬────────┘
                 │                     │
           ┌─────────────────┐   ┌─────────────────┐
           │ ModelVersions   │   │ TrainingRuns    │
           └─────────────────┘   └─────────────────┘
           │ version_id (PK) │   │ run_id (PK)     │
           │ model_id (FK)   │   │ model_id (FK)   │
           │ metrics JSONB   │   │ dataset_id (FK) │
           │ hyperparams JSONB│  │ start_time      │
           └─────────────────┘   └─────────────────┘
```

## Best Practices for AI/ML Engineers

1. **Design for reproducibility**: Include versioning at all levels
2. **Separate concerns**: Different tables for metadata vs. features vs. results
3. **Plan for growth**: Start with appropriate normalization, but be ready to denormalize
4. **Use constraints**: Enforce data integrity with foreign keys and check constraints
5. **Consider temporal needs**: Track when data was created/updated for ML lineage

This foundation enables you to design robust data systems that support complex ML workflows while maintaining data integrity and performance.
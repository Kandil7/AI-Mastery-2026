# Normalization Forms

Normalization is the process of organizing data to minimize redundancy and prevent update anomalies. Each normal form represents a specific level of organization, with higher forms providing stronger guarantees against data inconsistencies.

## Overview

For senior AI/ML engineers, understanding normalization is crucial for designing databases that support reliable data pipelines, model training, and production inference systems. Proper normalization ensures data integrity while allowing for flexible schema evolution.

## First Normal Form (1NF)

Each column contains only atomic (indivisible) values. No repeating groups or arrays.

### Requirements
- Each column contains only single values
- Each row is unique
- No repeating groups

### Example - Violation and Fix

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

## Second Normal Form (2NF)

Must be in 1NF, and all non-key attributes must fully functionally depend on the entire primary key (no partial dependencies).

### Requirements
- Must be in 1NF
- No partial dependencies on part of a composite primary key

### Example - Violation and Fix

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

## Third Normal Form (3NF)

Must be in 2NF, and no transitive dependencies (non-key attributes should depend only on the primary key).

### Requirements
- Must be in 2NF
- No transitive dependencies

### Example - Violation and Fix

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

## Boyce-Codd Normal Form (BCNF)

A stronger version of 3NF where every determinant must be a candidate key.

### Requirements
- Must be in 3NF
- For every functional dependency X → Y, X must be a superkey

### Example - BCNF Violation

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

## Fourth Normal Form (4NF) and Fifth Normal Form (5NF)

### Fourth Normal Form (4NF)
- Eliminates multi-valued dependencies
- Requires that no table contains two or more independent multi-valued facts about an entity

### Fifth Normal Form (5NF)
- Eliminates join dependencies
- Ensures that every join dependency is implied by the candidate keys

### When to Stop Normalizing
Most practical database designs stop at 3NF or BCNF because:
- Higher normal forms can lead to excessive joins
- Performance may suffer due to increased complexity
- Modern databases provide alternative optimization techniques

## Normalization Summary Table

| Normal Form | Goal | Key Rule | Practical Consideration |
|-------------|------|----------|-------------------------|
| 1NF | Eliminate repeating groups | Atomic values | Basic requirement for all databases |
| 2NF | Eliminate partial dependencies | Full functional dependency on PK | Important for composite keys |
| 3NF | Eliminate transitive dependencies | Non-key attributes depend only on PK | Most common target for production systems |
| BCNF | Strengthen 3NF | Every determinant is a candidate key | Good for complex schemas with multiple candidate keys |
| 4NF | Eliminate multi-valued dependencies | No independent multi-valued facts | Rarely needed in practice |
| 5NF | Eliminate join dependencies | Join dependencies implied by keys | Theoretical, rarely implemented |

## Denormalization Considerations

While normalization reduces redundancy, denormalization is often necessary for performance optimization:

### When to Denormalize
- **Read-heavy workloads**: Where query speed is critical
- **Reporting/analytics**: Requiring complex aggregations
- **Caching frequently accessed data**: From multiple tables
- **Reducing join complexity**: For common queries

### Common Denormalization Patterns
1. **Pre-Joining Tables**: Store joined data to avoid expensive joins
2. **Storing Derived Data**: Pre-compute and store calculated values
3. **Storing Duplicate Data**: Copy frequently accessed columns

### Example - Denormalized Order Table
```sql
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
```

## AI/ML Engineering Applications

### Model Registry Design
- **Normalized**: Separate tables for models, versions, experiments
- **Denormalized**: Combined view for fast dashboard queries
- **Hybrid**: Normalized core + materialized views for analytics

### Feature Store Design
- **1NF**: Individual feature values
- **2NF**: Separate feature definitions from values
- **3NF**: Normalize metadata and statistics

### Training Data Management
- **Normalization**: Reduce redundancy in dataset versions
- **Denormalization**: Optimize for batch processing performance

## Best Practices for AI/ML Databases

1. **Normalize first, denormalize later**: Start with a normalized design
2. **Document denormalization**: Make it explicit in code and docs
3. **Use triggers/materialized views**: Automate consistency
4. **Monitor for drift**: Check for data inconsistencies
5. **Consider alternatives**: Indexes, caching, query optimization first

## Related Resources

- [Entity-Relationship Modeling] - Foundation for understanding relationships
- [Schema Design Patterns] - Common patterns for different use cases
- [Data Modeling for AI/ML] - Specialized patterns for machine learning applications
- [Performance Optimization] - Balancing normalization with performance needs
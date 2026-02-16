# Schema Design Patterns

Different schema patterns suit different use cases. Understanding these patterns helps senior AI/ML engineers choose the right approach for their applications, from data warehousing to real-time analytics and ML model management.

## Overview

Schema design patterns provide proven solutions to common database design problems. For AI/ML applications, selecting the appropriate pattern is critical for supporting complex data relationships, high-performance queries, and scalable architecture.

## Star Schema

The star schema organizes data into fact and dimension tables, with the fact table at the center connected to dimension tables.

### Components
- **Fact Table**: Contains quantitative data (metrics) to be analyzed
- **Dimension Tables**: Contain descriptive attributes for filtering and labeling

### Best For
- Data warehousing
- Business intelligence
- Analytics workloads
- Reporting systems

### Example - E-Commerce Star Schema

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

### Advantages
- Simple queries with fewer joins
- Fast aggregation queries
- Easy to understand and maintain
- Excellent for BI tools and reporting

### Disadvantages
- Data redundancy in dimension tables
- Less normalized than other schemas
- Can become complex with many dimensions

## Snowflake Schema

The snowflake schema is a normalized version of the star schema where dimension tables are further normalized into multiple related tables.

### Example - Snowflake Schema

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

### Advantages
- Reduced data redundancy
- Easier data maintenance
- Better storage efficiency
- More normalized structure

### Disadvantages
- More complex queries (more joins)
- Slower query performance
- Harder to understand for business users
- Increased complexity for BI tools

## Galaxy Schema (Fact Constellation)

Multiple fact tables share dimension tables, useful for complex analytical requirements.

### Use Cases
- Multiple business processes sharing common dimensions
- Different granularities of analysis
- Complex enterprise data warehouses

### Example Structure
```
          dim_time
             |
   ------------+------------
   |                       |
fact_sales        fact_inventory
   |                       |
dim_product           dim_product
   |                       |
dim_customer        dim_supplier
```

## Entity-Attribute-Value (EAV)

Used for highly variable schemas where entities can have different attributes.

### Structure
```sql
CREATE TABLE product_attributes (
    entity_id INT NOT NULL,
    attribute_name VARCHAR(50) NOT NULL,
    value VARCHAR(255),
    PRIMARY KEY (entity_id, attribute_name)
);
```

### Best For
- Product catalogs with varying attributes
- CMS systems
- Medical records with variable fields
- Configuration management

### Considerations
- Complex queries required for filtering
- Performance overhead due to pivoting
- Difficult to enforce data types and constraints
- Requires application-level validation

## Document Model (NoSQL)

For flexible schemas, document databases allow varying structures.

### Example - Flexible Product Catalog
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

### Advantages
- Extreme flexibility for evolving schemas
- Natural fit for JSON-based APIs
- Good for semi-structured data
- Horizontal scalability

### Disadvantages
- No ACID guarantees (typically)
- Complex joins across documents
- Query optimization challenges
- Schema evolution requires careful migration

## AI/ML Specific Schema Patterns

### Model Registry Schema
```sql
-- Core model metadata
CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    framework VARCHAR(50),
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'staging'
);

-- Model versions with artifacts
CREATE TABLE model_versions (
    version_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    version_number INT NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics JSONB,
    hyperparameters JSONB,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- Experiment tracking
CREATE TABLE experiments (
    experiment_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    parameters JSONB,
    results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- Prediction logs
CREATE TABLE predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_version_id UUID NOT NULL,
    input_features JSONB NOT NULL,
    output_predictions JSONB NOT NULL,
    prediction_proba JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (model_version_id) REFERENCES model_versions(version_id)
);
```

### Feature Store Schema
```sql
-- Feature definitions
CREATE TABLE feature_definitions (
    feature_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    data_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature values (time-series)
CREATE TABLE feature_values (
    feature_id UUID NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (feature_id, entity_id, timestamp),
    FOREIGN KEY (feature_id) REFERENCES feature_definitions(feature_id)
);

-- Feature views (materialized aggregates)
CREATE TABLE feature_views (
    view_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    definition JSONB NOT NULL,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
```

## Pattern Selection Guide

### Choose Star Schema When:
- You need fast analytical queries
- Your data has clear fact/dimension relationships
- You're building BI dashboards
- Query simplicity is more important than storage efficiency

### Choose Snowflake Schema When:
- You have significant data redundancy concerns
- Your organization requires strict data governance
- You have complex hierarchies in dimension data
- Storage efficiency is critical

### Choose Document Model When:
- Your data schema evolves frequently
- You have highly variable attributes
- You need horizontal scalability
- You're building modern web/mobile applications

### Choose EAV When:
- You have extremely variable data structures
- You need maximum flexibility for new attributes
- Performance is secondary to flexibility
- You can handle the query complexity

## Related Resources

- [Entity-Relationship Modeling] - Foundation for understanding relationships
- [Normalization Forms] - Formal rules for database design
- [Data Modeling for AI/ML] - Specialized patterns for machine learning applications
- [Performance Optimization] - How schema choices affect query performance
# Getting Started with Database Systems for AI/ML Engineers

## Executive Summary

This comprehensive getting started guide provides foundational knowledge for AI/ML engineers who need to work with database systems. Designed specifically for senior AI/ML engineers transitioning into database-intensive roles, this tutorial covers the essential concepts and practical skills needed to be productive quickly.

**Key Features**:
- Beginner-friendly introduction to database fundamentals
- AI/ML-specific context and examples
- Hands-on exercises with real-world scenarios
- Integration with existing AI/ML workflows
- Practical troubleshooting guidance

## Learning Path Overview

### Phase 1: Database Fundamentals (Weeks 1-2)
- Understanding relational vs. NoSQL databases
- SQL basics for data extraction and transformation
- Data modeling for ML datasets
- Basic performance optimization

### Phase 2: AI/ML Integration (Weeks 3-4)
- Vector databases for embedding storage
- Feature stores for ML pipelines
- Real-time inference database patterns
- RAG system fundamentals

### Phase 3: Production Readiness (Weeks 5-6)
- Security and compliance for AI/ML data
- Monitoring and observability
- CI/CD for database changes
- SRE practices for database reliability

## Step-by-Step Getting Started Guide

### 1. Setting Up Your Development Environment

**Local Setup Options**:
```bash
# Option 1: Docker-based (recommended)
docker run --name postgres-dev -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:14

# Option 2: Local installation (PostgreSQL)
# Download from https://www.postgresql.org/download/
# Install with default settings, password: 'password'

# Option 3: Cloud-based (AWS RDS)
# Create PostgreSQL instance, note endpoint and credentials
```

**Python Environment Setup**:
```bash
# Create virtual environment
python -m venv db-ai-env
source db-ai-env/bin/activate  # On Windows: db-ai-env\Scripts\activate

# Install essential packages
pip install psycopg2-binary sqlalchemy pandas numpy scikit-learn
pip install milvus-client langchain openai

# Verify setup
python -c "import psycopg2; print('PostgreSQL client installed')"
python -c "import milvus; print('Milvus client installed')"
```

### 2. Essential SQL for AI/ML Engineers

**Core SQL Concepts**:
```sql
-- 1. Data extraction for ML training
SELECT 
    user_id,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_transaction,
    MAX(timestamp) as last_activity
FROM transactions 
WHERE timestamp >= '2024-01-01'
GROUP BY user_id
HAVING COUNT(*) > 5;

-- 2. Feature engineering with SQL
SELECT 
    user_id,
    -- Time-based features
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    EXTRACT(DOW FROM timestamp) as day_of_week,
    -- Behavioral features
    COUNT(*) FILTER (WHERE amount > 100) as high_value_transactions,
    -- Aggregation features
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_amount
FROM transactions
GROUP BY user_id;

-- 3. Joining multiple data sources
SELECT 
    u.user_id,
    u.age,
    u.location,
    t.transaction_count,
    t.avg_amount,
    f.feature_vector
FROM users u
LEFT JOIN (
    SELECT user_id, COUNT(*) as transaction_count, AVG(amount) as avg_amount
    FROM transactions GROUP BY user_id
) t ON u.user_id = t.user_id
LEFT JOIN feature_store f ON u.user_id = f.entity_id;
```

### 3. Hands-On Exercise: Build a Simple Recommendation System

**Exercise Overview**: Create a basic collaborative filtering system using SQL and Python.

**Step 1: Create sample data**
```sql
-- Users table
CREATE TABLE users (user_id INT PRIMARY KEY, name VARCHAR(100));
INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');

-- Items table  
CREATE TABLE items (item_id INT PRIMARY KEY, name VARCHAR(100));
INSERT INTO items VALUES (1, 'Movie A'), (2, 'Movie B'), (3, 'Movie C');

-- Ratings table
CREATE TABLE ratings (user_id INT, item_id INT, rating FLOAT);
INSERT INTO ratings VALUES 
(1, 1, 5), (1, 2, 3), (1, 3, 4),
(2, 1, 4), (2, 2, 5), (2, 3, 2),
(3, 1, 2), (3, 2, 3), (3, 3, 5);
```

**Step 2: Calculate user similarities**
```sql
-- Calculate cosine similarity between users
WITH user_vectors AS (
    SELECT 
        user_id,
        ARRAY_AGG(rating ORDER BY item_id) as ratings_vector
    FROM ratings
    GROUP BY user_id
),
similarity_matrix AS (
    SELECT 
        u1.user_id as user1,
        u2.user_id as user2,
        -- Cosine similarity calculation
        (SUM(u1.rating * u2.rating) / 
         (SQRT(SUM(u1.rating * u1.rating)) * SQRT(SUM(u2.rating * u2.rating)))
        ) as similarity
    FROM ratings u1
    JOIN ratings u2 ON u1.item_id = u2.item_id AND u1.user_id < u2.user_id
    GROUP BY u1.user_id, u2.user_id
)
SELECT * FROM similarity_matrix ORDER BY similarity DESC;
```

**Step 3: Implement recommendation logic in Python**
```python
import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(user_id, conn):
    """Get movie recommendations for a user"""
    # Get user ratings
    cursor = conn.cursor()
    cursor.execute("""
        SELECT item_id, rating FROM ratings WHERE user_id = %s
    """, (user_id,))
    user_ratings = dict(cursor.fetchall())
    
    # Get all items
    cursor.execute("SELECT item_id FROM items")
    all_items = [row[0] for row in cursor.fetchall()]
    
    # Calculate predictions
    recommendations = []
    for item_id in all_items:
        if item_id in user_ratings:
            continue
            
        # Simple collaborative filtering
        predicted_rating = 3.0  # baseline
        # Add more sophisticated logic here
        
        recommendations.append((item_id, predicted_rating))
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

# Test the function
conn = psycopg2.connect(
    host="localhost", database="postgres", 
    user="postgres", password="password"
)
print(get_recommendations(1, conn))
conn.close()
```

### 4. Common Pitfalls and Troubleshooting

**SQL Performance Issues**:
- **Problem**: Queries taking too long on large datasets
- **Solution**: Add appropriate indexes, use EXPLAIN ANALYZE, consider materialized views

**Data Quality Issues**:
- **Problem**: NULL values causing ML model failures
- **Solution**: Use COALESCE, CASE statements, or preprocessing pipelines

**Connection Issues**:
- **Problem**: "Too many connections" errors
- **Solution**: Use connection pooling, increase max_connections, optimize query patterns

### 5. Next Steps and Resources

**Immediate Next Steps**:
1. Complete the hands-on exercise above
2. Explore the `01_foundations/` directory for deeper learning
3. Try the vector database tutorial in `06_tutorials/02_ai_ml_integration/`

**Recommended Resources**:
- PostgreSQL Documentation: https://www.postgresql.org/docs/
- SQL for Data Scientists: https://www.sqlfordata.science/
- Milvus Quick Start: https://milvus.io/docs/install_standalone-docker.md
- LangChain Database Integration: https://python.langchain.com/docs/integrations/vectorstores/milvus

## Conclusion

This getting started guide provides the foundation you need to begin working effectively with database systems as an AI/ML engineer. The key is to start simple, focus on practical applications, and gradually build your database expertise alongside your AI/ML skills.

Remember: You don't need to become a database expert overnight. Focus on understanding the core concepts and how they apply to your AI/ML work, then expand your knowledge as needed.
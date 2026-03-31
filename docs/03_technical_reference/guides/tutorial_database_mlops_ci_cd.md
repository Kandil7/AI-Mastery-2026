# Database Integration with MLOps and CI/CD Pipelines Tutorial

## Overview

This tutorial focuses on integrating databases with MLOps practices and CI/CD pipelines. We'll cover database versioning, model registry integration, automated testing, deployment automation, and production monitoring specifically for senior AI/ML engineers building robust ML systems.

## Prerequisites
- Python 3.8+
- MLflow, DVC, or similar MLOps tools
- Git, GitHub Actions/GitLab CI/Bitbucket Pipelines
- PostgreSQL/MySQL with proper backup and migration capabilities
- Basic understanding of MLOps concepts

## Tutorial Structure
1. **Database Versioning** - Schema and data versioning
2. **Model Registry Integration** - Database-backed model metadata
3. **Automated Testing** - Database-integrated testing
4. **CI/CD Pipeline Integration** - Automated deployment workflows
5. **Production Monitoring** - Database-backed observability
6. **Performance Benchmarking** - MLOps pipeline efficiency

## Section 1: Database Versioning

### Step 1: Schema versioning with Alembic
```python
from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Base model for database schema
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingData(Base):
    __tablename__ = 'training_data'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    features = Column(JSON)
    label = Column(Boolean)
    split = Column(String(10))  # 'train', 'val', 'test'
    created_at = Column(DateTime, default=datetime.utcnow)

# Alembic environment configuration
def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=Base.metadata
        )

        with context.begin_transaction():
            context.run_migrations()

# Migration script example
"""Create training data table

Revision ID: abc123
Revises: xyz789
Create Date: 2024-01-01 12:00:00
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'training_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('label', sa.Boolean(), nullable=True),
        sa.Column('split', sa.String(length=10), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('training_data')
```

### Step 2: Data versioning with DVC
```python
import dvc.api
import pandas as pd
import os

class DatabaseDataVersioning:
    def __init__(self, dvc_repo_path: str = "."):
        self.dvc_repo_path = dvc_repo_path
    
    def version_database_export(self, db_config: Dict, query: str, 
                              dataset_name: str, description: str = ""):
        """Version database export using DVC"""
        # Export data from database
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Save to file
        filename = f"data/{dataset_name}.parquet"
        os.makedirs("data", exist_ok=True)
        df.to_parquet(filename, index=False)
        
        # Add to DVC
        os.system(f"dvc add {filename}")
        
        # Create DVC file with metadata
        dvc_yaml = f"""
{dataset_name}:
  desc: {description}
  type: dataset
  labels:
    - ml
    - training
  meta:
    source: database
    query: "{query}"
    rows: {len(df)}
    columns: {list(df.columns)}
    created_at: {pd.Timestamp.now().isoformat()}
"""
        
        with open(f"{filename}.dvc", 'w') as f:
            f.write(dvc_yaml)
        
        return filename
    
    def load_versioned_data(self, dataset_name: str):
        """Load versioned data from DVC"""
        try:
            # Use DVC API to get data
            with dvc.api.open(f"data/{dataset_name}.parquet", mode='rb') as f:
                df = pd.read_parquet(f)
            return df
        except Exception as e:
            # Fallback to local file
            filename = f"data/{dataset_name}.parquet"
            if os.path.exists(filename):
                return pd.read_parquet(filename)
            raise e
    
    def track_data_lineage(self, dataset_name: str, upstream_datasets: List[str] = None):
        """Track data lineage for versioned datasets"""
        lineage_file = f"data/{dataset_name}_lineage.json"
        
        lineage_data = {
            'dataset': dataset_name,
            'created_at': pd.Timestamp.now().isoformat(),
            'upstream_datasets': upstream_datasets or [],
            'transformation_steps': [
                {
                    'step': 'database_export',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {'source': 'database'}
                }
            ]
        }
        
        with open(lineage_file, 'w') as f:
            json.dump(lineage_data, f, indent=2)
        
        return lineage_file

# Usage example
data_versioning = DatabaseDataVersioning()

# Version database export
db_config = {
    'host': 'localhost',
    'database': 'ai_db',
    'user': 'postgres',
    'password': 'password'
}

query = """
SELECT user_id, age, engagement_score, session_count, label
FROM training_data 
WHERE split = 'train'
"""

dataset_file = data_versioning.version_database_export(
    db_config, query, "training_dataset_v1",
    "Training dataset for user engagement prediction"
)

print(f"Dataset versioned: {dataset_file}")

# Load versioned data
df = data_versioning.load_versioned_data("training_dataset_v1")
print(f"Loaded {len(df)} rows")
```

## Section 2: Model Registry Integration

### Step 1: MLflow integration with database metadata
```python
import mlflow
import mlflow.pytorch
import pandas as pd
from typing import Dict, Any

class DatabaseModelRegistry:
    def __init__(self, db_config: Dict, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.db_config = db_config
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = mlflow.MlflowClient()
    
    def log_model_with_database_info(self, model, model_name: str,
                                   database_info: Dict, metrics: Dict):
        """Log model with comprehensive database metadata"""
        with mlflow.start_run() as run:
            # Log model
            mlflow.pytorch.log_model(model, model_name)
            
            # Log database information
            mlflow.log_param("database_type", database_info.get("type", "unknown"))
            mlflow.log_param("database_version", database_info.get("version", "unknown"))
            mlflow.log_param("table_name", database_info.get("table_name", "unknown"))
            mlflow.log_param("query_complexity", database_info.get("query_complexity", "medium"))
            mlflow.log_param("data_version", database_info.get("data_version", "unknown"))
            
            # Log feature information
            mlflow.log_param("num_features", database_info.get("num_features", 0))
            mlflow.log_param("feature_types", str(database_info.get("feature_types", [])))
            
            # Log training data info
            mlflow.log_param("training_data_size", database_info.get("training_data_size", 0))
            mlflow.log_param("data_split", database_info.get("data_split", "unknown"))
            mlflow.log_param("data_timestamp", database_info.get("data_timestamp", "unknown"))
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model architecture
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("model_params", str(model.__dict__.get('state_dict', {}))[:100])
    
    def register_model_from_database(self, model_name: str, version: str,
                                   source_query: str, description: str = ""):
        """Register model from database source with versioning"""
        # Get model info from database
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                model_id, model_name, version, created_at, accuracy, 
                training_data_version, data_source
            FROM models 
            WHERE model_name = %s AND version = %s
        """, (model_name, version))
        
        model_info = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not model_info:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Create model version with database metadata
        model_version = self.client.create_model_version(
            name=model_name,
            source=f"models/{model_name}",
            run_id="dummy_run_id",
            description=description,
            tags={
                "database_source": model_info[6],
                "training_data_version": model_info[5],
                "accuracy": str(model_info[4]),
                "created_at": model_info[3].isoformat()
            }
        )
        
        return model_version
    
    def get_model_by_database_version(self, database_version: str):
        """Get model by database version"""
        # Search MLflow for models with specific database version tag
        models = self.client.search_model_versions(
            filter_string=f"tags.database_version='{database_version}'"
        )
        
        if models:
            return models[0]
        return None

# Usage example
model_registry = DatabaseModelRegistry(db_config)

# Log model with database info
database_info = {
    "type": "PostgreSQL",
    "version": "14.7",
    "table_name": "training_data",
    "query_complexity": "high",
    "data_version": "v2024-01-15",
    "num_features": 5,
    "feature_types": ["int", "int", "float", "int", "bool"],
    "training_data_size": 1000000,
    "data_split": "80-10-10",
    "data_timestamp": "2024-01-15T12:00:00Z"
}

metrics = {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90,
    "latency_ms": 120.5
}

# In practice, replace None with actual model
model_registry.log_model_with_database_info(
    model=None,
    model_name="user_engagement_predictor",
    database_info=database_info,
    metrics=metrics
)
```

### Step 2: Custom model registry with database storage
```python
class CustomModelRegistry:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_model_registry_tables(self):
        """Create tables for custom model registry"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            description TEXT,
            model_type VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            status VARCHAR(20) DEFAULT 'active',
            UNIQUE(name, version)
        );
        """)
        
        # Model versions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES models(id),
            version VARCHAR(50) NOT NULL,
            model_path TEXT,
            metrics JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            is_current BOOLEAN DEFAULT FALSE,
            UNIQUE(model_id, version)
        );
        """)
        
        # Model deployments table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_deployments (
            id SERIAL PRIMARY KEY,
            model_version_id INTEGER REFERENCES model_versions(id),
            endpoint_url TEXT,
            deployment_status VARCHAR(20),
            deployed_at TIMESTAMP DEFAULT NOW(),
            environment VARCHAR(50)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Model registry tables created successfully"
    
    def register_model_version(self, model_name: str, version: str,
                             model_path: str, metrics: Dict, metadata: Dict):
        """Register model version in custom registry"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Insert or get model
        cursor.execute("""
            INSERT INTO models (name, version, description, model_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name, version) DO UPDATE 
            SET updated_at = NOW()
            RETURNING id
        """, (model_name, version, metadata.get('description', ''), metadata.get('model_type', '')))
        
        model_id = cursor.fetchone()[0]
        
        # Insert model version
        cursor.execute("""
            INSERT INTO model_versions (model_id, version, model_path, metrics, metadata, is_current)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (model_id, version, model_path, json.dumps(metrics), json.dumps(metadata), True))
        
        version_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return version_id
    
    def get_latest_model_version(self, model_name: str):
        """Get latest model version"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT mv.*
            FROM model_versions mv
            JOIN models m ON mv.model_id = m.id
            WHERE m.name = %s AND mv.is_current = TRUE
            ORDER BY mv.created_at DESC
            LIMIT 1
        """, (model_name,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result

# Usage example
custom_registry = CustomModelRegistry(db_config)
custom_registry.create_model_registry_tables()

# Register model version
version_id = custom_registry.register_model_version(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    model_path="/models/user_engagement_v1.2.3.pkl",
    metrics={
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90
    },
    metadata={
        "description": "Random Forest classifier for user engagement prediction",
        "model_type": "RandomForestClassifier",
        "training_data_version": "v2024-01-15",
        "features": ["user_id", "age", "engagement_score", "session_count", "label"]
    }
)

print(f"Model version registered: {version_id}")
```

## Section 3: Automated Testing

### Step 1: Database-integrated unit tests
```python
import unittest
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import psycopg2

class DatabaseIntegratedTests:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def test_database_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            return True, "Database connection successful"
        except Exception as e:
            return False, f"Database connection failed: {e}"
    
    def test_data_integrity(self, table_name: str, required_columns: List[str]):
        """Test data integrity for a table"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Check table exists
            cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'")
            if cursor.fetchone()[0] == 0:
                return False, f"Table '{table_name}' does not exist"
            
            # Check required columns
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
            columns = [row[0] for row in cursor.fetchall()]
            
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            
            # Check for null values in critical columns
            critical_columns = [col for col in required_columns if col != 'id']
            for col in critical_columns:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    return False, f"{null_count} null values in column '{col}'"
            
            return True, f"Data integrity check passed for '{table_name}'"
        
        finally:
            cursor.close()
            conn.close()
    
    def test_model_performance_on_database_data(self, model, 
                                              table_name: str,
                                              feature_cols: List[str],
                                              target_col: str,
                                              test_size: float = 0.2):
        """Test model performance on database data"""
        conn = psycopg2.connect(**self.db_config)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Check for overfitting
        overfitting = abs(train_score - test_score) > 0.1
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'overfitting_detected': overfitting,
            'performance_ratio': test_score / train_score if train_score > 0 else 0
        }

# Usage example
db_tests = DatabaseIntegratedTests(db_config)

# Run tests
success, message = db_tests.test_database_connection()
print(f"Database connection: {success} - {message}")

integrity_success, integrity_message = db_tests.test_data_integrity(
    "training_data",
    ["user_id", "age", "engagement_score", "session_count", "label"]
)
print(f"Data integrity: {integrity_success} - {integrity_message}")

# Test model performance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
performance = db_tests.test_model_performance_on_database_data(
    rf_model,
    "training_data",
    ["age", "engagement_score", "session_count"],
    "label"
)
print(f"Model performance: {performance}")
```

### Step 2: Integration tests for MLOps pipelines
```python
import requests
import time
import json

class MLOpsIntegrationTests:
    def __init__(self, api_base_url: str, db_config: Dict):
        self.api_base_url = api_base_url
        self.db_config = db_config
    
    def test_model_deployment(self, model_endpoint: str):
        """Test model deployment endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/{model_endpoint}/health")
            if response.status_code == 200:
                return True, "Endpoint is healthy"
            else:
                return False, f"Endpoint returned status {response.status_code}"
        except Exception as e:
            return False, f"Endpoint unreachable: {e}"
    
    def test_prediction_endpoint(self, endpoint: str, test_data: Dict):
        """Test prediction endpoint with sample data"""
        try:
            response = requests.post(
                f"{self.api_base_url}/{endpoint}/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'prediction' in result:
                    return True, f"Prediction successful: {result['prediction']}"
                else:
                    return False, "Response missing prediction field"
            else:
                return False, f"Prediction failed with status {response.status_code}"
        
        except Exception as e:
            return False, f"Prediction request failed: {e}"
    
    def test_database_integration_in_pipeline(self, pipeline_name: str):
        """Test database integration in MLOps pipeline"""
        # Simulate pipeline execution
        try:
            # Check if pipeline can access database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result[0] == 1:
                # Test data extraction
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM training_data")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                return True, f"Database integration successful. Found {count} training records."
            else:
                return False, "Database query returned unexpected result"
        
        except Exception as e:
            return False, f"Database integration failed: {e}"
    
    def run_comprehensive_mlops_test_suite(self):
        """Run comprehensive MLOps test suite"""
        tests = [
            ("database_connection", self.test_database_connection),
            ("data_integrity", lambda: self.test_data_integrity("training_data", ["user_id", "label"])),
            ("model_deployment", lambda: self.test_model_deployment("user-engagement")),
            ("prediction_endpoint", lambda: self.test_prediction_endpoint(
                "user-engagement", 
                {"user_id": 123, "age": 25, "engagement_score": 0.8, "session_count": 5}
            )),
            ("pipeline_integration", self.test_database_integration_in_pipeline)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success, message = test_func()
                results.append({
                    'test': test_name,
                    'success': success,
                    'message': message,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
            except Exception as e:
                results.append({
                    'test': test_name,
                    'success': False,
                    'message': f"Exception: {e}",
                    'timestamp': pd.Timestamp.now().isoformat()
                })
        
        return results

# Usage example
mlops_tests = MLOpsIntegrationTests("http://localhost:8000", db_config)

# Run comprehensive test suite
test_results = mlops_tests.run_comprehensive_mlops_test_suite()
for result in test_results:
    print(f"{result['test']}: {'✅' if result['success'] else '❌'} {result['message']}")
```

## Section 4: CI/CD Pipeline Integration

### Step 1: GitHub Actions workflow for database-integrated ML
```yaml
# .github/workflows/ml-ci-cd.yml
name: ML CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_DB: ai_db
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
        ports:
          - 5432:5432
        options: --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Wait for PostgreSQL
      run: |
        until pg_isready -h localhost -U postgres; do
          echo "Waiting for PostgreSQL..."
          sleep 2
        done
    
    - name: Initialize database
      run: |
        psql -h localhost -U postgres -d ai_db -c "
          CREATE TABLE IF NOT EXISTS training_data (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            age INTEGER,
            engagement_score FLOAT,
            session_count INTEGER,
            label BOOLEAN,
            created_at TIMESTAMP DEFAULT NOW()
          );
          INSERT INTO training_data (user_id, age, engagement_score, session_count, label) 
          VALUES (1, 25, 0.8, 5, true), (2, 30, 0.6, 3, false);
        "
    
    - name: Run unit tests
      run: |
        python -m pytest tests/test_database_integration.py
    
    - name: Run integration tests
      run: |
        python -m pytest tests/test_mlops_integration.py

  build:
    needs: test
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_DB: ai_db
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
        ports:
          - 5432:5432
        options: --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Wait for PostgreSQL
      run: |
        until pg_isready -h localhost -U postgres; do
          echo "Waiting for PostgreSQL..."
          sleep 2
        done
    
    - name: Train model
      run: |
        python scripts/train_model.py --db-host=localhost --db-port=5432 --db-name=ai_db
    
    - name: Log model to MLflow
      run: |
        python scripts/log_model_to_mlflow.py
    
    - name: Create Docker image
      run: |
        docker build -t ${{ github.repository }}:latest .
    
    - name: Push to Docker Hub
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ github.repository }}:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v4
      with:
        manifests: |
          manifests/deployment.yaml
          manifests/service.yaml
        images: |
          ${{ github.repository }}:${{ github.sha }}
        kubectl-version: 'latest'
    
    - name: Verify deployment
      run: |
        kubectl get pods -n ai-ml
        kubectl get services -n ai-ml
```

### Step 2: GitLab CI pipeline for database ML workflows
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  POSTGRES_DB: ai_db
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: password

services:
  postgres:
    image: postgres:14
    variables:
      POSTGRES_DB: $POSTGRES_DB
      POSTGRES_USER: $POSTGRES_USER
      POSTGRES_PASSWORD: $POSTGRES_PASSWORD

test:
  stage: test
  image: python:3.9
  before_script:
    - pip install -r requirements.txt
    - apt-get update && apt-get install -y postgresql-client
  script:
    - psql -h postgres -U $POSTGRES_USER -d $POSTGRES_DB -c "CREATE TABLE IF NOT EXISTS training_data (id SERIAL PRIMARY KEY, user_id INTEGER, label BOOLEAN);"
    - psql -h postgres -U $POSTGRES_USER -d $POSTGRES_DB -c "INSERT INTO training_data (user_id, label) VALUES (1, true), (2, false);"
    - python -m pytest tests/

build:
  stage: build
  image: python:3.9
  services:
    - postgres:14
  script:
    - python scripts/train_model.py --db-host=postgres --db-name=$POSTGRES_DB
    - python scripts/log_model.py
    - docker build -t registry.gitlab.com/$CI_PROJECT_PATH:$CI_COMMIT_SHA .
  after_script:
    - docker login -u gitlab-ci-token -p $CI_JOB_TOKEN registry.gitlab.com
    - docker push registry.gitlab.com/$CI_PROJECT_PATH:$CI_COMMIT_SHA

deploy:
  stage: deploy
  image: alpine:latest
  services:
    - postgres:14
  before_script:
    - apk add --no-cache curl
  script:
    - |
      curl --request POST \
        --url "https://api.gitlab.com/v4/projects/$CI_PROJECT_ID/deploy_tokens" \
        --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
        --data "name=deploy-token&scopes=api"
    - kubectl apply -f k8s/deployment.yaml
    - kubectl rollout status deployment/ai-model -n ai-ml
```

## Section 5: Production Monitoring

### Step 1: Database-backed observability
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import psycopg2
import time
from datetime import datetime

# Prometheus metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total number of predictions', ['model_name', 'environment'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds', ['model_name'])
MODEL_VERSION = Gauge('model_version', 'Current model version', ['model_name', 'version'])
DATABASE_CONNECTIONS = Gauge('database_connections', 'Database connections', ['database'])

class DatabaseObservability:
    def __init__(self, db_config: Dict, model_name: str = "user_engagement"):
        self.db_config = db_config
        self.model_name = model_name
        self.last_model_version = None
    
    def update_model_version_metric(self, version: str):
        """Update model version metric"""
        if self.last_model_version != version:
            MODEL_VERSION.labels(model_name=self.model_name, version=version).set(1)
            if self.last_model_version:
                MODEL_VERSION.labels(model_name=self.model_name, version=self.last_model_version).set(0)
            self.last_model_version = version
    
    def record_prediction(self, latency: float, environment: str = "production"):
        """Record prediction metrics"""
        PREDICTION_COUNT.labels(model_name=self.model_name, environment=environment).inc()
        PREDICTION_LATENCY.labels(model_name=self.model_name).observe(latency)
    
    def monitor_database_health(self):
        """Monitor database health metrics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get connection count
            cursor.execute("SELECT COUNT(*) FROM pg_stat_activity;")
            connections = cursor.fetchone()[0]
            
            DATABASE_CONNECTIONS.labels(database=self.db_config['database']).set(connections)
            
            # Get database size
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database()));
            """)
            db_size = cursor.fetchone()[0]
            
            # Get query latency
            cursor.execute("""
                SELECT 
                    avg(total_exec_time) / 1000 as avg_latency_ms
                FROM pg_stat_statements 
                WHERE query LIKE '%training_data%' 
                ORDER BY total_exec_time DESC 
                LIMIT 1;
            """)
            avg_latency = cursor.fetchone()
            if avg_latency:
                # In practice, expose this as a metric
                pass
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Database monitoring error: {e}")
    
    def log_prediction_to_database(self, prediction_data: Dict):
        """Log prediction to database for auditing"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO predictions_log (
            model_name, version, user_id, prediction, confidence,
            features_used, created_at, environment
        ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
        """
        
        cursor.execute(insert_query, (
            prediction_data['model_name'],
            prediction_data['version'],
            prediction_data['user_id'],
            prediction_data['prediction'],
            prediction_data['confidence'],
            json.dumps(prediction_data.get('features_used', {})),
            prediction_data.get('environment', 'production')
        ))
        
        conn.commit()
        cursor.close()
        conn.close()

# Usage example
observability = DatabaseObservability(db_config)

# Update model version
observability.update_model_version_metric("v1.2.3")

# Record prediction
observability.record_prediction(latency=0.12, environment="production")

# Monitor database health
observability.monitor_database_health()

# Log prediction
observability.log_prediction_to_database({
    'model_name': 'user_engagement_predictor',
    'version': 'v1.2.3',
    'user_id': 123,
    'prediction': 0.85,
    'confidence': 0.92,
    'features_used': {'age': 25, 'engagement_score': 0.8},
    'environment': 'production'
})
```

### Step 2: Alerting and anomaly detection
```python
class MLOpsAlerting:
    def __init__(self, db_config: Dict, alert_thresholds: Dict = None):
        self.db_config = db_config
        self.alert_thresholds = alert_thresholds or {
            'prediction_latency': 1.0,  # seconds
            'error_rate': 0.05,  # 5%
            'database_latency': 0.5,  # seconds
            'model_drift': 0.1  # 10% drift
        }
        self.error_count = 0
        self.total_predictions = 0
    
    def check_prediction_anomalies(self, latency: float, prediction: float):
        """Check for prediction anomalies"""
        alerts = []
        
        # High latency
        if latency > self.alert_thresholds['prediction_latency']:
            alerts.append({
                'severity': 'warning',
                'message': f'High prediction latency: {latency:.3f}s',
                'metric': 'prediction_latency',
                'value': latency
            })
        
        # Extreme predictions
        if prediction < 0.01 or prediction > 0.99:
            alerts.append({
                'severity': 'info',
                'message': f'Extreme prediction value: {prediction:.3f}',
                'metric': 'prediction_value',
                'value': prediction
            })
        
        return alerts
    
    def check_database_anomalies(self):
        """Check for database anomalies"""
        alerts = []
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check for slow queries
            cursor.execute("""
                SELECT 
                    query,
                    total_exec_time / calls as avg_exec_time
                FROM pg_stat_statements 
                WHERE calls > 100
                ORDER BY avg_exec_time DESC
                LIMIT 5;
            """)
            
            slow_queries = cursor.fetchall()
            for query, avg_time in slow_queries:
                if avg_time > self.alert_thresholds['database_latency'] * 1000:  # convert to ms
                    alerts.append({
                        'severity': 'warning',
                        'message': f'Slow query: {query[:50]}... ({avg_time:.2f}ms)',
                        'metric': 'database_latency',
                        'value': avg_time
                    })
            
            # Check for high connection count
            cursor.execute("SELECT COUNT(*) FROM pg_stat_activity;")
            connections = cursor.fetchone()[0]
            if connections > 100:
                alerts.append({
                    'severity': 'warning',
                    'message': f'High database connections: {connections}',
                    'metric': 'database_connections',
                    'value': connections
                })
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            alerts.append({
                'severity': 'critical',
                'message': f'Database monitoring failed: {e}',
                'metric': 'database_health',
                'value': 'error'
            })
        
        return alerts
    
    def log_alert(self, alert: Dict):
        """Log alert to database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO alerts (
            severity, message, metric, value, created_at, resolved
        ) VALUES (%s, %s, %s, %s, NOW(), FALSE)
        """
        
        cursor.execute(insert_query, (
            alert['severity'],
            alert['message'],
            alert['metric'],
            str(alert['value']),
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle"""
        alerts = []
        
        # Check prediction anomalies (simulated)
        alerts.extend(self.check_prediction_anomalies(1.2, 0.95))
        
        # Check database anomalies
        alerts.extend(self.check_database_anomalies())
        
        # Log alerts
        for alert in alerts:
            self.log_alert(alert)
            print(f"[{alert['severity'].upper()}] {alert['message']}")
        
        return alerts

# Usage example
alerting = MLOpsAlerting(db_config)

# Run monitoring cycle
alerts = alerting.run_monitoring_cycle()
print(f"Found {len(alerts)} alerts")
```

## Section 6: Performance Benchmarking

### Step 1: MLOps pipeline benchmarking
```python
import time
import pandas as pd
from typing import List, Dict, Callable

class MLOpsBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_pipeline_stage(self, methods: List[Callable], 
                               stage_names: List[str] = ["data_extraction", "model_training", "deployment"]):
        """Benchmark MLOps pipeline stages"""
        for method in methods:
            for stage in stage_names:
                start_time = time.time()
                
                try:
                    method(stage)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'pipeline_stage',
                        'method': method.__name__,
                        'stage': stage,
                        'duration_seconds': duration,
                        'throughput_operations_per_second': 1.0 / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'pipeline_stage',
                        'method': method.__name__,
                        'stage': stage,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_database_integration(self, methods: List[Callable],
                                     data_sizes: List[int] = [1000, 10000, 100000]):
        """Benchmark database integration performance"""
        for method in methods:
            for size in data_sizes:
                start_time = time.time()
                
                try:
                    method(size)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'database_integration',
                        'method': method.__name__,
                        'data_size': size,
                        'duration_seconds': duration,
                        'throughput_rows_per_second': size / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'database_integration',
                        'method': method.__name__,
                        'data_size': size,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_mlops_benchmark_report(self):
        """Generate comprehensive MLOps benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method', 'stage']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'throughput_rows_per_second': ['mean', 'std'],
            'throughput_operations_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best pipeline stage
        if 'pipeline_stage' in df['benchmark'].values:
            best_stage = df[df['benchmark'] == 'pipeline_stage'].loc[
                df[df['benchmark'] == 'pipeline_stage']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best pipeline stage: {best_stage['method']} ({best_stage['stage']}) "
                f"({best_stage['duration_seconds']:.2f}s)"
            )
        
        # Best database integration
        if 'database_integration' in df['benchmark'].values:
            best_db = df[df['benchmark'] == 'database_integration'].loc[
                df[df['benchmark'] == 'database_integration']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best database integration: {best_db['method']} "
                f"({best_db['duration_seconds']:.2f}s for {best_db['data_size']} rows)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'mlops_tips': [
                "Use DVC for data versioning and reproducibility",
                "Implement comprehensive testing before deployment",
                "Monitor database health alongside model performance",
                "Use CI/CD to automate MLOps workflows",
                "Track model lineage and data provenance",
                "Set up alerting for production anomalies",
                "Optimize database queries for ML workloads",
                "Version both models and database schemas"
            ]
        }

# Usage example
benchmark = MLOpsBenchmark()

# Define test methods
def test_data_extraction(stage: str):
    """Test data extraction"""
    time.sleep(0.5)

def test_model_training(stage: str):
    """Test model training"""
    time.sleep(2.0)

def test_deployment(stage: str):
    """Test deployment"""
    time.sleep(1.0)

def test_db_integration(size: int):
    """Test database integration"""
    time.sleep(0.1 * (size / 1000))

# Run benchmarks
benchmark.benchmark_pipeline_stage(
    [test_data_extraction, test_model_training, test_deployment],
    ["data_extraction", "model_training", "deployment"]
)

benchmark.benchmark_database_integration(
    [test_db_integration],
    [1000, 10000, 100000]
)

report = benchmark.generate_mlops_benchmark_report()
print("MLOps Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Database versioning
1. Set up Alembic for schema versioning
2. Implement DVC for data versioning
3. Create data lineage tracking
4. Test version rollback scenarios

### Exercise 2: Model registry integration
1. Set up MLflow with database metadata
2. Implement custom model registry
3. Test model versioning and promotion
4. Integrate with CI/CD pipeline

### Exercise 3: Automated testing
1. Write database-integrated unit tests
2. Implement integration tests for MLOps pipelines
3. Set up test coverage reporting
4. Test failure scenarios and recovery

### Exercise 4: CI/CD pipeline integration
1. Create GitHub Actions workflow
2. Implement GitLab CI pipeline
3. Test automated deployment
4. Set up canary releases and rollbacks

### Exercise 5: Production monitoring
1. Set up Prometheus metrics
2. Implement database health monitoring
3. Create alerting system
4. Build dashboard for MLOps observability

## Best Practices Summary

1. **Version Control**: Version both database schemas and data alongside models
2. **Testing**: Implement comprehensive testing at all stages of the MLOps pipeline
3. **Automation**: Automate CI/CD pipelines to reduce human error
4. **Monitoring**: Monitor both model performance and database health
5. **Alerting**: Set up proactive alerting for production issues
6. **Security**: Secure database connections in MLOps pipelines
7. **Reproducibility**: Ensure end-to-end reproducibility with versioned artifacts
8. **Cost Optimization**: Monitor and optimize MLOps infrastructure costs

This tutorial provides practical, hands-on experience with database integration for MLOps and CI/CD pipelines. Complete all exercises to master these critical skills for building robust, production-grade ML systems.
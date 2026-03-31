# Database Integration with Data Science Tools Tutorial

## Overview

This tutorial focuses on integrating databases with core data science tools: Pandas, Scikit-learn, and Jupyter notebooks. We'll cover efficient data analysis, feature engineering, model development, and production deployment patterns specifically for AI/ML engineers.

## Prerequisites
- Python 3.8+
- Pandas 1.5+
- Scikit-learn 1.2+
- Jupyter Notebook/Lab
- SQLAlchemy or psycopg2/MySQLdb
- Basic SQL knowledge

## Tutorial Structure
1. **Efficient Data Loading** - Optimized database-to-Pandas workflows
2. **Advanced Feature Engineering** - Database-powered feature creation
3. **Scikit-learn Integration** - Database-backed ML pipelines
4. **Jupyter Optimization** - Interactive database analysis
5. **Production Deployment** - From notebook to production
6. **Performance Benchmarking** - Measuring integration efficiency

## Section 1: Efficient Data Loading

### Step 1: Optimized pandas data loading
```python
import pandas as pd
import sqlalchemy
import psycopg2
from typing import Union, Dict, List
import time

class OptimizedDatabaseLoader:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
        self.connection_string = connection_string
    
    def load_large_table(self, table_name: str, columns: List[str] = None, 
                        chunk_size: int = 10000, use_polars: bool = False):
        """Load large tables efficiently"""
        if use_polars:
            import polars as pl
            # Polars is faster for large datasets
            query = f"SELECT {', '.join(columns) if columns else '*'} FROM {table_name}"
            return pl.read_database(query, self.connection_string)
        
        # Standard pandas approach
        if columns:
            query = f"SELECT {', '.join(columns)} FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"
        
        # Use chunked reading for large tables
        chunks = []
        offset = 0
        
        while True:
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            
            start_time = time.time()
            chunk = pd.read_sql(chunk_query, self.engine)
            load_time = time.time() - start_time
            
            if len(chunk) == 0:
                break
                
            chunks.append(chunk)
            offset += chunk_size
            
            print(f"Loaded chunk {len(chunks)}: {len(chunk)} rows in {load_time:.2f}s")
        
        return pd.concat(chunks, ignore_index=True)
    
    def load_with_query_optimization(self, query: str, index_col: str = None):
        """Load data with query optimization techniques"""
        # Add EXPLAIN ANALYZE for performance insights
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS) {query}"
        
        # Execute explain first to check performance
        try:
            explain_result = pd.read_sql(explain_query, self.engine)
            print("Query execution plan:")
            print(explain_result.head())
        except Exception as e:
            print(f"Explain failed: {e}")
        
        # Load actual data
        return pd.read_sql(query, self.engine, index_col=index_col)

# Usage example
loader = OptimizedDatabaseLoader("postgresql://user:password@localhost:5432/ai_db")

# Load large training dataset
df = loader.load_large_table(
    "training_data", 
    columns=["user_id", "age", "engagement_score", "label"],
    chunk_size=50000
)

print(f"Loaded {len(df)} rows")
```

### Step 2: Memory-efficient loading for very large datasets
```python
def memory_efficient_loading(connection_string: str, query: str, 
                           dtype_mapping: Dict[str, str] = None):
    """Load data with memory optimization"""
    import pandas as pd
    import numpy as np
    
    # Define optimal dtypes to reduce memory usage
    if dtype_mapping is None:
        dtype_mapping = {
            'user_id': 'int32',
            'age': 'int8',
            'engagement_score': 'float32',
            'label': 'bool',
            'session_count': 'int16'
        }
    
    # Load in chunks with dtype optimization
    chunks = []
    chunk_size = 100000
    offset = 0
    
    engine = sqlalchemy.create_engine(connection_string)
    
    while True:
        chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
        
        try:
            chunk = pd.read_sql(chunk_query, engine, dtype=dtype_mapping)
            
            if len(chunk) == 0:
                break
                
            chunks.append(chunk)
            offset += chunk_size
            
        except Exception as e:
            print(f"Error loading chunk at offset {offset}: {e}")
            break
    
    if chunks:
        # Concatenate with memory optimization
        result = pd.concat(chunks, ignore_index=True)
        
        # Further memory optimization
        for col in result.columns:
            if result[col].dtype == 'object':
                # Try to convert object columns to categorical
                if result[col].nunique() / len(result) < 0.5:
                    result[col] = result[col].astype('category')
        
        return result
    else:
        return pd.DataFrame()

# Usage
optimized_df = memory_efficient_loading(
    "postgresql://user:password@localhost:5432/ai_db",
    "SELECT user_id, age, engagement_score, session_count, label FROM training_data",
    dtype_mapping={
        'user_id': 'int32',
        'age': 'int8',
        'engagement_score': 'float32',
        'session_count': 'int16',
        'label': 'bool'
    }
)

print(f"Memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

## Section 2: Advanced Feature Engineering

### Step 1: Database-powered feature engineering
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sqlparse

class DatabaseFeatureEngineer:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
        self.feature_registry = {}
    
    def create_feature_view_sql(self, feature_name: str, base_table: str, 
                              feature_columns: List[str], aggregations: List[str]):
        """Generate SQL for feature views"""
        # Build SELECT clause
        select_parts = []
        for col in feature_columns:
            if col in aggregations:
                select_parts.append(f"AVG({col}) as avg_{col}")
            else:
                select_parts.append(col)
        
        # Build GROUP BY clause
        group_by = "user_id"  # Assuming user-level features
        
        sql = f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS feature_{feature_name}
        AS
        SELECT 
            {', '.join(select_parts)},
            COUNT(*) as record_count
        FROM {base_table}
        GROUP BY {group_by};
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_feature_{feature_name}_user_id 
        ON feature_{feature_name} ({group_by});
        """
        
        return sql
    
    def compute_window_features(self, table_name: str, partition_by: str, 
                              order_by: str, window_functions: List[str]):
        """Compute window functions for time-series features"""
        window_clauses = []
        for func in window_functions:
            window_clauses.append(f"{func} OVER (PARTITION BY {partition_by} ORDER BY {order_by}) as {func.replace(' ', '_')}")
        
        sql = f"""
        SELECT 
            *,
            {', '.join(window_clauses)}
        FROM {table_name}
        """
        
        return sql
    
    def generate_feature_matrix(self, feature_views: List[str], 
                              target_column: str = "label"):
        """Generate feature matrix from multiple feature views"""
        # Join feature views
        join_conditions = []
        for i, view in enumerate(feature_views):
            if i == 0:
                join_conditions.append(f"SELECT * FROM feature_{view}")
            else:
                join_conditions.append(f"JOIN feature_{view} USING (user_id)")
        
        full_query = " ".join(join_conditions)
        
        # Add target column
        if target_column:
            full_query = f"SELECT *, (SELECT {target_column} FROM training_data t2 WHERE t2.user_id = t1.user_id) as target FROM ({full_query}) t1"
        
        return pd.read_sql(full_query, self.engine)

# Usage example
feature_engineer = DatabaseFeatureEngineer("postgresql://user:password@localhost:5432/ai_db")

# Create engagement features
engagement_sql = feature_engineer.create_feature_view_sql(
    "user_engagement",
    "user_events",
    ["clicks", "time_spent", "session_duration"],
    ["clicks", "time_spent"]
)

print("Engagement feature SQL:")
print(engagement_sql)

# Compute window features
window_sql = feature_engineer.compute_window_features(
    "user_events",
    "user_id",
    "event_timestamp",
    ["AVG(clicks) as rolling_avg_clicks", "COUNT(*) as session_count_7d"]
)

print("\nWindow features SQL:")
print(window_sql)
```

### Step 2: Time-series feature engineering
```python
class TimeSeriesFeatureEngineer:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
    
    def create_time_series_features(self, table_name: str, timestamp_col: str,
                                  user_id_col: str, value_cols: List[str],
                                  windows: List[int] = [7, 30, 90]):
        """Create time-series features with different window sizes"""
        features = []
        
        for window in windows:
            for col in value_cols:
                # Rolling average
                features.append(f"""
                    AVG({col}) OVER (
                        PARTITION BY {user_id_col} 
                        ORDER BY {timestamp_col}
                        RANGE BETWEEN INTERVAL '{window} days' PRECEDING AND CURRENT ROW
                    ) as {col}_rolling_{window}d
                """)
                
                # Rolling sum
                features.append(f"""
                    SUM({col}) OVER (
                        PARTITION BY {user_id_col} 
                        ORDER BY {timestamp_col}
                        RANGE BETWEEN INTERVAL '{window} days' PRECEDING AND CURRENT ROW
                    ) as {col}_sum_{window}d
                """)
                
                # Rolling count
                features.append(f"""
                    COUNT({col}) OVER (
                        PARTITION BY {user_id_col} 
                        ORDER BY {timestamp_col}
                        RANGE BETWEEN INTERVAL '{window} days' PRECEDING AND CURRENT ROW
                    ) as {col}_count_{window}d
                """)
        
        # Build final query
        select_clause = ",\n    ".join(features)
        sql = f"""
        SELECT 
            {user_id_col},
            {timestamp_col},
            {select_clause}
        FROM {table_name}
        ORDER BY {user_id_col}, {timestamp_col}
        """
        
        return sql

# Usage
ts_engineer = TimeSeriesFeatureEngineer("postgresql://user:password@localhost:5432/ai_db")

time_series_sql = ts_engineer.create_time_series_features(
    "user_events",
    "event_timestamp",
    "user_id",
    ["clicks", "time_spent", "conversion_rate"],
    windows=[7, 30]
)

print("Time-series features SQL:")
print(time_series_sql)
```

## Section 3: Scikit-learn Integration

### Step 1: Database-backed ML pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DatabaseMLPipeline:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
        self.pipeline = None
    
    def create_pipeline_from_database(self, query: str, 
                                    feature_cols: List[str], 
                                    target_col: str):
        """Create ML pipeline directly from database query"""
        # Load data
        df = pd.read_sql(query, self.engine)
        
        # Prepare features and target
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            # Store encoder for later use
            setattr(self, f'le_{col}', le)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train pipeline
        self.pipeline.fit(X, y)
        
        return self.pipeline
    
    def predict_from_database(self, query: str, feature_cols: List[str]):
        """Make predictions using database query"""
        if not self.pipeline:
            raise ValueError("Pipeline not trained yet")
        
        # Load prediction data
        df = pd.read_sql(query, self.engine)
        
        # Prepare features
        X = df[feature_cols]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if hasattr(self, f'le_{col}'):
                le = getattr(self, f'le_{col}')
                X[col] = le.transform(X[col].astype(str))
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'features_used': feature_cols
        }
    
    def save_pipeline_to_database(self, pipeline_name: str, description: str = ""):
        """Save pipeline to database for versioning"""
        import pickle
        import json
        
        # Serialize pipeline
        pipeline_bytes = pickle.dumps(self.pipeline)
        
        # Get metadata
        metadata = {
            'name': pipeline_name,
            'description': description,
            'created_at': pd.Timestamp.now().isoformat(),
            'features': getattr(self, 'feature_cols', []),
            'model_type': type(self.pipeline.named_steps['classifier']).__name__
        }
        
        # Insert into database
        insert_query = """
        INSERT INTO ml_pipelines (name, description, pipeline_data, metadata, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        
        with self.engine.connect() as conn:
            conn.execute(insert_query, [
                pipeline_name,
                description,
                pipeline_bytes,
                json.dumps(metadata)
            ])

# Usage example
pipeline_manager = DatabaseMLPipeline("postgresql://user:password@localhost:5432/ai_db")

# Create pipeline
train_query = """
SELECT user_id, age, avg_clicks, avg_time_spent, session_count, engagement_score, label
FROM training_data WHERE split = 'train'
"""

feature_cols = ['age', 'avg_clicks', 'avg_time_spent', 'session_count', 'engagement_score']
target_col = 'label'

pipeline = pipeline_manager.create_pipeline_from_database(
    train_query, feature_cols, target_col
)

print("Pipeline created successfully")

# Save pipeline
pipeline_manager.save_pipeline_to_database(
    "user_engagement_classifier",
    "Random Forest classifier for user engagement prediction"
)
```

### Step 2: Cross-validation with database sampling
```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

class DatabaseCrossValidator:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
    
    def stratified_kfold_cv_from_database(self, query: str, 
                                        feature_cols: List[str],
                                        target_col: str,
                                        n_splits: int = 5,
                                        scoring: str = 'accuracy'):
        """Perform stratified k-fold cross-validation using database sampling"""
        # Load data
        df = pd.read_sql(query, self.engine)
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'cv_results': scores
        }
    
    def database_sampling_validation(self, table_name: str, 
                                   sample_sizes: List[int] = [1000, 10000, 100000]):
        """Validate model performance with different sample sizes"""
        results = []
        
        for size in sample_sizes:
            # Sample from database
            query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {size}"
            df = pd.read_sql(query, self.engine)
            
            # Train and evaluate
            X = df.drop(columns=['label'])
            y = df['label']
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            results.append({
                'sample_size': size,
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'training_time': 0  # Would measure in real implementation
            })
        
        return results

# Usage
cv_validator = DatabaseCrossValidator("postgresql://user:password@localhost:5432/ai_db")

# Perform cross-validation
cv_results = cv_validator.stratified_kfold_cv_from_database(
    "SELECT * FROM training_data WHERE split = 'train'",
    ['age', 'engagement_score', 'session_count'],
    'label',
    n_splits=5
)

print(f"Cross-validation results: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
```

## Section 4: Jupyter Optimization

### Step 1: Interactive database analysis in Jupyter
```python
# In Jupyter notebook
%load_ext sql
%config SqlMagic.autocommit=False

# Connect to database
%sql postgresql://user:password@localhost:5432/ai_db

# Magic commands for database interaction
%%sql
-- Get dataset statistics
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT user_id) as unique_users,
    MIN(event_timestamp) as min_date,
    MAX(event_timestamp) as max_date
FROM user_events;

%%sql
-- Quick feature distribution
SELECT 
    engagement_score,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM training_data
GROUP BY engagement_score
ORDER BY engagement_score;
```

### Step 2: Custom Jupyter magic for database operations
```python
from IPython.core.magic import register_line_magic, register_cell_magic
import pandas as pd
import sqlalchemy

@register_line_magic
def dbinfo(line):
    """Display database information"""
    engine = sqlalchemy.create_engine(line)
    with engine.connect() as conn:
        result = conn.execute("SELECT version();")
        version = result.fetchone()[0]
        print(f"Database version: {version}")
        
        # Get table info
        result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        tables = [row[0] for row in result]
        print(f"Tables: {tables}")

@register_cell_magic
def dbquery(line, cell):
    """Execute database query and return DataFrame"""
    engine = sqlalchemy.create_engine(line)
    query = cell.strip()
    
    # Add timing
    start_time = time.time()
    df = pd.read_sql(query, engine)
    end_time = time.time()
    
    print(f"Query executed in {end_time - start_time:.2f}s")
    print(f"Returned {len(df)} rows")
    
    return df

# Usage in Jupyter:
# %dbinfo postgresql://user:password@localhost:5432/ai_db
# 
# %%dbquery postgresql://user:password@localhost:5432/ai_db
# SELECT user_id, age, engagement_score 
# FROM training_data 
# WHERE split = 'train' 
# LIMIT 1000
```

## Section 5: Production Deployment

### Step 1: From notebook to production pipeline
```python
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ProductionDeploymentManager:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
    
    def export_notebook_model(self, pipeline: Pipeline, model_name: str):
        """Export model from notebook to production format"""
        # Save model
        joblib.dump(pipeline, f"{model_name}.pkl")
        
        # Save feature metadata
        feature_metadata = {
            'feature_names': list(pipeline.named_steps['scaler'].feature_names_in_),
            'model_type': type(pipeline.named_steps['classifier']).__name__,
            'created_at': pd.Timestamp.now().isoformat(),
            'version': '1.0'
        }
        
        with open(f"{model_name}_metadata.json", 'w') as f:
            json.dump(feature_metadata, f)
        
        return f"Model {model_name} exported successfully"
    
    def deploy_to_api(self, model_path: str, api_endpoint: str):
        """Deploy model to REST API"""
        # Load model
        pipeline = joblib.load(model_path)
        
        # Create FastAPI app
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="Database-Integrated ML API")
        
        class PredictionRequest(BaseModel):
            features: dict
        
        @app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                # Convert features to DataFrame
                features_df = pd.DataFrame([request.features])
                
                # Make prediction
                prediction = pipeline.predict(features_df)
                probability = pipeline.predict_proba(features_df)
                
                return {
                    "prediction": prediction[0],
                    "probability": probability[0].tolist(),
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Run server
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    def monitor_production_model(self, model_name: str, monitoring_interval: int = 300):
        """Monitor production model performance"""
        import time
        
        while True:
            try:
                # Check database for new data
                query = f"SELECT COUNT(*) FROM predictions_log WHERE model_name = '{model_name}'"
                result = pd.read_sql(query, self.engine)
                prediction_count = result.iloc[0, 0]
                
                # Check model drift
                drift_query = f"""
                SELECT 
                    AVG(engagement_score) as current_avg,
                    (SELECT AVG(engagement_score) FROM training_data) as baseline_avg
                FROM predictions_log 
                WHERE model_name = '{model_name}' 
                AND created_at > NOW() - INTERVAL '1 day'
                """
                
                drift_result = pd.read_sql(drift_query, self.engine)
                if not drift_result.empty:
                    current_avg = drift_result['current_avg'].iloc[0]
                    baseline_avg = drift_result['baseline_avg'].iloc[0]
                    drift_ratio = abs(current_avg - baseline_avg) / baseline_avg
                    
                    print(f"Model drift: {drift_ratio:.2%}")
                    
                    if drift_ratio > 0.1:  # 10% drift threshold
                        print("⚠️  Significant model drift detected!")
                
                print(f"Predictions: {prediction_count}")
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(monitoring_interval)

# Usage example
deployment_manager = ProductionDeploymentManager("postgresql://user:password@localhost:5432/ai_db")

# Export model from notebook
# pipeline = ... # your trained pipeline
# deployment_manager.export_notebook_model(pipeline, "user_engagement_model")

# Deploy to API
# deployment_manager.deploy_to_api("user_engagement_model.pkl", "http://localhost:8000")
```

## Section 6: Performance Benchmarking

### Step 1: Comprehensive benchmarking framework
```python
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Callable

class DataScienceBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_data_loading_methods(self, methods: List[Callable], 
                                     dataset_sizes: List[int]):
        """Benchmark different data loading methods"""
        for method in methods:
            for size in dataset_sizes:
                start_time = time.time()
                
                try:
                    method(size)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'data_loading',
                        'method': method.__name__,
                        'dataset_size': size,
                        'duration_seconds': duration,
                        'throughput_rows_per_second': size / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'data_loading',
                        'method': method.__name__,
                        'dataset_size': size,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_feature_engineering(self, methods: List[Callable], 
                                    feature_counts: List[int]):
        """Benchmark feature engineering methods"""
        for method in methods:
            for count in feature_counts:
                start_time = time.time()
                
                try:
                    method(count)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'feature_engineering',
                        'method': method.__name__,
                        'feature_count': count,
                        'duration_seconds': duration,
                        'throughput_features_per_second': count / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'feature_engineering',
                        'method': method.__name__,
                        'feature_count': count,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_comprehensive_report(self):
        """Generate comprehensive benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'throughput_rows_per_second': ['mean', 'std'],
            'throughput_features_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best data loading method
        if 'data_loading' in df['benchmark'].values:
            best_loading = df[df['benchmark'] == 'data_loading'].loc[
                df[df['benchmark'] == 'data_loading']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best data loading: {best_loading['method']} "
                f"({best_loading['duration_seconds']:.2f}s for {best_loading['dataset_size']} rows)"
            )
        
        # Best feature engineering
        if 'feature_engineering' in df['benchmark'].values:
            best_feature = df[df['benchmark'] == 'feature_engineering'].loc[
                df[df['benchmark'] == 'feature_engineering']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best feature engineering: {best_feature['method']} "
                f"({best_feature['duration_seconds']:.2f}s for {best_feature['feature_count']} features)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'performance_tips': [
                "Use chunked loading for datasets > 1M rows",
                "Precompute complex features in database",
                "Use Polars for very large datasets (> 10M rows)",
                "Cache frequently used feature views",
                "Optimize database indexes for ML queries"
            ]
        }

# Usage example
benchmark = DataScienceBenchmark()

# Define test methods
def test_pandas_load(size: int):
    """Test pandas loading"""
    time.sleep(0.1 * (size / 10000))

def test_sqlalchemy_load(size: int):
    """Test SQLAlchemy loading"""
    time.sleep(0.05 * (size / 10000))

def test_polars_load(size: int):
    """Test Polars loading"""
    time.sleep(0.03 * (size / 10000))

# Run benchmarks
benchmark.benchmark_data_loading_methods(
    [test_pandas_load, test_sqlalchemy_load, test_polars_load],
    [10000, 100000, 1000000]
)

report = benchmark.generate_comprehensive_report()
print("Data Science Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Optimize data loading
1. Implement the `OptimizedDatabaseLoader` class
2. Test with different dataset sizes (10K, 100K, 1M rows)
3. Compare pandas vs. Polars performance
4. Measure memory usage improvements

### Exercise 2: Build feature engineering pipeline
1. Create materialized views for key features
2. Implement time-series feature engineering
3. Test with real user event data
4. Compare performance of different approaches

### Exercise 3: Scikit-learn integration
1. Create database-backed ML pipeline
2. Implement cross-validation with database sampling
3. Save pipeline to database for versioning
4. Test end-to-end workflow

### Exercise 4: Jupyter optimization
1. Set up Jupyter with database magic commands
2. Create interactive dashboards with database queries
3. Implement real-time monitoring
4. Export notebook to production pipeline

## Best Practices Summary

1. **Memory Efficiency**: Use dtype optimization and chunked loading
2. **Feature Caching**: Precompute and cache complex features
3. **Database Optimization**: Use materialized views and proper indexing
4. **Version Control**: Version both models and database schemas
5. **Monitoring**: Track model performance and data drift
6. **Security**: Secure database connections in notebooks
7. **Testing**: Test integration with realistic data volumes
8. **Documentation**: Document database-ML integration patterns

This tutorial provides practical, hands-on experience with database integration for data science workflows. Complete all exercises to master these critical skills for AI/ML engineering.
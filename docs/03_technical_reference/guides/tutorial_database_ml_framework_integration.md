# Database Integration with ML Frameworks Tutorial for AI/ML Systems

## Overview

This hands-on tutorial teaches senior AI/ML engineers how to integrate databases with popular ML frameworks (PyTorch, TensorFlow) and data science tools. We'll cover efficient data loading, feature engineering, model training integration, and inference optimization.

## Prerequisites
- Python 3.8+
- PyTorch 2.0+ or TensorFlow 2.12+
- PostgreSQL 14+ or MySQL 8+
- Pandas, NumPy, Scikit-learn
- Basic understanding of ML workflows

## Tutorial Structure
This tutorial is divided into 6 progressive sections:
1. **Data Loading Patterns** - Efficient database-to-ML data pipelines
2. **Feature Engineering Integration** - Database-backed feature engineering
3. **Model Training Integration** - Database-integrated training workflows
4. **Inference Optimization** - Real-time inference with database integration
5. **Cross-Platform Integration** - Integrating with MLflow, Kubeflow, etc.
6. **Performance Benchmarking** - Measuring integration performance

## Section 1: Data Loading Patterns

### Step 1: Efficient data loading from databases
```python
import torch
import pandas as pd
import psycopg2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional

class DatabaseDataset(Dataset):
    """Efficient dataset class that loads data directly from database"""
    
    def __init__(self, db_config: Dict, query: str, batch_size: int = 1000):
        self.db_config = db_config
        self.query = query
        self.batch_size = batch_size
        self._total_rows = None
        
        # Get total row count for length
        self._get_total_rows()
    
    def _get_total_rows(self):
        """Get total number of rows in query result"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Count rows without loading all data
        count_query = f"SELECT COUNT(*) FROM ({self.query}) AS subquery"
        cursor.execute(count_query)
        self._total_rows = cursor.fetchone()[0]
        conn.close()
    
    def __len__(self):
        return self._total_rows
    
    def __getitem__(self, idx):
        """Load single item by index"""
        # Calculate offset for pagination
        offset = idx * self.batch_size
        limit = self.batch_size
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Execute query with pagination
        paginated_query = f"{self.query} LIMIT {limit} OFFSET {offset}"
        cursor.execute(paginated_query)
        
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        if not rows:
            raise IndexError(f"Index {idx} out of range")
        
        # Convert to tensor
        data = np.array(rows)
        return torch.tensor(data, dtype=torch.float32)

# Usage example
db_config = {
    'host': 'localhost',
    'database': 'ai_db',
    'user': 'postgres',
    'password': 'password'
}

query = """
SELECT 
    user_id,
    age,
    engagement_score,
    session_count,
    conversion_rate,
    label
FROM training_data 
WHERE split = 'train'
"""

dataset = DatabaseDataset(db_config, query, batch_size=1000)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### Step 2: Streaming data loading for large datasets
```python
class StreamingDatabaseDataset(Dataset):
    """Streaming dataset for very large datasets"""
    
    def __init__(self, db_config: Dict, query: str, chunk_size: int = 10000):
        self.db_config = db_config
        self.query = query
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.current_data = None
        self.current_index = 0
        
        # Initialize first chunk
        self._load_chunk(0)
    
    def _load_chunk(self, chunk_num: int):
        """Load a chunk of data from database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        offset = chunk_num * self.chunk_size
        limit = self.chunk_size
        
        paginated_query = f"{self.query} LIMIT {limit} OFFSET {offset}"
        cursor.execute(paginated_query)
        
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        if rows:
            self.current_data = np.array(rows)
            self.current_index = 0
        else:
            self.current_data = None
    
    def __len__(self):
        # Estimate length based on typical dataset size
        # In production, you'd calculate this properly
        return 1000000  # Example: 1M rows
    
    def __getitem__(self, idx):
        """Get item with streaming capability"""
        # Calculate which chunk and position within chunk
        chunk_num = idx // self.chunk_size
        pos_in_chunk = idx % self.chunk_size
        
        # Load new chunk if needed
        if chunk_num != self.current_chunk:
            self.current_chunk = chunk_num
            self._load_chunk(chunk_num)
        
        if self.current_data is None or pos_in_chunk >= len(self.current_data):
            raise IndexError(f"Index {idx} out of range")
        
        return torch.tensor(self.current_data[pos_in_chunk], dtype=torch.float32)

# Usage with PyTorch DataLoader
dataset = StreamingDatabaseDataset(db_config, query)
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, persistent_workers=True)
```

## Section 2: Feature Engineering Integration

### Step 1: Database-backed feature engineering
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sqlparse

class DatabaseFeatureEngineer:
    def __init__(self, db_connection, feature_registry=None):
        self.db = db_connection
        self.feature_registry = feature_registry or {}
    
    def create_feature_view(self, feature_name: str, sql_definition: str, description: str = ""):
        """Create a feature view in the database"""
        # Parse SQL to extract dependencies
        parsed = sqlparse.parse(sql_definition)[0]
        
        # Create materialized view
        create_view_sql = f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS feature_{feature_name}
        AS {sql_definition};
        
        -- Create index for performance
        CREATE INDEX IF NOT EXISTS idx_feature_{feature_name}_id 
        ON feature_{feature_name} (id);
        """
        
        cursor = self.db.cursor()
        cursor.execute(create_view_sql)
        self.db.commit()
        cursor.close()
        
        # Register feature
        self.feature_registry[feature_name] = {
            'sql': sql_definition,
            'description': description,
            'created_at': pd.Timestamp.now(),
            'last_refreshed': None
        }
        
        return f"Feature view 'feature_{feature_name}' created"
    
    def refresh_feature_view(self, feature_name: str):
        """Refresh a materialized view"""
        cursor = self.db.cursor()
        cursor.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY feature_{feature_name};")
        self.db.commit()
        cursor.close()
        
        self.feature_registry[feature_name]['last_refreshed'] = pd.Timestamp.now()
        return f"Feature view 'feature_{feature_name}' refreshed"
    
    def get_feature_data(self, feature_name: str, limit: int = 1000):
        """Get feature data for ML training"""
        cursor = self.db.cursor()
        cursor.execute(f"SELECT * FROM feature_{feature_name} LIMIT {limit};")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        
        return pd.DataFrame(rows, columns=columns)

# Usage example
feature_engineer = DatabaseFeatureEngine(db_connection)

# Create engagement score feature
engagement_sql = """
SELECT 
    user_id,
    AVG(clicks) as avg_clicks,
    AVG(time_spent) as avg_time_spent,
    COUNT(*) as session_count,
    (AVG(clicks) * 0.4 + AVG(time_spent) * 0.6) as engagement_score
FROM user_events
GROUP BY user_id
"""

feature_engineer.create_feature_view(
    "user_engagement", 
    engagement_sql,
    "User engagement score based on clicks and time spent"
)

# Refresh and get data
feature_engineer.refresh_feature_view("user_engagement")
feature_df = feature_engineer.get_feature_data("user_engagement", limit=10000)
```

### Step 2: Real-time feature computation
```python
class RealTimeFeatureEngineer:
    def __init__(self, db_connection, redis_client=None):
        self.db = db_connection
        self.redis = redis_client
    
    def compute_real_time_features(self, user_id: int, event_data: dict):
        """Compute real-time features for inference"""
        # Get historical features from database
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT 
                avg_clicks, avg_time_spent, session_count, engagement_score
            FROM feature_user_engagement 
            WHERE user_id = %s
        """, (user_id,))
        
        historical_features = cursor.fetchone()
        cursor.close()
        
        # Compute real-time features from current event
        real_time_features = {
            'current_click': event_data.get('clicks', 0),
            'current_time_spent': event_data.get('time_spent', 0),
            'session_duration': event_data.get('session_duration', 0),
            'event_timestamp': event_data.get('timestamp', pd.Timestamp.now())
        }
        
        # Combine historical and real-time features
        if historical_features:
            combined_features = {
                'historical_avg_clicks': historical_features[0],
                'historical_avg_time_spent': historical_features[1],
                'historical_session_count': historical_features[2],
                'historical_engagement_score': historical_features[3],
                **real_time_features
            }
        else:
            # Default values for new users
            combined_features = {
                'historical_avg_clicks': 0.0,
                'historical_avg_time_spent': 0.0,
                'historical_session_count': 0.0,
                'historical_engagement_score': 0.0,
                **real_time_features
            }
        
        return combined_features
    
    def cache_features(self, user_id: int, features: dict, ttl_seconds: int = 300):
        """Cache computed features in Redis"""
        if self.redis:
            self.redis.setex(f"features:{user_id}", ttl_seconds, json.dumps(features))

# Usage example
real_time_engineer = RealTimeFeatureEngine(db_connection, redis_client)

# Process real-time event
event_data = {
    'clicks': 5,
    'time_spent': 120,
    'session_duration': 300,
    'timestamp': pd.Timestamp.now()
}

features = real_time_engineer.compute_real_time_features(123, event_data)
real_time_engineer.cache_features(123, features)
```

## Section 3: Model Training Integration

### Step 1: Database-integrated training workflow
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

class DatabaseTrainer:
    def __init__(self, db_config: Dict, model: nn.Module, optimizer: optim.Optimizer):
        self.db_config = db_config
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_epoch(self, query: str, batch_size: int = 32, num_workers: int = 4):
        """Train for one epoch using database data"""
        dataset = DatabaseDataset(self.db_config, query, batch_size=1000)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, pin_memory=True)
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move data to device
            batch = batch.to(self.device)
            
            # Forward pass
            inputs = batch[:, :-1]  # All columns except last (label)
            targets = batch[:, -1]  # Last column is label
            
            outputs = self.model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train_with_validation(self, train_query: str, val_query: str, 
                            epochs: int = 10, patience: int = 3):
        """Train with early stopping and validation"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_query)
            
            # Validate
            val_loss = self.validate(val_query)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load("best_model.pth"))
    
    def validate(self, query: str, batch_size: int = 32):
        """Validate model on database data"""
        dataset = DatabaseDataset(self.db_config, query, batch_size=1000)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                inputs = batch[:, :-1]
                targets = batch[:, -1]
                
                outputs = self.model(inputs)
                loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

# Usage example
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleMLP(input_dim=6)  # 6 features
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = DatabaseTrainer(db_config, model, optimizer)

train_query = """
SELECT user_id, age, avg_clicks, avg_time_spent, session_count, engagement_score, label
FROM training_data WHERE split = 'train'
"""

val_query = """
SELECT user_id, age, avg_clicks, avg_time_spent, session_count, engagement_score, label
FROM training_data WHERE split = 'val'
"""

trainer.train_with_validation(train_query, val_query, epochs=20)
```

## Section 4: Inference Optimization

### Step 1: Real-time inference with database integration
```python
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class InferenceRequest(BaseModel):
    user_id: int
    event_data: dict

class DatabaseInferenceService:
    def __init__(self, db_connection, model_path: str, redis_client=None):
        self.db = db_connection
        self.redis = redis_client
        self.model = self._load_model(model_path)
        self.feature_engineer = RealTimeFeatureEngineer(db_connection, redis_client)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        model = SimpleMLP(input_dim=10)  # Adjust based on your model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    async def predict(self, user_id: int, event_data: dict):
        """Make prediction with database integration"""
        # Get cached features first
        if self.redis:
            cached_features = self.redis.get(f"features:{user_id}")
            if cached_features:
                features = json.loads(cached_features)
            else:
                features = self.feature_engineer.compute_real_time_features(user_id, event_data)
                self.feature_engineer.cache_features(user_id, features)
        else:
            features = self.feature_engineer.compute_real_time_features(user_id, event_data)
        
        # Convert to tensor
        feature_values = list(features.values())
        input_tensor = torch.tensor([feature_values], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            probability = torch.sigmoid(prediction).item()
        
        return {
            'user_id': user_id,
            'prediction': probability,
            'features_used': list(features.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }

# Initialize service
inference_service = DatabaseInferenceService(db_connection, "best_model.pth", redis_client)

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        result = await inference_service.predict(request.user_id, request.event_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Section 5: Cross-Platform Integration

### Step 1: MLflow integration
```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os

class MLflowDatabaseIntegration:
    def __init__(self, db_config: Dict, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.db_config = db_config
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()
    
    def log_model_with_database_info(self, model, model_name: str, 
                                   database_info: Dict, metrics: Dict):
        """Log model with database metadata"""
        with mlflow.start_run() as run:
            # Log model
            mlflow.pytorch.log_model(model, model_name)
            
            # Log database information
            mlflow.log_param("database_type", database_info.get("type", "unknown"))
            mlflow.log_param("database_version", database_info.get("version", "unknown"))
            mlflow.log_param("table_name", database_info.get("table_name", "unknown"))
            mlflow.log_param("query_complexity", database_info.get("query_complexity", "medium"))
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature information
            if "features" in database_info:
                mlflow.log_param("num_features", len(database_info["features"]))
                mlflow.log_param("feature_types", str(database_info.get("feature_types", [])))
            
            # Log training data info
            mlflow.log_param("training_data_size", database_info.get("training_data_size", 0))
            mlflow.log_param("data_split", database_info.get("data_split", "unknown"))
    
    def register_model_from_database(self, model_name: str, version: str, 
                                   source_query: str, description: str = ""):
        """Register model from database source"""
        # Get model info from database
        cursor = psycopg2.connect(**self.db_config).cursor()
        cursor.execute("""
            SELECT 
                model_id, model_name, version, created_at, accuracy, 
                training_data_version
            FROM models 
            WHERE model_name = %s AND version = %s
        """, (model_name, version))
        
        model_info = cursor.fetchone()
        cursor.close()
        
        if not model_info:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Create model version
        model_version = self.client.create_model_version(
            name=model_name,
            source=f"models/{model_name}",
            run_id="dummy_run_id",
            description=description
        )
        
        return model_version

# Usage example
mlflow_integration = MLflowDatabaseIntegration(db_config)

# Log model with database info
database_info = {
    "type": "PostgreSQL",
    "version": "14.7",
    "table_name": "training_data",
    "query_complexity": "high",
    "features": ["user_id", "age", "engagement_score", "session_count"],
    "feature_types": ["int", "int", "float", "int"],
    "training_data_size": 1000000,
    "data_split": "80-10-10"
}

metrics = {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90
}

mlflow_integration.log_model_with_database_info(
    model=model,
    model_name="user_engagement_predictor",
    database_info=database_info,
    metrics=metrics
)
```

### Step 2: Kubeflow integration
```python
# kubeflow_integration.py
from kfp import dsl
from kfp.components import func_to_container_op
import os

@func_to_container_op
def database_training_op(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    model_name: str,
    epochs: int = 10
):
    """Kubeflow component for database-integrated training"""
    
    import torch
    import psycopg2
    import pandas as pd
    
    # Connect to database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    # Load data
    query = f"SELECT * FROM training_data WHERE split = 'train' LIMIT 10000"
    df = pd.read_sql(query, conn)
    
    # Train model (simplified)
    print(f"Training {model_name} for {epochs} epochs with {len(df)} samples")
    
    # Save model
    model_path = f"/mnt/output/{model_name}.pth"
    torch.save({"dummy": "model"}, model_path)
    
    # Log metrics
    with open("/mnt/output/metrics.json", "w") as f:
        json.dump({"accuracy": 0.92, "loss": 0.08}, f)
    
    conn.close()
    print("Training completed")

@dsl.pipeline(
    name='Database-Integrated Training Pipeline',
    description='Pipeline for training models with database integration'
)
def database_training_pipeline(
    db_host: str = 'postgres-service',
    db_port: int = 5432,
    db_name: str = 'ai_db',
    db_user: str = 'postgres',
    db_password: str = 'password',
    model_name: str = 'user_engagement_model',
    epochs: int = 10
):
    """Define the pipeline"""
    
    train_op = database_training_op(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        model_name=model_name,
        epochs=epochs
    )
    
    # Add evaluation step
    evaluate_op = database_evaluation_op(
        model_path=train_op.output,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    )

# Usage in Kubeflow
# client = kfp.Client()
# client.create_run_from_pipeline_func(database_training_pipeline, arguments={})
```

## Section 6: Performance Benchmarking

### Step 1: Comprehensive benchmarking framework
```python
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Callable

class DatabaseMLBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_data_loading(self, methods: List[Callable], dataset_sizes: List[int]):
        """Benchmark different data loading methods"""
        for method in methods:
            for size in dataset_sizes:
                start_time = time.time()
                
                try:
                    # Execute method with given dataset size
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
    
    def benchmark_feature_engineering(self, methods: List[Callable], feature_counts: List[int]):
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
                        'throughput_features_per_second': count / duration if duration > 0 else iglia
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'feature_engineering',
                        'method': method.__name__,
                        'feature_count': count,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'throughput_rows_per_second': ['mean', 'std'],
            'throughput_features_per_second': ['mean', 'std']
        }).round(2)
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': self._generate_recommendations(df)
        }
    
    def _generate_recommendations(self, df):
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Data loading recommendations
        data_loading = df[df['benchmark'] == 'data_loading']
        if not data_loading.empty:
            best_method = data_loading.loc[data_loading['duration_seconds'].idxmin()]
            recommendations.append(
                f"Best data loading method: {best_method['method']} "
                f"({best_method['duration_seconds']:.2f}s for {best_method['dataset_size']} rows)"
            )
        
        # Feature engineering recommendations
        feature_eng = df[df['benchmark'] == 'feature_engineering']
        if not feature_eng.empty:
            best_feature_method = feature_eng.loc[feature_eng['duration_seconds'].idxmin()]
            recommendations.append(
                f"Best feature engineering method: {best_feature_method['method']} "
                f"({best_feature_method['duration_seconds']:.2f}s for {best_feature_method['feature_count']} features)"
            )
        
        return recommendations

# Usage example
benchmark = DatabaseMLBenchmark()

# Define test methods
def test_pandas_loading(size: int):
    """Test pandas loading"""
    # Simulate loading from database
    time.sleep(0.1 * (size / 10000))

def test_torch_dataset_loading(size: int):
    """Test torch dataset loading"""
    # Simulate loading with DatabaseDataset
    time.sleep(0.05 * (size / 10000))

def test_streaming_loading(size: int):
    """Test streaming loading"""
    # Simulate streaming loading
    time.sleep(0.03 * (size / 10000))

# Run benchmarks
benchmark.benchmark_data_loading(
    [test_pandas_loading, test_torch_dataset_loading, test_streaming_loading],
    [10000, 100000, 1000000]
)

report = benchmark.generate_report()
print("Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Implement database dataset class
1. Create a `DatabaseDataset` class for your database
2. Implement efficient pagination and streaming
3. Test with different batch sizes
4. Compare performance with traditional pandas loading

### Exercise 2: Build feature engineering pipeline
1. Create materialized views for key features
2. Implement real-time feature computation
3. Cache features in Redis
4. Test with simulated user events

### Exercise 3: Integrate with ML framework
1. Connect your database to PyTorch/TensorFlow training
2. Implement database-integrated training loop
3. Add MLflow logging with database metadata
4. Test end-to-end training and inference

### Exercise 4: Cross-platform integration
1. Set up Kubeflow pipeline with database integration
2. Implement MLflow model registration from database
3. Create monitoring dashboard for database-ML integration
4. Benchmark different integration patterns

## Best Practices Summary

1. **Efficient Data Loading**: Use streaming and pagination for large datasets
2. **Feature Caching**: Cache computed features to reduce database load
3. **Database-Backed Features**: Use materialized views for complex features
4. **Real-time Integration**: Combine historical and real-time features
5. **Monitoring**: Track database-ML integration performance
6. **Versioning**: Version both models and database schemas
7. **Security**: Secure database connections in ML pipelines
8. **Testing**: Test integration thoroughly with realistic data volumes

This tutorial provides practical, hands-on experience with database integration specifically for AI/ML systems. Complete all exercises to master these critical integration skills.
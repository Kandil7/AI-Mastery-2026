# AI-Mastery-2026: API Usage Tutorial

## Introduction

This tutorial demonstrates how to use the production API module to serve machine learning models. The API is built with FastAPI and includes features like input validation, performance monitoring, and health checks.

## Setting Up the API

First, let's look at how to create and run the API:

```python
from src.production.api import create_app
import uvicorn

# Create the API application
app = create_app(title="My ML Model API", version="1.0.0")

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Using the API

### Health Check

The API provides health check endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Metrics endpoint
curl http://localhost:8000/metrics
```

### Making Predictions

The API provides endpoints for making predictions:

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, -0.3, 2.1],
    "model_name": "default"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [[0.5, 1.2], [0.3, 0.8]],
    "model_name": "default"
  }'
```

## Programmatic API Usage

You can also use the API programmatically:

```python
import requests
import json

# Base URL of your API
BASE_URL = "http://localhost:8000"

# Single prediction
def make_single_prediction(features, model_name="default"):
    url = f"{BASE_URL}/predict"
    payload = {
        "features": features,
        "model_name": model_name
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Batch prediction
def make_batch_prediction(instances, model_name="default"):
    url = f"{BASE_URL}/predict/batch"
    payload = {
        "instances": instances,
        "model_name": model_name
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
features = [0.5, 1.2, -0.3, 2.1]
single_result = make_single_prediction(features)
print(f"Single prediction result: {single_result}")

instances = [[0.5, 1.2], [0.3, 0.8]]
batch_result = make_batch_prediction(instances)
print(f"Batch prediction result: {batch_result}")
```

## Adding Your Own Model

To integrate your own model with the API:

```python
from src.production.api import model_cache
from src.ml.classical import RandomForestScratch
import numpy as np

# Train your model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
my_model = RandomForestScratch(n_estimators=50, max_depth=10)
my_model.fit(X, y)

# Load your model into the cache
model_cache.load_model("my_custom_model", my_model)

# Now you can use it with the API by specifying model_name="my_custom_model"
```

## Custom Model Integration

For more complex models, you can create a wrapper:

```python
from src.production.api import create_app
from fastapi import FastAPI
import numpy as np

class ModelWrapper:
    """Wrapper for your custom model"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, features):
        """Make a prediction with your model"""
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        return float(prediction)

# Create your model wrapper
my_model_wrapper = ModelWrapper(my_model)

# You can add custom endpoints if needed
app = create_app()

@app.post("/custom_predict")
async def custom_predict(features: list):
    prediction = my_model_wrapper.predict(features)
    return {"prediction": prediction, "model": "custom_model"}
```

## Monitoring and Metrics

The API automatically collects metrics:

```python
from src.production.api import metrics

# Get current metrics
current_metrics = metrics.get_metrics()
print(f"Total requests: {current_metrics['total_requests']}")
print(f"Error rate: {current_metrics['error_rate']:.3f}")
print(f"Latency P95: {current_metrics['latency_p95_ms']:.2f}ms")
```

## Error Handling

The API includes comprehensive error handling:

```python
from fastapi import HTTPException

# The API will automatically handle validation errors
# For example, if features are empty:
try:
    response = make_single_prediction([])
except HTTPException as e:
    print(f"API Error: {e.detail}")
```

## Complete Example: Serving a Trained Model

Here's a complete example of training a model and serving it with the API:

```python
from src.production.api import create_app, model_cache
from src.ml.classical import RandomForestScratch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import uvicorn

# 1. Train your model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_size=0.2, random_state=42)

model = RandomForestScratch(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)

# 2. Load model into cache
model_cache.load_model("production_model", model)

# 3. Create and run API
app = create_app(title="Production ML API", version="1.0.0")

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

## Testing the API

You can test the API endpoints:

```python
import pytest
from fastapi.testclient import TestClient
from src.production.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post("/predict", json={
        "features": [0.5, 1.2, -0.3, 2.1]
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_batch_prediction():
    response = client.post("/predict/batch", json={
        "instances": [[0.5, 1.2], [0.3, 0.8]]
    })
    assert response.status_code == 200
    assert "predictions" in response.json()
```

## Production Deployment

For production deployment, consider:

1. Using a production ASGI server like Gunicorn:
```bash
gunicorn src.production.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. Setting up a reverse proxy with Nginx
3. Implementing proper logging
4. Setting up monitoring and alerting
5. Using environment variables for configuration

## Conclusion

The AI-Mastery-2026 API module provides a robust foundation for serving machine learning models in production. It includes features like:

- Input validation with Pydantic
- Performance monitoring
- Health checks
- Error handling
- Model caching
- Async request handling
- OpenAPI documentation

This allows you to quickly deploy your models while maintaining production best practices.
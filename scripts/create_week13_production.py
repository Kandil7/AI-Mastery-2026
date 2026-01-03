#!/usr/bin/env python3
"""
Week 13: Production ML & MLOps - High-Value Sprint Phase 3 (Final)
Complete production deployment and monitoring notebooks
"""

import json
from pathlib import Path

BASE_DIR = Path("k:/learning/technical/ai-ml/AI-Mastery-2026/notebooks/week_13")

def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}

def md(c): 
    return {"cell_type": "markdown", "metadata": {}, "source": c if isinstance(c, list) else [c]}

def code(c): 
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c if isinstance(c, list) else [c]}

# MODEL SERVING WITH FASTAPI
fastapi_cells = [
    md(["# 🚀 Model Serving with FastAPI\n\n## Production ML Deployment\n\nTurn your ML model into a production API in minutes.\n\n---\n"]),
    
    code(["print('✅ FastAPI serving guide ready!')\n"]),
    
    md(["## Why FastAPI for ML?\n\n✅ **Fast**: Async/await support\n✅ **Auto docs**: Swagger UI built-in\n✅ **Type hints**: Pydantic validation\n✅ **Production-ready**: Used by Uber, Netflix\n\n## Basic ML API\n\n```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\nimport joblib\n\napp = FastAPI()\nmodel = joblib.load('model.pkl')\n\nclass PredictionInput(BaseModel):\n    features: list[float]\n\n@app.post('/predict')\ndef predict(input: PredictionInput):\n    prediction = model.predict([input.features])\n    return {'prediction': prediction[0]}\n```\n\n## Running\n\n```bash\nuvicorn main:app --reload\n```\n\nAPI now at: http://localhost:8000\nDocs at: http://localhost:8000/docs\n"]),
    
    md(["## Production Best Practices\n\n### 1. Request Validation\n\n```python\nclass Features(BaseModel):\n    age: int = Field(ge=0, le=120)\n    income: float = Field(gt=0)\n    \n    @validator('age')\n    def check_age(cls, v):\n        if v < 18:\n            raise ValueError('Must be 18+')\n        return v\n```\n\n### 2. Error Handling\n\n```python\n@app.exception_handler(Exception)\nasync def global_exception_handler(request, exc):\n    return JSONResponse(\n        status_code=500,\n        content={'error': str(exc)}\n    )\n```\n\n### 3. Health Check\n\n```python\n@app.get('/health')\ndef health():\n    return {'status': 'healthy', 'model_loaded': model is not None}\n```\n\n### 4. Batch Predictions\n\n```python\n@app.post('/predict/batch')\ndef predict_batch(inputs: list[PredictionInput]):\n    features = [inp.features for inp in inputs]\n    predictions = model.predict(features)\n    return {'predictions': predictions.tolist()}\n```\n"]),
    
    md(["## Performance Optimization\n\n### Async Endpoints\n\n```python\n@app.post('/predict')\nasync def predict(input: PredictionInput):\n    # Use asyncio for I/O-bound ops\n    result = await async_model_predict(input)\n    return result\n```\n\n### Caching\n\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=1000)\ndef cached_predict(features_tuple):\n    return model.predict([list(features_tuple)])[0]\n```\n\n### Model Loading\n\n```python\n@app.on_event('startup')\nasync def load_model():\n    global model\n    model = joblib.load('model.pkl')\n```\n"]),
]

# MONITORING & OBSERVABILITY
monitoring_cells = [
    md(["# 📊 Monitoring & Observability\n\n## Don't Deploy and Pray - Monitor!\n\n---\n"]),
    
    code(["print('✅ Monitoring concepts ready!')\n"]),
    
    md(["## The Three Pillars\n\n### 1. Metrics\n**What to monitor**:\n- **Latency**: p50, p95, p99\n- **Throughput**: Requests/second\n- **Error rate**: 5xx errors\n- **Model metrics**: Accuracy, drift\n\n### 2. Logs\n**What to log**:\n- Request/response\n- Errors and exceptions\n- Model predictions\n- Feature values\n\n### 3. Traces\n**Distributed tracing**:\n- Request flow through services\n- Bottleneck identification\n- Dependency mapping\n"]),
    
    md(["## Prometheus + Grafana\n\n### Prometheus Setup\n\n```python\nfrom prometheus_client import Counter, Histogram, make_asgi_app\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n# Metrics\nPREDICTIONS = Counter('predictions_total', 'Total predictions')\nLATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')\n\n@app.post('/predict')\n@LATENCY.time()\ndef predict(input: PredictionInput):\n    PREDICTIONS.inc()\n    return model.predict([input.features])\n\n# Expose metrics\nmetrics_app = make_asgi_app()\napp.mount('/metrics', metrics_app)\n```\n\n### Grafana Dashboard\n\n1. Add Prometheus as data source\n2. Create dashboard\n3. Add panels:\n   - Prediction rate\n   - Latency percentiles\n   - Error rate\n"]),
    
    md(["## Model Monitoring\n\n### Data Drift Detection\n\n**Concept**: Input distribution changes over time\n\n```python\ndef detect_drift(current_data, reference_data):\n    from scipy import stats\n    \n    # Kolmogorov-Smirnov test\n    statistic, p_value = stats.ks_2samp(current_data, reference_data)\n    \n    if p_value < 0.05:\n        return {'drift_detected': True, 'p_value': p_value}\n    return {'drift_detected': False}\n```\n\n### Model Performance Monitoring\n\n```python\n# Log predictions for later analysis\n@app.post('/predict')\ndef predict(input: PredictionInput):\n    pred = model.predict([input.features])[0]\n    \n    # Log to DB/monitoring\n    log_prediction(\n        features=input.features,\n        prediction=pred,\n        timestamp=datetime.now()\n    )\n    \n    return {'prediction': pred}\n```\n"]),
]

# DOCKER DEPLOYMENT
docker_cells = [
    md(["# 🐳 Docker Deployment\n\n## Containerize Your ML Service\n\n---\n"]),
    
    code(["print('✅ Docker deployment ready!')\n"]),
    
    md(["## Dockerfile for ML\n\n```dockerfile\nFROM python:3.10-slim\n\nWORKDIR /app\n\n# Install dependencies\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\n# Copy model and code\nCOPY model.pkl .\nCOPY main.py .\n\n# Expose port\nEXPOSE 8000\n\n# Run\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n```\n\n## Build & Run\n\n```bash\n# Build\ndocker build -t ml-api .\n\n# Run\ndocker run -p 8000:8000 ml-api\n```\n"]),
    
    md(["## Docker Compose\n\n```yaml\nversion: '3.8'\n\nservices:\n  api:\n    build: .\n    ports:\n      - \"8000:8000\"\n    environment:\n      - MODEL_PATH=/app/model.pkl\n    volumes:\n      - ./models:/app/models\n  \n  prometheus:\n    image: prom/prometheus\n    ports:\n      - \"9090:9090\"\n    volumes:\n      - ./prometheus.yml:/etc/prometheus/prometheus.yml\n  \n  grafana:\n    image: grafana/grafana\n    ports:\n      - \"3000:3000\"\n```\n\n## Multi-Stage Build (Optimization)\n\n```dockerfile\n# Build stage\nFROM python:3.10 AS builder\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --user --no-cache-dir -r requirements.txt\n\n# Runtime stage\nFROM python:3.10-slim\nWORKDIR /app\nCOPY --from=builder /root/.local /root/.local\nCOPY . .\nENV PATH=/root/.local/bin:$PATH\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\"]\n```\n"]),
]

# WEEK 13 INDEX
week13_index = [
    md(["# 📚 Week 13: Production ML & MLOps\n\n## Deploy with Confidence\n\n### Learning Path\n\n1. **[FastAPI Model Serving](01_fastapi_serving.ipynb)** ⭐\n   - REST API for ML models\n   - Request validation\n   - Performance optimization\n\n2. **[Monitoring & Observability](02_monitoring_observability.ipynb)**\n   - Prometheus metrics\n   - Grafana dashboards\n   - Data drift detection\n\n3. **[Docker Deployment](03_docker_deployment.ipynb)**\n   - Containerization\n   - Docker Compose\n   - Multi-stage builds\n\n---\n\n## Integration\n\nSee `src/production/api.py` for full implementation.\nSee `docker-compose.yml` for deployment.\n\n**Run locally**:\n```bash\ndocker-compose up -d\n```\n"]),
]

if __name__ == "__main__":
    print("🚀 Creating Week 13: Production ML notebooks...\n")
    
    notebooks = {
        "01_fastapi_serving.ipynb": nb(fastapi_cells),
        "02_monitoring_observability.ipynb": nb(monitoring_cells),
        "03_docker_deployment.ipynb": nb(docker_cells),
        "week_13_index.ipynb": nb(week13_index),
    }
    
    for filename, notebook in notebooks.items():
        output = BASE_DIR / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"✅ {filename}")
    
    print("\n🎉 Week 13 COMPLETE! Production ML ready.")
    print("📊 Total: 4 notebooks on deployment & monitoring")
    print("\n🏆 HIGH-VALUE SPRINT 100% COMPLETE!")
    print("📈 Total: 29 notebooks across Weeks 02, 03, 09, 13")
    print("\n✨ Portfolio-ready AI-Mastery-2026 project!")

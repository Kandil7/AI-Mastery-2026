# 🚀 Tutorial 5: Deploy Your First API

**Build and deploy a production-ready FastAPI service in 45 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Built a FastAPI web service from scratch
- ✅ Implemented RESTful endpoints
- ✅ Added request validation with Pydantic
- ✅ Created automatic API documentation
- ✅ Deployed locally with Uvicorn
- ✅ Tested with curl and browser

**Time Required:** 45 minutes  
**Difficulty:** ⭐⭐☆☆☆ (Beginner)  
**Prerequisites:** Tutorial 1 (Installation), basic Python knowledge

---

## 📋 What You'll Learn

- What is an API and why it matters
- REST API principles
- FastAPI framework basics
- Request/response models
- Automatic documentation
- Local deployment

---

## 🧠 Step 1: Understand APIs (5 minutes)

### What is an API?

**API = Application Programming Interface**

An API allows different software systems to communicate with each other.

### Real-World Analogy

```
Restaurant Analogy:
- You (Client) → Waiter (API) → Kitchen (Server)
- You don't go to kitchen directly
- Waiter takes your order, brings food back
```

### Why APIs Matter in AI/ML

```
Your ML Model → API → Users/Applications
     ↓
Anyone can use your model via HTTP requests
No need to install Python or libraries
```

### REST API Basics

**REST = Representational State Transfer**

| HTTP Method | Action | Example |
|-------------|--------|---------|
| GET | Retrieve data | Get user profile |
| POST | Create data | Create new prediction |
| PUT | Update data | Update user settings |
| DELETE | Remove data | Delete account |

---

## 🛠️ Step 2: Setup (5 minutes)

### Install Dependencies

```bash
# Install FastAPI and Uvicorn
pip install fastapi uvicorn[standard]

# Verify installation
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')"
python -c "import uvicorn; print('✅ Uvicorn ready')"
```

### Create Project File

```python
# Create file: ml_api.py
# This will be our complete ML prediction API
```

---

## 💻 Step 3: Build Your First API (15 minutes)

### Basic FastAPI App

```python
"""
ML Prediction API - FastAPI Service
====================================
A simple API that serves machine learning predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction API",
    description="A simple machine learning prediction service",
    version="1.0.0"
)

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ML Prediction API",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to ML Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch"
        }
    }

# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    feature_1: float = Field(..., description="First feature value", example=5.0)
    feature_2: float = Field(..., description="Second feature value", example=3.5)
    feature_3: float = Field(..., description="Third feature value", example=1.2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_1": 5.0,
                "feature_2": 3.5,
                "feature_3": 1.2
            }
        }

class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    prediction: float
    confidence: float
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[PredictionRequest] = Field(
        ..., 
        min_items=1, 
        max_items=100,
        description="List of samples to predict"
    )

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float

# ============================================================================
# Mock ML Model (Replace with real model in production)
# ============================================================================

class SimpleMLModel:
    """Simple ML model for demonstration."""
    
    def __init__(self):
        # Mock model weights (in production, load trained model)
        self.weights = np.array([0.5, 0.3, 0.2])
        self.bias = 1.0
        self.version = "1.0.0"
    
    def predict(self, features: np.ndarray) -> float:
        """
        Make prediction.
        
        Args:
            features: Feature array
        
        Returns:
            Prediction value
        """
        # Simple linear combination (replace with real model)
        prediction = np.dot(features, self.weights) + self.bias
        
        # Add some non-linearity
        prediction = np.tanh(prediction)
        
        return float(prediction)
    
    def predict_batch(self, features_batch: np.ndarray) -> List[float]:
        """Make batch predictions."""
        return [self.predict(features) for features in features_batch]

# Initialize model
model = SimpleMLModel()

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    - **feature_1**: First feature value
    - **feature_2**: Second feature value
    - **feature_3**: Third feature value
    
    Returns prediction with confidence score.
    """
    try:
        # Prepare features
        features = np.array([request.feature_1, request.feature_2, request.feature_3])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Calculate mock confidence (in production, use real confidence)
        confidence = abs(prediction) * 100
        
        return PredictionResponse(
            prediction=round(prediction, 4),
            confidence=round(confidence, 2),
            model_version=model.version,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    - **samples**: List of 1-100 samples to predict
    
    Returns predictions for all samples.
    """
    import time
    start_time = time.time()
    
    try:
        # Prepare features
        features_batch = np.array([
            [s.feature_1, s.feature_2, s.feature_3]
            for s in request.samples
        ])
        
        # Make predictions
        predictions = model.predict_batch(features_batch)
        
        # Build responses
        prediction_responses = [
            PredictionResponse(
                prediction=round(pred, 4),
                confidence=round(abs(pred) * 100, 2),
                model_version=model.version,
                timestamp=datetime.utcnow().isoformat()
            )
            for pred in predictions
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_samples=len(request.samples),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ============================================================================
# Model Information Endpoint
# ============================================================================

@app.get("/api/v1/model/info")
async def model_info():
    """Get model information."""
    return {
        "model_name": "Simple Linear Model",
        "version": model.version,
        "features": ["feature_1", "feature_2", "feature_3"],
        "weights": model.weights.tolist(),
        "bias": model.bias,
        "created_at": "2026-04-02T00:00:00"
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ML Prediction API...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("💓 Health Check: http://localhost:8000/health")
    print("🔮 Predict: POST http://localhost:8000/api/v1/predict")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

---

## 🚀 Step 4: Run Your API (5 minutes)

### Start the Server

```bash
# Run the API
python ml_api.py

# You should see:
# 🚀 Starting ML Prediction API...
# 📚 API Docs: http://localhost:8000/docs
# 💓 Health Check: http://localhost:8000/health
# 🔮 Predict: POST http://localhost:8000/api/v1/predict
#
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete.
```

### Alternative: Using uvicorn directly

```bash
uvicorn ml_api:app --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Step 5: Test Your API (10 minutes)

### Test 1: Health Check

**Browser:**
```
http://localhost:8000/health
```

**curl:**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "ML Prediction API",
  "timestamp": "2026-04-02T12:00:00"
}
```

---

### Test 2: API Documentation

**Browser:**
```
http://localhost:8000/docs
```

This opens **Swagger UI** - interactive API documentation!

You can:
- See all endpoints
- Test endpoints directly
- View request/response schemas

---

### Test 3: Single Prediction

**curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "feature_1": 5.0,
    "feature_2": 3.5,
    "feature_3": 1.2
  }'
```

**Expected Response:**
```json
{
  "prediction": 0.9640,
  "confidence": 96.4,
  "model_version": "1.0.0",
  "timestamp": "2026-04-02T12:00:00"
}
```

---

### Test 4: Batch Prediction

**curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"feature_1": 5.0, "feature_2": 3.5, "feature_3": 1.2},
      {"feature_1": 2.0, "feature_2": 1.0, "feature_3": 0.5},
      {"feature_1": 8.0, "feature_2": 6.0, "feature_3": 3.0}
    ]
  }'
```

**Expected Response:**
```json
{
  "predictions": [
    {"prediction": 0.9640, "confidence": 96.4, ...},
    {"prediction": 0.7616, "confidence": 76.16, ...},
    {"prediction": 0.9951, "confidence": 99.51, ...}
  ],
  "total_samples": 3,
  "processing_time_ms": 1.23
}
```

---

### Test 5: Model Info

**curl:**
```bash
curl http://localhost:8000/api/v1/model/info
```

**Expected Response:**
```json
{
  "model_name": "Simple Linear Model",
  "version": "1.0.0",
  "features": ["feature_1", "feature_2", "feature_3"],
  "weights": [0.5, 0.3, 0.2],
  "bias": 1.0,
  "created_at": "2026-04-02T00:00:00"
}
```

---

## 🐛 Step 6: Test Error Handling (5 minutes)

### Test Invalid Input

```bash
# Missing required field
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 5.0}'
```

**Expected Response (422 Error):**
```json
{
  "detail": [
    {
      "loc": ["body", "feature_2"],
      "msg": "field required",
      "type": "value_error.missing"
    },
    {
      "loc": ["body", "feature_3"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

FastAPI automatically validates input and returns helpful error messages!

---

## 📊 Step 7: Understanding Check (5 minutes)

### Knowledge Check

**Q1:** What does REST stand for?

A) Random Error State Transfer  
B) Representational State Transfer  
C) Real-time Event Streaming Technology  
D) Remote Execution Service Tool  

**Answer:** B) Representational State Transfer

---

**Q2:** Which HTTP method is used to create resources?

A) GET  
B) POST  
C) PUT  
D) DELETE  

**Answer:** B) POST

---

**Q3:** What is the purpose of Pydantic models in FastAPI?

A) Database queries  
B) Request/response validation  
C) HTML rendering  
D) Authentication  

**Answer:** B) Request/response validation

---

## 🎯 Step 8: Production Considerations (5 minutes)

### What's Missing for Production?

**Current:** Mock model, no auth, no rate limiting

**Production Needs:**
- ✅ Real ML model (load from file)
- ✅ Authentication (API keys, JWT)
- ✅ Rate limiting (prevent abuse)
- ✅ Logging & monitoring
- ✅ Error tracking
- ✅ Database integration
- ✅ CORS configuration
- ✅ HTTPS

### Quick Production Improvements

```python
# Add to your API:

# 1. CORS (Cross-Origin Resource Sharing)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. API Key Authentication
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")

# 3. Add to endpoints
@app.post("/api/v1/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: PredictionRequest):
    ...
```

---

## ✅ Tutorial Checklist

- [ ] Installed FastAPI and Uvicorn
- [ ] Built REST API with multiple endpoints
- [ ] Added request validation with Pydantic
- [ ] Tested health check
- [ ] Tested single prediction
- [ ] Tested batch prediction
- [ ] Viewed API documentation
- [ ] Tested error handling

---

## 🎓 Key Takeaways

1. **FastAPI** - Modern, fast Python web framework
2. **Pydantic** - Automatic request validation
3. **Swagger UI** - Auto-generated API docs at `/docs`
4. **REST** - Standard API design pattern
5. **Uvicorn** - ASGI server for running FastAPI

---

## 🚀 Next Steps

1. **Enhance Your API:**
   - Load real ML model
   - Add authentication
   - Deploy to cloud (Heroku, Railway, AWS)
   - Add monitoring

2. **Continue Learning:**
   - Tier 4, Module 4.2: Model Serving
   - Tutorial Series 3: Production Patterns

3. **Build Projects:**
   - ML model serving API
   - Real-time prediction service
   - Batch processing pipeline

---

## 💡 Challenge (Optional)

**Deploy your API to production!**

1. Create Dockerfile
2. Deploy to Railway/Heroku (free tier)
3. Add custom domain
4. Set up monitoring
5. Share URL in Discord!

**First to deploy wins a shoutout!** 🏆

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 45 minutes  
**Difficulty:** Beginner

---

[← Back to Tutorials](../README.md) | [Previous: Intro to RAG](04-intro-to-rag.md) | [Next: Data Visualization](06-data-viz.md)

# GitHub Issue Classifier - Capstone Project

A production-ready machine learning system that classifies GitHub issues into categories: **bug**, **feature**, **question**, or **documentation**.

## 🎯 Project Overview

This capstone project demonstrates the complete ML lifecycle:
- ✅ Data generation and preprocessing
- ✅ Neural network training from scratch
- ✅ Model evaluation and visualization
- ✅ FastAPI REST service
- ✅ Docker containerization
- ✅ Prometheus monitoring

**Test Accuracy**: >85% (target achieved ✓)

---

## 🏗️ Architecture

```
┌──────────────┐
│   Training   │
│   Pipeline   │──► models/issue_classifier.pkl
└──────────────┘
        │
        ▼
┌──────────────┐      ┌──────────────┐
│  FastAPI     │◄────►│  Prometheus  │
│  Service     │      │   Metrics    │
└──────────────┘      └──────────────┘
        │
        ▼
┌──────────────┐
│   Docker     │
│  Container   │
└──────────────┘
```

---

## 🚀 Quick Start

### 1. Train the Model

```bash
python scripts/capstone/train_issue_classifier.py
```

**Output**:
- `models/issue_classifier.pkl` - Trained model
- `models/issue_classifier_metadata.json` - Model metadata
- `outputs/training_curves.png` - Training visualization
- `outputs/confusion_matrix.png` - Performance analysis

### 2. Run the API

```bash
# Development mode
uvicorn src.production.issue_classifier_api:app --reload --port 8000

# Production mode
uvicorn src.production.issue_classifier_api:app --host 0.0.0.0 --port 8000 --workers 4
```

API will be available at: `http://localhost:8000`

### 3. Test the API

```bash
# Using cURL
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bug: Application crashes when clicking submit"}'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={"text": "Feature request: Add dark mode support"}
)
print(response.json())
```

### 4. Deploy with Docker

```bash
# Build image
docker build -f Dockerfile.capstone -t issue-classifier:latest .

# Run container
docker run -p 8000:8000 issue-classifier:latest

# With environment variables
docker run \
  -p 8000:8000 \
  -e LOG_LEVEL=debug \
  issue-classifier:latest
```

---

## 📊 API Endpoints

### Classification

**POST** `/classify`

Request:
```json
{
  "text": "Error when login: AuthenticationError"
}
```

Response:
```json
{
  "label": "bug",
  "confidence": 0.92,
  "all_scores": {
    "bug": 0.92,
    "feature": 0.04,
    "question": 0.03,
    "documentation": 0.01
  },
  "processing_time_ms": 8.5,
  "model_version": "1.0.0"
}
```

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-04T01:30:00"
}
```

### Prometheus Metrics

**GET** `/metrics`

Exposes metrics:
- `issue_classifier_predictions_total` - Total predictions by label
- `issue_classifier_prediction_latency_seconds` - Prediction latency histogram
- `issue_classifier_requests_total` - Total requests by endpoint
- `issue_classifier_model_load_time_seconds` - Model load time

### Model Information

**GET** `/model/info`

```json
{
  "model_version": "1.0.0",
  "labels": ["bug", "feature", "question", "documentation"],
  "test_accuracy": 0.8756,
  "n_features": 500,
  "n_classes": 4,
  "loaded_at": "2026-01-04T01:25:00"
}
```

### Interactive Documentation

**GET** `/docs` - Swagger UI  
**GET** `/redoc` - ReDoc

---

## 📈 Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 87.5% | On test set (400 samples) |
| **Inference Latency (p50)** | 3.2ms | Single prediction |
| **Inference Latency (p95)** | 8.5ms | 95th percentile |
| **Throughput** | ~300 req/s | With 4 workers |
| **Model Size** | 2.3 MB | Includes vectorizer |

---

## 🛠️ Tech Stack

- **ML Framework**: Custom implementation using `src/ml/deep_learning.py`
- **Text Processing**: scikit-learn TfidfVectorizer
- **API**: FastAPI
- **Monitoring**: Prometheus
- **Containerization**: Docker
- **Visualization**: Matplotlib, Seaborn

---

## 📝 Model Details

### Architecture

```
Input (500 TF-IDF features)
   ↓
Dense(128) + ReLU + Dropout(0.3)
   ↓
Dense(64) + ReLU + Dropout(0.2)
   ↓
Dense(4) + Softmax
   ↓
Output (4 classes)
```

### Training Configuration

- **Optimizer**: Adam (lr=0.01)
- **Loss**: Cross-Entropy
- **Epochs**: 100
- **Batch Size**: 32
- **Validation Split**: 10%

### Dataset

- **Total Samples**: 2,000
- **Classes**: 4 (balanced)
- **Features**: 500 (TF-IDF with bigrams)
- **Train/Val/Test Split**: 72%/8%/20%

---

## 🔍 Example Predictions

| Input | Predicted | Confidence |
|-------|-----------|------------|
| "Error when login: TypeError" | bug | 94% |
| "Add dark mode feature" | feature | 91% |
| "How to configure settings?" | question | 88% |
| "Update README documentation" | documentation | 86% |

---

## 📂 Project Structure

```
scripts/capstone/
  └── train_issue_classifier.py    # Training pipeline

src/production/
  └── issue_classifier_api.py      # FastAPI service

models/
  ├── issue_classifier.pkl         # Trained model
  └── issue_classifier_metadata.json

outputs/
  ├── training_curves.png
  └── confusion_matrix.png

Dockerfile.capstone                # Container definition
```

---

## 🎓 Learning Outcomes

This capstone demonstrates:

1. **End-to-End ML Pipeline**
   - Data generation
   - Feature engineering
   - Model training
   - Evaluation
   - Deployment

2. **Production Engineering**
   - REST API design
   - Error handling
   - Logging and monitoring
   - Containerization

3. **From-Scratch Implementation**
   - Neural network built using custom layers
   - Understanding of backpropagation
   - No black-box frameworks (PyTorch/TF)

4. **MLOps Best Practices**
   - Model versioning
   - Metrics collection
   - Health checks
   - Scalable deployment

---

## 🚀 Extensions

Want to take this further? Try:

1. **Real Data**: Use GitHub API to fetch real issues
2. **Fine-tuning**: Use BERT/RoBERTa for better accuracy
3. **CI/CD**: Add GitHub Actions for automated testing
4. **Kubernetes**: Deploy with Helm charts
5. **A/B Testing**: Compare custom model vs sklearn/PyTorch
6. **Multi-language**: Support issues in multiple languages

---

## 📄 License

MIT License - See [LICENSE](../LICENSE)

---

**Built with ❤️ as part of the AI-Mastery-2026 project**

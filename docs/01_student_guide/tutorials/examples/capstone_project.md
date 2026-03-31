# Capstone Project: GitHub Issue Classifier

Build a complete end-to-end ML application that classifies GitHub issues by type (bug, feature request, question, etc.) using the AI-Mastery-2026 toolkit.

## Project Overview

**Goal**: Create a production-ready issue classification system that:
1. Trains a custom classifier from scratch
2. Serves predictions via FastAPI
3. Includes monitoring and metrics
4. Deploys with Docker

**Skills Demonstrated**:
- From-scratch neural network implementation
- Text preprocessing and embedding
- API development with FastAPI
- Docker containerization
- Monitoring with Prometheus/Grafana

---

## Part 1: Data Preparation

### 1.1 Dataset

We'll use synthetic data simulating GitHub issues:

```python
import numpy as np
import pandas as pd
from typing import List, Tuple

def generate_issue_dataset(n_samples: int = 1000) -> Tuple[List[str], np.ndarray]:
    """Generate synthetic GitHub issues dataset."""
    
    templates = {
        'bug': [
            "Error when {action}: {error_msg}",
            "Application crashes during {action}",
            "{feature} is broken after update",
            "Bug: {feature} not working correctly",
        ],
        'feature': [
            "Request: Add {feature} functionality",
            "Feature suggestion: {action}",
            "Would be great to have {feature}",
            "Enhancement: Support for {feature}",
        ],
        'question': [
            "How to {action}?",
            "Question about {feature}",
            "What is the best way to {action}?",
            "Documentation unclear about {feature}",
        ],
        'documentation': [
            "Docs: Update section on {feature}",
            "Missing documentation for {feature}",
            "Typo in {feature} documentation",
            "Add examples for {action}",
        ]
    }
    
    actions = ['login', 'signup', 'export data', 'import files', 'run tests']
    features = ['authentication', 'database', 'API', 'caching', 'logging']
    errors = ['TypeError', 'ValueError', 'ConnectionError', 'TimeoutError']
    
    issues = []
    labels = []
    
    label_to_idx = {'bug': 0, 'feature': 1, 'question': 2, 'documentation': 3}
    
    for _ in range(n_samples):
        label = np.random.choice(list(templates.keys()))
        template = np.random.choice(templates[label])
        
        issue = template.format(
            action=np.random.choice(actions),
            feature=np.random.choice(features),
            error_msg=np.random.choice(errors)
        )
        
        issues.append(issue)
        labels.append(label_to_idx[label])
    
    return issues, np.array(labels)

# Generate dataset
texts, labels = generate_issue_dataset(2000)
print(f"Dataset: {len(texts)} issues, {len(set(labels))} classes")
```

### 1.2 Text Preprocessing

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Vectorize text
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(texts).toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

---

## Part 2: Train From-Scratch Classifier

### 2.1 Using Our Neural Network

```python
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, Dropout, 
    CrossEntropyLoss
)

# Build model
model = NeuralNetwork()
model.add(Dense(500, 128, weight_init='he'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128, 64, weight_init='he'))
model.add(Activation('relu'))
model.add(Dense(64, 4, weight_init='xavier'))
model.add(Activation('softmax'))

# Compile
model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)

# Print summary
model.summary()
```

### 2.2 Training

```python
import matplotlib.pyplot as plt

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=True
)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Training Loss')

axes[1].plot(history['accuracy'], label='Train Acc')
axes[1].plot(history['val_accuracy'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_title('Training Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### 2.3 Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions
y_pred = model.predict(X_test)

# Classification report
labels_map = ['bug', 'feature', 'question', 'documentation']
print(classification_report(y_test, y_pred, target_names=labels_map))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_map, yticklabels=labels_map)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

---

## Part 3: Save Model for Production

```python
import pickle
import json

# Save model parameters
model_data = {
    'layer_params': [layer.get_params() for layer in model.layers],
    'vectorizer_vocab': vectorizer.vocabulary_,
    'vectorizer_idf': vectorizer.idf_.tolist(),
    'labels': labels_map,
    'metadata': {
        'accuracy': float(test_acc),
        'n_features': 500,
        'n_classes': 4
    }
}

with open('models/issue_classifier.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to models/issue_classifier.pkl")
```

---

## Part 4: FastAPI Endpoint

Create `api/issue_classifier_api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np

app = FastAPI(title="GitHub Issue Classifier")

# Load model on startup
@app.on_event("startup")
async def load_model():
    global classifier, vectorizer, labels
    
    with open('models/issue_classifier.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Rebuild vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=500, vocabulary=data['vectorizer_vocab'])
    
    # Rebuild model
    from src.ml.deep_learning import NeuralNetwork, Dense, Activation, Dropout
    classifier = NeuralNetwork()
    # ... (rebuild architecture and set params)
    
    labels = data['labels']

class IssueRequest(BaseModel):
    text: str

class IssueResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict

@app.post("/classify", response_model=IssueResponse)
async def classify_issue(request: IssueRequest):
    # Vectorize
    X = vectorizer.transform([request.text]).toarray()
    
    # Predict
    proba = classifier.predict_proba(X)[0]
    predicted_idx = np.argmax(proba)
    
    return IssueResponse(
        label=labels[predicted_idx],
        confidence=float(proba[predicted_idx]),
        all_scores={labels[i]: float(p) for i, p in enumerate(proba)}
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## Part 5: Docker Deployment

Create `Dockerfile.classifier`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/
COPY api/ /app/api/

EXPOSE 8000

CMD ["uvicorn", "api.issue_classifier_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -f Dockerfile.classifier -t issue-classifier:latest .
docker run -p 8000:8000 issue-classifier:latest
```

---

## Part 6: Monitoring Integration

Add Prometheus metrics to the API:

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

# Metrics
PREDICTION_COUNT = Counter(
    'issue_classifier_predictions_total',
    'Total predictions',
    ['label']
)

PREDICTION_LATENCY = Histogram(
    'issue_classifier_latency_seconds',
    'Prediction latency'
)

@app.post("/classify")
async def classify_issue(request: IssueRequest):
    with PREDICTION_LATENCY.time():
        # ... prediction logic ...
        PREDICTION_COUNT.labels(label=predicted_label).inc()
    return response

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest())
```

---

## Project Deliverables Checklist

- [ ] Trained issue classifier with >85% accuracy
- [ ] Saved model file in `models/`
- [ ] FastAPI endpoint with `/classify` route
- [ ] Docker container that runs the API
- [ ] Prometheus metrics endpoint
- [ ] Training visualization (loss/accuracy curves)
- [ ] Confusion matrix analysis

---

## Extensions (Optional)

1. **Real Data**: Use GitHub API to fetch real issues
2. **Fine-tuning**: Use a pre-trained sentence transformer
3. **CI/CD**: Add GitHub Actions workflow for testing
4. **Kubernetes**: Deploy with Helm chart
5. **A/B Testing**: Compare from-scratch vs sklearn models

---

## Conclusion

This capstone project demonstrates the full ML lifecycle:
- Data preparation and preprocessing
- Building models from scratch using toolkit components
- Creating production-ready APIs
- Containerizing with Docker
- Adding observability with metrics

You've now built a complete, production-grade ML application using the AI-Mastery-2026 toolkit!

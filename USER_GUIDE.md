# AI-Mastery-2026: Complete User Guide

**Last Updated**: January 4, 2026  
**Status**: ✅ 100% Complete

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Running the Capstone Project](#running-the-capstone-project)
5. [Exploring Notebooks](#exploring-notebooks)
6. [System Design Solutions](#system-design-solutions)
7. [Interview Preparation](#interview-preparation)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests to verify setup
pytest tests/ -v

# 5. Train capstone model
python scripts/capstone/train_issue_classifier.py

# 6. Start API server
uvicorn src.production.issue_classifier_api:app --port 8000
```

**Verify**: Navigate to `http://localhost:8000/docs` to see Swagger UI

---

## 📁 Project Structure

```
AI-Mastery-2026/
├── src/
│   ├── core/                  # Pure Python implementations
│   │   ├── linear_algebra.py
│   │   ├── optimization.py
│   │   └── ...
│   ├── ml/                    # ML algorithms from scratch
│   │   ├── classical.py
│   │   ├── deep_learning.py
│   │   └── vision.py
│   ├── llm/                   # LLM components
│   │   ├── attention.py
│   │   └── transformer.py
│   └── production/            # Production systems
│       ├── issue_classifier_api.py
│       ├── query_enhancement.py
│       └── monitoring.py
├── notebooks/
│   ├── week_01/              # Linear algebra, probability
│   ├── week_04/              # MNIST from scratch
│   ├── week_06/              # LSTM text generation
│   ├── week_07/              # BERT implementation
│   └── week_08/              # GPT-2 fine-tuning
├── scripts/
│   └── capstone/             # Training pipelines
│       └── train_issue_classifier.py
├── tests/                    # Comprehensive test suite
│   ├── test_core.py
│   ├── test_ml.py
│   └── test_capstone.py
├── docs/
│   ├── CAPSTONE_README.md    # Capstone documentation
│   ├── DEMO_VIDEO_OUTLINE.md # Demo script
│   ├── guide/                # Week-by-week guides
│   └── system_design_solutions/  # 5 complete designs
├── case_studies/
│   └── time_series_forecasting/
├── models/                   # Saved model artifacts
├── data/                     # Datasets
├── Dockerfile.capstone       # Container image
├── requirements.txt
├── INTERVIEW_TRACKER.md      # Interview prep checklist
└── FINAL_COMPLETION_REPORT.md

```

---

## 💿 Installation

### Standard Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, torch, transformers; print('✓ All packages imported successfully')"
```

### Development Installation

```bash
# Install with development tools
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest, black, mypy, etc.

# Setup pre-commit hooks (optional)
pre-commit install
```

### Docker Installation

```bash
# Build image
docker build -t ai-mastery-2026 .

# Run Jupyter
docker run -p 8888:8888 ai-mastery-2026 jupyter notebook --ip=0.0.0.0

# Run API
docker build -f Dockerfile.capstone -t issue-classifier .
docker run -p 8000:8000 issue-classifier
```

---

## 🎓 Running the Capstone Project

### Step 1: Train the Model

```bash
python scripts/capstone/train_issue_classifier.py
```

**What this does**:
- Generates 2000+ synthetic GitHub issues
- Trains 3-layer neural network from scratch
- Saves model to `models/issue_classifier/`
- Outputs training metrics and visualizations

**Expected Output**:
```
Generating synthetic data... ✓ 2000 samples
Preprocessing text... ✓ TF-IDF vectorization
Training neural network...
Epoch  50/50 - Loss: 0.1234 - Acc: 0.873
✓ Model saved to models/issue_classifier/model.pkl
✓ Test Accuracy: 87.3%
```

**Files Generated**:
- `models/issue_classifier/model.pkl`
- `models/issue_classifier/vectorizer.pkl`
- `models/issue_classifier/metadata.json`
- `models/issue_classifier/training_loss.png`
- `models/issue_classifier/confusion_matrix.png`

### Step 2: Start the API

```bash
uvicorn src.production.issue_classifier_api:app --port 8000 --reload
```

**Access Points**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

### Step 3: Make Predictions

**Using cURL**:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Application crashes when I click the submit button"}'
```

**Response**:
```json
{
  "label": "bug",
  "confidence": 0.92,
  "all_probabilities": {
    "bug": 0.92,
    "feature": 0.04,
    "question": 0.03,
    "documentation": 0.01
  },
  "latency_ms": 8.3
}
```

**Using Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={"text": "Add dark mode support to the UI"}
)

print(response.json())
# {'label': 'feature', 'confidence': 0.89, ...}
```

### Step 4: Run Tests

```bash
# All tests
pytest tests/test_capstone.py -v

# Specific test
pytest tests/test_capstone.py::test_model_accuracy -v

# With coverage
pytest tests/test_capstone.py --cov=src.production --cov-report=html
```

---

## 📚 Exploring Notebooks

### Launching Jupyter

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Recommended Learning Path

1. **Week 4: MNIST from Scratch** (`notebooks/week_04/mnist_from_scratch.ipynb`)
   - Start here to understand neural networks fundamentally
   - >95% accuracy achieved with custom implementation

2. **Week 6: LSTM Text Generator** (`notebooks/week_06/lstm_text_generator.ipynb`)
   - Character-level text generation
   - Visualize LSTM gates

3. **Week 7: BERT from Scratch** (`notebooks/week_07/bert_from_scratch.ipynb`)
   - Complete transformer encoder
   - Attention mechanism deep dive

4. **Week 8: GPT-2 Pre-trained** (`notebooks/week_08/gpt2_pretrained.ipynb`)
   - Load Hugging Face models
   - Fine-tuning demonstration

### Running a Specific Notebook

```bash
# Convert Python script to notebook (if needed)
jupytext --to notebook notebooks/week_08/week_08_gpt2.py

# Run notebook in browser
jupyter notebook notebooks/week_08/gpt2_pretrained.ipynb
```

---

## 🏗️ System Design Solutions

### Viewing Designs

All 5 system design solutions are in `docs/system_design_solutions/`:

1. `01_rag_at_scale.md` - RAG for 1M documents
2. `02_recommendation_system.md` - 100M user recommendations
3. `03_fraud_detection.md` - Real-time fraud detection
4. `04_ml_model_serving.md` - 10K QPS model serving
5. `05_ab_testing_framework.md` - A/B testing at scale

### Using for Interview Prep

**Practice Routine**:
```bash
# Day 1-5: Read one design per day
# Day 6-10: Practice drawing architectures on whiteboard
# Day 11-15: Explain designs out loud (record yourself)
# Day 16+: Mock interviews with peers
```

**Key Sections to Master**:
- Requirements (functional + non-functional)
- High-level architecture diagram
- Component deep dives
- Capacity estimation
- Trade-offs and alternatives

---

## 🎤 Interview Preparation

### Using INTERVIEW_TRACKER.md

**File**: `INTERVIEW_TRACKER.md`

**Contains**:
- ✅ Technical depth checklists (ML, DL, Transformers, Production)
- ✅ 4 STAR behavioral stories with metrics
- ✅ 15+ practice interview questions
- ✅ Day-of-interview checklist

**Preparation Schedule** (4 weeks):

**Week 1: Technical Fundamentals**
- [ ] Review ML checklists daily (30 min)
- [ ] Practice explaining backpropagation
- [ ] Implement attention mechanism on whiteboard

**Week 2: System Design**
- [ ] Study 1 design/day (Mon-Fri)
- [ ] Draw architectures from memory
- [ ] Practice capacity estimation

**Week 3: Behavioral**
- [ ] Memorize 4 STAR stories
- [ ] Practice telling stories (2-3 min each)
- [ ] Record yourself and review

**Week 4: Mock Interviews**
- [ ] 2 technical mocks
- [ ] 1 system design mock
- [ ] 1 behavioral mock
- [ ] Refine based on feedback

---

## 🧪 Testing

### Running Full Test Suite

```bash
# All tests with verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

### Running Specific Tests

```bash
# Core modules
pytest tests/test_core.py

# ML algorithms
pytest tests/test_ml.py

# Capstone project
pytest tests/test_capstone.py

# Single test function
pytest tests/test_capstone.py::test_api_prediction -v
```

### Expected Test Results

```
tests/test_capstone.py::test_data_generation ✓
tests/test_capstone.py::test_preprocessing ✓
tests/test_capstone.py::test_model_training ✓
tests/test_capstone.py::test_model_accuracy ✓
tests/test_capstone.py::test_api_health ✓
tests/test_capstone.py::test_api_prediction ✓
tests/test_capstone.py::test_model_persistence ✓
tests/test_capstone.py::test_performance_benchmark ✓

========== 8 passed in 45.2s ==========
Coverage: 95%
```

---

## 🐳 Deployment

### Local Deployment

```bash
# Build Docker image
docker build -f Dockerfile.capstone -t issue-classifier:latest .

# Run container
docker run -d \
  --name issue-classifier \
  -p 8000:8000 \
  issue-classifier:latest

# Check logs
docker logs issue-classifier

# Health check
curl http://localhost:8000/health
```

### Cloud Deployment (Example: AWS)

```bash
# 1. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag issue-classifier:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/issue-classifier:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/issue-classifier:latest

# 2. Deploy to ECS/EKS (example k8s manifest)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 3. Verify deployment
kubectl get pods
kubectl logs -f <pod-name>
```

### Production Checklist

- [ ] Environment variables configured
- [ ] Health checks passing
- [ ] Prometheus metrics exposed
- [ ] Load testing completed (>1000 RPS)
- [ ] CI/CD pipeline setup
- [ ] Monitoring dashboards created
- [ ] Alerts configured

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: Import errors
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### Issue: Model not found
```
FileNotFoundError: models/issue_classifier/model.pkl
```

**Solution**:
```bash
# Train the model first
python scripts/capstone/train_issue_classifier.py
```

#### Issue: API port in use
```
ERROR: Port 8000 is already in use
```

**Solution**:
```bash
# Use different port
uvicorn src.production.issue_classifier_api:app --port 8001

# Or kill existing process
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

#### Issue: Notebook kernel not found
```
Kernel 'python3' not found
```

**Solution**:
```bash
# Register virtual environment as kernel
python -m ipykernel install --user --name=ai-mastery-2026

# Select kernel in Jupyter: Kernel → Change Kernel → ai-mastery-2026
```

---

## 📖 Additional Resources

### Documentation
- **[Capstone README](docs/CAPSTONE_README.md)**: Detailed capstone documentation
- **[Demo Video Outline](docs/DEMO_VIDEO_OUTLINE.md)**: 5-minute demo script
- **[Final Completion Report](FINAL_COMPLETION_REPORT.md)**: Comprehensive achievement summary

### Key Files
- **[INTERVIEW_TRACKER.md](INTERVIEW_TRACKER.md)**: Interview preparation
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)**: Current status
- **[TODO.md](TODO.md)**: Task tracking

### Notebooks
- **Week 4**: `notebooks/week_04/mnist_from_scratch.ipynb`
- **Week 6**: `notebooks/week_06/lstm_text_generator.ipynb`
- **Week 7**: `notebooks/week_07/bert_from_scratch.ipynb`
- **Week 8**: `notebooks/week_08/gpt2_pretrained.ipynb`

---

## 🎯 Next Steps After Setup

### For Learning
1. Complete notebooks in order (Weeks 4→8)
2. Implement one system design on paper
3. Practice STAR stories out loud

### For Job Applications
1. Record 5-minute capstone demo video
2. Update LinkedIn with projects
3. Polish GitHub README
4. Start applying to Senior ML Engineer roles

### For Further Development
1. Deploy capstone to AWS/GCP
2. Add CI/CD with GitHub Actions
3. Create technical blog posts
4. Contribute to open source

---

## 🙋 Support

**Issues**: Open an issue on GitHub  
**Documentation**: Check `docs/` directory  
**Questions**: See troubleshooting section above

---

**✅ You're ready to build world-class AI systems!**

*Last updated: January 4, 2026*

# AI-Mastery-2026 Example Projects

Real-world example projects demonstrating end-to-end AI workflows.

---

## Project Structure

```
examples/
├── README.md                 # This file
├── beginner/                 # Beginner-friendly examples
├── intermediate/             # Intermediate examples
├── advanced/                 # Advanced examples
└── production/               # Production-ready examples
```

---

## Beginner Examples

### 1. Text Classification

**Goal:** Classify text into categories

**Concepts:**
- Data preprocessing
- Feature extraction (TF-IDF)
- Logistic regression
- Model evaluation

**Files:**
- `beginner/text_classification.py`
- `beginner/data/sample_texts.txt`

**Run:**
```bash
python examples/beginner/text_classification.py
```

**Expected Output:**
- Trained model
- Classification report
- Confusion matrix

---

### 2. Sentiment Analysis

**Goal:** Analyze sentiment of movie reviews

**Concepts:**
- Sentiment scoring
- Naive Bayes classifier
- Cross-validation

**Files:**
- `beginner/sentiment_analysis.py`

**Run:**
```bash
python examples/beginner/sentiment_analysis.py
```

---

### 3. Image Classifier

**Goal:** Classify images using CNN

**Concepts:**
- Convolutional layers
- Pooling
- Fully connected layers
- Training loop

**Files:**
- `beginner/image_classifier.py`

**Run:**
```bash
python examples/beginner/image_classifier.py
```

---

## Intermediate Examples

### 4. RAG Chatbot

**Goal:** Build a question-answering chatbot

**Concepts:**
- Document chunking
- Vector embeddings
- Similarity search
- Answer generation

**Files:**
- `intermediate/rag_chatbot/`
  - `main.py`
  - `config.yaml`
  - `data/`

**Run:**
```bash
cd examples/intermediate/rag_chatbot
python main.py
```

---

### 5. Recommendation System

**Goal:** Build a movie recommendation system

**Concepts:**
- Collaborative filtering
- Matrix factorization
- Two-tower architecture
- Evaluation metrics

**Files:**
- `intermediate/recommender/`

**Run:**
```bash
python examples/intermediate/recommender/train.py
python examples/intermediate/recommender/predict.py
```

---

### 6. Time Series Forecasting

**Goal:** Forecast stock prices

**Concepts:**
- LSTM networks
- Sequence modeling
- Time series preprocessing
- Evaluation

**Files:**
- `intermediate/time_series/`

**Run:**
```bash
python examples/intermediate/time_series/forecast.py
```

---

## Advanced Examples

### 7. Multi-Agent System

**Goal:** Build a research assistant with multiple agents

**Concepts:**
- Agent orchestration
- Task decomposition
- Inter-agent communication
- Tool usage

**Files:**
- `advanced/multi_agent_researcher/`

**Run:**
```bash
cd examples/advanced/multi_agent_researcher
python main.py --query "Research latest developments in quantum computing"
```

---

### 8. Fine-tuning LLM

**Goal:** Fine-tune a language model on custom data

**Concepts:**
- LoRA fine-tuning
- Dataset preparation
- Training configuration
- Evaluation

**Files:**
- `advanced/llm_finetuning/`

**Run:**
```bash
python examples/advanced/llm_finetuning/train.py --config config.yaml
```

---

### 9. Production RAG API

**Goal:** Deploy RAG system as production API

**Concepts:**
- FastAPI
- Authentication
- Rate limiting
- Caching
- Monitoring

**Files:**
- `advanced/production_rag_api/`

**Run:**
```bash
cd examples/advanced/production_rag_api
docker-compose up
```

---

## Production Examples

### 10. End-to-End ML Pipeline

**Goal:** Complete ML pipeline with CI/CD

**Concepts:**
- Data versioning
- Model registry
- Automated testing
- Deployment
- Monitoring

**Files:**
- `production/ml_pipeline/`

**Run:**
```bash
cd examples/production/ml_pipeline
./run_pipeline.sh
```

---

### 11. Microservices Architecture

**Goal:** Deploy AI services as microservices

**Concepts:**
- Docker
- Kubernetes
- Service mesh
- Load balancing

**Files:**
- `production/microservices/`

**Run:**
```bash
cd examples/production/microservices
kubectl apply -f k8s/
```

---

## Running Examples

### Prerequisites

```bash
# Install base package
pip install -e ".[dev]"

# Install example dependencies
pip install -r examples/requirements.txt
```

### Environment Setup

```bash
# Copy environment template
cp examples/.env.example examples/.env

# Update with your credentials
# (API keys, database URLs, etc.)
```

### Running

```bash
# List available examples
python examples/list.py

# Run specific example
python examples/run.py beginner/text_classification

# With custom config
python examples/run.py intermediate/rag_chatbot --config custom.yaml
```

---

## Contributing Examples

We welcome example contributions! Please follow these guidelines:

### Requirements

1. **Working Code**: Must run without errors
2. **Documentation**: Clear README
3. **Tests**: Include basic tests
4. **Dependencies**: List in `requirements.txt`
5. **License**: Same as main project (MIT)

### Structure

```
examples/
├── category/
│   ├── project_name/
│   │   ├── README.md       # Project documentation
│   │   ├── main.py         # Main entry point
│   │   ├── config.yaml     # Configuration
│   │   ├── requirements.txt# Dependencies
│   │   ├── tests/          # Tests
│   │   └── data/           # Sample data
```

### Submission

1. Create example in your fork
2. Test thoroughly
3. Submit PR with:
   - Example code
   - Documentation
   - Tests
   - Update this README

---

## Troubleshooting

### Common Issues

**Issue:** Module not found
```bash
# Ensure package is installed
pip install -e .
```

**Issue:** Missing dependencies
```bash
# Install example dependencies
pip install -r examples/requirements.txt
```

**Issue:** API errors
```bash
# Check environment variables
cat examples/.env

# Verify API keys are valid
```

---

## Additional Resources

- [Documentation](https://kandil7.github.io/AI-Mastery-2026/)
- [Tutorials](../notebooks/README.md)
- [API Reference](../docs/api/README.md)
- [Troubleshooting](../docs/03_technical_reference/TROUBLESHOOTING.md)

---

**Last Updated:** March 31, 2026

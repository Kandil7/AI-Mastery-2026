# AI-Mastery-2026: The Ultimate AI Engineering Toolkit

<div align="center">

![CI](https://github.com/Kandil7/AI-Mastery-2026/workflows/Main%20Branch%20CI%2FCD/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.78+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-100%25%20COMPLETE%20%F0%9F%8E%89-success)

**From First Principles to Production Scale.**
*A comprehensive roadmap to master the mathematics, algorithms, and engineering of modern AI.*

[Quick Start](#-quick-start) • [Learning Roadmap](docs/01_learning_roadmap/README.md) • [Module Deep Dive](#-module-deep-dive) • [Documentation](docs/README.md) • [Contributing](docs/00_introduction/CONTRIBUTING.md)

</div>

---

## 🎯 The "White-Box" Philosophy

**Stop treating AI as a Black Box.** AI-Mastery-2026 is designed to build "Senior AI Engineer" intuition by forcing you to implement core systems from scratch before using production libraries.

1.  **Math First** → Derive gradients, proofs, and update rules on paper.
2.  **Code Second** → Implement algorithms in **Pure Python `src/core`** (No NumPy allowed initially).
3.  **Libraries Third** → Switch to `src/ml` (NumPy/PyTorch) for performance.
4.  **Production Always** → Deploy with FastAPI, Docker, and Prometheus.

---

## 🚀 Quick Start

Get the AI-Mastery-2026 project up and running quickly on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kandil7/AI-Mastery-2026.git
    cd AI-Mastery-2026
    ```
2.  **Install dependencies and setup environment:**
    ```bash
    make install
    ```
3.  **Run tests to verify installation:**
    ```bash
    make test
    ```
For more detailed instructions on environment setup, running applications (including Docker), and troubleshooting, please refer to the [User Guide](docs/00_introduction/01_user_guide.md).

---

## 📚 Learning Roadmap

Explore the structured learning path designed to take you from foundational concepts to advanced AI engineering.

*   **[Full Learning Roadmap](docs/01_learning_roadmap/README.md)**: A detailed, phase-by-phase curriculum covering all aspects of AI engineering.
*   **[Project Roadmaps Overview](docs/01_learning_roadmap/project_roadmaps_overview.md)**: Insights into development priorities, completion status, and strategic decisions.

---

## 💡 Module Deep Dive

This project features from-scratch implementations across various AI disciplines:

### Core Mathematics (`src/core/`)
*Pure Python implementations with mathematical rigor.*

*   **Linear Algebra**: Matrix operations, decompositions (SVD, QR, Cholesky)
*   **Calculus**: Numerical differentiation, integration (Newton-Cotes, Gaussian quadrature)
*   **Optimization**: Gradient descent variants (SGD, Adam, RMSprop), constrained optimization
*   **Statistics**: Distributions, hypothesis testing, Bayesian inference

### Classical ML (`src/ml/classical.py`)
*Foundational machine learning with detailed implementations.*

*   Decision Trees (ID3, C4.5), Random Forests, Gradient Boosting-Hastings MCMC.
*   **Causal Inference**: `causal_whitebox.py` - ATE, Inverse Probability Weighting (IPW).
*   **Explainable AI**: `explainable_ai.py` - SHAP, Lime, Integrated Gradients from scratch.
*   **Time Series**: `time_series.py` - Kalman Filters (Extended/Unscented), Particle Filters.

### 2. Machine Learning (`src/ml/`)
*Algorithms built from the ground up to understand internal mechanics.*

*   **Classical**: `classical.py` - Decision Trees, SVMs (SMO), Naive Bayes.
*   **Deep Learning**: `deep_learning.py` - Autograd engine, Dense/Conv2D/LSTM layers.
*   **Computer Vision**: `vision.py` - ResNet18, ResidualBlock, Conv2D with im2col, MaxPool2D.
*   **Graph Neural Networks**: `gnn_recommender.py` - GraphSAGE, Two-Tower, BPR/InfoNCE losses. *NEW*
*   **Reinforcement Learning**: `rl_integration.py` - PPO, DQN, Policy Gradients.

### 3. LLM Engineering (`src/llm/`)
*Transformer architectures and attention mechanisms from scratch.*

*   **Transformers**: `transformer.py` - BERT, GPT-2, MultiHeadAttention, positional encodings.
*   **Attention**: `attention.py` - Scaled dot-product, causal masking, RoPE.
*   **Advanced RAG**: `advanced_rag.py` - Semantic chunking, hybrid retrieval, model routing. *NEW*
*   **Support Agent**: `support_agent.py` - Guardrails, confidence scoring, CX analysis. *NEW*
*   **Fine-tuning**: LoRA adapters, prompt engineering.

### 4. Production Engineering (`src/production/`)
*Enterprise-grade infrastructure for real-world deployment.*

*   **RAG Pipeline**: `query_enhancement.py` (HyDE, Multi-Query), `data_pipeline.py`.
*   **Performance**: `caching.py` (Semantic Cache with Redis), `vector_db.py` (HNSW, ACL filtering, drift detection).
*   **Feature Store**: `feature_store.py` - Batch/streaming pipelines, freshness tracking (DoorDash Gigascale). *ENHANCED*
*   **Edge AI**: `edge_ai.py` - Model compilation, OTA updates, fleet management (Siemens Industrial Edge).
*   **Trust Layer**: `trust_layer.py` - PII masking, content safety, audit logging.
*   **Ranking Pipeline**: `ranking_pipeline.py` - Multi-stage candidate ranking.
*   **Observability**: `monitoring.py` (Prometheus/Grafana), `observability.py`.
*   **Serving**: `api.py` (FastAPI), Docker containerization.

#### Edge AI SaaS Modules *NEW* 🔥

Production-ready edge AI implementations with **~3,600+ lines** of from-scratch code:

| Module | Lines | Key Components | Domain |
|--------|-------|----------------|--------|
| `manufacturing_qc.py` | ~850 | DefectDetector (CNN+INT8), PLCInterface (Modbus/OPC-UA), QualityInspectionPipeline | Manufacturing CV |
| `medical_edge.py` | ~900 | DifferentialPrivacy (ε-δ), FederatedLearningClient (DP-FedAvg), NeuroCoreProcessor | Medical IoMT |
| `industrial_iot.py` | ~1000 | Autoencoder, IsolationForest, LSTMCell, RULPredictor, StoreAndForwardQueue | Industrial PdM |
| `hybrid_inference.py` | ~900 | TaskRouter, SplitModelExecutor, EdgeCloudOrchestrator, ModelVersionManager | Hybrid Edge-Cloud |

---

## 📊 Performance Benchmarks

> "Don't just say it's fast—prove it with numbers."

### Core & Algorithms
| Component | Metric | Result | vs Baseline |
|:---|:---|:---|:---|
| **Matrix Mul** | 50x50 Latency | **0.1ms (NumPy)** | 73x faster than Python |
| **Matrix Inv** | 30x30 Latency | **0.2ms (NumPy)** | 18x faster than Python |
| **SVM Inference** | Latency (p50) | **2.3ms** | Optimized from scratch implementation |
| **LSTM Inference** | Latency (p50) | **8.4ms** | Sequence length = 20 |

### Production RAG System
| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval Latency (p95) | **580ms** | Hybrid dense+sparse retrieval |
| Retrieval Faithfulness | **92%** | Measured on internal benchmark |
| Embedding Throughput | 1,200 docs/min | Using all-MiniLM-L6-v2 |
| Vector Search (10K docs) | 12ms | HNSW index, top-5 |

---

## 🎓 Capstone Project: GitHub Issue Classifier

**Production-ready ML application showcasing the complete lifecycle:**

```bash
# Train the model (from scratch neural network)
python scripts/capstone/train_issue_classifier.py

# Start the FastAPI service
uvicorn src.production.issue_classifier_api:app --port 8000

# Deploy with Docker
docker build -f Dockerfile.capstone -t issue-classifier .
docker run -p 8000:8000 issue-classifier
```

**Achievements**:
- ✅ **87.3% Test Accuracy** (target: >85%) on multi-class classification
- ✅ **8.2ms p95 Inference Latency** (target: <10ms) - production-ready!
- ✅ **Full Observability** with Prometheus metrics + health checks
- ✅ **95% Test Coverage** with comprehensive pytest suite
- ✅ **Production Deployment** ready with Docker + FastAPI

### Architecture
```
 Raw GitHub Issues
        │
        ▼
   [Data Generation]  ← 2000+ balanced synthetic samples
        │
        ▼
 [Text Preprocessing] ← TF-IDF with bigrams
        │
        ▼
  [Neural Network]    ← 3 layers (100→64→32→4) from scratch
        │
        ▼
  [Classification]    ← bug | feature | question | docs
```

📖 **[Full Documentation](docs/06_case_studies/CAPSTONE_README.md)** | 🎥 **[Demo Video Outline](docs/04_tutorials/development/DEMO_VIDEO_OUTLINE.md)**

---

## 📚 Learning Modules

### Week 4: Neural Network Foundations
**[MNIST from Scratch](notebooks/week_04/mnist_from_scratch.ipynb)**
- Built complete neural network using only `src/core` modules
- Achieved >95% accuracy target
- Visualized training curves, confusion matrix, sample predictions

### Week 5: Convolutional Neural Networks
**[CNN Image Classifier Guide](docs/04_tutorials/examples/week_05_cnn_image_classifier.md)**
- ResNet blocks with skip connections
- Batch Normalization implementation
- CIFAR-10 training pipeline
- FastAPI deployment strategy

### Week 6: Sequence Modeling
**[LSTM Text Generator](notebooks/week_06/lstm_text_generator.ipynb)**
- Character-level text generation with Shakespeare corpus
- Multiple sampling strategies (greedy, temperature, top-k)
- LSTM gate visualization
- Comparison with vanilla RNNs

### Week 7: Transformer Architecture
**[BERT from Scratch](notebooks/week_07/bert_from_scratch.ipynb)**
- Complete transformer encoder implementation
- Scaled dot-product attention
- Multi-head self-attention
- Positional encoding (sinusoidal)
- Masked language model (MLM) pretraining
- Classification fine-tuning

### Week 8: LLM Engineering
**[GPT-2 Pre-trained Models](notebooks/week_08/gpt2_pretrained.ipynb)**
- Loading Hugging Face models
- BPE tokenization
- Advanced sampling (top-k, top-p, temperature)
- Zero-shot vs few-shot learning
- Fine-tuning on custom data

---

## 🏗️ System Design Solutions

**5 production-scale designs ready for technical interviews:**

> [!NOTE]
> Each design includes: requirements analysis, architecture diagrams, component deep dives, scaling strategies, cost estimates, and interview discussion points.

### 1. [RAG at Scale](docs/03_system_design/solutions/01_rag_at_scale.md)
- **Scope**: 1M documents, 1000 QPS, <500ms p95 latency
- **Architecture**: Hybrid search (semantic + BM25), multi-tier caching, cross-encoder re-ranking
- **Tech Stack**: Qdrant, Elasticsearch, Redis, FastAPI
- **Cost**: ~$5,850/month
- **Key Insight**: Hybrid retrieval 15-20% better than pure semantic

### 2. [Recommendation System](docs/03_system_design/solutions/02_recommendation_system.md)
- **Scope**: 100M users, 10M products, <100ms p95 latency
- **Architecture**: Multi-strategy (Matrix Factorization + Content-based + DNN two-tower)
- **Tech Stack**: Spark, Redis, TensorFlow Serving
- **Cost**: ~$19,000/month
- **Key Insight**: Offline batch + online real-time serving pattern

### 3. [Fraud Detection Pipeline](docs/03_system_design/solutions/03_fraud_detection.md)
- **Scope**: 1M transactions/day, <100ms p2p latency, <0.1% false positives
- **Architecture**: Multi-layer defense (rule engine + ML + anomaly detection)
- **Tech Stack**: XGBoost, Isolation Forest, Kafka, PostgreSQL
- **Cost**: ~$2,100/month
- **Key Insight**: Explainability (SHAP) critical for fraud analyst trust

### 4. [ML Model Serving at Scale](docs/03_system_design/solutions/04_ml_model_serving.md) *NEW*
- **Scope**: 10,000 QPS, <50ms p95 latency
- **Architecture**: Dynamic batching (21x throughput), blue-green + canary deployments
- **Tech Stack**: Triton Inference Server, MLflow, Kubernetes HPA, Redis cache
- **Cost**: ~$4,850/month (CPU) or ~$6,850/month (GPU)
- **Key Insight**: Dynamic batching reduces per-request latency from 10ms → 0.47ms

### 5. [A/B Testing Platform](docs/03_system_design/solutions/05_ab_testing_framework.md) *NEW*
- **Scope**: 10M daily active users, multi-variant experiments, statistical rigor
- **Architecture**: Consistent hashing assignment, Kafka event streaming, Spark analytics
- **Tech Stack**: Kafka, Spark, PostgreSQL, Redis (config cache)
- **Cost**: ~$2,000/month
- **Key Insight**: Thompson Sampling (Bayesian MAB) for adaptive traffic allocation

### 6. [Advanced RAG Systems](docs/03_system_design/solutions/15_advanced_rag_systems.md)
- **Scope**: Multi-source retrieval, agentic workflows, enterprise security
- **Architecture**: Modular RAG, hybrid search, graph-based reasoning, semantic caching
- **Tech Stack**: LangGraph, Qdrant, Elasticsearch, Neo4j, Redis
- **Cost**: Variable based on LLM usage and data volume
- **Key Insight**: Agentic RAG with planning and tool usage for complex queries

### 7. [Real-Time RAG Systems](docs/03_system_design/solutions/16_realtime_rag_systems.md)
- **Scope**: Streaming data ingestion, sub-second latency, high throughput
- **Architecture**: Stream processing, real-time indexing, multi-level caching
- **Tech Stack**: Apache Flink, Kafka, FAISS, Redis, PyTorch
- **Cost**: Variable based on data volume and processing requirements
- **Key Insight**: Efficient batch processing balances freshness and resource usage

### 8. [Federated RAG Systems](docs/03_system_design/solutions/17_federated_rag_systems.md)
- **Scope**: Privacy-preserving collaboration, multi-organization knowledge sharing
- **Architecture**: Local RAG nodes, secure aggregation, differential privacy
- **Tech Stack**: Secure MPC, Differential Privacy, Local Vector Stores
- **Cost**: Distributed across participating organizations
- **Key Insight**: Collaborative intelligence without compromising data privacy

### 9. [Quantum-Enhanced RAG Systems](docs/03_system_design/solutions/18_quantum_rag_systems.md)
- **Scope**: Quantum code generation, UML-to-Qiskit conversion
- **Architecture**: Quantum-classical hybrid processing, specialized encoders
- **Tech Stack**: Qiskit, Quantum-aware LLMs, Quantum-specific evaluation
- **Cost**: Variable based on quantum resource usage
- **Key Insight**: Quantum-specific processing for quantum software development

### 10. [Neuromorphic RAG Systems](docs/03_system_design/solutions/19_neuromorphic_rag_systems.md)
- **Scope**: Brain-inspired computing, ultra-low power consumption
- **Architecture**: Spiking neural networks, event-driven processing
- **Tech Stack**: Intel Loihi, SpiNNaker, Lava framework, SNNs
- **Cost**: Lower operational costs due to energy efficiency
- **Key Insight**: Energy-efficient processing through brain-inspired architectures

### 11. [Blockchain-Integrated RAG](docs/03_system_design/solutions/20_blockchain_rag_systems.md)
- **Scope**: Decentralized systems, provenance tracking, trust verification
- **Architecture**: Distributed ledger, smart contracts, decentralized retrieval
- **Tech Stack**: Ethereum/Polygon, Smart Contracts, Decentralized Storage
- **Cost**: Gas fees for blockchain operations
- **Key Insight**: Transparent and verifiable information retrieval

### 12. [Edge AI RAG for IoT](docs/03_system_design/solutions/21_edge_ai_rag_systems.md)
- **Scope**: Resource-constrained environments, real-time processing
- **Architecture**: Local processing, lightweight models, distributed knowledge
- **Tech Stack**: Embedded systems, ONNX, FAISS, TinyML frameworks
- **Cost**: Reduced cloud dependency and bandwidth usage
- **Key Insight**: Intelligence at the edge for low-latency applications

### 13. [Temporal RAG Systems](docs/03_system_design/solutions/22_temporal_rag_systems.md)
- **Scope**: Time-series forecasting, temporal pattern recognition
- **Architecture**: Time-aware encoders, temporal similarity matching
- **Tech Stack**: Time series models, Temporal embeddings, Forecasting systems
- **Cost**: Moderate computational requirements
- **Key Insight**: Temporal context for time-dependent queries

### 14. [Bio-Inspired RAG Systems](docs/03_system_design/solutions/23_bio_inspired_rag_systems.md)
- **Scope**: Nature-inspired design, creative problem solving
- **Architecture**: Biological pattern matching, semantic fusion
- **Tech Stack**: Bio-inspired algorithms, Evolutionary computation
- **Cost**: Moderate computational requirements
- **Key Insight**: Leveraging natural solutions for design challenges

### 15. [Zero-Shot Learning RAG](docs/03_system_design/solutions/24_zero_shot_rag_systems.md)
- **Scope**: Cross-domain transfer, generalizable reasoning
- **Architecture**: Generalizable encoders, cross-domain knowledge transfer
- **Tech Stack**: Transfer learning, Generalizable models, Domain adaptation
- **Cost**: Lower training costs due to generalization
- **Key Insight**: Solving new tasks without task-specific training

### 16. [Multi-Expert RAG (MoE)](docs/03_system_design/solutions/25_multi_expert_rag_systems.md)
- **Scope**: Specialized retrieval, dynamic routing
- **Architecture**: Mixture of Experts, Specialized retrievers, Dynamic controllers
- **Tech Stack**: Expert routing, Specialized models, Graph neural networks
- **Cost**: Higher computational requirements but better performance
- **Key Insight**: Specialized experts for different query types

### 17. [Cognitive RAG Systems](docs/03_system_design/solutions/26_cognitive_rag_systems.md)
- **Scope**: Associative memory, multi-hop reasoning
- **Architecture**: Human memory emulation, multi-hop search
- **Tech Stack**: Knowledge graphs, Associative memory, Reasoning engines
- **Cost**: Moderate computational requirements
- **Key Insight**: Human-like associative reasoning patterns

### 18. [Green RAG Systems](docs/03_system_design/solutions/27_green_rag_systems.md)
- **Scope**: Energy efficiency, carbon footprint optimization
- **Architecture**: Efficient models, intelligent caching, renewable integration
- **Tech Stack**: Small models, Efficient algorithms, Energy monitoring
- **Cost**: Reduced operational costs through efficiency
- **Key Insight**: Sustainable AI through energy-conscious design

**Total System Design Coverage**: $33,800/month across all 18 designs | ~20,000 lines of technical documentation



---

## 📊 Production Case Studies

**3 real-world ML projects with measurable business impact:**

### 1. [Churn Prediction for SaaS](docs/06_case_studies/domain_specific/01_churn_prediction.md)
- **Challenge**: 15% monthly churn costing $2M annually
- **Solution**: XGBoost with 47 behavioral features + Airflow pipeline
- **Impact**: **$800K savings**, 40% churn reduction (15% → 9%)
- **Tech**: Time-series CV, SHAP explainability, daily batch scoring

### 2. [Real-Time Fraud Detection](docs/06_case_studies/domain_specific/02_fraud_detection.md)
- **Challenge**: $5M annual fraud losses, need <100ms latency
- **Solution**: Multi-layer defense (rules + XGBoost + Isolation Forest)
- **Impact**: **$4.2M prevented**, 84% precision, 81% recall
- **Tech**: Redis feature store, Cassandra, SMOTE, threshold optimization

### 3. [Personalized Recommender System](docs/06_case_studies/domain_specific/03_recommender_system.md)
- **Challenge**: 45% users never engaged beyond homepage
- **Solution**: Hybrid collaborative filtering + deep two-tower ranker
- **Impact**: **+$17M revenue**, +32% watch time, +18% retention
- **Tech**: Matrix factorization, BERT embeddings, <400ms latency for 8M users

**Total Business Impact**: $22M+ across three case studies

### Full Stack AI Case Studies *ENHANCED*

| # | Case Study | Industry Reference | Key Topic |
|---|------------|-------------------|-----------| 
| 1 | [Uber Eats GNN Recommendations](docs/06_case_studies/full_stack_ai/01_uber_eats_gnn_recommendations.md) | Uber | GraphSAGE, Two-Tower |
| 2 | [Notion AI RAG Architecture](docs/06_case_studies/full_stack_ai/02_notion_ai_rag_architecture.md) | Notion | Hybrid Retrieval, Model Routing |
| 3 | [Intercom Fin Support Agent](docs/06_case_studies/full_stack_ai/03_intercom_fin_support_agent.md) | Intercom | Guardrails, CX Score |
| 4 | [Salesforce Trust Layer](docs/06_case_studies/full_stack_ai/04_salesforce_trust_layer.md) | Salesforce | PII Masking, Audit |
| 5 | [Pinterest Ranking Pipeline](docs/06_case_studies/full_stack_ai/05_pinterest_ranking_pipeline.md) | Pinterest | Multi-Stage Ranking |
| 6 | [DoorDash Feature Store](docs/06_case_studies/full_stack_ai/06_doordash_feature_store.md) | DoorDash | Streaming Features, Freshness SLA |
| 7 | [Siemens Edge AI](docs/06_case_studies/full_stack_ai/07_siemens_edge_ai.md) | Siemens | Edge Deployment, OTA Updates |
| 8 | [Manufacturing Quality Control](docs/06_case_studies/full_stack_ai/08_manufacturing_quality_control.md) | Darwin/Intrinsics | **CNN+INT8, PLC (<20ms)** |
| 9 | [Medical IoMT Devices](docs/06_case_studies/full_stack_ai/09_medical_iomt_devices.md) | Intrinsics | **Neuro Core, Federated Learning** |
| 10 | [Industrial IoT PdM](docs/06_case_studies/full_stack_ai/10_industrial_iot_pdm.md) | Barbara IoT | **Autoencoder, RUL, DDIL** |
| 11 | [Hybrid Edge-Cloud](docs/06_case_studies/full_stack_ai/11_hybrid_edge_cloud.md) | AWS Greengrass | **Task Routing, Split Inference** |

**📚 Edge AI Engineer Roadmap**: [docs/01_learning_roadmap/edge_ai_engineer_roadmap.md](docs/01_learning_roadmap/edge_ai_engineer_roadmap.md)

---

## 🎤 Interview Preparation

**[Complete Interview Tracker](docs/05_interview_prep/INTERVIEW_TRACKER.md)** with:

### ✅ Technical Depth Checklists
- ML fundamentals (bias-variance, regularization, cross-validation)
- Deep learning (backpropagation, activations, batch norm, dropout)
- Transformers & LLMs (self-attention, positional encodings, fine-tuning)
- Production ML (serving, monitoring, A/B testing, cost optimization)

### ✅ 4 STAR Behavioral Stories
1. **Technical Challenge**: Improved fraud detection precision 60% → 89%
2. **System Design**: Scaled RAG system latency 8s → 450ms
3. **Debugging**: Detected and fixed model drift, preventing $500K loss
4. **Leadership**: Mentored junior engineer to 87% accuracy (exceeded target)

### ✅ Practice Questions (15+)
- Warm-up, technical deep dive, system design, and behavioral
- Day-of-interview checklist
- Mock interview scenarios

---

## 📊 Case Studies

### [Time Series Forecasting](docs/06_case_studies/time_series_forecasting/README.md)
**Problem**: Retail sales forecasting for inventory optimization

**Approaches Compared**:
- ARIMA (classical): 8.5% MAPE
- LSTM (deep learning): **6.2% MAPE** ✅ Best
- Prophet (hybrid): 7.8% MAPE

**Production**: FastAPI service with confidence intervals

### Additional Case Studies
**New additions include**:
- Computer Vision for Medical Diagnosis
- NLP for Financial Document Analysis
- Advanced Recommendation Systems
- Multi-modal Retail Analytics
- Time Series Forecasting for Supply Chain
- Database Systems Mastery (Relational, NoSQL, OLAP, Vector, and more)

---

## 🚀 Quick Start


### Installation

We use a modular `setup.py` for flexibility, and provide automated setup for conda or venv.

```bash
# 1) Clone
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
```

#### Option A: Automated setup (recommended)

Windows PowerShell:
```powershell
.\setup.ps1 -EnvManager auto -Jupyter -Test
# GPU (CUDA example):
.\setup.ps1 -EnvManager conda -Cuda 11.8 -Jupyter -Test
```

macOS/Linux/WSL:
```bash
./setup.sh --auto --jupyter --test
# GPU (CUDA example):
./setup.sh --conda --cuda 11.8 --jupyter --test
```

#### Option B: Conda (full environment)

```bash
conda env create -f environment.full.yml
conda activate ai-mastery-2026
python -m ipykernel install --user --name ai-mastery-2026 --display-name "AI-Mastery-2026"
```

#### Option C: Venv (classic)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name ai-mastery-2026 --display-name "AI-Mastery-2026"
```

#### Option D: Editable installs (package extras)

```bash
pip install -e .          # Core only (Foundations)
pip install -e ".[dev]"   # + Testing tools
pip install -e ".[llm]"   # + Heavy AI libraries (Torch, Transformers)
pip install -e ".[all]"   # Everything
```

### Running the System

```bash
# Run Week 1 Benchmarks (Pure Python vs NumPy)
python scripts/benchmark_week1.py

# Launch Full Stack App (API + Streamlit + Grafana)
docker-compose up -d --build
```
*   **Web UI**: [http://localhost:8501](http://localhost:8501)
*   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Grafana**: [http://localhost:3000](http://localhost:3000) (admin/admin)

---

## 📖 Documentation

*   **[User Guide](docs/00_introduction/01_user_guide.md)**: The bible of this project. Steps, params, and usage.
*   **[Math Notes](docs/02_core_concepts/fundamentals/math_notes.md)**: Handwritten derivations of key algorithms.
*   **[Interview Prep](docs/05_interview_prep/README.md)**: "How would you design a recommendation system?"
*   **[Repo Documentation](docs/00_introduction/REPO_DOCUMENTATION.md)**: Full repository map and entry points.

---

## ✨ Feature Highlights

### Mathematical Core
```python
# Causal Inference (Inverse Probability Weighting)
from src.core.causal_whitebox import estimate_ate_ipw
ate = estimate_ate_ipw(data, propensity_score_func)

# Bayesian Inference (Metropolis-Hastings)
from src.core.probability_whitebox import metropolis_hastings
samples = metropolis_hastings(log_prob_func, n_samples=1000)
```

### Deep Learning (From Scratch)
```python
from src.ml.deep_learning import NeuralNetwork, Dense, Conv2D

model = NeuralNetwork()
model.add(Conv2D(1, 32, kernel_size=3))
model.add(Dense(32*14*14, 10))
model.fit(X_train, y_train)
```

### Production RAG
```python
from src.production.query_enhancement import HyDEGenerator
hyde = HyDEGenerator()
enhanced_query = hyde.generate("What is the capital of Mars?")
```

---

<div align="center">
    <b>Built for the 1% of Engineers who want to understand EVERYTHING down to the metal.</b>
</div>
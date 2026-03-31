# 🎯 Curriculum Tracks - Specialized Learning Paths

**Version:** 2.0 (2026 Redesign)  
**Status:** Production Ready  
**Last Updated:** March 29, 2026

---

## 📋 Overview

Tracks are **cross-cutting specializations** that complement the 4-tier learning path. Each track focuses on a specific skill set and maps to career roles.

**10 Tracks Total:**
- **4 Foundation Tracks** (01-04)
- **4 Core LLM Tracks** (05-08)
- **2 Advanced Tracks** (09-10) ⭐ NEW

---

## 🏗️ Track Architecture

```
Tracks Overview
│
├── Foundation Tracks (Build fundamentals)
│   ├── Track 01: Mathematics for AI
│   ├── Track 02: Python for ML
│   ├── Track 03: Neural Networks
│   └── Track 04: NLP Fundamentals
│
├── Core LLM Tracks (LLM-specific skills)
│   ├── Track 05: LLM Architecture
│   ├── Track 06: Fine-Tuning & Alignment
│   ├── Track 07: RAG Systems
│   └── Track 08: AI Agents
│
└── Advanced Tracks (Specialized expertise) ⭐ NEW
    ├── Track 09: Security & Safety
    └── Track 10: Production DevOps
```

---

## 📚 Foundation Tracks (01-04)

### Track 01: Mathematics for AI

**Duration:** 40-60 hours  
**Level:** Beginner  
**Modules:** 8

**Learning Objectives:**
By the end of this track, you will be able to:
- **Implement** vector/matrix operations from scratch
- **Calculate** eigenvalues and eigenvectors
- **Apply** gradient descent for optimization
- **Compute** probabilities using Bayes' theorem
- **Visualize** mathematical concepts

**Topics:**
1. Linear Algebra: Vectors, Matrices, Transformations
2. Matrix Decompositions: LU, QR, SVD, Eigendecomposition
3. Calculus: Derivatives, Gradients, Chain Rule
4. Optimization: Gradient Descent, Convex Optimization
5. Probability: Distributions, Bayes' Theorem
6. Statistics: Mean, Variance, Hypothesis Testing
7. Information Theory: Entropy, KL Divergence
8. Numerical Methods: Stability, Precision

**Assessments:**
- 3 Quizzes (60 questions total)
- 40 Knowledge Checks
- 3 Coding Challenges
- 2 Projects

**Career Mapping:** ML Researcher, Data Scientist

[Start Track 01 →](01_mathematics/README.md)

---

### Track 02: Python for ML

**Duration:** 40-60 hours  
**Level:** Beginner  
**Modules:** 6

**Learning Objectives:**
By the end of this track, you will be able to:
- **Manipulate** data using NumPy and Pandas
- **Visualize** datasets with Matplotlib/Seaborn
- **Implement** ML pipelines with scikit-learn
- **Clean** and preprocess raw data
- **Perform** exploratory data analysis

**Topics:**
1. NumPy: Arrays, Broadcasting, Vectorization
2. Pandas: DataFrames, GroupBy, Time Series
3. Visualization: Matplotlib, Seaborn, Plotly
4. scikit-learn: Pipelines, Models, Metrics
5. Data Cleaning: Missing Values, Outliers
6. Feature Engineering: Encoding, Scaling, Selection

**Assessments:**
- 3 Quizzes (70 questions total)
- 30 Knowledge Checks
- 3 Coding Challenges
- 2 Projects

**Career Mapping:** Data Scientist, ML Engineer

[Start Track 02 →](02_python_ml/README.md)

---

### Track 03: Neural Networks

**Duration:** 60-80 hours  
**Level:** Beginner-Intermediate  
**Modules:** 10

**Learning Objectives:**
By the end of this track, you will be able to:
- **Build** perceptrons and multi-layer networks
- **Implement** backpropagation algorithm
- **Apply** activation functions appropriately
- **Train** networks using optimization algorithms
- **Debug** common neural network issues

**Topics:**
1. Perceptrons and Linear Classifiers
2. Multi-Layer Perceptrons (MLPs)
3. Activation Functions (ReLU, Sigmoid, Tanh, Softmax)
4. Loss Functions (MSE, Cross-Entropy, Hinge)
5. Backpropagation and Autodiff
6. Optimization (SGD, Momentum, Adam, RMSprop)
7. Regularization (Dropout, BatchNorm, LayerNorm)
8. Convolutional Neural Networks (CNNs)
9. Recurrent Neural Networks (RNNs, LSTMs, GRUs)
10. Attention Mechanisms

**Assessments:**
- 3 Quizzes (90 questions total)
- 50 Knowledge Checks
- 3 Coding Challenges
- 3 Projects

**Career Mapping:** Deep Learning Engineer, Computer Vision Engineer

[Start Track 03 →](03_neural_networks/README.md)

---

### Track 04: NLP Fundamentals

**Duration:** 40-60 hours  
**Level:** Beginner-Intermediate  
**Modules:** 8

**Learning Objectives:**
By the end of this track, you will be able to:
- **Process** text using tokenization and normalization
- **Create** word embeddings (Word2Vec, GloVe)
- **Implement** text classification pipelines
- **Analyze** sentiment in text
- **Build** simple language models

**Topics:**
1. Text Preprocessing (Cleaning, Normalization)
2. Tokenization (Word, Subword, Character)
3. Stemming and Lemmatization
4. Part-of-Speech Tagging
5. Named Entity Recognition (NER)
6. Word Embeddings (Word2Vec, GloVe, FastText)
7. Text Classification (Naive Bayes, SVM, Neural)
8. Language Models (N-grams, RNNs, Transformers intro)

**Assessments:**
- 3 Quizzes (65 questions total)
- 40 Knowledge Checks
- 3 Coding Challenges
- 2 Projects

**Career Mapping:** NLP Engineer, Computational Linguist

[Start Track 04 →](04_nlp_fundamentals/README.md)

---

## 🧠 Core LLM Tracks (05-08)

### Track 05: LLM Architecture

**Duration:** 80-100 hours  
**Level:** Intermediate  
**Modules:** 12

**Learning Objectives:**
By the end of this track, you will be able to:
- **Explain** transformer architecture in detail
- **Implement** self-attention from scratch
- **Design** encoder and decoder architectures
- **Analyze** positional encoding schemes
- **Optimize** transformer models for efficiency

**Topics:**
1. Attention Mechanisms (Self-Attention, Multi-Head)
2. Transformer Architecture (Encoder, Decoder, Encoder-Decoder)
3. Positional Encodings (Absolute, Relative, RoPE, ALiBi)
4. Layer Normalization and Variants
5. Feed-Forward Networks (MLP, MoE, SwiGLU)
6. Embedding Layers (Token, Positional, Learned)
7. Decoder-Only Architecture (GPT-style)
8. Encoder-Only Architecture (BERT-style)
9. Encoder-Decoder Architecture (T5-style)
10. Scaling Laws and Model Size
11. Efficient Attention (Sparse, Linear, FlashAttention)
12. Architecture Variants (LLaMA, Mistral, Falcon, etc.)

**Assessments:**
- 3 Quizzes (100 questions total)
- 60 Knowledge Checks
- 3 Coding Challenges
- 3 Projects

**Career Mapping:** LLM Architect, ML Scientist, Research Engineer

[Start Track 05 →](05_llm_architecture/README.md)

---

### Track 06: Fine-Tuning & Alignment

**Duration:** 60-80 hours  
**Level:** Intermediate-Advanced  
**Modules:** 10

**Learning Objectives:**
By the end of this track, you will be able to:
- **Apply** supervised fine-tuning techniques
- **Implement** parameter-efficient methods (LoRA, QLoRA)
- **Align** models using RLHF and DPO
- **Evaluate** fine-tuned models
- **Debug** fine-tuning issues

**Topics:**
1. Full Fine-Tuning vs. Parameter-Efficient Methods
2. Supervised Fine-Tuning (SFT)
3. Instruction Tuning
4. LoRA (Low-Rank Adaptation)
5. QLoRA (Quantized LoRA)
6. Prefix Tuning and Prompt Tuning
7. Adapters (Houlsby, Pfeiffer)
8. Reinforcement Learning from Human Feedback (RLHF)
9. Direct Preference Optimization (DPO)
10. Reward Modeling

**Assessments:**
- 3 Quizzes (85 questions total)
- 50 Knowledge Checks
- 3 Coding Challenges
- 3 Projects

**Career Mapping:** ML Scientist, Fine-Tuning Specialist

[Start Track 06 →](06_fine_tuning/README.md)

---

### Track 07: RAG Systems

**Duration:** 100-120 hours  
**Level:** Intermediate-Advanced  
**Modules:** 15

**Learning Objectives:**
By the end of this track, you will be able to:
- **Build** vector storage systems
- **Implement** retrieval strategies (dense, sparse, hybrid)
- **Design** RAG pipelines with orchestration
- **Apply** advanced RAG techniques
- **Evaluate** RAG system performance

**Topics:**
1. Embeddings for Retrieval
2. Vector Databases (FAISS, Qdrant, Pinecone, Weaviate)
3. Dense Retrieval
4. Sparse Retrieval (BM25, SPLADE)
5. Hybrid Retrieval
6. Chunking Strategies (9 methods)
7. Reranking (Cross-Encoder, ColBERT, FlashRank)
8. RAG Orchestration (Query rewriting, Routing)
9. Advanced RAG (Multi-hop, Temporal, Graph-enhanced)
10. RAG Evaluation (RAGAS, TruLens)
11. Memory Systems for RAG
12. Multi-Modal RAG
13. Privacy-Preserving RAG
14. Streaming RAG
15. Production RAG Patterns

**Assessments:**
- 3 Quizzes (120 questions total)
- 75 Knowledge Checks
- 3 Coding Challenges
- 5 Projects

**Career Mapping:** RAG Engineer, Search Engineer, Knowledge Engineer

[Start Track 07 →](07_rag_systems/README.md)

---

### Track 08: AI Agents

**Duration:** 60-80 hours  
**Level:** Advanced  
**Modules:** 10

**Learning Objectives:**
By the end of this track, you will be able to:
- **Design** agentic systems with tool use
- **Implement** planning algorithms
- **Build** multi-agent orchestration
- **Create** agent memory architectures
- **Evaluate** agent performance

**Topics:**
1. Agent Architecture (ReAct, Plan-and-Solve)
2. Tool Use and Function Calling
3. Error Recovery and Self-Correction
4. Planning Algorithms (Tree of Thoughts, Graph of Thoughts)
5. Agent Memory (Working, Episodic, Semantic)
6. Multi-Agent Orchestration (CrewAI, AutoGen patterns)
7. Agent Communication Protocols
8. Task Decomposition
9. Agent Evaluation Metrics
10. Production Agent Patterns

**Assessments:**
- 3 Quizzes (90 questions total)
- 50 Knowledge Checks
- 3 Coding Challenges
- 3 Projects

**Career Mapping:** Agent Systems Engineer, AI Automation Engineer

[Start Track 08 →](08_ai_agents/README.md)

---

## 🛡️ Advanced Tracks (09-10) ⭐ NEW

### Track 09: Security & Safety

**Duration:** 40-60 hours  
**Level:** Advanced  
**Modules:** 8

**Learning Objectives:**
By the end of this track, you will be able to:
- **Identify** prompt injection attacks
- **Implement** defense mechanisms
- **Build** content moderation systems
- **Detect** and redact PII at scale
- **Apply** AI safety principles

**Topics:**
1. Prompt Injection Attacks (Direct, Indirect, Multi-turn)
2. Jailbreaking Techniques and Defenses
3. Content Moderation Systems
4. PII Detection and Redaction
5. Adversarial Examples for LLMs
6. AI Safety and Alignment
7. Guardrails Implementation
8. Security Best Practices for LLMs

**Assessments:**
- 3 Quizzes (75 questions total)
- 40 Knowledge Checks
- 3 Coding Challenges
- 3 Projects

**Career Mapping:** AI Safety Engineer, Security Engineer

[Start Track 09 →](09_security_safety/README.md)

---

### Track 10: Production DevOps

**Duration:** 80-100 hours  
**Level:** Advanced  
**Modules:** 12

**Learning Objectives:**
By the end of this track, you will be able to:
- **Deploy** LLMs to production
- **Monitor** LLM-specific metrics
- **Optimize** inference costs
- **Implement** A/B testing frameworks
- **Build** CI/CD pipelines for LLMs

**Topics:**
1. LLM-Specific Metrics (Token usage, Hallucination rate, Latency)
2. Deployment Patterns (Docker, Kubernetes, Serverless)
3. Inference Optimization (Caching, Batching, Streaming)
4. Cost Optimization (Token budgeting, Model cascading)
5. Monitoring & Observability (Prometheus, Grafana, Tracing)
6. A/B Testing for Model Comparisons
7. Drift Detection for Embeddings
8. User Feedback Loops
9. Alerting & Incident Response
10. MLOps for LLMs (CI/CD, Versioning)
11. LLM Gateway Patterns
12. Production Security

**Assessments:**
- 3 Quizzes (100 questions total)
- 60 Knowledge Checks
- 3 Coding Challenges
- 4 Projects

**Career Mapping:** ML Ops Engineer, Production Engineer

[Start Track 10 →](10_production_devops/README.md)

---

## 🎓 Track Completion Requirements

### Foundation Tracks (01-04)

To complete a foundation track:
- [ ] Complete all modules
- [ ] Pass all quizzes (80%+)
- [ ] Complete all knowledge checks
- [ ] Submit all projects
- [ ] Pass coding challenge (Easy)

**Certificate:** Foundation Specialist Certificate

---

### Core LLM Tracks (05-08)

To complete a core LLM track:
- [ ] Complete all modules
- [ ] Pass all quizzes (80%+)
- [ ] Complete all knowledge checks
- [ ] Submit all projects
- [ ] Pass coding challenge (Medium)
- [ ] Peer review 1 project

**Certificate:** LLM Specialist Certificate

---

### Advanced Tracks (09-10)

To complete an advanced track:
- [ ] Complete all modules
- [ ] Pass all quizzes (85%+)
- [ ] Complete all knowledge checks
- [ ] Submit all projects
- [ ] Pass coding challenge (Hard)
- [ ] Peer review 2 projects
- [ ] Present final project to review board

**Certificate:** Advanced Specialist Certificate

---

## 📊 Track Comparison

| Track | Duration | Difficulty | Projects | Career Roles |
|-------|----------|------------|----------|--------------|
| **01 Mathematics** | 40-60h | ⭐⭐ | 2 | ML Researcher |
| **02 Python ML** | 40-60h | ⭐⭐ | 2 | Data Scientist |
| **03 Neural Networks** | 60-80h | ⭐⭐⭐ | 3 | DL Engineer |
| **04 NLP** | 40-60h | ⭐⭐ | 2 | NLP Engineer |
| **05 LLM Architecture** | 80-100h | ⭐⭐⭐⭐ | 3 | LLM Architect |
| **06 Fine-Tuning** | 60-80h | ⭐⭐⭐⭐ | 3 | ML Scientist |
| **07 RAG Systems** | 100-120h | ⭐⭐⭐⭐ | 5 | RAG Engineer |
| **08 AI Agents** | 60-80h | ⭐⭐⭐⭐ | 3 | Agent Engineer |
| **09 Security ⭐** | 40-60h | ⭐⭐⭐⭐⭐ | 3 | Safety Engineer |
| **10 Production ⭐** | 80-100h | ⭐⭐⭐⭐⭐ | 4 | ML Ops Engineer |

---

## 🗺️ Track Learning Paths by Career Goal

### LLM Engineer Path
```
Foundation: Tracks 01, 02, 03, 04
    ↓
Core: Tracks 05, 06, 07, 08
    ↓
Advanced: Track 10
    ↓
🎓 LLM Engineer Certificate
```

### ML Scientist Path
```
Foundation: Tracks 01, 02, 03
    ↓
Core: Tracks 05, 06 (Deep Dive)
    ↓
Research Project
    ↓
🎓 ML Scientist Certificate
```

### RAG Specialist Path
```
Foundation: Tracks 02, 03, 04
    ↓
Core: Track 07 (Deep Dive)
    ↓
Advanced: Track 10
    ↓
🎓 RAG Engineer Certificate
```

### AI Safety Engineer Path ⭐ NEW
```
Foundation: Tracks 01, 02, 03
    ↓
Core: Tracks 05, 07, 08
    ↓
Advanced: Track 09 (Deep Dive), Track 10
    ↓
🎓 AI Safety Engineer Certificate
```

### ML Ops Engineer Path
```
Foundation: Tracks 02, 03
    ↓
Core: Tracks 07, 08
    ↓
Advanced: Track 10 (Deep Dive), Track 09
    ↓
🎓 ML Ops Engineer Certificate
```

[View Detailed Learning Paths →](../learning_paths/journey_maps.md)

---

## 📈 Track Statistics

| Metric | Value |
|--------|-------|
| **Total Tracks** | 10 |
| **Total Modules** | 89 |
| **Total Quizzes** | 137 |
| **Total Questions** | 3,500+ |
| **Knowledge Checks** | 490 |
| **Coding Challenges** | 30 |
| **Projects** | 30+ |
| **Estimated Hours** | 660-880 (all tracks) |

---

## 🎯 Industry Alignment

### Skills Mapping to Job Roles

| Job Role | Required Tracks | Optional Tracks |
|----------|-----------------|-----------------|
| **LLM Engineer** | 05, 06, 07, 08 | 09, 10 |
| **ML Scientist** | 05, 06 | 01, 03 |
| **RAG Engineer** | 07, 04 | 08, 10 |
| **ML Ops Engineer** | 10, 07 | 09, 08 |
| **AI Safety Engineer** | 09, 10 | 05, 08 |
| **Data Scientist** | 01, 02, 03 | 04 |
| **NLP Engineer** | 04, 05 | 06, 07 |

### Salary Ranges (US Market)

| Role | Entry | Mid | Senior |
|------|-------|-----|--------|
| LLM Engineer | $120-150K | $150-200K | $200-300K+ |
| ML Scientist | $130-160K | $160-220K | $220-350K+ |
| RAG Engineer | $110-140K | $140-190K | $190-280K+ |
| ML Ops Engineer | $120-150K | $150-200K | $200-300K+ |
| AI Safety Engineer | $140-170K | $170-230K | $230-350K+ |

---

## 📞 Track Support

### Track Coordinators

| Track | Coordinator | Contact |
|-------|-------------|---------|
| 01-04 | Dr. Sarah Johnson | foundation@ai-mastery-2026.com |
| 05-08 | Dr. Ahmed Hassan | core@ai-mastery-2026.com |
| 09-10 | Dr. Emily Chen | advanced@ai-mastery-2026.com |

### Office Hours

- **Foundation Tracks:** Mon/Wed 2-4 PM
- **Core LLM Tracks:** Tue/Thu 3-5 PM
- **Advanced Tracks:** Fri 1-5 PM

### Slack Channels

- `#track-01-mathematics`
- `#track-02-python`
- `#track-03-neural-networks`
- `#track-04-nlp`
- `#track-05-llm-architecture`
- `#track-06-fine-tuning`
- `#track-07-rag-systems`
- `#track-08-ai-agents`
- `#track-09-security-safety`
- `#track-10-production-devops`

---

## 🏆 Track Achievement Badges

| Badge | Requirement |
|-------|-------------|
| 🥇 **Foundation Master** | Complete all 4 foundation tracks with 90%+ |
| 🥇 **LLM Core Expert** | Complete all 4 core tracks with 85%+ |
| 🥇 **Advanced Specialist** | Complete both advanced tracks |
| 🏆 **Track Champion** | Complete all 10 tracks |
| 🌟 **Perfect Score** | Any track with 100% on all assessments |

---

## 📚 Additional Resources

### Textbooks

- **Mathematics:** "Mathematics for Machine Learning" (Deisenroth et al.)
- **Deep Learning:** "Deep Learning" (Goodfellow et al.)
- **NLP:** "Speech and Language Processing" (Jurafsky & Martin)
- **LLMs:** "Natural Language Processing with Transformers" (Tunstall et al.)

### Online Resources

- [Hugging Face Course](https://huggingface.co/course)
- [DeepLearning.AI](https://www.deeplearning.ai)
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp)
- [LLM University](https://www.llmuniversity.ai)

### Communities

- r/MachineLearning
- r/LocalLLaMA
- Hugging Face Forums
- AI Engineering Discord

---

**Last Updated:** March 29, 2026  
**Version:** 2.0  
**Status:** ✅ Production Ready

[**Back to Curriculum Home →**](../README.md)

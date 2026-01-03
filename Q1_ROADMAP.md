# Q1 2026 Roadmap: Foundations & Core Architecture

**Theme:** "White-Box Mastery" – Build everything from scratch, then productionize.
**Goal:** Land a Senior AI Engineer role by building a portfolio of deep technical understanding.

---

## 📅 Block 1: The Mathematician's Forge (Weeks 1-4)
**Focus:** Linear Algebra, Calculus, Optimization, Classical ML from scratch.
**Shippable Artifact:** `src/core` Python Library (published to PyPI or strongly documented).

- **Week 1: Foundations & Linear Algebra**
    - *Concept:* Vectors, matrices, eigenvalues without NumPy (initially), then optimized usage.
    - *Code:* `src/core/math_operations.py`, `src/core/linear_algebra.py`.
    - *Deliverable:* A "Numpy-lite" implementation and a benchmark script comparing it to real NumPy.

- **Week 2: Optimization & Probability**
    - *Concept:* Gradients, SGD, Adam, Lagrange Multipliers.
    - *Code:* `src/core/optimization.py` (Adam/RMSProp from scratch).
    - *Deliverable:* Interactive visualization of convergence rates for different optimizers.

- **Week 3: Classical Machine Learning**
    - *Concept:* Regression, SVM, Trees, Integration (Monte Carlo).
    - *Code:* `src/ml/classical.py` (implementing `fit`/`predict` patterns).
    - *Deliverable:* A simplified Scikit-Learn clone (`ai-mastery-sklearn`).

- **Week 4: Neural Foundations**
    - *Concept:* Backpropagation, Dense Layers, Activations.
    - *Code:* `src/ml/deep_learning.py` (The "MicroGrad" moment).
    - *Deliverable:* Training a neural network to solve MNIST using only `src/core`.

---

## 📅 Block 2: Deep Learning & Transformers (Weeks 5-8)
**Focus:** Architecture internals, Attention mechanisms, and Efficient Training.
**Shippable Artifact:** A "Mini-Llama" training/inference script.

- **Week 5: CNNs & Computer Vision**
    - *Concept:* Convolutions, kernels, pooling, ResNets.
    - *Code:* `src/ml/vision.py`.
    - *Deliverable:* An image classifier API using `src/production/api.py`.

- **Week 6: Sequences & RNNs**
    - *Concept:* RNNs, LSTMs, Vanishing gradients.
    - *Code:* `src/ml/sequence.py`.
    - *Deliverable:* A stock price predictor or text generator.

- **Week 7: The Transformer Implementation**
    - *Concept:* Self-Attention, Multi-Head, Positional Encodings.
    - *Code:* `src/llm/attention.py`, `src/llm/transformer.py`.
    - *Deliverable:* A "Build BERT from scratch" notebook.

- **Week 8: LLM Engineering**
    - *Concept:* Tokenization, RoPE, KV-Cache.
    - *Code:* `src/llm/model.py`.
    - *Deliverable:* A script that loads pre-trained GPT-2 weights into your custom architecture.

---

## 📅 Block 3: RAG & Production Systems (Weeks 9-12)
**Focus:** Retrieval, Vector DBs, MLOps, and Deployment.
**Shippable Artifact:** End-to-End Enterprise RAG System.

- **Week 9: Vector Search & Embeddings**
    - *Concept:* HNSW, LSH, Contrastive Loss.
    - *Code:* `src/production/vector_db.py`.
    - *Deliverable:* A custom vector database that passes recall tests.

- **Week 10: Advanced RAG Strategies**
    - *Concept:* Hybrid Search, Re-ranking, Context Window management.
    - *Code:* `src/llm/rag.py`.
    - *Deliverable:* RAG Evaluation framework (Faithfulness/Relevance metrics).

- **Week 11: MLOps & Orchestration**
    - *Concept:* Docker, CI/CD, Prometheus, Grafana.
    - *Code:* `.github/workflows/`, `docker-compose.yml`.
    - *Deliverable:* A fully monitored deployment dashboard.

- **Week 12: Capstone & Portfolio Polish**
    - *Concept:* Documentation, Video Demos, Social Proof.
    - *Action:* Clean up `READMEs`, record "Loom" walkthroughs.
    - *Deliverable:* The "AI-Mastery-2026" repository reaches v1.0.

---

## 📈 Key Metrics (Units of Progress)
- **Core Algo:** 1 core algorithm implemented from scratch/week.
- **Paper:** 1 research paper read & summarized code-wise/week.
- **App:** 1 feature shipped to production (local or cloud)/week.

## 🛠 Tech Stack
- **Languages:** Python, SQL, C++ (for extensions later).
- **Libraries:** NumPy (Core), PyTorch (DL), FastAPI (Web), Docker (Ops).

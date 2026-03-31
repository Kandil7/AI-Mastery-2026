# Interactive Tutorial Notebooks

<div align="center">

![Notebooks](https://img.shields.io/badge/notebooks-50+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=jupyter)
![Last Updated](https://img.shields.io/badge/updated-March%2028%2C%202026-green.svg)

**Hands-on interactive notebooks for learning AI engineering**

[Getting Started](#-getting-started) • [By Topic](#-notebooks-by-topic) • [By Level](#-notebooks-by-level) • [Solutions](#-exercises--solutions)

</div>

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install Jupyter and dependencies
pip install jupyterlab notebook
pip install -r requirements.txt

# Or use make
make install-notebooks
```

### Running Notebooks

```bash
# Start JupyterLab
cd notebooks
jupyter lab

# Or classic notebook
jupyter notebook
```

### Notebook Structure

Each notebook includes:
- 📖 **Learning Objectives** - What you'll learn
- 📝 **Theory** - Key concepts explained
- 💻 **Code Examples** - Working implementations
- ✏️ **Exercises** - Hands-on practice
- ✅ **Solutions** - Reference solutions

---

## 📚 Notebooks by Topic

### Mathematical Foundations

| Notebook | Level | Duration | Topics |
|----------|-------|----------|--------|
| [01_linear_algebra.ipynb](01_mathematical_foundations/01_linear_algebra.ipynb) | Beginner | 2h | Vectors, matrices, operations |
| [02_calculus.ipynb](01_mathematical_foundations/02_calculus.ipynb) | Beginner | 2h | Derivatives, gradients, optimization |
| [03_probability.ipynb](01_mathematical_foundations/03_probability.ipynb) | Beginner | 2h | Distributions, Bayes, expectation |
| [04_statistics.ipynb](01_mathematical_foundations/04_statistics.ipynb) | Intermediate | 3h | Hypothesis testing, confidence intervals |

**Learning Outcomes:**
- Understand linear algebra for ML
- Compute gradients and derivatives
- Apply probability theory
- Perform statistical analysis

### Classical Machine Learning

| Notebook | Level | Duration | Topics |
|----------|-------|----------|--------|
| [01_linear_regression.ipynb](02_classical_ml/01_linear_regression.ipynb) | Beginner | 2h | Regression, gradient descent |
| [02_logistic_regression.ipynb](02_classical_ml/02_logistic_regression.ipynb) | Beginner | 2h | Classification, sigmoid, cross-entropy |
| [03_decision_trees.ipynb](02_classical_ml/03_decision_trees.ipynb) | Intermediate | 3h | ID3, C4.5, random forests |
| [04_svm.ipynb](02_classical_ml/04_svm.ipynb) | Intermediate | 3h | Support vector machines, kernels |
| [05_clustering.ipynb](02_classical_ml/05_clustering.ipynb) | Intermediate | 2h | K-means, DBSCAN, hierarchical |
| [06_pca.ipynb](02_classical_ml/06_pca.ipynb) | Intermediate | 2h | Dimensionality reduction, SVD |

**Learning Outcomes:**
- Implement ML algorithms from scratch
- Understand when to use each algorithm
- Tune hyperparameters effectively
- Evaluate model performance

### Deep Learning

| Notebook | Level | Duration | Topics |
|----------|-------|----------|--------|
| [01_neural_networks_basics.ipynb](03_deep_learning/01_neural_networks_basics.ipynb) | Intermediate | 3h | Perceptrons, activation functions |
| [02_backpropagation.ipynb](03_deep_learning/02_backpropagation.ipynb) | Intermediate | 4h | Chain rule, gradient computation |
| [03_cnn.ipynb](03_deep_learning/03_cnn.ipynb) | Advanced | 4h | Convolution, pooling, ResNet |
| [04_rnn_lstm.ipynb](03_deep_learning/04_rnn_lstm.ipynb) | Advanced | 4h | Sequential models, LSTM, GRU |
| [05_transformers.ipynb](03_deep_learning/05_transformers.ipynb) | Advanced | 5h | Self-attention, transformer architecture |
| [06_transfer_learning.ipynb](03_deep_learning/06_transfer_learning.ipynb) | Advanced | 3h | Pre-trained models, fine-tuning |

**Learning Outcomes:**
- Build neural networks from scratch
- Implement backpropagation
- Work with CNNs and RNNs
- Understand transformer architecture

### LLM Engineering

| Notebook | Level | Duration | Topics |
|----------|-------|----------|--------|
| [01_tokenization.ipynb](04_llm/01_tokenization.ipynb) | Intermediate | 2h | BPE, WordPiece, sentencepiece |
| [02_attention_mechanisms.ipynb](04_llm/02_attention_mechanisms.ipynb) | Advanced | 4h | Self-attention, multi-head attention |
| [03_transformer_implementation.ipynb](04_llm/03_transformer_implementation.ipynb) | Advanced | 5h | Build transformer from scratch |
| [04_llm_finetuning.ipynb](04_llm/04_llm_finetuning.ipynb) | Advanced | 4h | SFT, instruction tuning |
| [05_lora_qlora.ipynb](04_llm/05_lora_qlora.ipynb) | Advanced | 4h | Parameter-efficient fine-tuning |
| [06_prompt_engineering.ipynb](04_llm/06_prompt_engineering.ipynb) | Intermediate | 3h | Prompt design, few-shot learning |

**Learning Outcomes:**
- Understand tokenization
- Implement attention mechanisms
- Fine-tune LLMs efficiently
- Master prompt engineering

### RAG Systems

| Notebook | Level | Duration | Topics |
|----------|-------|----------|--------|
| [01_rag_basics.ipynb](RAG/01_rag_basics.ipynb) | Intermediate | 3h | RAG architecture, basic implementation |
| [02_embedding_models.ipynb](RAG/02_embedding_models.ipynb) | Intermediate | 3h | Embedding models, comparison |
| [03_vector_databases.ipynb](RAG/03_vector_databases.ipynb) | Intermediate | 3h | Chroma, FAISS, Pinecone |
| [04_advanced_rag.ipynb](RAG/04_advanced_rag.ipynb) | Advanced | 4h | Hybrid search, re-ranking |
| [05_multimodal_rag.ipynb](RAG/05_multimodal_rag.ipynb) | Advanced | 4h | Text + images, multi-modal embeddings |
| [06_temporal_rag.ipynb](RAG/06_temporal_rag.ipynb) | Advanced | 4h | Time-aware retrieval |
| [07_rag_evaluation.ipynb](RAG/07_rag_evaluation.ipynb) | Advanced | 3h | RAGAS, evaluation metrics |

**Learning Outcomes:**
- Build production RAG systems
- Implement advanced retrieval strategies
- Evaluate RAG performance
- Handle multi-modal data

### AI Agents

| Notebook | Level | Duration | Topics |
|----------|-------|----------|--------|
| [01_agent_basics.ipynb](agents/01_agent_basics.ipynb) | Intermediate | 3h | Agent architecture, tools |
| [02_langchain_agents.ipynb](agents/02_langchain_agents.ipynb) | Intermediate | 4h | LangChain agent framework |
| [03_crewai_agents.ipynb](agents/03_crewai_agents.ipynb) | Advanced | 4h | Multi-agent collaboration |
| [04_langgraph_workflows.ipynb](agents/04_langgraph_workflows.ipynb) | Advanced | 4h | State-based workflows |
| [05_tool_use.ipynb](agents/05_tool_use.ipynb) | Advanced | 3h | Custom tools, function calling |

**Learning Outcomes:**
- Build AI agents
- Implement tool use
- Create multi-agent systems
- Design complex workflows

---

## 🎯 Notebooks by Level

### Beginner (0-3 months experience)

```
notebooks/
├── 01_foundations/
│   ├── 01_python_for_ai.ipynb
│   ├── 02_numpy_essentials.ipynb
│   └── 03_matplotlib_basics.ipynb
├── 01_mathematical_foundations/
│   ├── 01_linear_algebra.ipynb
│   ├── 02_calculus.ipynb
│   └── 03_probability.ipynb
└── 02_classical_ml/
    ├── 01_linear_regression.ipynb
    └── 02_logistic_regression.ipynb
```

**Total:** 8 notebooks, ~20 hours

### Intermediate (3-6 months experience)

```
notebooks/
├── 02_classical_ml/
│   ├── 03_decision_trees.ipynb
│   ├── 04_svm.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_pca.ipynb
├── 03_deep_learning/
│   ├── 01_neural_networks_basics.ipynb
│   └── 02_backpropagation.ipynb
├── 04_llm/
│   ├── 01_tokenization.ipynb
│   └── 06_prompt_engineering.ipynb
└── RAG/
    ├── 01_rag_basics.ipynb
    ├── 02_embedding_models.ipynb
    └── 03_vector_databases.ipynb
```

**Total:** 12 notebooks, ~35 hours

### Advanced (6+ months experience)

```
notebooks/
├── 03_deep_learning/
│   ├── 03_cnn.ipynb
│   ├── 04_rnn_lstm.ipynb
│   ├── 05_transformers.ipynb
│   └── 06_transfer_learning.ipynb
├── 04_llm/
│   ├── 02_attention_mechanisms.ipynb
│   ├── 03_transformer_implementation.ipynb
│   ├── 04_llm_finetuning.ipynb
│   └── 05_lora_qlora.ipynb
├── RAG/
│   ├── 04_advanced_rag.ipynb
│   ├── 05_multimodal_rag.ipynb
│   ├── 06_temporal_rag.ipynb
│   └── 07_rag_evaluation.ipynb
└── agents/
    ├── 03_crewai_agents.ipynb
    ├── 04_langgraph_workflows.ipynb
    └── 05_tool_use.ipynb
```

**Total:** 15 notebooks, ~50 hours

---

## 📝 Exercise Format

Each notebook follows a consistent structure:

### 1. Learning Objectives

```markdown
## Learning Objectives

By the end of this notebook, you will:
- ✅ Understand [concept]
- ✅ Implement [algorithm]
- ✅ Apply [technique] to [problem]
```

### 2. Theory Section

```markdown
## Theory

### Key Concept 1

Explanation with formulas and diagrams...

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

### 3. Code Examples

```python
# Example: Self-Attention Implementation
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, V, K, Q, mask=None):
        batch_size = Q.shape[0]
        
        # Split into heads
        V = self.values(V).reshape(batch_size, -1, self.heads, self.head_dim)
        K = self.keys(K).reshape(batch_size, -1, self.heads, self.head_dim)
        Q = self.queries(Q).reshape(batch_size, -1, self.heads, self.head_dim)
        
        # Attention computation
        attention = self.attention(Q, K, V, mask)
        
        # Concatenate and project
        concat = attention.reshape(batch_size, -1, self.embed_size)
        out = self.fc_out(concat)
        
        return out
```

### 4. Exercises

```markdown
## Exercise 1: Implement Gradient Descent

**Difficulty:** ⭐⭐ (Medium)
**Time:** 15 minutes

Implement gradient descent from scratch:

```python
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Implement gradient descent for linear regression.
    
    Args:
        X: Input features (m, n)
        y: Target values (m,)
        learning_rate: Step size
        iterations: Number of iterations
    
    Returns:
        weights: Learned parameters (n,)
        loss_history: Loss at each iteration
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    loss_history = []
    
    # TODO: Implement gradient descent
    # 1. Compute predictions
    # 2. Compute gradients
    # 3. Update parameters
    # 4. Track loss
    
    return weights, bias, loss_history
```

**Hints:**
- Gradient of MSE loss: ∂L/∂w = (2/m) * X^T(Xw - y)
- Update rule: w = w - learning_rate * gradient
```

### 5. Solutions

```python
# Solution: Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    loss_history = []
    
    for i in range(iterations):
        # Compute predictions
        y_pred = X @ weights + bias
        
        # Compute gradients
        dw = (2/m) * X.T @ (y_pred - y)
        db = (2/m) * np.sum(y_pred - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Track loss
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)
    
    return weights, bias, loss_history
```

---

## 🏆 Real-World Examples

### Example 1: Building a Chatbot

**Notebook:** [RAG/01_rag_basics.ipynb](RAG/01_rag_basics.ipynb)

**What You'll Build:**
```python
from src.rag import RAGSystem

# Initialize RAG
rag = RAGSystem(
    embedding_model="all-MiniLM-L6-v2",
    vector_db="chroma",
    llm="gpt-3.5-turbo"
)

# Add your documentation
rag.add_documents([
    {"id": "doc1", "content": "Product documentation..."},
    {"id": "doc2", "content": "FAQ content..."}
])

# Query your chatbot
response = rag.query("How do I reset my password?")
print(response.answer)
print(f"Sources: {response.sources}")
```

### Example 2: Fine-Tuning for Arabic

**Notebook:** [04_llm/05_lora_qlora.ipynb](04_llm/05_lora_qlora.ipynb)

**What You'll Build:**
```python
from src.llm.finetuning import QLoRATrainer

# Initialize trainer
trainer = QLoRATrainer(
    model_name="meta-llama/Llama-2-7b",
    lora_r=16,
    quantization="4bit"
)

# Train on Arabic dataset
trainer.train(
    data_path="data/arabic_sft.jsonl",
    epochs=3,
    batch_size=8
)

# Test fine-tuned model
response = trainer.generate("مرحباً، كيف حالك؟")
print(response)
```

### Example 3: Multi-Agent System

**Notebook:** [agents/03_crewai_agents.ipynb](agents/03_crewai_agents.ipynb)

**What You'll Build:**
```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover new insights",
    backstory="Expert in data analysis"
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content",
    backstory="Skilled technical writer"
)

# Define tasks
research_task = Task(
    description="Research AI trends in 2026",
    agent=researcher
)

write_task = Task(
    description="Write article based on research",
    agent=writer
)

# Execute crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task]
)

result = crew.kickoff()
print(result)
```

---

## 📊 Progress Tracking

### Completion Checklist

Use this checklist to track your progress:

#### Foundations (8 notebooks)
- [ ] Python for AI
- [ ] NumPy Essentials
- [ ] Matplotlib Basics
- [ ] Linear Algebra
- [ ] Calculus
- [ ] Probability
- [ ] Statistics
- [ ] Linear Regression

#### Classical ML (6 notebooks)
- [ ] Logistic Regression
- [ ] Decision Trees
- [ ] SVM
- [ ] Clustering
- [ ] PCA
- [ ] Ensemble Methods

#### Deep Learning (6 notebooks)
- [ ] Neural Networks Basics
- [ ] Backpropagation
- [ ] CNN
- [ ] RNN/LSTM
- [ ] Transformers
- [ ] Transfer Learning

#### LLM Engineering (6 notebooks)
- [ ] Tokenization
- [ ] Attention Mechanisms
- [ ] Transformer Implementation
- [ ] LLM Fine-Tuning
- [ ] LoRA/QLoRA
- [ ] Prompt Engineering

#### RAG Systems (7 notebooks)
- [ ] RAG Basics
- [ ] Embedding Models
- [ ] Vector Databases
- [ ] Advanced RAG
- [ ] Multi-Modal RAG
- [ ] Temporal RAG
- [ ] RAG Evaluation

#### AI Agents (5 notebooks)
- [ ] Agent Basics
- [ ] LangChain Agents
- [ ] CrewAI Agents
- [ ] LangGraph Workflows
- [ ] Tool Use

**Total:** 38 notebooks, ~105 hours

---

## 🎓 Certification Path

Complete notebooks to earn certificates:

| Certificate | Required Notebooks | Badge |
|-------------|-------------------|-------|
| **Foundations** | 8 foundation notebooks | 🥉 Bronze |
| **ML Practitioner** | 6 classical ML + 6 DL | 🥈 Silver |
| **LLM Engineer** | 6 LLM + 7 RAG | 🥇 Gold |
| **AI Architect** | All 38 notebooks | 🏆 Platinum |

---

## 🔗 Additional Resources

### Related Documentation

- [Learning Roadmap](../01_learning_roadmap/README.md)
- [API Reference](../api/)
- [Knowledge Base](../kb/)
- [FAQ](../faq/)

### External Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Hugging Face Course](https://huggingface.co/course)
- [Fast.ai](https://course.fast.ai/)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

---

## 🤝 Contributing

Want to add a notebook? See our [Notebook Contribution Guide](../00_introduction/NOTEBOOK_CONTRIBUTING.md) for:

- Notebook structure and format
- Code style guidelines
- Exercise design principles
- Review process

---

**Last Updated:** March 28, 2026  
**Total Notebooks:** 50+  
**Total Hours:** 100+  
**Difficulty Levels:** Beginner, Intermediate, Advanced

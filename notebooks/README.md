# AI-Mastery-2026 Example Notebooks

This directory contains example notebooks demonstrating key features of AI-Mastery-2026.

## Quick Start Examples

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [01_getting_started.ipynb](01_getting_started.ipynb) | Installation and first steps | Beginner |
| [02_configuration.ipynb](02_configuration.ipynb) | Using the configuration system | Beginner |
| [03_types_and_protocols.ipynb](03_types_and_protocols.ipynb) | Understanding type definitions | Intermediate |

## Core Mathematics

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [core/01_linear_algebra.ipynb](core/01_linear_algebra.ipynb) | Linear algebra from scratch | Beginner |
| [core/02_calculus.ipynb](core/02_calculus.ipynb) | Numerical calculus | Intermediate |
| [core/03_optimization.ipynb](core/03_optimization.ipynb) | Optimization algorithms | Intermediate |
| [core/04_probability.ipynb](core/04_probability.ipynb) | Probability and statistics | Intermediate |
| [core/05_mcmc.ipynb](core/05_mcmc.ipynb) | MCMC sampling | Advanced |

## Machine Learning

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [ml/01_linear_regression.ipynb](ml/01_linear_regression.ipynb) | Linear regression from scratch | Beginner |
| [ml/02_logistic_regression.ipynb](ml/02_logistic_regression.ipynb) | Logistic regression | Beginner |
| [ml/03_decision_trees.ipynb](ml/03_decision_trees.ipynb) | Decision trees (ID3, C4.5, CART) | Intermediate |
| [ml/04_random_forest.ipynb](ml/04_random_forest.ipynb) | Random forests | Intermediate |
| [ml/05_svm.ipynb](ml/05_svm.ipynb) | Support vector machines | Advanced |
| [ml/06_neural_network.ipynb](ml/06_neural_network.ipynb) | Neural networks from scratch | Intermediate |
| [ml/07_cnn.ipynb](ml/07_cnn.ipynb) | Convolutional neural networks | Advanced |
| [ml/08_lstm.ipynb](ml/08_lstm.ipynb) | LSTM networks | Advanced |
| [ml/09_resnet.ipynb](ml/09_resnet.ipynb) | ResNet architecture | Advanced |
| [ml/10_gnn_recommender.ipynb](ml/10_gnn_recommender.ipynb) | Graph neural networks | Advanced |

## LLM Engineering

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [llm/01_attention.ipynb](llm/01_attention.ipynb) | Attention mechanisms | Intermediate |
| [llm/02_transformer.ipynb](llm/02_transformer.ipynb) | Transformer architecture | Advanced |
| [llm/03_bert.ipynb](llm/03_bert.ipynb) | BERT from scratch | Advanced |
| [llm/04_gpt2.ipynb](llm/04_gpt2.ipynb) | GPT-2 from scratch | Advanced |
| [llm/05_fine_tuning.ipynb](llm/05_fine_tuning.ipynb) | Fine-tuning (LoRA, Adapters) | Advanced |
| [llm/06_evaluation.ipynb](llm/06_evaluation.ipynb) | LLM evaluation | Intermediate |

## RAG Systems

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [rag/01_chunking.ipynb](rag/01_chunking.ipynb) | Document chunking strategies | Beginner |
| [rag/02_retrieval.ipynb](rag/02_retrieval.ipynb) | Retrieval methods | Intermediate |
| [rag/03_reranking.ipynb](rag/03_reranking.ipynb) | Reranking techniques | Intermediate |
| [rag/04_rag_pipeline.ipynb](rag/04_rag_pipeline.ipynb) | Complete RAG pipeline | Advanced |
| [rag/05_specialized_rags.ipynb](rag/05_specialized_rags.ipynb) | Specialized RAG architectures | Advanced |

## AI Agents

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [agents/01_orchestration.ipynb](agents/01_orchestration.ipynb) | Agent orchestration | Intermediate |
| [agents/02_multi_agent.ipynb](agents/02_multi_agent.ipynb) | Multi-agent systems | Advanced |
| [agents/03_support_agent.ipynb](agents/03_support_agent.ipynb) | Support agent example | Advanced |
| [agents/04_tools.ipynb](agents/04_tools.ipynb) | Agent tools integration | Advanced |

## Production

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [production/01_api_deployment.ipynb](production/01_api_deployment.ipynb) | FastAPI deployment | Intermediate |
| [production/02_caching.ipynb](production/02_caching.ipynb) | Caching strategies | Intermediate |
| [production/03_monitoring.ipynb](production/03_monitoring.ipynb) | Monitoring and observability | Advanced |
| [production/04_docker.ipynb](production/04_docker.ipynb) | Docker containerization | Intermediate |

## Capstone Projects

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [capstone/01_issue_classifier.ipynb](capstone/01_issue_classifier.ipynb) | GitHub issue classifier | Advanced |
| [capstone/02_rag_chatbot.ipynb](capstone/02_rag_chatbot.ipynb) | RAG-based chatbot | Advanced |
| [capstone/03_recommendation_system.ipynb](capstone/03_recommendation_system.ipynb) | Recommendation system | Advanced |

---

## Running Notebooks

### Local Jupyter

```bash
# Install Jupyter
pip install jupyter jupyterlab

# Start Jupyter
jupyter lab

# Or classic notebook
jupyter notebook
```

### VS Code

1. Install Python extension
2. Open notebook file
3. Select Python kernel
4. Run cells

### Google Colab

```python
# Clone repository
!git clone https://github.com/Kandil7/AI-Mastery-2026.git
%cd AI-Mastery-2026

# Install package
!pip install -e .

# Run notebook
# Upload notebook to Colab and run
```

---

## Contributing Examples

We welcome example notebooks! Please follow these guidelines:

1. **Structure**: Follow the existing notebook structure
2. **Documentation**: Include clear explanations
3. **Tests**: Ensure code runs without errors
4. **Formatting**: Use consistent formatting
5. **Output**: Clear notebook outputs before committing

### Notebook Template

```python
"""
Notebook Title
==============

Brief description of what this notebook demonstrates.

Prerequisites:
- Package versions
- Required knowledge

Learning Objectives:
- What you will learn
"""

# Setup
from src.module import Component

# Example code
component = Component()
result = component.process(data)

# Visualization
import matplotlib.pyplot as plt
plt.plot(result)
plt.show()
```

---

**Last Updated:** March 31, 2026

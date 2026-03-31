# AI-Mastery-2026 API Usage Examples

Practical examples demonstrating how to use AI-Mastery-2026 components.

---

## Table of Contents

1. [Configuration](#configuration)
2. [Core Mathematics](#core-mathematics)
3. [Machine Learning](#machine-learning)
4. [LLM Engineering](#llm-engineering)
5. [RAG Systems](#rag-systems)
6. [AI Agents](#ai-agents)
7. [Production](#production)

---

## Configuration

### Basic Configuration

```python
from src.config import get_settings, Settings

# Get global settings (singleton)
settings = get_settings()
print(f"Environment: {settings.environment}")
print(f"Debug mode: {settings.debug}")
print(f"API port: {settings.api_port}")
```

### Custom Configuration

```python
from src.config import Settings, Environment

settings = Settings(
    environment=Environment.DEVELOPMENT,
    debug=True,
    batch_size=16,
    device="cuda"
)
```

### Model Configuration

```python
from src.config import TransformerConfig, TrainingConfig

# Transformer config
model_config = TransformerConfig(
    hidden_dim=768,
    num_heads=12,
    num_layers=12,
    dropout=0.1
)

# Training config
train_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
    warmup_steps=1000
)
```

---

## Core Mathematics

### Matrix Operations

```python
import numpy as np
from src.core.math_operations import (
    matrix_multiply,
    matrix_inverse,
    svd,
    eigenvalues_eigenvectors
)

# Matrix multiplication
A = np.random.randn(10, 20)
B = np.random.randn(20, 15)
C = matrix_multiply(A, B)

# SVD decomposition
U, S, Vt = svd(A)

# Eigen decomposition
M = np.random.randn(10, 10)
M = M @ M.T  # Make symmetric
eigenvalues, eigenvectors = eigenvalues_eigenvectors(M)
```

### Optimization

```python
from src.core.optimization import Adam, GradientDescent, minimize

# Define loss function
def loss_fn(params):
    return np.sum(params ** 2)

# Initialize parameters
params = np.random.randn(10)

# Use Adam optimizer
optimizer = Adam(learning_rate=0.01)
for i in range(100):
    grad = compute_gradient(params)
    params = optimizer.step(params, grad)

# Or use minimize function
result = minimize(loss_fn, params, method='adam')
```

---

## Machine Learning

### Linear Regression

```python
from src.ml import LinearRegression
import numpy as np

# Generate data
X = np.random.randn(100, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict(X_test)
```

### Decision Tree

```python
from src.ml import DecisionTree

# Create model
model = DecisionTree(
    max_depth=5,
    min_samples_split=10,
    criterion='gini'
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Feature importance
importances = model.feature_importances_
```

### Neural Network

```python
from src.ml import NeuralNetwork

# Create network
model = NeuralNetwork(
    input_dim=10,
    hidden_dims=[64, 32],
    output_dim=4,
    activations=['relu', 'relu', 'softmax']
)

# Train
model.fit(
    X_train, y_train,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Predict
predictions = model.predict(X_test)
```

### CNN for Image Classification

```python
from src.ml.vision import ResNet18

# Create model
model = ResNet18(num_classes=10)

# Train
model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50
)

# Predict
predictions = model.predict(X_test)
```

---

## LLM Engineering

### Multi-Head Attention

```python
from src.llm import MultiHeadAttention
import numpy as np

# Create attention layer
attention = MultiHeadAttention(
    d_model=768,
    num_heads=12,
    dropout=0.1
)

# Input sequences
batch_size = 4
seq_len = 128
d_model = 768

query = np.random.randn(batch_size, seq_len, d_model)
key = np.random.randn(batch_size, seq_len, d_model)
value = np.random.randn(batch_size, seq_len, d_model)

# Apply attention
output, attention_weights = attention(query, key, value)
```

### BERT

```python
from src.llm import BERT

# Create BERT model
bert = BERT(
    vocab_size=30522,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=512
)

# Forward pass
input_ids = np.random.randint(0, 30522, (4, 128))
attention_mask = np.ones((4, 128))

outputs = bert(input_ids, attention_mask)
last_hidden_state = outputs.last_hidden_state
pooler_output = outputs.pooler_output
```

### Fine-tuning with LoRA

```python
from src.llm import FineTuner, FineTuningConfig, FineTuningMethod

# Configure fine-tuning
config = FineTuningConfig(
    method=FineTuningMethod.LORA,
    lora_rank=8,
    lora_alpha=16,
    learning_rate=1e-4
)

# Create fine-tuner
tuner = FineTuner(model, config)

# Fine-tune
tuner.fine_tune(
    train_dataset,
    eval_dataset,
    epochs=3,
    batch_size=16
)
```

---

## RAG Systems

### Basic RAG Pipeline

```python
from src.rag import RAGPipeline, Document, SemanticChunker

# Create pipeline
pipeline = RAGPipeline(
    chunker=SemanticChunker(chunk_size=512, overlap=50)
)

# Add documents
documents = [
    Document(id="1", content="AI is transforming industries."),
    Document(id="2", content="Machine learning powers modern AI."),
    Document(id="3", content="Deep learning enables computer vision."),
]
pipeline.add_documents(documents)

# Query
results = pipeline.query(
    "How does AI work?",
    k=5
)

for result in results:
    print(f"ID: {result.id}")
    print(f"Content: {result.content}")
    print(f"Score: {result.score}")
```

### Advanced RAG with Reranking

```python
from src.rag import RAGPipeline, RAGConfig
from src.rag.reranking import CrossEncoderReranker

# Configure RAG
config = RAGConfig(
    chunk_size=512,
    chunk_overlap=50,
    top_k=20,  # Retrieve more for reranking
    rerank=True,
    rerank_top_k=5  # Return top 5 after reranking
)

# Create pipeline with reranker
pipeline = RAGPipeline(config=config)
pipeline.reranker = CrossEncoderReranker()

# Add documents and query
pipeline.add_documents(documents)
results = pipeline.query("What is machine learning?")
```

### Custom Chunking

```python
from src.rag.chunking import HierarchicalChunker, ChunkingConfig

# Configure chunking
config = ChunkingConfig(
    parent_chunk_size=2000,
    child_chunk_size=500,
    overlap=50
)

# Create chunker
chunker = HierarchicalChunker(config)

# Chunk document
chunks = chunker.chunk({
    "id": "doc1",
    "content": "Long document content..."
})
```

---

## AI Agents

### Simple Agent

```python
from src.agents import ReActAgent, Tool, ToolRegistry

# Define tools
def calculator(expression: str) -> str:
    return str(eval(expression))

calculator_tool = Tool(
    name="calculator",
    description="Evaluate mathematical expressions",
    func=calculator
)

# Create tool registry
registry = ToolRegistry()
registry.register(calculator_tool)

# Create agent
agent = ReActAgent(
    model="gpt-4",
    tools=registry,
    max_iterations=10
)

# Run agent
response = agent.run("What is 2 + 2 * 3?")
print(response)
```

### Multi-Agent System

```python
from src.agents import MultiAgent, AgentRole, Task

# Create agents
researcher = MultiAgent.create_agent(
    role=AgentRole.RESEARCHER,
    goal="Find relevant information",
    backstory="Expert researcher with access to databases"
)

writer = MultiAgent.create_agent(
    role=AgentRole.WRITER,
    goal="Write clear content",
    backstory="Professional writer with expertise in technical content"
)

# Create task
task = Task(
    description="Write a blog post about AI",
    expected_output="1000-word blog post",
    agents=[researcher, writer]
)

# Execute
result = MultiAgent.execute(task)
```

---

## Production

### FastAPI Service

```python
from fastapi import FastAPI
from src.production.api import create_app

# Create app
app = create_app()

# Run
# uvicorn src.production.api:app --reload
```

### Caching

```python
from src.production.caching import SemanticCache, CacheEntry

# Create cache
cache = SemanticCache(
    similarity_threshold=0.9,
    max_size=1000
)

# Add to cache
cache.put(
    key="What is AI?",
    value="AI stands for Artificial Intelligence..."
)

# Query cache
result = cache.get("What is artificial intelligence?")
if result:
    print(f"Cache hit: {result.value}")
else:
    print("Cache miss")
```

### Monitoring

```python
from src.production.observability import LatencyTracker, MetricsCollector

# Create tracker
tracker = LatencyTracker()

# Track operation
with tracker.track("model_inference"):
    result = model.predict(data)

# Get metrics
metrics = tracker.get_metrics()
print(f"P50: {metrics.p50}ms")
print(f"P95: {metrics.p95}ms")
print(f"P99: {metrics.p99}ms")
```

---

## Type Usage

```python
from src.types import (
    DocumentProtocol,
    EmbeddingVector,
    ModelOutput,
    Trainable,
)
import numpy as np

# Type-annotated function
def process_document(doc: DocumentProtocol) -> EmbeddingVector:
    """Process document and return embedding."""
    return np.random.randn(384)

# Type-annotated model
class MyModel(Trainable):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MyModel":
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.random.randn(len(X))
```

---

## Error Handling

```python
from src.core.utils.errors import (
    AIError,
    ConfigurationError,
    ModelError,
    handle_error
)

try:
    result = model.predict(data)
except ModelError as e:
    handle_error(e)
    print(f"Model error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except AIError as e:
    print(f"AI error: {e}")
```

---

## Logging

```python
from src.core.utils.logging import (
    get_logger,
    log_execution_time,
    debug,
    info,
    warning,
    error
)

# Get logger
logger = get_logger(__name__)

# Log messages
logger.info("Starting process")
logger.debug("Debug info")
logger.warning("Warning message")
logger.error("Error occurred")

# Decorator for timing
@log_execution_time
def slow_function():
    # ... long operation
    pass
```

---

**Last Updated:** March 31, 2026

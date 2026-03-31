# AI-Mastery-2026: End-to-End Examples and Tutorials

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundations Example](#mathematical-foundations-example)
3. [Classical ML Example](#classical-ml-example)
4. [Deep Learning Example](#deep-learning-example)
5. [LLM Engineering Example](#llm-engineering-example)
6. [Production Deployment Example](#production-deployment-example)
7. [Complete RAG Pipeline Example](#complete-rag-pipeline-example)

## Introduction

This tutorial demonstrates how to use the AI-Mastery-2026 toolkit to build complete AI solutions from mathematical foundations to production deployment. Each example follows the "White-Box Approach" philosophy, emphasizing understanding of mathematical foundations before using abstractions.

## Mathematical Foundations Example

This example demonstrates implementing and using core mathematical operations:

```python
import numpy as np
from src.core.math_operations import (
    dot_product, magnitude, normalize, cosine_similarity,
    matrix_multiply, PCA, softmax, sigmoid, relu
)
from src.core.optimization import Adam, gradient_descent_train

# Example 1: Vector operations
v1 = [1, 2, 3]
v2 = [4, 5, 6]

print(f"Dot product: {dot_product(v1, v2)}")
print(f"Magnitude of v1: {magnitude(v1)}")
print(f"Cosine similarity: {cosine_similarity(v1, v2)}")

# Example 2: Matrix operations
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = matrix_multiply(A, B)
print(f"Matrix multiplication result: {result}")

# Example 3: Principal Component Analysis
X = np.random.randn(100, 5)  # 100 samples, 5 features
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")

# Example 4: Activation functions
x = np.array([1.0, 2.0, 3.0])
print(f"Softmax: {softmax(x)}")
print(f"Sigmoid: {sigmoid(x)}")
print(f"ReLU: {relu(x)}")

# Example 5: Optimization with gradient descent
def mse_loss(X, y, w):
    """Mean squared error loss function"""
    pred = X @ w
    loss = np.mean((pred - y) ** 2)
    grad = 2/len(y) * X.T @ (pred - y)
    return loss, grad

# Generate synthetic data
X = np.random.randn(100, 3)
true_weights = np.array([1.5, -2.0, 0.5])
y = X @ true_weights + np.random.randn(100) * 0.1

# Train with gradient descent
optimizer = Adam(learning_rate=0.01)
params, history = gradient_descent_train(
    X, y, mse_loss, np.zeros(3), optimizer, epochs=100, verbose=False
)
print(f"Learned weights: {params}")
print(f"Final loss: {history[-1]}")
```

## Classical ML Example

This example demonstrates implementing and using classical machine learning algorithms:

```python
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.ml.classical import (
    LinearRegressionScratch, LogisticRegressionScratch,
    DecisionTreeScratch, RandomForestScratch
)

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                          n_redundant=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Example 1: Logistic Regression
print("=== Logistic Regression ===")
log_reg = LogisticRegressionScratch(n_iterations=1000, learning_rate=0.1)
log_reg.fit(X_train_scaled, y_train)
log_reg_accuracy = log_reg.score(X_test_scaled, y_test)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.3f}")

# Example 2: Decision Tree
print("\n=== Decision Tree ===")
dt = DecisionTreeScratch(max_depth=10, min_samples_split=5)
dt.fit(X_train, y_train)
dt_accuracy = dt.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy:.3f}")

# Example 3: Random Forest
print("\n=== Random Forest ===")
rf = RandomForestScratch(n_estimators=50, max_depth=10, min_samples_split=5)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Example 4: Linear Regression
print("\n=== Linear Regression ===")
lin_reg = LinearRegressionScratch(method='gradient_descent', learning_rate=0.01, n_iterations=1000)
lin_reg.fit(X_reg_train, y_reg_train)
lin_reg_score = lin_reg.score(X_reg_test, y_reg_test)
print(f"Linear Regression R² Score: {lin_reg_score:.3f}")
```

## Deep Learning Example

This example demonstrates building and training neural networks:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, Dropout, BatchNormalization,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
)

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                          n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to appropriate format for binary classification
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Create a neural network
model = NeuralNetwork()
model.add(Dense(input_size=20, output_size=64, weight_init='he'))
model.add(BatchNormalization(n_features=64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(input_size=64, output_size=32, weight_init='he'))
model.add(BatchNormalization(n_features=32))
model.add(Activation('relu'))
model.add(Dense(input_size=32, output_size=1, weight_init='xavier'))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss=BinaryCrossEntropyLoss(), learning_rate=0.001)

print("=== Neural Network Training ===")
# Train the model
history = model.fit(
    X_train_scaled, y_train_reshaped,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_reshaped),
    verbose=True
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test_reshaped)
print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}")

# Make predictions
predictions = model.predict(X_test_scaled)
print(f"Sample predictions: {predictions[:5].flatten()}")
```

## LLM Engineering Example

This example demonstrates using attention mechanisms and fine-tuning techniques:

```python
import torch
import numpy as np
from src.llm.attention import (
    scaled_dot_product_attention, MultiHeadAttention,
    TransformerBlock, FeedForwardNetwork, LayerNorm
)
from src.llm.fine_tuning import LoRALayer, LinearWithLoRA

# Example 1: Scaled Dot-Product Attention
print("=== Scaled Dot-Product Attention ===")
Q = torch.randn(2, 10, 64)  # batch=2, seq=10, d_k=64
K = torch.randn(2, 15, 64)  # batch=2, seq=15, d_k=64
V = torch.randn(2, 15, 64)  # batch=2, seq=15, d_v=64

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Example 2: Multi-Head Attention
print("\n=== Multi-Head Attention ===")
mha = MultiHeadAttention(d_model=512, num_heads=8)
X = torch.randn(4, 20, 512)  # batch=4, seq=20, d_model=512
output = mha(X, X, X)
print(f"Multi-Head Attention output shape: {output.shape}")

# Example 3: Transformer Block
print("\n=== Transformer Block ===")
block = TransformerBlock(d_model=256, num_heads=8, d_ff=512)
X = torch.randn(2, 15, 256)  # batch=2, seq=15, d_model=256
output = block(X)
print(f"Transformer Block output shape: {output.shape}")

# Example 4: LoRA (Low-Rank Adaptation)
print("\n=== LoRA (Low-Rank Adaptation) ===")
lora = LoRALayer(in_features=128, out_features=64, r=8, alpha=16.0)
x = np.random.randn(10, 128)
lora_output = lora.forward(x)
print(f"LoRA output shape: {lora_output.shape}")

# Example 5: Linear layer with LoRA
linear_with_lora = LinearWithLoRA(in_features=128, out_features=64, r=8)
x_tensor = torch.randn(10, 128)
# Note: This would require proper PyTorch integration
print("Linear layer with LoRA initialized")
```

## Production Deployment Example

This example demonstrates how to deploy models in production:

```python
import numpy as np
from src.production.caching import LRUCache, EmbeddingCache
from src.production.monitoring import DriftDetector, PerformanceMonitor
from src.production.deployment import ModelSerializer, ModelVersionManager
from src.ml.classical import RandomForestScratch

# Example 1: Model Serialization and Versioning
print("=== Model Serialization and Versioning ===")
# Create a sample model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestScratch(n_estimators=50, max_depth=10)
model.fit(X, y)

# Serialize the model
serializer = ModelSerializer()
model_path = serializer.save(model, "models/my_model.pkl", format="pickle")
print(f"Model saved to: {model_path}")

# Load the model
loaded_model = serializer.load("models/my_model.pkl", format="pickle")
print("Model loaded successfully")

# Model versioning
version_manager = ModelVersionManager("models/")
version_manager.register("v1.0", model, metrics={"accuracy": 0.92})
print("Model version registered")

# Example 2: Caching
print("\n=== Caching ===")
cache = LRUCache(max_size=100)
cache.set("model_prediction", [0.8, 0.2])
cached_result = cache.get("model_prediction")
print(f"Cached result: {cached_result}")

# Embedding cache
embedding_cache = EmbeddingCache(cache_backend=LRUCache(max_size=1000))
text = "This is a sample text for embedding"
embedding = np.random.randn(384)  # Simulated embedding
embedding_cache.set(text, embedding.tolist())
cached_embedding = embedding_cache.get(text)
print(f"Embedding cached: {cached_embedding is not None}")

# Example 3: Monitoring
print("\n=== Monitoring ===")
# Drift detection
drift_detector = DriftDetector(method='ks', threshold=0.05)
ref_data = np.random.normal(0, 1, (100, 5))  # Reference data
drift_detector.set_reference(ref_data)

# Current data (should be similar to reference)
cur_data = np.random.normal(0, 1, (50, 5))
drift_results = drift_detector.detect_drift(cur_data)
print(f"Drift detected in {sum(1 for r in drift_results if r.drift_detected)} features")

# Performance monitoring
perf_monitor = PerformanceMonitor(window_size=1000)
for i in range(100):
    # Simulate predictions
    y_true = np.random.randint(0, 2, 10)
    y_pred = np.random.randint(0, 2, 10)
    latency = np.random.uniform(10, 100)  # ms
    perf_monitor.record_prediction(y_true=y_true[0], y_pred=y_pred[0], latency_ms=latency)

metrics = perf_monitor.get_metrics(task='classification')
print(f"Current accuracy: {metrics.accuracy:.3f}")
print(f"Latency p95: {metrics.latency_p95_ms:.2f}ms")
```

## Complete RAG Pipeline Example

This example demonstrates building a complete RAG (Retrieval-Augmented Generation) pipeline:

```python
import numpy as np
from src.llm.rag import (
    Document, TextChunker, EmbeddingModel, Retriever,
    Reranker, ContextAssembler, RAGPipeline
)

# Example: Building a RAG pipeline for question answering
print("=== Complete RAG Pipeline ===")

# Create sample documents
documents = [
    Document(content="Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."),
    Document(content="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."),
    Document(content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning."),
    Document(content="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."),
    Document(content="Computer vision is an interdisciplinary scientific field that deals with how computers can be made to gain high-level understanding from digital images or videos."),
]

# Create embedding model (using dummy embeddings for this example)
def dummy_embed(texts):
    """Dummy embedding function for demonstration"""
    return np.random.randn(len(texts), 128)

embedding_model = EmbeddingModel(model_fn=dummy_embed, dim=128)

# Create the RAG pipeline
rag_pipeline = RAGPipeline(
    embedding_model=embedding_model,
    # For this example, we'll use a simple mock LLM function
    llm_fn=lambda prompt: f"Based on the context: {prompt[:100]}... AI is a fascinating field."
)

# Add documents to the pipeline
rag_pipeline.add_documents(documents)

# Query the pipeline
query = "What is Artificial Intelligence?"
response = rag_pipeline.query(query, k=3, return_sources=True)

print(f"Query: {query}")
print(f"Answer: {response['answer']}")
print(f"Number of sources used: {len(response['sources'])}")

# Show sources
for i, source in enumerate(response['sources']):
    print(f"Source {i+1}: {source['content'][:100]}...")

# Example with chunking for longer documents
print("\n=== RAG with Text Chunking ===")
long_document = Document(content="""
    The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen.
    The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as a symbolic system.
    This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.
    This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.
    
    The field of AI research was founded at a Dartmouth College workshop in 1956. 
    The attendees became the leaders of AI research for decades, and they were optimistic about the future of the field.
    Herbert Simon predicted, 'machines will be capable, within twenty years, of doing any work a man can do.'
    Marvin Minsky agreed, writing, 'within a generation... the problem of creating 'artificial intelligence' will substantially be solved.'
    
    Researchers in the 1960s and early 1970s were working on all the fundamental areas: problem solving, planning, reasoning, knowledge representation, learning, natural language processing, perception, and robotics.
    They succeeded in many specific sub-problems but the general problem of artificial intelligence remained unsolved.
    The complexity of the problem was underestimated, and progress was much slower than expected.
    
    The next few decades saw cycles of optimism and disappointment, with periods of reduced funding known as 'AI winters'.
    However, the field continued to advance, with new techniques and applications emerging.
    The development of machine learning algorithms, particularly neural networks, led to significant breakthroughs in the 2010s.
    
    Today, AI is used in a wide range of applications, from recommendation systems to autonomous vehicles.
    Deep learning has enabled significant advances in computer vision, natural language processing, and other areas.
    The field continues to evolve rapidly, with new architectures and techniques being developed regularly.
""")

# Use text chunker to split the long document
chunker = TextChunker(chunk_size=100, overlap=20, strategy='semantic')
chunks = chunker.chunk(long_document.content, metadata={"source": "history_of_ai"})

print(f"Original document length: {len(long_document.content)} characters")
print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.content)} characters")

# Create a new RAG pipeline with the chunks
rag_with_chunks = RAGPipeline(
    llm_fn=lambda prompt: f"Based on the provided context: {prompt[:150]}... The history of AI is fascinating."
)

rag_with_chunks.add_documents(chunks)

# Query about the history of AI
history_query = "Tell me about the history of artificial intelligence"
history_response = rag_with_chunks.query(history_query, k=2, return_sources=True)

print(f"\nQuery: {history_query}")
print(f"Answer: {history_response['answer']}")
print(f"Sources used: {len(history_response['sources'])}")
```

## Complete End-to-End Example: Building a Production ML System

This example demonstrates how to build a complete machine learning system from data preprocessing to deployment:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from datetime import datetime

from src.ml.classical import RandomForestScratch
from src.production.caching import LRUCache
from src.production.monitoring import PerformanceMonitor, DriftDetector
from src.production.deployment import ModelSerializer, ModelVersionManager

class MLProductionSystem:
    """
    A complete ML production system that demonstrates the full pipeline
    from model training to deployment and monitoring.
    """
    
    def __init__(self, model_dir="models", cache_size=1000):
        self.model_dir = model_dir
        self.cache = LRUCache(max_size=cache_size)
        self.performance_monitor = PerformanceMonitor(window_size=1000)
        self.drift_detector = DriftDetector(method='ks', threshold=0.05)
        self.version_manager = ModelVersionManager(model_dir)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_data(self, X, y):
        """Prepare data by scaling features"""
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train(self, X, y, model_params=None):
        """Train the model"""
        if model_params is None:
            model_params = {"n_estimators": 100, "max_depth": 10}
        
        # Prepare data
        X_prepared, y_prepared = self.prepare_data(X, y)
        
        # Initialize and train model
        self.model = RandomForestScratch(**model_params)
        self.model.fit(X_prepared, y_prepared)
        self.is_trained = True
        
        # Set reference data for drift detection
        self.drift_detector.set_reference(X_prepared)
        
        print("Model trained successfully")
        
        # Evaluate on training data
        train_pred = self.model.predict(X_prepared)
        train_acc = accuracy_score(y_prepared, train_pred)
        print(f"Training accuracy: {train_acc:.3f}")
        
        return train_acc
    
    def predict(self, X):
        """Make predictions with caching and monitoring"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare data
        X_scaled = self.scaler.transform(X)
        
        # Create cache key
        cache_key = f"pred_{hash(str(X_scaled.flatten()[:10]))}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            print("Retrieved prediction from cache")
            return cached_result
        
        # Make prediction
        start_time = datetime.now()
        prediction = self.model.predict(X_scaled)
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cache the result
        self.cache.set(cache_key, prediction.tolist())
        
        # Record performance metrics
        # Note: In a real system, we'd need actual labels for accuracy
        self.performance_monitor.record_prediction(
            y_pred=prediction[0] if len(prediction) > 0 else None,
            latency_ms=latency_ms
        )
        
        return prediction
    
    def evaluate_drift(self, X):
        """Evaluate data drift"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluating drift")
        
        X_scaled = self.scaler.transform(X)
        drift_results = self.drift_detector.detect_drift(X_scaled)
        drifted_features = [r.feature_name for r in drift_results if r.drift_detected]
        
        if drifted_features:
            print(f"Drift detected in features: {drifted_features}")
        else:
            print("No significant drift detected")
        
        return drift_results
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        return self.performance_monitor.get_metrics(task='classification')
    
    def save_model(self, version="v1.0"):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create metadata
        metrics = self.get_performance_metrics()
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "accuracy": metrics.accuracy,
            "latency_p95": metrics.latency_p95_ms
        }
        
        # Save model
        model_path = self.version_manager.register(
            version, self.model, 
            metrics={"accuracy": metrics.accuracy} if metrics.accuracy else {}
        )
        
        # Save scaler
        import joblib
        scaler_path = os.path.join(self.model_dir, f"scaler_{version}.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata_path = os.path.join(self.model_dir, f"metadata_{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved as version {version}")
        return model_path
    
    def load_model(self, version):
        """Load a trained model"""
        # Load model
        self.model = self.version_manager.load_active()
        if self.model is None:
            raise ValueError(f"Model version {version} not found")
        
        # Load scaler
        import joblib
        scaler_path = os.path.join(self.model_dir, f"scaler_{version}.pkl")
        self.scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        print(f"Model version {version} loaded successfully")

# Example usage of the complete system
print("=== Complete ML Production System ===")

# Generate sample data
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the production system
ml_system = MLProductionSystem()

# Train the model
print("\n1. Training the model...")
train_acc = ml_system.train(X_train, y_train)

# Make predictions
print("\n2. Making predictions...")
sample_predictions = ml_system.predict(X_test[:5])
print(f"Sample predictions: {sample_predictions}")

# Evaluate on test set
test_predictions = ml_system.predict(X_test)
test_acc = accuracy_score(y_test, test_predictions)
print(f"Test accuracy: {test_acc:.3f}")

# Check for data drift
print("\n3. Checking for data drift...")
drift_results = ml_system.evaluate_drift(X_test[:100])

# Get performance metrics
print("\n4. Performance metrics:")
metrics = ml_system.get_performance_metrics()
print(f"Accuracy: {metrics.accuracy:.3f}")
print(f"Latency P95: {metrics.latency_p95_ms:.2f}ms")

# Save the model
print("\n5. Saving the model...")
ml_system.save_model("v1.0")

print("\nComplete ML Production System example finished!")
```

## Conclusion

This tutorial demonstrated how to use the AI-Mastery-2026 toolkit to build complete AI solutions:

1. **Mathematical Foundations**: Implementing core mathematical operations from scratch
2. **Classical ML**: Building and using classical machine learning algorithms
3. **Deep Learning**: Creating neural networks with various components
4. **LLM Engineering**: Using attention mechanisms and fine-tuning techniques
5. **Production Deployment**: Deploying models with caching, monitoring, and versioning
6. **RAG Pipeline**: Building a complete retrieval-augmented generation system
7. **Complete System**: End-to-end ML production system

Each example follows the "White-Box Approach" philosophy, emphasizing understanding of mathematical foundations before using abstractions. The toolkit provides implementations from scratch while also offering production-ready components for real-world applications.
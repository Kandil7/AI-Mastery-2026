"""
Component-Specific Benchmarking Tools
=====================================

This module provides benchmarking tools for specific components of the AI-Mastery-2026 toolkit,
including mathematical operations, ML algorithms, deep learning components, and LLM components.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os
from pathlib import Path

from src.core.math_operations import (
    dot_product, magnitude, cosine_similarity, matrix_multiply,
    PCA, softmax, sigmoid, relu
)
from src.core.optimization import Adam, SGD
from src.ml.classical import (
    LinearRegressionScratch, LogisticRegressionScratch,
    DecisionTreeScratch, RandomForestScratch
)
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, MSELoss, CrossEntropyLoss
)
from src.llm.attention import (
    scaled_dot_product_attention, MultiHeadAttention,
    TransformerBlock
)
from src.llm.rag import RAGPipeline, Document
from src.production.caching import LRUCache
from src.benchmarks.performance_evaluation import (
    BenchmarkSuite, ModelBenchmark, AlgorithmBenchmark,
    ScalabilityBenchmark, PerformanceMetrics
)


@dataclass
class ComponentBenchmarkResult:
    """Result from benchmarking a specific component."""
    component: str
    operation: str
    input_size: int
    execution_time: float
    memory_usage: float
    throughput: float
    accuracy: float = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class MathOperationsBenchmark:
    """Benchmark mathematical operations."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_dot_product(self, sizes: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark dot product operation."""
        results = []
        
        for size in sizes:
            # Create test vectors
            v1 = np.random.random(size)
            v2 = np.random.random(size)
            
            # Measure execution time
            start_time = time.time()
            result = dot_product(v1, v2)
            execution_time = time.time() - start_time
            
            # Calculate throughput (operations per second)
            throughput = 1.0 / execution_time if execution_time > 0 else float('inf')
            
            # Estimate memory usage (v1, v2, result)
            memory_usage = (size * 2 + 1) * 8 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="MathOperations",
                operation="dot_product",
                input_size=size,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_matrix_multiply(self, sizes: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark matrix multiplication operation."""
        results = []
        
        for size in sizes:
            # Create test matrices
            A = np.random.random((size, size))
            B = np.random.random((size, size))
            
            # Measure execution time
            start_time = time.time()
            result = matrix_multiply(A, B)
            execution_time = time.time() - start_time
            
            # Calculate throughput (operations per second)
            throughput = 1.0 / execution_time if execution_time > 0 else float('inf')
            
            # Estimate memory usage (A, B, result)
            memory_usage = (size * size * 3) * 8 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="MathOperations",
                operation="matrix_multiply",
                input_size=size,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_pca(self, n_samples_list: List[int], n_features: int = 100) -> List[ComponentBenchmarkResult]:
        """Benchmark PCA operation."""
        results = []
        
        for n_samples in n_samples_list:
            # Create test data
            X = np.random.random((n_samples, n_features))
            
            # Initialize PCA
            pca = PCA(n_components=min(10, n_features))
            
            # Measure execution time
            start_time = time.time()
            X_reduced = pca.fit_transform(X)
            execution_time = time.time() - start_time
            
            # Calculate throughput (samples processed per second)
            throughput = n_samples / execution_time if execution_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (X.size + X_reduced.size) * 8 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="MathOperations",
                operation="pca",
                input_size=n_samples,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results


class ClassicalMLBenchmark:
    """Benchmark classical ML algorithms."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_linear_regression(self, n_samples_list: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark Linear Regression."""
        results = []
        
        for n_samples in n_samples_list:
            # Generate data
            X, y = make_regression(n_samples=n_samples, n_features=10, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize model
            model = LinearRegressionScratch(method='gradient_descent', n_iterations=100)
            
            # Measure training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Measure inference time
            inference_start = time.time()
            predictions = model.predict(X_test)
            inference_time = time.time() - inference_start
            
            # Calculate accuracy (R² score for regression)
            from sklearn.metrics import r2_score
            accuracy = r2_score(y_test, predictions)
            
            # Calculate throughput
            throughput = len(X_test) / inference_time if inference_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (X_train.size + y_train.size + X_test.size + len(predictions)) * 8 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="ClassicalML",
                operation="linear_regression",
                input_size=n_samples,
                execution_time=training_time + inference_time,
                memory_usage=memory_usage,
                throughput=throughput,
                accuracy=accuracy
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_logistic_regression(self, n_samples_list: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark Logistic Regression."""
        results = []
        
        for n_samples in n_samples_list:
            # Generate data
            X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize model
            model = LogisticRegressionScratch(n_iterations=100)
            
            # Measure training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Measure inference time
            inference_start = time.time()
            predictions = model.predict(X_test)
            inference_time = time.time() - inference_start
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, predictions)
            
            # Calculate throughput
            throughput = len(X_test) / inference_time if inference_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (X_train.size + y_train.size + X_test.size + len(predictions)) * 8 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="ClassicalML",
                operation="logistic_regression",
                input_size=n_samples,
                execution_time=training_time + inference_time,
                memory_usage=memory_usage,
                throughput=throughput,
                accuracy=accuracy
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_random_forest(self, n_samples_list: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark Random Forest."""
        results = []
        
        for n_samples in n_samples_list:
            # Generate data
            X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize model
            model = RandomForestScratch(n_estimators=10, max_depth=5)
            
            # Measure training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Measure inference time
            inference_start = time.time()
            predictions = model.predict(X_test)
            inference_time = time.time() - inference_start
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, predictions)
            
            # Calculate throughput
            throughput = len(X_test) / inference_time if inference_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (X_train.size + y_train.size + X_test.size + len(predictions)) * 8 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="ClassicalML",
                operation="random_forest",
                input_size=n_samples,
                execution_time=training_time + inference_time,
                memory_usage=memory_usage,
                throughput=throughput,
                accuracy=accuracy
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results


class DeepLearningBenchmark:
    """Benchmark deep learning components."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_neural_network(self, n_samples_list: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark Neural Network."""
        results = []
        
        for n_samples in n_samples_list:
            # Generate data
            X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2, random_state=42)
            X = X.astype(np.float32)
            y = y.astype(np.int64)
            
            # Convert to appropriate format for binary classification
            y = y.reshape(-1, 1).astype(np.float32)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create neural network
            model = NeuralNetwork()
            model.add(Dense(input_size=20, output_size=64))
            model.add(Activation('relu'))
            model.add(Dense(input_size=64, output_size=32))
            model.add(Activation('relu'))
            model.add(Dense(input_size=32, output_size=1))
            model.add(Activation('sigmoid'))
            
            model.compile(loss=model._get_loss('binary_crossentropy'), learning_rate=0.001)
            
            # Measure training time
            start_time = time.time()
            history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=False)
            training_time = time.time() - start_time
            
            # Measure inference time
            inference_start = time.time()
            predictions = model.predict(X_test)
            inference_time = time.time() - inference_start
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score
            y_pred_binary = (predictions > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test.flatten(), y_pred_binary)
            
            # Calculate throughput
            throughput = len(X_test) / inference_time if inference_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (X_train.size + y_train.size + X_test.size + len(predictions)) * 4 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="DeepLearning",
                operation="neural_network",
                input_size=n_samples,
                execution_time=training_time + inference_time,
                memory_usage=memory_usage,
                throughput=throughput,
                accuracy=accuracy
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_layer_components(self, batch_sizes: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark individual layer components."""
        results = []
        
        for batch_size in batch_sizes:
            # Test Dense layer
            dense_layer = Dense(input_size=128, output_size=64)
            x = np.random.randn(batch_size, 128).astype(np.float32)
            
            start_time = time.time()
            output = dense_layer.forward(x)
            execution_time = time.time() - start_time
            
            memory_usage = (x.size + output.size) * 4 / (1024 * 1024)  # MB
            throughput = batch_size / execution_time if execution_time > 0 else float('inf')
            
            result_obj = ComponentBenchmarkResult(
                component="DeepLearning",
                operation="dense_layer",
                input_size=batch_size,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
            
            # Test Activation layer
            activation_layer = Activation('relu')
            
            start_time = time.time()
            output = activation_layer.forward(x)
            execution_time = time.time() - start_time
            
            memory_usage = (x.size + output.size) * 4 / (1024 * 1024)  # MB
            throughput = batch_size / execution_time if execution_time > 0 else float('inf')
            
            result_obj = ComponentBenchmarkResult(
                component="DeepLearning",
                operation="activation_layer",
                input_size=batch_size,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results


class LLMComponentsBenchmark:
    """Benchmark LLM components."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_attention_mechanisms(self, sequence_lengths: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark attention mechanisms."""
        results = []
        
        for seq_len in sequence_lengths:
            # Test scaled dot-product attention
            Q = torch.randn(2, seq_len, 64)
            K = torch.randn(2, seq_len, 64)
            V = torch.randn(2, seq_len, 64)
            
            start_time = time.time()
            output, attention_weights = scaled_dot_product_attention(Q, K, V)
            execution_time = time.time() - start_time
            
            # Calculate throughput (tokens processed per second)
            tokens_per_batch = 2 * seq_len  # batch_size * seq_len
            throughput = tokens_per_batch / execution_time if execution_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (Q.numel() + K.numel() + V.numel() + output.numel()) * 4 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="LLM",
                operation="scaled_dot_product_attention",
                input_size=seq_len,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_transformer_components(self, sequence_lengths: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark transformer components."""
        results = []
        
        for seq_len in sequence_lengths:
            # Test Multi-Head Attention
            mha = MultiHeadAttention(d_model=512, num_heads=8)
            x = torch.randn(2, seq_len, 512)  # batch_size=2, seq_len, d_model=512
            
            start_time = time.time()
            output = mha(x, x, x)
            execution_time = time.time() - start_time
            
            # Calculate throughput (tokens processed per second)
            tokens_per_batch = 2 * seq_len  # batch_size * seq_len
            throughput = tokens_per_batch / execution_time if execution_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (x.numel() + output.numel()) * 4 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="LLM",
                operation="multihead_attention",
                input_size=seq_len,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
            
            # Test Transformer Block
            block = TransformerBlock(d_model=256, num_heads=8, d_ff=512)
            x = torch.randn(2, seq_len, 256)
            
            start_time = time.time()
            output = block(x)
            execution_time = time.time() - start_time
            
            # Calculate throughput (tokens processed per second)
            tokens_per_batch = 2 * seq_len  # batch_size * seq_len
            throughput = tokens_per_batch / execution_time if execution_time > 0 else float('inf')
            
            # Estimate memory usage
            memory_usage = (x.numel() + output.numel()) * 4 / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="LLM",
                operation="transformer_block",
                input_size=seq_len,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results


class ProductionComponentsBenchmark:
    """Benchmark production components."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_caching(self, n_operations_list: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark caching mechanisms."""
        results = []
        
        for n_ops in n_operations_list:
            # Test LRU Cache
            cache = LRUCache(max_size=1000)
            
            start_time = time.time()
            for i in range(n_ops):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i % 100}")  # Get some values to test hit rate
            execution_time = time.time() - start_time
            
            throughput = n_ops / execution_time if execution_time > 0 else float('inf')
            
            # Memory usage is managed internally by the cache
            memory_usage = 0  # We don't have direct access to internal memory
            
            result_obj = ComponentBenchmarkResult(
                component="Production",
                operation="lru_cache",
                input_size=n_ops,
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results
    
    def benchmark_rag_pipeline(self, n_documents_list: List[int]) -> List[ComponentBenchmarkResult]:
        """Benchmark RAG pipeline."""
        results = []
        
        for n_docs in n_documents_list:
            # Create documents
            documents = [
                Document(content=f"This is document {i} with some content for RAG testing. " * 5)
                for i in range(n_docs)
            ]
            
            # Create RAG pipeline
            rag = RAGPipeline()
            
            # Measure indexing time
            indexing_start = time.time()
            rag.add_documents(documents)
            indexing_time = time.time() - indexing_start
            
            # Measure query time
            query_start = time.time()
            response = rag.query("What is this collection about?", k=min(5, n_docs))
            query_time = time.time() - query_start
            
            total_time = indexing_time + query_time
            throughput = 1 / query_time if query_time > 0 else float('inf')  # queries per second
            
            # Estimate memory usage (very rough)
            content_size = sum(len(doc.content) for doc in documents)
            memory_usage = content_size / (1024 * 1024)  # MB
            
            result_obj = ComponentBenchmarkResult(
                component="Production",
                operation="rag_pipeline",
                input_size=n_docs,
                execution_time=total_time,
                memory_usage=memory_usage,
                throughput=throughput
            )
            
            results.append(result_obj)
            self.results.append(result_obj)
        
        return results


def run_comprehensive_benchmarks(output_dir: str = "benchmark_results"):
    """Run comprehensive benchmarks for all components."""
    print("Running Comprehensive Benchmarks...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize benchmarkers
    math_bench = MathOperationsBenchmark()
    ml_bench = ClassicalMLBenchmark()
    dl_bench = DeepLearningBenchmark()
    llm_bench = LLMComponentsBenchmark()
    prod_bench = ProductionComponentsBenchmark()
    
    # Run math benchmarks
    print("Running Math Operations Benchmarks...")
    math_results = math_bench.benchmark_dot_product([100, 500, 1000, 2000])
    math_results += math_bench.benchmark_matrix_multiply([50, 100, 200])
    math_results += math_bench.benchmark_pca([100, 500, 1000])
    
    # Run ML benchmarks
    print("Running Classical ML Benchmarks...")
    ml_results = ml_bench.benchmark_linear_regression([100, 500, 1000])
    ml_results += ml_bench.benchmark_logistic_regression([100, 500, 1000])
    ml_results += ml_bench.benchmark_random_forest([100, 500, 1000])
    
    # Run Deep Learning benchmarks
    print("Running Deep Learning Benchmarks...")
    dl_results = dl_bench.benchmark_neural_network([100, 500, 1000])
    dl_results += dl_bench.benchmark_layer_components([32, 64, 128, 256])
    
    # Run LLM benchmarks
    print("Running LLM Component Benchmarks...")
    llm_results = llm_bench.benchmark_attention_mechanisms([64, 128, 256, 512])
    llm_results += llm_bench.benchmark_transformer_components([64, 128, 256])
    
    # Run Production benchmarks
    print("Running Production Component Benchmarks...")
    prod_results = prod_bench.benchmark_caching([100, 500, 1000, 2000])
    prod_results += prod_bench.benchmark_rag_pipeline([10, 50, 100, 200])
    
    # Combine all results
    all_results = math_results + ml_results + dl_results + llm_results + prod_results
    
    # Create a DataFrame for analysis
    df_data = []
    for result in all_results:
        df_data.append({
            'component': result.component,
            'operation': result.operation,
            'input_size': result.input_size,
            'execution_time': result.execution_time,
            'memory_usage': result.memory_usage,
            'throughput': result.throughput,
            'accuracy': result.accuracy,
            'timestamp': result.timestamp
        })
    
    df = pd.DataFrame(df_data)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "comprehensive_benchmarks.csv")
    df.to_csv(csv_path, index=False)
    print(f"Benchmark results saved to {csv_path}")
    
    # Create summary statistics
    summary = df.groupby(['component', 'operation']).agg({
        'execution_time': ['mean', 'std'],
        'throughput': ['mean', 'std'],
        'memory_usage': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)
    
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary.to_csv(summary_path)
    print(f"Benchmark summary saved to {summary_path}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print(summary)
    
    # Create visualizations
    create_benchmark_visualizations(df, output_dir)
    
    return df


def create_benchmark_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualizations for benchmark results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Figure 1: Execution time by component and operation
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='component', y='execution_time', hue='operation')
    plt.title('Execution Time by Component and Operation')
    plt.xlabel('Component')
    plt.ylabel('Execution Time (seconds)')
    plt.yscale('log')  # Log scale for better visualization
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_by_component.png'))
    plt.close()
    
    # Figure 2: Throughput by component
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='component', y='throughput')
    plt.title('Throughput by Component')
    plt.xlabel('Component')
    plt.ylabel('Throughput (operations/second)')
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_by_component.png'))
    plt.close()
    
    # Figure 3: Memory usage by component
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='component', y='memory_usage')
    plt.title('Memory Usage by Component')
    plt.xlabel('Component')
    plt.ylabel('Memory Usage (MB)')
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage_by_component.png'))
    plt.close()
    
    # Figure 4: Accuracy by component (for ML components)
    ml_df = df[df['accuracy'].notna()]
    if not ml_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=ml_df, x='component', y='accuracy')
        plt.title('Accuracy by Component (ML Components Only)')
        plt.xlabel('Component')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_component.png'))
        plt.close()
    
    # Figure 5: Scalability - Execution time vs Input Size
    plt.figure(figsize=(12, 8))
    for component in df['component'].unique():
        comp_data = df[df['component'] == component]
        plt.scatter(comp_data['input_size'], comp_data['execution_time'], 
                   label=component, alpha=0.7)
    plt.title('Scalability: Execution Time vs Input Size')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_execution_time.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def run_specific_component_benchmarks():
    """Run benchmarks for specific components and print results."""
    print("Running Specific Component Benchmarks...")
    
    # Math Operations
    print("\n1. Math Operations Benchmark")
    math_bench = MathOperationsBenchmark()
    results = math_bench.benchmark_dot_product([1000, 2000, 5000])
    for r in results:
        print(f"  Dot product (size {r.input_size}): {r.execution_time:.4f}s, "
              f"Throughput: {r.throughput:.2f} ops/s")
    
    # Classical ML
    print("\n2. Classical ML Benchmark")
    ml_bench = ClassicalMLBenchmark()
    results = ml_bench.benchmark_random_forest([500, 1000])
    for r in results:
        print(f"  Random Forest (samples {r.input_size}): {r.execution_time:.4f}s, "
              f"Accuracy: {r.accuracy:.3f}, Throughput: {r.throughput:.2f} inferences/s")
    
    # Deep Learning
    print("\n3. Deep Learning Benchmark")
    dl_bench = DeepLearningBenchmark()
    results = dl_bench.benchmark_layer_components([64, 128])
    for r in results:
        print(f"  {r.operation} (batch {r.input_size}): {r.execution_time:.4f}s, "
              f"Throughput: {r.throughput:.2f} batches/s")
    
    # LLM Components
    print("\n4. LLM Components Benchmark")
    llm_bench = LLMComponentsBenchmark()
    results = llm_bench.benchmark_attention_mechanisms([128, 256])
    for r in results:
        print(f"  Attention (seq_len {r.input_size}): {r.execution_time:.4f}s, "
              f"Throughput: {r.throughput:.2f} tokens/s")
    
    # Production Components
    print("\n5. Production Components Benchmark")
    prod_bench = ProductionComponentsBenchmark()
    results = prod_bench.benchmark_caching([1000, 2000])
    for r in results:
        print(f"  Cache (ops {r.input_size}): {r.execution_time:.4f}s, "
              f"Throughput: {r.throughput:.2f} ops/s")


if __name__ == "__main__":
    # Run specific component benchmarks
    run_specific_component_benchmarks()
    
    # Run comprehensive benchmarks
    df = run_comprehensive_benchmarks()
    
    print("\nAll benchmarks completed successfully!")
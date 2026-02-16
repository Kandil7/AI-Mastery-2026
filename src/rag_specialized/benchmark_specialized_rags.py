"""
Performance Benchmarks for Specialized RAG Architectures

This module provides comprehensive performance benchmarks for all five specialized RAG architectures:
- Adaptive Multi-Modal RAG
- Temporal-Aware RAG
- Graph-Enhanced RAG
- Privacy-Preserving RAG
- Continual Learning RAG

Benchmarks include:
- Query latency measurements
- Memory usage analysis
- Throughput testing
- Scalability evaluation
- Accuracy validation
- Resource utilization metrics
"""

import time
import psutil
import tracemalloc
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import datetime
import hashlib
import gc
import os
import pandas as pd
from pathlib import Path
import json

# Import specialized RAG architectures
from src.rag_specialized.adaptive_multimodal.adaptive_multimodal_rag import (
    AdaptiveMultiModalRAG, MultiModalDocument, MultiModalQuery, ModalityType
)
from src.rag_specialized.temporal_aware.temporal_aware_rag import (
    TemporalAwareRAG, TemporalDocument, TemporalQuery, TemporalScope
)
from src.rag_specialized.graph_enhanced.graph_enhanced_rag import (
    GraphEnhancedRAG, GraphDocument, GraphQuery
)
from src.rag_specialized.privacy_preserving.privacy_preserving_rag import (
    PrivacyPreservingRAG, PrivacyDocument, PrivacyQuery, PrivacyConfig, PrivacyLevel
)
from src.rag_specialized.continual_learning.continual_learning_rag import (
    ContinualLearningRAG, ContinualDocument, ContinualQuery, ForgettingMechanism
)
from src.rag_specialized.integration_layer import (
    UnifiedRAGInterface, UnifiedDocument, UnifiedQuery, RAGArchitecture
)


@dataclass
class BenchmarkResult:
    """Structure for storing benchmark results."""
    architecture: str
    operation: str
    avg_time_ms: float
    std_dev: float
    min_time_ms: float
    max_time_ms: float
    memory_peak_mb: float
    memory_avg_mb: float
    throughput_per_second: float
    sample_size: int
    timestamp: datetime.datetime
    additional_metrics: Dict[str, Any] = None


class RAGBenchmarkSuite:
    """Comprehensive benchmark suite for RAG architectures."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Initialize all RAG systems
        self.systems = {
            "Adaptive Multi-Modal": self._init_adaptive_multimodal(),
            "Temporal-Aware": self._init_temporal_aware(),
            "Graph-Enhanced": self._init_graph_enhanced(),
            "Privacy-Preserving": self._init_privacy_preserving(),
            "Continual Learning": self._init_continual_learning(),
            "Unified Interface": self._init_unified_interface()
        }
        
        print("Initialized benchmark suite with all RAG architectures")
    
    def _init_adaptive_multimodal(self):
        """Initialize Adaptive Multi-Modal RAG system."""
        return AdaptiveMultiModalRAG()
    
    def _init_temporal_aware(self):
        """Initialize Temporal-Aware RAG system."""
        return TemporalAwareRAG()
    
    def _init_graph_enhanced(self):
        """Initialize Graph-Enhanced RAG system."""
        return GraphEnhancedRAG()
    
    def _init_privacy_preserving(self):
        """Initialize Privacy-Preserving RAG system."""
        config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        return PrivacyPreservingRAG(config=config)
    
    def _init_continual_learning(self):
        """Initialize Continual Learning RAG system."""
        return ContinualLearningRAG(forgetting_mechanism=ForgettingMechanism.EXPERIENCE_REPLAY)
    
    def _init_unified_interface(self):
        """Initialize Unified Interface system."""
        return UnifiedRAGInterface()
    
    def _measure_memory(self) -> Tuple[float, float]:
        """Measure current and peak memory usage in MB."""
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # For peak memory, we'll use tracemalloc if available
        try:
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                peak_memory = peak / 1024 / 1024  # Convert to MB
            else:
                peak_memory = current_memory
        except:
            peak_memory = current_memory
            
        return current_memory, peak_memory
    
    def _warm_up_system(self, system, doc_type: str):
        """Warm up the system with dummy operations."""
        if doc_type == "multimodal":
            doc = MultiModalDocument(
                id="warmup",
                text_content="This is a warmup document for testing purposes.",
                metadata={"source": "test"}
            )
            query = MultiModalQuery(text_query="What is this?")
        elif doc_type == "temporal":
            doc = TemporalDocument(
                id="warmup",
                content="This is a warmup document for testing purposes.",
                timestamp=datetime.datetime.now()
            )
            query = TemporalQuery(text="What is this?")
        elif doc_type == "graph":
            doc = GraphDocument(
                id="warmup",
                content="This is a warmup document for testing purposes.",
                metadata={"source": "test"}
            )
            query = GraphQuery(text="What is this?")
        elif doc_type == "privacy":
            doc = PrivacyDocument(
                id="warmup",
                content="This is a warmup document for testing purposes.",
                privacy_level=PrivacyLevel.PUBLIC
            )
            query = PrivacyQuery(text="What is this?")
        elif doc_type == "continual":
            doc = ContinualDocument(
                id="warmup",
                content="This is a warmup document for testing purposes.",
                metadata={"source": "test"}
            )
            query = ContinualQuery(text="What is this?")
        else:  # unified
            doc = UnifiedDocument(
                id="warmup",
                content="This is a warmup document for testing purposes.",
                metadata={"source": "test"}
            )
            query = UnifiedQuery(text="What is this?")
        
        # Add document and perform query to warm up
        try:
            if doc_type == "multimodal":
                system.add_documents([doc])
                query_text_hash = hashlib.md5(query.text_query.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                system.query(query, query_embedding, k=1)
            elif doc_type == "temporal":
                system.add_documents([doc])
                query_text_hash = hashlib.md5(query.text.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                system.query(query, query_embedding, k=1)
            elif doc_type == "graph":
                system.add_documents([doc])
                query_text_hash = hashlib.md5(query.text.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                system.query(query, query_embedding, k=1)
            elif doc_type == "privacy":
                system.add_documents([doc])
                query_text_hash = hashlib.md5(query.text.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                system.query(query, query_embedding, k=1)
            elif doc_type == "continual":
                system.add_documents([doc])
                query_text_hash = hashlib.md5(query.text.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                system.query(query, query_embedding, k=1)
            else:  # unified
                system.add_documents([doc])
                system.query(query, k=1)
        except:
            pass  # Ignore errors during warmup
    
    def benchmark_add_documents(self, iterations: int = 10) -> List[BenchmarkResult]:
        """Benchmark document addition performance."""
        print(f"Benchmarking document addition (iterations: {iterations})")
        results = []
        
        for name, system in self.systems.items():
            print(f"  Testing {name}...")
            
            # Prepare test documents based on system type
            if name == "Adaptive Multi-Modal":
                docs = [
                    MultiModalDocument(
                        id=f"doc_{i}",
                        text_content=f"This is test document {i} with some content for benchmarking purposes.",
                        metadata={"source": "benchmark", "iteration": i}
                    ) for i in range(iterations)
                ]
                doc_type = "multimodal"
            elif name == "Temporal-Aware":
                docs = [
                    TemporalDocument(
                        id=f"doc_{i}",
                        content=f"This is test document {i} with some content for benchmarking purposes.",
                        timestamp=datetime.datetime.now() - datetime.timedelta(days=i),
                        metadata={"source": "benchmark", "iteration": i}
                    ) for i in range(iterations)
                ]
                doc_type = "temporal"
            elif name == "Graph-Enhanced":
                docs = [
                    GraphDocument(
                        id=f"doc_{i}",
                        content=f"This is test document {i} with some content for benchmarking purposes.",
                        metadata={"source": "benchmark", "iteration": i}
                    ) for i in range(iterations)
                ]
                doc_type = "graph"
            elif name == "Privacy-Preserving":
                docs = [
                    PrivacyDocument(
                        id=f"doc_{i}",
                        content=f"This is test document {i} with some content for benchmarking purposes.",
                        privacy_level=PrivacyLevel.PUBLIC,
                        metadata={"source": "benchmark", "iteration": i}
                    ) for i in range(iterations)
                ]
                doc_type = "privacy"
            elif name == "Continual Learning":
                docs = [
                    ContinualDocument(
                        id=f"doc_{i}",
                        content=f"This is test document {i} with some content for benchmarking purposes.",
                        metadata={"source": "benchmark", "iteration": i}
                    ) for i in range(iterations)
                ]
                doc_type = "continual"
            else:  # Unified Interface
                docs = [
                    UnifiedDocument(
                        id=f"doc_{i}",
                        content=f"This is test document {i} with some content for benchmarking purposes.",
                        metadata={"source": "benchmark", "iteration": i}
                    ) for i in range(iterations)
                ]
                doc_type = "unified"
            
            # Warm up the system
            self._warm_up_system(system, doc_type)
            
            # Measure memory before
            mem_before, _ = self._measure_memory()
            
            # Start tracing memory
            tracemalloc.start()
            
            # Time the operation
            times = []
            for i in range(iterations):
                start_time = time.perf_counter()
                
                # Add single document
                if name == "Unified Interface":
                    result = system.add_documents([docs[i]])
                else:
                    result = system.add_documents([docs[i]])
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Stop tracing memory
            _, peak_mem = tracemalloc.get_traced_memory()
            peak_mem_mb = peak_mem / 1024 / 1024
            tracemalloc.stop()
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_dev = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            throughput = iterations / (sum(times) / 1000)  # per second
            
            # Measure memory after
            mem_after, _ = self._measure_memory()
            avg_mem_mb = (mem_before + mem_after) / 2
            
            result = BenchmarkResult(
                architecture=name,
                operation="add_documents",
                avg_time_ms=avg_time,
                std_dev=std_dev,
                min_time_ms=min_time,
                max_time_ms=max_time,
                memory_peak_mb=peak_mem_mb,
                memory_avg_mb=avg_mem_mb,
                throughput_per_second=throughput,
                sample_size=iterations,
                timestamp=datetime.datetime.now()
            )
            
            results.append(result)
            self.benchmark_results.append(result)
            
            print(f"    Avg time: {avg_time:.2f}ms, Peak memory: {peak_mem_mb:.2f}MB, Throughput: {throughput:.2f}/sec")
        
        return results
    
    def benchmark_query_performance(self, iterations: int = 10) -> List[BenchmarkResult]:
        """Benchmark query performance."""
        print(f"Benchmarking query performance (iterations: {iterations})")
        results = []
        
        for name, system in self.systems.items():
            print(f"  Testing {name}...")
            
            # Prepare test documents and queries based on system type
            if name == "Adaptive Multi-Modal":
                docs = [
                    MultiModalDocument(
                        id=f"doc_{i}",
                        text_content=f"Content for document {i} discussing topic {i}. This is sample content for benchmarking purposes.",
                        metadata={"topic": f"topic_{i}", "id": i}
                    ) for i in range(5)  # Fewer docs for quicker setup
                ]
                system.add_documents(docs)
                
                queries = [
                    MultiModalQuery(text_query=f"What is about topic {i % 5}?")
                    for i in range(iterations)
                ]
                doc_type = "multimodal"
            elif name == "Temporal-Aware":
                base_time = datetime.datetime.now()
                docs = [
                    TemporalDocument(
                        id=f"doc_{i}",
                        content=f"Content for document {i} discussing topic {i}.",
                        timestamp=base_time - datetime.timedelta(days=i),
                        metadata={"topic": f"topic_{i}", "id": i}
                    ) for i in range(5)
                ]
                system.add_documents(docs)
                
                queries = [
                    TemporalQuery(
                        text=f"What is about topic {i % 5}?",
                        reference_time=base_time,
                        temporal_scope=TemporalScope.ALL_TIME
                    )
                    for i in range(iterations)
                ]
                doc_type = "temporal"
            elif name == "Graph-Enhanced":
                docs = [
                    GraphDocument(
                        id=f"doc_{i}",
                        content=f"Content for document {i} discussing topic {i}. John works at Company {i}.",
                        metadata={"topic": f"topic_{i}", "id": i}
                    ) for i in range(5)
                ]
                system.add_documents(docs)
                
                queries = [
                    GraphQuery(text=f"What is about topic {i % 5}?")
                    for i in range(iterations)
                ]
                doc_type = "graph"
            elif name == "Privacy-Preserving":
                docs = [
                    PrivacyDocument(
                        id=f"doc_{i}",
                        content=f"Content for document {i} discussing topic {i}.",
                        privacy_level=PrivacyLevel.PUBLIC,
                        metadata={"topic": f"topic_{i}", "id": i}
                    ) for i in range(5)
                ]
                system.add_documents(docs)
                
                queries = [
                    PrivacyQuery(text=f"What is about topic {i % 5}?")
                    for i in range(iterations)
                ]
                doc_type = "privacy"
            elif name == "Continual Learning":
                docs = [
                    ContinualDocument(
                        id=f"doc_{i}",
                        content=f"Content for document {i} discussing topic {i}.",
                        metadata={"topic": f"topic_{i}", "id": i}
                    ) for i in range(5)
                ]
                system.add_documents(docs)
                
                queries = [
                    ContinualQuery(text=f"What is about topic {i % 5}?", domain="general")
                    for i in range(iterations)
                ]
                doc_type = "continual"
            else:  # Unified Interface
                docs = [
                    UnifiedDocument(
                        id=f"doc_{i}",
                        content=f"Content for document {i} discussing topic {i}.",
                        metadata={"topic": f"topic_{i}", "id": i}
                    ) for i in range(5)
                ]
                system.add_documents([UnifiedDocument(
                    id="setup_doc",
                    content="Setup document for unified interface",
                    metadata={"source": "setup"}
                )])
                
                queries = [
                    UnifiedQuery(text=f"What is about topic {i % 5}?")
                    for i in range(iterations)
                ]
                doc_type = "unified"
            
            # Warm up the system
            self._warm_up_system(system, doc_type)
            
            # Measure memory before
            mem_before, _ = self._measure_memory()
            
            # Start tracing memory
            tracemalloc.start()
            
            # Time the operation
            times = []
            for i in range(iterations):
                start_time = time.perf_counter()
                
                # Create embedding for query (for systems that need it)
                query_text_hash = hashlib.md5(queries[i].text.encode() if hasattr(queries[i], 'text') else queries[i].text_query.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                
                # Execute query
                if name == "Adaptive Multi-Modal":
                    result = system.query(queries[i], query_embedding, k=2)
                elif name == "Temporal-Aware":
                    result = system.query(queries[i], query_embedding, k=2)
                elif name == "Graph-Enhanced":
                    result = system.query(queries[i], query_embedding, k=2)
                elif name == "Privacy-Preserving":
                    result = system.query(queries[i], query_embedding, k=2)
                elif name == "Continual Learning":
                    result = system.query(queries[i], query_embedding, k=2)
                else:  # Unified Interface
                    result = system.query(queries[i], k=2)
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Stop tracing memory
            _, peak_mem = tracemalloc.get_traced_memory()
            peak_mem_mb = peak_mem / 1024 / 1024
            tracemalloc.stop()
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_dev = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            throughput = iterations / (sum(times) / 1000)  # per second
            
            # Measure memory after
            mem_after, _ = self._measure_memory()
            avg_mem_mb = (mem_before + mem_after) / 2
            
            result = BenchmarkResult(
                architecture=name,
                operation="query",
                avg_time_ms=avg_time,
                std_dev=std_dev,
                min_time_ms=min_time,
                max_time_ms=max_time,
                memory_peak_mb=peak_mem_mb,
                memory_avg_mb=avg_mem_mb,
                throughput_per_second=throughput,
                sample_size=iterations,
                timestamp=datetime.datetime.now()
            )
            
            results.append(result)
            self.benchmark_results.append(result)
            
            print(f"    Avg time: {avg_time:.2f}ms, Peak memory: {peak_mem_mb:.2f}MB, Throughput: {throughput:.2f}/sec")
        
        return results
    
    def benchmark_scalability(self, doc_counts: List[int] = [10, 50, 100]) -> List[BenchmarkResult]:
        """Benchmark scalability with varying document counts."""
        print(f"Benchmarking scalability with document counts: {doc_counts}")
        results = []
        
        for name, system in self.systems.items():
            if name == "Unified Interface":
                continue  # Skip for brevity in this example
                
            print(f"  Testing {name} scalability...")
            
            for doc_count in doc_counts:
                print(f"    With {doc_count} documents...")
                
                # Prepare documents based on system type
                if name == "Adaptive Multi-Modal":
                    docs = [
                        MultiModalDocument(
                            id=f"scale_doc_{i}",
                            text_content=f"Scalability test document {i} with content for benchmarking. " * 5,
                            metadata={"topic": f"scale_topic_{i % 10}", "id": i}
                        ) for i in range(doc_count)
                    ]
                    query = MultiModalQuery(text_query="What are these documents about?")
                    doc_type = "multimodal"
                elif name == "Temporal-Aware":
                    base_time = datetime.datetime.now()
                    docs = [
                        TemporalDocument(
                            id=f"scale_doc_{i}",
                            content=f"Scalability test document {i} with content for benchmarking. " * 5,
                            timestamp=base_time - datetime.timedelta(hours=i),
                            metadata={"topic": f"scale_topic_{i % 10}", "id": i}
                        ) for i in range(doc_count)
                    ]
                    query = TemporalQuery(text="What are these documents about?")
                    doc_type = "temporal"
                elif name == "Graph-Enhanced":
                    docs = [
                        GraphDocument(
                            id=f"scale_doc_{i}",
                            content=f"Scalability test document {i} about Topic_{i % 10}. John knows about this topic.",
                            metadata={"topic": f"scale_topic_{i % 10}", "id": i}
                        ) for i in range(doc_count)
                    ]
                    query = GraphQuery(text="What are these documents about?")
                    doc_type = "graph"
                elif name == "Privacy-Preserving":
                    docs = [
                        PrivacyDocument(
                            id=f"scale_doc_{i}",
                            content=f"Scalability test document {i} with content for benchmarking. " * 5,
                            privacy_level=PrivacyLevel.PUBLIC,
                            metadata={"topic": f"scale_topic_{i % 10}", "id": i}
                        ) for i in range(doc_count)
                    ]
                    query = PrivacyQuery(text="What are these documents about?")
                    doc_type = "privacy"
                elif name == "Continual Learning":
                    docs = [
                        ContinualDocument(
                            id=f"scale_doc_{i}",
                            content=f"Scalability test document {i} with content for benchmarking. " * 5,
                            metadata={"topic": f"scale_topic_{i % 10}", "id": i}
                        ) for i in range(doc_count)
                    ]
                    query = ContinualQuery(text="What are these documents about?", domain="general")
                    doc_type = "continual"
                
                # Add documents
                system.add_documents(docs)
                
                # Warm up
                self._warm_up_system(system, doc_type)
                
                # Measure memory before
                mem_before, _ = self._measure_memory()
                
                # Start tracing memory
                tracemalloc.start()
                
                # Time a single query
                start_time = time.perf_counter()
                
                # Create embedding for query
                query_text_hash = hashlib.md5(query.text.encode() if hasattr(query, 'text') else query.text_query.encode()).hexdigest()
                query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
                if len(query_embedding) < 384:
                    query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
                elif len(query_embedding) > 384:
                    query_embedding = query_embedding[:384]
                
                # Execute query
                if name == "Adaptive Multi-Modal":
                    result = system.query(query, query_embedding, k=3)
                elif name == "Temporal-Aware":
                    result = system.query(query, query_embedding, k=3)
                elif name == "Graph-Enhanced":
                    result = system.query(query, query_embedding, k=3)
                elif name == "Privacy-Preserving":
                    result = system.query(query, query_embedding, k=3)
                elif name == "Continual Learning":
                    result = system.query(query, query_embedding, k=3)
                
                end_time = time.perf_counter()
                query_time_ms = (end_time - start_time) * 1000
                
                # Stop tracing memory
                _, peak_mem = tracemalloc.get_traced_memory()
                peak_mem_mb = peak_mem / 1024 / 1024
                tracemalloc.stop()
                
                # Measure memory after
                mem_after, _ = self._measure_memory()
                avg_mem_mb = (mem_before + mem_after) / 2
                
                result = BenchmarkResult(
                    architecture=name,
                    operation=f"query_scalability_{doc_count}",
                    avg_time_ms=query_time_ms,
                    std_dev=0.0,  # Single measurement
                    min_time_ms=query_time_ms,
                    max_time_ms=query_time_ms,
                    memory_peak_mb=peak_mem_mb,
                    memory_avg_mb=avg_mem_mb,
                    throughput_per_second=1 / (query_time_ms / 1000),
                    sample_size=1,
                    timestamp=datetime.datetime.now(),
                    additional_metrics={"document_count": doc_count}
                )
                
                results.append(result)
                self.benchmark_results.append(result)
                
                print(f"      Query time: {query_time_ms:.2f}ms, Peak memory: {peak_mem_mb:.2f}MB")
        
        return results
    
    def generate_benchmark_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.benchmark_results:
            return "No benchmark results available."
        
        report = []
        report.append("# RAG Architectures Performance Benchmark Report\n")
        report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary table
        report.append("## Executive Summary\n")
        summary_data = []
        for result in self.benchmark_results:
            if result.operation == "query":
                summary_data.append({
                    'Architecture': result.architecture,
                    'Avg Latency (ms)': f"{result.avg_time_ms:.2f}",
                    'Throughput (/s)': f"{result.throughput_per_second:.2f}",
                    'Peak Memory (MB)': f"{result.memory_peak_mb:.2f}"
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            report.append(df.to_string(index=False))
            report.append("\n")
        
        # Detailed results by operation
        operations = set(r.operation for r in self.benchmark_results)
        for op in operations:
            report.append(f"## {op.replace('_', ' ').title()} Results\n")
            
            op_results = [r for r in self.benchmark_results if r.operation == op]
            op_data = []
            for result in op_results:
                op_data.append({
                    'Architecture': result.architecture,
                    'Avg Time (ms)': f"{result.avg_time_ms:.2f}",
                    'Std Dev (ms)': f"{result.std_dev:.2f}",
                    'Min Time (ms)': f"{result.min_time_ms:.2f}",
                    'Max Time (ms)': f"{result.max_time_ms:.2f}",
                    'Throughput (/s)': f"{result.throughput_per_second:.2f}",
                    'Peak Memory (MB)': f"{result.memory_peak_mb:.2f}",
                    'Avg Memory (MB)': f"{result.memory_avg_mb:.2f}"
                })
            
            if op_data:
                df_op = pd.DataFrame(op_data)
                report.append(df_op.to_string(index=False))
                report.append("\n")
        
        # Scalability results if available
        scalability_results = [r for r in self.benchmark_results if "scalability" in r.operation]
        if scalability_results:
            report.append("## Scalability Analysis\n")
            scale_data = []
            for result in scalability_results:
                doc_count = result.additional_metrics.get('document_count', 0) if result.additional_metrics else 0
                scale_data.append({
                    'Architecture': result.architecture,
                    'Documents': doc_count,
                    'Query Time (ms)': f"{result.avg_time_ms:.2f}",
                    'Peak Memory (MB)': f"{result.memory_peak_mb:.2f}"
                })
            
            if scale_data:
                df_scale = pd.DataFrame(scale_data)
                report.append(df_scale.to_string(index=False))
                report.append("\n")
        
        return "\n".join(report)
    
    def plot_benchmark_results(self):
        """Generate visualizations of benchmark results."""
        if not self.benchmark_results:
            print("No benchmark results to plot.")
            return
        
        # Create figures directory
        fig_dir = self.results_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # Filter results for query operation only for cleaner plots
        query_results = [r for r in self.benchmark_results if r.operation == "query"]
        
        if not query_results:
            print("No query benchmark results to plot.")
            return
        
        # Prepare data for plotting
        architectures = [r.architecture for r in query_results]
        avg_times = [r.avg_time_ms for r in query_results]
        peak_memory = [r.memory_peak_mb for r in query_results]
        throughput = [r.throughput_per_second for r in query_results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG Architectures Performance Comparison', fontsize=16)
        
        # Plot 1: Average Query Time
        bars1 = axes[0, 0].bar(architectures, avg_times, color='skyblue', edgecolor='navy', linewidth=1.2)
        axes[0, 0].set_title('Average Query Time (ms)')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_times):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Peak Memory Usage
        bars2 = axes[0, 1].bar(architectures, peak_memory, color='lightgreen', edgecolor='darkgreen', linewidth=1.2)
        axes[0, 1].set_title('Peak Memory Usage (MB)')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add value labels on bars
        for bar, value in zip(bars2, peak_memory):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peak_memory)*0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Throughput
        bars3 = axes[1, 0].bar(architectures, throughput, color='salmon', edgecolor='darkred', linewidth=1.2)
        axes[1, 0].set_title('Throughput (queries/second)')
        axes[1, 0].set_ylabel('Queries per Second')
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add value labels on bars
        for bar, value in zip(bars3, throughput):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput)*0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Time vs Memory Scatter
        scatter = axes[1, 1].scatter(avg_times, peak_memory, s=100, c=range(len(architectures)), cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('Average Query Time (ms)')
        axes[1, 1].set_ylabel('Peak Memory (MB)')
        axes[1, 1].set_title('Time vs Memory Trade-off')
        
        # Add architecture labels to scatter points
        for i, arch in enumerate(architectures):
            axes[1, 1].annotate(arch, (avg_times[i], peak_memory[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, ha='left')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison chart saved to {fig_dir / 'performance_comparison.png'}")
        
        # Create scalability plot if available
        scalability_results = [r for r in self.benchmark_results if "scalability" in r.operation]
        if scalability_results:
            # Group by architecture
            from collections import defaultdict
            scale_data = defaultdict(list)
            for r in scalability_results:
                if r.additional_metrics:
                    doc_count = r.additional_metrics.get('document_count', 0)
                    scale_data[r.architecture].append((doc_count, r.avg_time_ms))
            
            if scale_data:
                plt.figure(figsize=(12, 8))
                for arch, data in scale_data.items():
                    if data:  # Only plot if there's data
                        data_sorted = sorted(data, key=lambda x: x[0])
                        doc_counts, times = zip(*data_sorted)
                        plt.plot(doc_counts, times, marker='o', label=arch, linewidth=2, markersize=6)
                
                plt.xlabel('Number of Documents')
                plt.ylabel('Query Time (ms)')
                plt.title('Scalability: Query Time vs Document Count')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(fig_dir / "scalability_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Scalability analysis chart saved to {fig_dir / 'scalability_analysis.png'}")
    
    def export_results(self, filename: str = None):
        """Export benchmark results to JSON file."""
        if filename is None:
            filename = f"benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_path = self.results_dir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.benchmark_results:
            result_dict = {
                'architecture': result.architecture,
                'operation': result.operation,
                'avg_time_ms': result.avg_time_ms,
                'std_dev': result.std_dev,
                'min_time_ms': result.min_time_ms,
                'max_time_ms': result.max_time_ms,
                'memory_peak_mb': result.memory_peak_mb,
                'memory_avg_mb': result.memory_avg_mb,
                'throughput_per_second': result.throughput_per_second,
                'sample_size': result.sample_size,
                'timestamp': result.timestamp.isoformat(),
                'additional_metrics': result.additional_metrics
            }
            serializable_results.append(result_dict)
        
        with open(export_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Benchmark results exported to {export_path}")
    
    def run_complete_benchmark_suite(self):
        """Run the complete benchmark suite."""
        print("="*60)
        print("RUNNING COMPLETE BENCHMARK SUITE FOR SPECIALIZED RAG ARCHITECTURES")
        print("="*60)
        
        # Run all benchmarks
        print("\n1. Running document addition benchmark...")
        add_results = self.benchmark_add_documents(iterations=5)
        
        print("\n2. Running query performance benchmark...")
        query_results = self.benchmark_query_performance(iterations=10)
        
        print("\n3. Running scalability benchmark...")
        scale_results = self.benchmark_scalability(doc_counts=[10, 25, 50])
        
        print("\n4. Generating benchmark report...")
        report = self.generate_benchmark_report()
        
        print("\n5. Creating visualizations...")
        self.plot_benchmark_results()
        
        print("\n6. Exporting results...")
        self.export_results()
        
        print(f"\nBenchmark completed! {len(self.benchmark_results)} results recorded.")
        print(f"Results exported to {self.results_dir}")
        
        # Print summary
        print("\nSUMMARY:")
        print("-" * 40)
        for result in self.benchmark_results:
            if result.operation == "query":
                print(f"{result.architecture:20} | {result.avg_time_ms:6.2f}ms | {result.memory_peak_mb:6.2f}MB | {result.throughput_per_second:6.2f}q/s")
        
        return report


def run_benchmarks():
    """Run the complete benchmark suite and return results."""
    benchmark_suite = RAGBenchmarkSuite()
    report = benchmark_suite.run_complete_benchmark_suite()
    return report


if __name__ == "__main__":
    print("Starting performance benchmarks for specialized RAG architectures...")
    print("This may take several minutes to complete.\n")
    
    report = run_benchmarks()
    
    # Print final report
    print("\n" + "="*60)
    print("BENCHMARK REPORT")
    print("="*60)
    print(report)
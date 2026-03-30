"""
Production Engineering Module
=============================

Enterprise-ready components for production RAG systems.

Implements the 5 pillars of production RAG:
1. Data Pipeline - Semantic/hierarchical chunking, metadata extraction
2. Hybrid Retrieval - (uses src/retrieval module)
3. Query Enhancement - Query rewriting, HyDE, multi-query
4. Evaluation & Observability - Metrics, quality monitoring
5. Cost Optimization - Semantic caching, model routing

Reference: docs/PRODUCTION_RAG_GUIDE.md
"""

# Cost Optimization (Pillar 5)
from src.production.caching import (CacheEntry, CostOptimizer, CostReport,
                                    CostTracker, ModelRouter,
                                    QueryClassification, SemanticCache)
# Data Pipeline (Pillar 1)
from src.production.data_pipeline import (BaseChunker, Document, DocumentChunk,
                                          DocumentParser, FixedSizeChunker,
                                          HierarchicalChunker,
                                          MetadataExtractor,
                                          ProductionDataPipeline,
                                          SemanticChunker)
# Observability (Pillar 4)
from src.production.observability import (LatencyStats, LatencyTracker,
                                          MetricPoint, QualityMonitor,
                                          QualityScore, RAGMetrics,
                                          RAGObservability)
# Query Enhancement (Pillar 3)
from src.production.query_enhancement import (EnhancedQuery, HyDEGenerator,
                                              MultiQueryGenerator,
                                              QueryEnhancementPipeline,
                                              QueryRewriter, SynonymExpander)

__all__ = [
    # Data Pipeline
    "Document",
    "DocumentChunk",
    "BaseChunker",
    "FixedSizeChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "MetadataExtractor",
    "DocumentParser",
    "ProductionDataPipeline",
    # Query Enhancement
    "EnhancedQuery",
    "QueryRewriter",
    "HyDEGenerator",
    "MultiQueryGenerator",
    "SynonymExpander",
    "QueryEnhancementPipeline",
    # Cost Optimization
    "CacheEntry",
    "QueryClassification",
    "CostReport",
    "SemanticCache",
    "ModelRouter",
    "CostTracker",
    "CostOptimizer",
    # Observability
    "MetricPoint",
    "LatencyStats",
    "QualityScore",
    "LatencyTracker",
    "QualityMonitor",
    "RAGMetrics",
    "RAGObservability",
]

# Module 3: Graph-Enhanced RAG (Knowledge Graphs + Vectors)

## 📋 Module Overview

**Duration:** 3-4 weeks (20-25 hours)  
**Difficulty:** Advanced  
**Prerequisites:** Module 1-2 completion, Graph database basics

This module teaches you to combine knowledge graphs with vector retrieval for enhanced reasoning, relationship-aware search, and improved answer quality.

---

## 🎯 Learning Objectives

### Remember
- Define knowledge graphs and their components (entities, relations)
- Identify graph traversal operations
- Recall graph-vector hybrid architectures

### Understand
- Explain how graphs enhance vector retrieval
- Describe entity linking and relationship extraction
- Summarize graph-based query expansion techniques

### Apply
- Build knowledge graph from documents
- Implement graph-vector hybrid retrieval
- Create graph-based query expansion

### Analyze
- Compare pure vector vs. graph-enhanced retrieval
- Diagnose entity linking errors
- Evaluate graph traversal strategies

### Evaluate
- Assess when graph enhancement is beneficial
- Critique knowledge graph construction quality
- Judge reasoning capabilities of hybrid systems

### Create
- Design end-to-end graph-enhanced RAG architectures
- Develop custom graph traversal algorithms
- Build entity-aware retrieval pipelines

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              GRAPH-ENHANCED RAG ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Documents ──▶ [Knowledge Graph Construction]               │
│                  │                                          │
│                  ├──▶ Entity Extraction                     │
│                  ├──▶ Relation Extraction                   │
│                  └──▶ Graph Indexing                        │
│                                                             │
│  Query ──▶ [Query Understanding]                            │
│              │                                              │
│              ├──▶ Entity Recognition                        │
│              ├──▶ Intent Classification                     │
│              └──▶ Query Graph Construction                  │
│                                                             │
│              ▼                                              │
│         [Hybrid Retrieval]                                  │
│              │                                              │
│              ├──▶ Vector Search (semantic similarity)       │
│              ├──▶ Graph Traversal (relationship paths)      │
│              └──▶ Result Fusion                             │
│                                                             │
│              ▼                                              │
│         [Graph-Augmented Generation]                        │
│              │                                              │
│              ├──▶ Entity-aware context assembly             │
│              ├──▶ Relationship-aware reasoning              │
│              └──▶ Structured answer generation              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Knowledge Graph Structure

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Entity:
    id: str
    name: str
    type: str
    attributes: Dict[str, any]
    embeddings: Dict[str, List[float]]  # Multiple embedding types

@dataclass
class Relation:
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    weight: float
    metadata: Dict[str, any]

class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.adjacency_list: Dict[str, List[str]] = {}
```

### Graph-Vector Hybrid Search

```python
async def hybrid_search(query: str, vector_index, graph, top_k: int = 10):
    # Step 1: Extract entities from query
    query_entities = extract_entities(query)
    
    # Step 2: Vector search for semantic similarity
    vector_results = await vector_index.search(query, top_k=top_k)
    
    # Step 3: Graph traversal from query entities
    graph_results = []
    for entity in query_entities:
        if entity in graph.entities:
            # Traverse 2-hop neighborhood
            neighbors = graph.traverse(entity, max_hops=2)
            graph_results.extend(neighbors)
    
    # Step 4: Fuse results
    fused = reciprocal_rank_fusion([vector_results, graph_results])
    
    return fused[:top_k]
```

---

## 📚 Module Structure

```
module_3_graph_enhanced/
├── README.md
├── theory/
│   ├── 01_knowledge_graph_fundamentals.md
│   ├── 02_entity_relation_extraction.md
│   ├── 03_graph_vector_fusion.md
│   ├── 04_graph_traversal_strategies.md
│   └── 05_reasoning_with_graphs.md
├── labs/
│   ├── lab_1_kg_construction/
│   ├── lab_2_hybrid_retrieval/
│   └── lab_3_graph_reasoning/
├── knowledge_checks/
├── coding_challenges/
├── solutions/
└── further_reading.md
```

---

## Use Cases

| Use Case | Graph Benefit | Example |
|----------|--------------|---------|
| Multi-hop QA | Relationship traversal | "Who founded the company that acquired X?" |
| Recommendation | Similar entity discovery | "Products similar to X by same manufacturer" |
| Fact Verification | Path validation | "Verify claim through entity relationships" |
| Domain Expert | Structured knowledge | "Medical diagnosis with symptom-disease graph" |

---

*Last Updated: March 30, 2026*

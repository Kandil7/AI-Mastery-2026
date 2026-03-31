# Module 5: Continual Learning RAG

## 📋 Module Overview

**Duration:** 3-4 weeks (20-25 hours)  
**Difficulty:** Advanced  
**Prerequisites:** Module 1-4 completion, ML fundamentals

This module teaches you to build RAG systems that continuously learn from new data, user feedback, and interactions while avoiding catastrophic forgetting and maintaining retrieval quality.

---

## 🎯 Learning Objectives

### Remember
- Define continual learning and catastrophic forgetting
- Identify feedback types (explicit, implicit)
- Recall update strategies for retrieval systems

### Understand
- Explain the stability-plasticity dilemma
- Describe feedback loop architectures
- Summarize embedding update techniques

### Apply
- Implement feedback collection pipelines
- Build incremental index updates
- Create embedding fine-tuning workflows

### Analyze
- Compare batch vs. streaming updates
- Diagnose forgetting in retrieval quality
- Evaluate feedback signal quality

### Evaluate
- Assess when to trigger system updates
- Critique feedback weighting strategies
- Judge update frequency trade-offs

### Create
- Design end-to-end continual learning architectures
- Develop custom feedback integration algorithms
- Build quality monitoring systems

---

## 🔄 Continual Learning Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              CONTINUAL LEARNING RAG ARCHITECTURE            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Feedback Collection Layer]                                │
│              │                                              │
│              ├──▶ Explicit feedback (ratings, corrections)  │
│              ├──▶ Implicit feedback (clicks, dwell time)    │
│              └──▶ Conversation history                      │
│                                                             │
│              ▼                                              │
│         [Feedback Processing]                               │
│              │                                              │
│              ├──▶ Quality filtering                         │
│              ├──▶ Signal aggregation                        │
│              └──▶ Priority scoring                          │
│                                                             │
│              ▼                                              │
│         [Learning Strategies]                               │
│              │                                              │
│              ├──▶ Embedding fine-tuning                     │
│              ├──▶ Index updates (add/modify/delete)         │
│              └──▶ Router weight adjustment                  │
│                                                             │
│              ▼                                              │
│         [Quality Validation]                                │
│              │                                              │
│              ├──▶ A/B testing                               │
│              ├──▶ Regression testing                        │
│              └──▶ Forgetting detection                      │
│                                                             │
│              ▼                                              │
│         [Deployment]                                        │
│              │                                              │
│              ├──▶ Canary deployment                         │
│              ├──▶ Rollback capability                       │
│              └──▶ Monitoring dashboards                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Feedback Types and Collection

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

class FeedbackType(Enum):
    EXPLICIT_POSITIVE = "explicit_positive"    # Thumbs up, 5-star rating
    EXPLICIT_NEGATIVE = "explicit_negative"    # Thumbs down, correction
    IMPLICIT_POSITIVE = "implicit_positive"    # Click, long dwell time
    IMPLICIT_NEGATIVE = "implicit_negative"    # Skip, quick bounce
    CORRECTION = "correction"                   # User-provided correction

@dataclass
class Feedback:
    feedback_id: str
    query: str
    result_id: str
    feedback_type: FeedbackType
    value: float  # Normalized score -1 to 1
    timestamp: datetime
    user_id: Optional[str] = None
    metadata: Dict = None

class FeedbackCollector:
    def __init__(self):
        self.feedback_store = []
    
    def collect_explicit(self, query: str, result_id: str, 
                         rating: int, user_id: str = None) -> Feedback:
        """Collect explicit feedback (1-5 star rating)."""
        # Normalize rating to -1 to 1 scale
        value = (rating - 3) / 2  # 3 stars = 0, 5 stars = 1, 1 star = -1
        
        feedback = Feedback(
            feedback_id=f"fb_{datetime.now().timestamp()}",
            query=query,
            result_id=result_id,
            feedback_type=FeedbackType.EXPLICIT_POSITIVE if rating >= 4 
                          else FeedbackType.EXPLICIT_NEGATIVE,
            value=value,
            timestamp=datetime.now(),
            user_id=user_id
        )
        self.feedback_store.append(feedback)
        return feedback
    
    def collect_implicit(self, query: str, result_id: str,
                         clicked: bool, dwell_time_seconds: float,
                         user_id: str = None) -> Feedback:
        """Collect implicit feedback from user behavior."""
        # Calculate implicit score from behavior
        if not clicked:
            value = -0.3  # Skip indicates low relevance
        elif dwell_time_seconds < 5:
            value = -0.2  # Quick bounce indicates poor match
        elif dwell_time_seconds > 60:
            value = 0.8  # Long engagement indicates good match
        else:
            value = 0.3  # Moderate engagement
        
        feedback = Feedback(
            feedback_id=f"fb_{datetime.now().timestamp()}",
            query=query,
            result_id=result_id,
            feedback_type=FeedbackType.IMPLICIT_POSITIVE if value > 0
                          else FeedbackType.IMPLICIT_NEGATIVE,
            value=value,
            timestamp=datetime.now(),
            user_id=user_id,
            metadata={'clicked': clicked, 'dwell_time': dwell_time_seconds}
        )
        self.feedback_store.append(feedback)
        return feedback
```

### Incremental Index Updates

```python
import numpy as np
from typing import List, Dict

class IncrementalIndex:
    """
    Vector index that supports incremental updates without full rebuild.
    """
    
    def __init__(self, dimension: int, batch_size: int = 1000):
        self.dimension = dimension
        self.batch_size = batch_size
        self.vectors = []
        self.metadata = []
        self.pending_additions = []
        self.pending_deletions = set()
        
    def add(self, doc_id: str, vector: np.ndarray, metadata: dict):
        """Add document to pending additions."""
        self.pending_additions.append({
            'id': doc_id,
            'vector': vector,
            'metadata': metadata
        })
        
        # Trigger batch update if threshold reached
        if len(self.pending_additions) >= self.batch_size:
            self.commit_additions()
    
    def delete(self, doc_id: str):
        """Mark document for deletion."""
        self.pending_deletions.add(doc_id)
    
    def update(self, doc_id: str, vector: np.ndarray, metadata: dict):
        """Update existing document."""
        self.delete(doc_id)
        self.add(doc_id, vector, metadata)
    
    def commit_additions(self):
        """Commit pending additions to index."""
        for item in self.pending_additions:
            if item['id'] not in self.pending_deletions:
                self.vectors.append(item['vector'])
                self.metadata.append({
                    'id': item['id'],
                    **item['metadata']
                })
        
        self.pending_additions = []
        self.pending_deletions = set()
        
        # Rebuild index structure (HNSW, IVF, etc.)
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild underlying index structure."""
        # Implementation depends on vector DB (Pinecone, Qdrant, etc.)
        pass
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[dict]:
        """Search index (includes pending additions in search)."""
        # Search committed vectors
        results = self._search_committed(query_vector, top_k)
        
        # Also search pending additions
        pending_results = self._search_pending(query_vector, top_k // 2)
        
        # Merge and rerank
        merged = self._merge_results(results, pending_results)
        return merged[:top_k]
```

### Embedding Fine-Tuning with Feedback

```python
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

class FeedbackBasedFineTuning:
    """
    Fine-tune embedding model using collected feedback.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.training_data = []
    
    def prepare_training_data(self, feedback_list: List[Feedback],
                              queries: Dict[str, str],
                              documents: Dict[str, str]):
        """
        Prepare training pairs from feedback.
        
        Positive feedback → (query, document) as positive pair
        Negative feedback → (query, document) as negative pair
        """
        for feedback in feedback_list:
            query = queries.get(feedback.query, feedback.query)
            doc = documents.get(feedback.result_id, "")
            
            if feedback.value > 0.5:  # Strong positive
                self.training_data.append(
                    InputExample(texts=[query, doc], label=1.0)
                )
            elif feedback.value < -0.5:  # Strong negative
                self.training_data.append(
                    InputExample(texts=[query, doc], label=0.0)
                )
    
    def fine_tune(self, epochs: int = 3, batch_size: int = 16):
        """Fine-tune model on feedback data."""
        from sentence_transformers import losses
        
        train_loader = DataLoader(
            self.training_data, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=epochs,
            warmup_steps=100
        )
    
    def evaluate_improvement(self, test_queries: List[str],
                             expected_docs: List[str]) -> dict:
        """Evaluate model improvement after fine-tuning."""
        # Calculate recall@k before and after
        # Implementation depends on evaluation setup
        pass
```

---

## 📚 Module Structure

```
module_5_continual_learning/
├── README.md
├── theory/
│   ├── 01_continual_learning_fundamentals.md
│   ├── 02_feedback_collection.md
│   ├── 03_incremental_updates.md
│   ├── 04_embedding_finetuning.md
│   └── 05_quality_monitoring.md
├── labs/
│   ├── lab_1_feedback_pipeline/
│   ├── lab_2_incremental_index/
│   └── lab_3_model_finetuning/
├── knowledge_checks/
├── coding_challenges/
├── solutions/
└── further_reading.md
```

---

## Forgetting Detection

```python
class ForgettingDetector:
    """Detect catastrophic forgetting in retrieval quality."""
    
    def __init__(self, baseline_metrics: dict):
        self.baseline = baseline_metrics
        self.history = []
    
    def record_metrics(self, metrics: dict):
        """Record current metrics for comparison."""
        self.history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
    
    def detect_forgetting(self, threshold: float = 0.1) -> bool:
        """
        Detect if retrieval quality has degraded significantly.
        
        Returns True if forgetting detected.
        """
        if len(self.history) < 2:
            return False
        
        current = self.history[-1]['metrics']
        
        # Check key metrics against baseline
        for metric_name, baseline_value in self.baseline.items():
            current_value = current.get(metric_name, 0)
            degradation = (baseline_value - current_value) / baseline_value
            
            if degradation > threshold:
                return True
        
        return False
```

---

*Last Updated: March 30, 2026*

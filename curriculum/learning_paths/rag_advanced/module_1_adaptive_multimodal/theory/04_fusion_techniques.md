# Theory 4: Fusion Techniques for Multimodal Results

## 4.1 Overview

After retrieving results from multiple indexes, fusion techniques combine and rank them for optimal presentation.

## 4.2 Reciprocal Rank Fusion (RRF)

### 4.2.1 Algorithm

```python
def reciprocal_rank_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion.
    
    Args:
        results: List of result lists from different indexes
        k: Constant for RRF formula (typically 60)
    
    Returns:
        Fused and ranked results
    """
    score_map = {}
    
    for result_list in results:
        for rank, doc in enumerate(result_list):
            doc_id = doc['id']
            rrf_score = 1 / (k + rank + 1)
            
            if doc_id not in score_map:
                score_map[doc_id] = {**doc, 'rrf_score': 0}
            score_map[doc_id]['rrf_score'] += rrf_score
    
    # Sort by RRF score
    fused = sorted(score_map.values(), key=lambda x: x['rrf_score'], reverse=True)
    return fused
```

### 4.2.2 Weighted RRF

```python
def weighted_rrf(results: List[tuple], weights: List[float], k: int = 60) -> List[dict]:
    """RRF with index-specific weights."""
    score_map = {}
    
    for (result_list, weight) in zip(results, weights):
        for rank, doc in enumerate(result_list):
            doc_id = doc['id']
            rrf_score = weight / (k + rank + 1)
            
            if doc_id not in score_map:
                score_map[doc_id] = {**doc, 'rrf_score': 0}
            score_map[doc_id]['rrf_score'] += rrf_score
    
    return sorted(score_map.values(), key=lambda x: x['rrf_score'], reverse=True)
```

## 4.3 Score Normalization

### 4.3.1 Min-Max Normalization

```python
def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]
```

### 4.3.2 Z-Score Normalization

```python
import numpy as np

def z_score_normalize(scores: List[float]) -> List[float]:
    """Normalize using z-score."""
    arr = np.array(scores)
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        return [0.0] * len(scores)
    
    return ((arr - mean) / std).tolist()
```

## 4.4 Cross-Modal Relevance Scoring

```python
class CrossModalScorer:
    """Scores relevance across different modalities."""
    
    def __init__(self, modality_weights: Dict[str, float]):
        self.weights = modality_weights
    
    def score(self, query: str, results: List[dict]) -> List[dict]:
        """Apply cross-modal relevance scoring."""
        for doc in results:
            modality = doc.get('modality', 'text')
            base_score = doc.get('score', 0.5)
            
            # Adjust based on query-modality alignment
            alignment = self._calculate_alignment(query, modality)
            adjusted_score = base_score * alignment * self.weights.get(modality, 1.0)
            
            doc['adjusted_score'] = adjusted_score
        
        return sorted(results, key=lambda x: x['adjusted_score'], reverse=True)
    
    def _calculate_alignment(self, query: str, modality: str) -> float:
        """Calculate how well modality matches query intent."""
        query_lower = query.lower()
        
        alignment_scores = {
            'image': sum(1 for w in ['diagram', 'chart', 'visual'] if w in query_lower),
            'code': sum(1 for w in ['code', 'function', 'example'] if w in query_lower),
            'table': sum(1 for w in ['data', 'table', 'metrics'] if w in query_lower),
            'text': 1.0  # Default
        }
        
        base = alignment_scores.get(modality, 0.5)
        return 0.5 + (base * 0.5)  # Range: 0.5 - 1.0
```

## 4.5 Diversity Optimization

```python
def diversity_rerank(results: List[dict], diversity_factor: float = 0.3) -> List[dict]:
    """
    Rerank results to maximize diversity while maintaining relevance.
    Uses Maximal Marginal Relevance (MMR) approach.
    """
    if not results:
        return []
    
    selected = []
    remaining = results.copy()
    
    while remaining:
        # Find document with best MMR score
        best_doc = None
        best_mmr = -float('inf')
        
        for doc in remaining:
            relevance = doc.get('score', 0.5)
            
            # Calculate max similarity to already selected docs
            max_similarity = 0
            if selected:
                max_similarity = max(
                    self._calculate_similarity(doc, sel) for sel in selected
                )
            
            # MMR formula
            mmr = diversity_factor * relevance - (1 - diversity_factor) * max_similarity
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_doc = doc
        
        selected.append(best_doc)
        remaining.remove(best_doc)
    
    return selected
```

## 4.6 Summary

Key fusion techniques:
- RRF for combining ranked lists
- Score normalization for cross-index comparison
- Cross-modal relevance for query alignment
- Diversity optimization for varied results

# Lab 3: Multimodal Fusion and Ranking

## 🎯 Lab Objectives

By completing this lab, you will:
1. Implement Reciprocal Rank Fusion (RRF) for combining results
2. Build score normalization across different modalities
3. Create cross-modal relevance scoring
4. Implement diversity optimization with MMR
5. Evaluate fusion quality with metrics

## 📋 Prerequisites

- Completion of Labs 1 and 2
- Understanding of ranking algorithms
- Python 3.10+ with numpy

## ⏱️ Time Estimate: 3-4 hours

---

## Part 1: Reciprocal Rank Fusion

### Task 1.1: Basic RRF Implementation

```python
def reciprocal_rank_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion.
    
    RRF Formula: score = sum(1 / (k + rank)) for each result
    
    Args:
        results: List of result lists from different indexes
        k: Constant for RRF formula (typically 60)
    
    Returns:
        Fused and ranked results
    """
    # Your implementation here
    pass
```

### Task 1.2: Weighted RRF

```python
def weighted_rrf(results: List[tuple], weights: List[float], k: int = 60) -> List[dict]:
    """
    RRF with index-specific weights.
    
    Args:
        results: List of (result_list, weight) tuples
        k: RRF constant
    
    Returns:
        Fused and ranked results
    """
    # Your implementation here
    pass
```

### Task 1.3: Test RRF Implementation

```python
def test_rrf():
    # Create mock results from different indexes
    text_results = [
        {'id': 't1', 'content': 'Text doc 1', 'score': 0.9, 'index': 'text'},
        {'id': 't2', 'content': 'Text doc 2', 'score': 0.8, 'index': 'text'},
        {'id': 't3', 'content': 'Text doc 3', 'score': 0.7, 'index': 'text'},
    ]
    
    image_results = [
        {'id': 'i1', 'content': 'Image 1', 'score': 0.85, 'index': 'image'},
        {'id': 'i2', 'content': 'Image 2', 'score': 0.75, 'index': 'image'},
    ]
    
    code_results = [
        {'id': 'c1', 'content': 'Code snippet 1', 'score': 0.95, 'index': 'code'},
        {'id': 'c2', 'content': 'Code snippet 2', 'score': 0.65, 'index': 'code'},
    ]
    
    # Apply RRF
    fused = reciprocal_rank_fusion([text_results, image_results, code_results])
    
    print("Fused Results:")
    for i, doc in enumerate(fused):
        print(f"  {i+1}. {doc['id']} (RRF score: {doc['rrf_score']:.4f})")
```

---

## Part 2: Score Normalization

### Task 2.1: Min-Max Normalization

```python
def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] range using min-max scaling.
    
    Formula: normalized = (score - min) / (max - min)
    """
    # Your implementation here
    pass
```

### Task 2.2: Z-Score Normalization

```python
def z_score_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores using z-score (standard score).
    
    Formula: z = (score - mean) / std
    """
    # Your implementation here
    pass
```

### Task 2.3: Cross-Index Score Alignment

```python
def align_scores(results_by_index: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """
    Align scores across different indexes for fair comparison.
    
    Args:
        results_by_index: Dict mapping index name to results
    
    Returns:
        Results with normalized scores
    """
    # Your implementation here
    pass
```

---

## Part 3: Cross-Modal Relevance Scoring

### Task 3.1: Query-Modality Alignment

```python
class CrossModalScorer:
    """Scores relevance across different modalities."""
    
    MODALITY_WEIGHTS = {
        'text': 1.0,
        'image': 1.2,  # Slight boost for visual content
        'code': 1.1,
        'table': 1.15
    }
    
    QUERY_MODALITY_SIGNALS = {
        'image': ['diagram', 'chart', 'visual', 'picture', 'show me'],
        'code': ['code', 'function', 'example', 'snippet', 'implement'],
        'table': ['table', 'data', 'metrics', 'compare', 'statistics'],
    }
    
    def score(self, query: str, results: List[dict]) -> List[dict]:
        """
        Apply cross-modal relevance scoring.
        
        Args:
            query: Original user query
            results: List of results to score
        
        Returns:
            Results with adjusted scores
        """
        # Your implementation here
        pass
    
    def _calculate_alignment(self, query: str, modality: str) -> float:
        """Calculate how well modality matches query intent."""
        # Your implementation here
        pass
```

---

## Part 4: Diversity Optimization

### Task 4.1: Maximal Marginal Relevance (MMR)

```python
def diversity_rerank(results: List[dict], diversity_factor: float = 0.3) -> List[dict]:
    """
    Rerank results to maximize diversity while maintaining relevance.
    Uses Maximal Marginal Relevance (MMR) approach.
    
    MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    
    Args:
        results: List of results with embeddings or features
        diversity_factor: λ parameter (0 = pure diversity, 1 = pure relevance)
    
    Returns:
        Diversified result list
    """
    # Your implementation here
    pass
```

### Task 4.2: Modality Diversity

```python
def modality_balanced_rank(results: List[dict], target_distribution: Dict[str, float] = None) -> List[dict]:
    """
    Rerank to achieve balanced modality distribution.
    
    Args:
        results: List of results with modality field
        target_distribution: Desired modality distribution (default: equal)
    
    Returns:
        Balanced result list
    """
    # Your implementation here
    pass
```

---

## Part 5: Complete Fusion Pipeline

### Task 5.1: Integrate All Components

```python
class MultimodalFusionPipeline:
    """
    Complete fusion pipeline combining all techniques.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'rrf_k': 60,
            'diversity_factor': 0.3,
            'normalize_scores': True,
            'apply_cross_modal': True
        }
    
    def fuse(self, results_by_index: Dict[str, List[dict]], query: str) -> List[dict]:
        """
        Complete fusion pipeline:
        1. Normalize scores across indexes
        2. Apply RRF
        3. Apply cross-modal scoring
        4. Apply diversity optimization
        
        Args:
            results_by_index: Results from each index
            query: Original query
        
        Returns:
            Fused, ranked, diversified results
        """
        # Step 1: Normalize scores
        if self.config['normalize_scores']:
            normalized = self._normalize_all(results_by_index)
        else:
            normalized = results_by_index
        
        # Step 2: Apply RRF
        result_lists = list(normalized.values())
        fused = reciprocal_rank_fusion(result_lists, k=self.config['rrf_k'])
        
        # Step 3: Cross-modal scoring
        if self.config['apply_cross_modal']:
            scorer = CrossModalScorer()
            fused = scorer.score(query, fused)
        
        # Step 4: Diversity optimization
        diversified = diversity_rerank(fused, self.config['diversity_factor'])
        
        return diversified
    
    def _normalize_all(self, results_by_index: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        """Normalize scores across all indexes."""
        # Your implementation here
        pass
```

### Task 5.2: Evaluation Metrics

```python
def calculate_ndcg(results: List[dict], ideal_results: List[dict], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        results: Actual ranked results
        ideal_results: Ideal ranking (ground truth)
        k: Cut-off for evaluation
    
    Returns:
        NDCG@k score
    """
    # Your implementation here
    pass


def calculate_diversity(results: List[dict]) -> float:
    """
    Calculate diversity score based on modality distribution.
    
    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    # Your implementation here
    pass
```

---

## Part 6: End-to-End Testing

### Task 6.1: Create Test Dataset

```python
TEST_DATASET = {
    'text_index': [
        {'id': 't1', 'content': 'System architecture overview', 'score': 0.92, 'modality': 'text'},
        {'id': 't2', 'content': 'Implementation guide', 'score': 0.85, 'modality': 'text'},
        {'id': 't3', 'content': 'API documentation', 'score': 0.78, 'modality': 'text'},
    ],
    'image_index': [
        {'id': 'i1', 'content': 'Architecture diagram', 'score': 0.88, 'modality': 'image'},
        {'id': 'i2', 'content': 'Flow chart', 'score': 0.75, 'modality': 'image'},
    ],
    'code_index': [
        {'id': 'c1', 'content': 'Main module code', 'score': 0.90, 'modality': 'code'},
        {'id': 'c2', 'content': 'Helper functions', 'score': 0.70, 'modality': 'code'},
    ],
    'table_index': [
        {'id': 'tb1', 'content': 'Performance metrics', 'score': 0.82, 'modality': 'table'},
    ],
}

TEST_QUERIES = [
    "Show me the architecture diagram",
    "Compare system performance metrics",
    "How to implement the main module",
]
```

### Task 6.2: Run Evaluation

```python
def evaluate_fusion():
    pipeline = MultimodalFusionPipeline()
    
    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        results = pipeline.fuse(TEST_DATASET, query)
        
        print("Fused Results:")
        for i, doc in enumerate(results[:5]):
            print(f"  {i+1}. [{doc['modality']}] {doc['content']}")
            print(f"     Score: {doc.get('adjusted_score', doc['score']):.4f}")
        
        # Calculate metrics
        diversity = calculate_diversity(results)
        print(f"\nDiversity Score: {diversity:.4f}")
        print(f"Modalities represented: {set(d['modality'] for d in results)}")
```

---

## 📝 Deliverables

1. Complete `solution.py` with all fusion components
2. RRF implementation with weighted support
3. Score normalization functions
4. Cross-modal scorer implementation
5. Diversity optimization with MMR
6. Complete fusion pipeline
7. Evaluation with metrics

## ✅ Success Criteria

- RRF correctly combines ranked lists
- Score normalization handles edge cases (all same scores)
- Cross-modal scoring boosts relevant modalities
- Diversity optimization improves modality distribution
- Fusion pipeline produces coherent results
- NDCG calculation matches expected values

## 🔍 Hints

- RRF is robust to score scale differences
- Use small epsilon to avoid division by zero
- MMR requires similarity calculation between documents
- Consider using embeddings for similarity if available
- Test with edge cases (empty results, single result)

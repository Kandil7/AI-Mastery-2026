"""
Lab 3 Solution: Multimodal Fusion and Ranking

This module implements complete fusion and ranking for multimodal RAG results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# ============================================================================
# RECIPROCAL RANK FUSION
# ============================================================================

def reciprocal_rank_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion.
    
    RRF Formula: score = sum(1 / (k + rank)) for each result
    
    Args:
        results: List of result lists from different indexes
        k: Constant for RRF formula (typically 60)
    
    Returns:
        Fused and ranked results with rrf_score field
    """
    score_map: Dict[str, dict] = {}
    
    for result_list in results:
        for rank, doc in enumerate(result_list):
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_id not in score_map:
                # Store a copy of the document
                score_map[doc_id] = {**doc, 'rrf_score': 0.0, 'rank_positions': []}
            
            score_map[doc_id]['rrf_score'] += rrf_score
            score_map[doc_id]['rank_positions'].append(rank + 1)
    
    # Sort by RRF score descending
    fused = sorted(score_map.values(), key=lambda x: x['rrf_score'], reverse=True)
    
    # Clean up temporary fields
    for doc in fused:
        if 'rank_positions' in doc:
            del doc['rank_positions']
    
    return fused


def weighted_rrf(results: List[tuple], weights: List[float], k: int = 60) -> List[dict]:
    """
    RRF with index-specific weights.
    
    Args:
        results: List of (result_list, weight) tuples
        weights: Weight for each result list
        k: RRF constant
    
    Returns:
        Fused and ranked results
    """
    score_map: Dict[str, dict] = {}
    
    for (result_list, weight) in zip(results, weights):
        for rank, doc in enumerate(result_list):
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            weighted_rrf_score = weight / (k + rank + 1)
            
            if doc_id not in score_map:
                score_map[doc_id] = {**doc, 'rrf_score': 0.0}
            
            score_map[doc_id]['rrf_score'] += weighted_rrf_score
    
    return sorted(score_map.values(), key=lambda x: x['rrf_score'], reverse=True)


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] range using min-max scaling.
    
    Formula: normalized = (score - min) / (max - min)
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle edge case: all scores are the same
    if max_score == min_score:
        return [0.5] * len(scores)
    
    epsilon = 1e-10  # Avoid division by zero
    normalized = [(s - min_score) / (max_score - min_score + epsilon) for s in scores]
    
    return normalized


def z_score_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores using z-score (standard score).
    
    Formula: z = (score - mean) / std
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        Z-score normalized values (shifted to [0, 1] range)
    """
    if not scores:
        return []
    
    arr = np.array(scores)
    mean = np.mean(arr)
    std = np.std(arr)
    
    # Handle edge case: zero standard deviation
    if std < 1e-10:
        return [0.5] * len(scores)
    
    z_scores = (arr - mean) / std
    
    # Shift to [0, 1] range using sigmoid-like transformation
    normalized = 1 / (1 + np.exp(-z_scores))
    
    return normalized.tolist()


def align_scores(results_by_index: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """
    Align scores across different indexes for fair comparison.
    
    Args:
        results_by_index: Dict mapping index name to results
        
    Returns:
        Results with normalized scores
    """
    aligned = {}
    
    for index_name, results in results_by_index.items():
        if not results:
            aligned[index_name] = []
            continue
        
        # Extract scores
        scores = [r.get('score', 0.5) for r in results]
        
        # Normalize
        normalized_scores = min_max_normalize(scores)
        
        # Create aligned results
        aligned_results = []
        for result, norm_score in zip(results, normalized_scores):
            aligned_result = {**result, 'normalized_score': norm_score}
            aligned_results.append(aligned_result)
        
        aligned[index_name] = aligned_results
    
    return aligned


# ============================================================================
# CROSS-MODAL RELEVANCE SCORING
# ============================================================================

class CrossModalScorer:
    """
    Scores relevance across different modalities based on query alignment.
    """
    
    MODALITY_WEIGHTS = {
        'text': 1.0,
        'image': 1.2,
        'code': 1.1,
        'table': 1.15
    }
    
    QUERY_MODALITY_SIGNALS = {
        'image': [
            'diagram', 'chart', 'graph', 'image', 'picture', 'photo',
            'screenshot', 'visual', 'illustration', 'architecture',
            'flow', 'map', 'schema', 'blueprint', 'drawing', 'figure',
            'show me', 'display', 'render'
        ],
        'code': [
            'code', 'function', 'method', 'class', 'snippet',
            'implementation', 'example', 'script', 'api',
            'endpoint', 'algorithm', 'pattern', 'implement',
            'write', 'create function', 'define'
        ],
        'table': [
            'table', 'data', 'metrics', 'statistics', 'comparison',
            'matrix', 'grid', 'specifications', 'parameters',
            'configuration', 'settings', 'csv', 'spreadsheet',
            'compare', 'values', 'numbers'
        ],
    }
    
    def __init__(self, modality_weights: Optional[Dict[str, float]] = None):
        self.weights = modality_weights or self.MODALITY_WEIGHTS
    
    def score(self, query: str, results: List[dict]) -> List[dict]:
        """
        Apply cross-modal relevance scoring.
        
        Args:
            query: Original user query
            results: List of results to score
            
        Returns:
            Results with adjusted_score field
        """
        query_lower = query.lower()
        
        for doc in results:
            modality = doc.get('modality', 'text')
            base_score = doc.get('score', 0.5)
            norm_score = doc.get('normalized_score', base_score)
            
            # Calculate query-modality alignment
            alignment = self._calculate_alignment(query_lower, modality)
            
            # Get modality weight
            modality_weight = self.weights.get(modality, 1.0)
            
            # Calculate adjusted score
            adjusted_score = norm_score * alignment * modality_weight
            
            doc['adjusted_score'] = round(adjusted_score, 4)
            doc['alignment_score'] = round(alignment, 4)
        
        # Sort by adjusted score
        return sorted(results, key=lambda x: x['adjusted_score'], reverse=True)
    
    def _calculate_alignment(self, query: str, modality: str) -> float:
        """
        Calculate how well modality matches query intent.
        
        Args:
            query: Lowercase query string
            modality: Document modality
            
        Returns:
            Alignment score (0.5 - 1.5)
        """
        signals = self.QUERY_MODALITY_SIGNALS.get(modality, [])
        
        # Count matching signals
        matches = sum(1 for signal in signals if signal in query)
        
        # Base alignment is 1.0, adjust based on matches
        # Each match adds 0.15, capped at 1.5
        alignment = 1.0 + (matches * 0.15)
        
        # If no signals for this modality but query has other signals, reduce score
        all_signals = set()
        for mod_signals in self.QUERY_MODALITY_SIGNALS.values():
            all_signals.update(mod_signals)
        
        has_any_signal = any(s in query for s in all_signals)
        if has_any_signal and matches == 0:
            alignment = 0.7  # Penalty for mismatched modality
        
        return min(max(alignment, 0.5), 1.5)


# ============================================================================
# DIVERSITY OPTIMIZATION
# ============================================================================

def calculate_similarity(doc1: dict, doc2: dict) -> float:
    """
    Calculate similarity between two documents.
    Uses content overlap as proxy for embedding similarity.
    """
    content1 = str(doc1.get('content', '')).lower()
    content2 = str(doc2.get('content', '')).lower()
    
    # Simple Jaccard similarity on words
    words1 = set(content1.split())
    words2 = set(content2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def diversity_rerank(results: List[dict], diversity_factor: float = 0.3) -> List[dict]:
    """
    Rerank results to maximize diversity while maintaining relevance.
    Uses Maximal Marginal Relevance (MMR) approach.
    
    MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    
    Args:
        results: List of results with scores
        diversity_factor: λ parameter (0 = pure diversity, 1 = pure relevance)
        
    Returns:
        Diversified result list
    """
    if not results:
        return []
    
    if len(results) == 1:
        return results
    
    selected = []
    remaining = results.copy()
    
    # Use adjusted_score if available, otherwise score
    score_key = 'adjusted_score' if 'adjusted_score' in results[0] else 'score'
    
    while remaining:
        best_doc = None
        best_mmr = -float('inf')
        
        for doc in remaining:
            relevance = doc.get(score_key, 0.5)
            
            # Calculate max similarity to already selected docs
            max_similarity = 0.0
            if selected:
                max_similarity = max(
                    calculate_similarity(doc, sel_doc) 
                    for sel_doc in selected
                )
            
            # MMR formula
            mmr = (diversity_factor * relevance - 
                   (1 - diversity_factor) * max_similarity)
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_doc = doc
        
        if best_doc:
            selected.append(best_doc)
            remaining.remove(best_doc)
    
    return selected


def modality_balanced_rank(results: List[dict], 
                           target_distribution: Optional[Dict[str, float]] = None) -> List[dict]:
    """
    Rerank to achieve balanced modality distribution.
    
    Args:
        results: List of results with modality field
        target_distribution: Desired modality distribution (default: equal)
        
    Returns:
        Balanced result list
    """
    if not results:
        return []
    
    # Default to equal distribution
    modalities = set(doc.get('modality', 'text') for doc in results)
    if target_distribution is None:
        target_distribution = {m: 1.0 / len(modalities) for m in modalities}
    
    # Group results by modality
    by_modality: Dict[str, List[dict]] = defaultdict(list)
    for doc in results:
        modality = doc.get('modality', 'text')
        by_modality[modality].append(doc)
    
    # Sort each group by score
    for modality in by_modality:
        score_key = 'adjusted_score' if 'adjusted_score' in by_modality[modality][0] else 'score'
        by_modality[modality].sort(key=lambda x: x.get(score_key, 0), reverse=True)
    
    # Interleave results based on target distribution
    balanced = []
    indices = {m: 0 for m in by_modality}
    
    total_slots = len(results)
    
    for i in range(total_slots):
        # Find modality that is most behind its target
        best_modality = None
        best_gap = -float('inf')
        
        for modality, target_prop in target_distribution.items():
            current_count = sum(1 for d in balanced if d.get('modality') == modality)
            current_prop = current_count / (i + 1) if i > 0 else 0
            gap = target_prop - current_prop
            
            # Check if this modality has remaining documents
            if indices.get(modality, 0) < len(by_modality.get(modality, [])):
                if gap > best_gap:
                    best_gap = gap
                    best_modality = modality
        
        # Add document from best modality
        if best_modality and by_modality.get(best_modality):
            idx = indices[best_modality]
            if idx < len(by_modality[best_modality]):
                balanced.append(by_modality[best_modality][idx])
                indices[best_modality] += 1
        else:
            # Fallback: add from any remaining
            for modality, docs in by_modality.items():
                idx = indices.get(modality, 0)
                if idx < len(docs):
                    balanced.append(docs[idx])
                    indices[modality] = idx + 1
                    break
    
    return balanced


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_dcg(results: List[dict], k: int = 10) -> float:
    """
    Calculate Discounted Cumulative Gain.
    
    DCG = sum(rel_i / log2(i + 1)) for i in 1..k
    """
    dcg = 0.0
    score_key = 'adjusted_score' if results and 'adjusted_score' in results[0] else 'score'
    
    for i, doc in enumerate(results[:k]):
        rel = doc.get(score_key, 0.5)
        dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
    
    return dcg


def calculate_idcg(ideal_results: List[dict], k: int = 10) -> float:
    """
    Calculate Ideal DCG (best possible DCG).
    """
    # Sort by relevance to get ideal ordering
    score_key = 'adjusted_score' if ideal_results and 'adjusted_score' in ideal_results[0] else 'score'
    sorted_results = sorted(ideal_results, key=lambda x: x.get(score_key, 0), reverse=True)
    return calculate_dcg(sorted_results, k)


def calculate_ndcg(results: List[dict], ideal_results: List[dict], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    NDCG = DCG / IDCG
    
    Args:
        results: Actual ranked results
        ideal_results: Ideal ranking (ground truth)
        k: Cut-off for evaluation
        
    Returns:
        NDCG@k score (0-1)
    """
    dcg = calculate_dcg(results, k)
    idcg = calculate_idcg(ideal_results, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_diversity(results: List[dict]) -> float:
    """
    Calculate diversity score based on modality distribution.
    Uses entropy-based measure.
    
    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    if not results:
        return 0.0
    
    # Count modalities
    modality_counts: Dict[str, int] = defaultdict(int)
    for doc in results:
        modality = doc.get('modality', 'text')
        modality_counts[modality] += 1
    
    # Calculate proportions
    total = len(results)
    proportions = [count / total for count in modality_counts.values()]
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
    
    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log2(len(modality_counts)) if len(modality_counts) > 1 else 1
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def calculate_modality_coverage(results: List[dict]) -> Dict[str, float]:
    """
    Calculate coverage of each modality in results.
    
    Returns:
        Dict mapping modality to coverage proportion
    """
    if not results:
        return {}
    
    modality_counts: Dict[str, int] = defaultdict(int)
    for doc in results:
        modality = doc.get('modality', 'text')
        modality_counts[modality] += 1
    
    total = len(results)
    return {m: c / total for m, c in modality_counts.items()}


# ============================================================================
# COMPLETE FUSION PIPELINE
# ============================================================================

@dataclass
class FusionConfig:
    """Configuration for fusion pipeline."""
    rrf_k: int = 60
    diversity_factor: float = 0.3
    normalize_scores: bool = True
    apply_cross_modal: bool = True
    balance_modalities: bool = False
    target_distribution: Optional[Dict[str, float]] = None


class MultimodalFusionPipeline:
    """
    Complete fusion pipeline combining all techniques.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.cross_modal_scorer = CrossModalScorer()
    
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
        if self.config.normalize_scores:
            normalized = align_scores(results_by_index)
        else:
            normalized = results_by_index
        
        # Step 2: Apply RRF
        result_lists = list(normalized.values())
        if not result_lists:
            return []
        
        fused = reciprocal_rank_fusion(result_lists, k=self.config.rrf_k)
        
        # Step 3: Cross-modal scoring
        if self.config.apply_cross_modal and fused:
            fused = self.cross_modal_scorer.score(query, fused)
        
        # Step 4: Diversity optimization
        if fused:
            if self.config.balance_modalities:
                fused = modality_balanced_rank(
                    fused, 
                    self.config.target_distribution
                )
            else:
                fused = diversity_rerank(fused, self.config.diversity_factor)
        
        return fused
    
    def evaluate(self, results: List[dict], 
                 ideal_results: Optional[List[dict]] = None,
                 k: int = 10) -> Dict[str, float]:
        """
        Evaluate fusion results.
        
        Args:
            results: Fused results
            ideal_results: Ground truth for NDCG calculation
            k: Cut-off for metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'diversity': calculate_diversity(results),
            'modality_coverage': calculate_modality_coverage(results),
            'num_results': len(results),
        }
        
        if ideal_results:
            metrics['ndcg'] = calculate_ndcg(results, ideal_results, k)
        
        return metrics


# ============================================================================
# TEST DATA AND EVALUATION
# ============================================================================

TEST_DATASET = {
    'text_index': [
        {'id': 't1', 'content': 'System architecture overview document', 'score': 0.92, 'modality': 'text'},
        {'id': 't2', 'content': 'Implementation guide for developers', 'score': 0.85, 'modality': 'text'},
        {'id': 't3', 'content': 'API documentation and examples', 'score': 0.78, 'modality': 'text'},
    ],
    'image_index': [
        {'id': 'i1', 'content': 'System architecture diagram with components', 'score': 0.88, 'modality': 'image'},
        {'id': 'i2', 'content': 'Data flow chart showing process', 'score': 0.75, 'modality': 'image'},
    ],
    'code_index': [
        {'id': 'c1', 'content': 'Main module implementation code', 'score': 0.90, 'modality': 'code'},
        {'id': 'c2', 'content': 'Helper functions and utilities', 'score': 0.70, 'modality': 'code'},
    ],
    'table_index': [
        {'id': 'tb1', 'content': 'Performance metrics comparison table', 'score': 0.82, 'modality': 'table'},
    ],
}

TEST_QUERIES = [
    "Show me the architecture diagram",
    "Compare system performance metrics",
    "How to implement the main module",
    "Explain the system architecture",
]


def evaluate_fusion():
    """Run fusion evaluation on test queries."""
    print("=" * 70)
    print("MULTIMODAL FUSION EVALUATION")
    print("=" * 70)
    
    pipeline = MultimodalFusionPipeline(FusionConfig(
        rrf_k=60,
        diversity_factor=0.3,
        normalize_scores=True,
        apply_cross_modal=True,
        balance_modalities=False
    ))
    
    for query in TEST_QUERIES:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)
        
        results = pipeline.fuse(TEST_DATASET, query)
        
        print("\nFused Results (Top 5):")
        for i, doc in enumerate(results[:5]):
            print(f"  {i+1}. [{doc['modality']}] {doc['content'][:50]}...")
            print(f"     Score: {doc.get('adjusted_score', doc['score']):.4f}")
            if 'alignment_score' in doc:
                print(f"     Alignment: {doc['alignment_score']:.4f}")
        
        # Calculate metrics
        metrics = pipeline.evaluate(results)
        print(f"\nMetrics:")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Modalities: {metrics['modality_coverage']}")
        print(f"  Total Results: {metrics['num_results']}")


def test_rrf():
    """Test RRF implementation."""
    print("\n" + "=" * 70)
    print("RRF TEST")
    print("=" * 70)
    
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
    
    fused = reciprocal_rank_fusion([text_results, image_results, code_results])
    
    print("\nFused Results:")
    for i, doc in enumerate(fused):
        print(f"  {i+1}. {doc['id']} (RRF score: {doc['rrf_score']:.4f})")


def test_normalization():
    """Test score normalization."""
    print("\n" + "=" * 70)
    print("NORMALIZATION TEST")
    print("=" * 70)
    
    scores = [0.9, 0.8, 0.7, 0.5, 0.3]
    
    minmax = min_max_normalize(scores)
    zscore = z_score_normalize(scores)
    
    print(f"\nOriginal: {scores}")
    print(f"Min-Max:  {[f'{s:.4f}' for s in minmax]}")
    print(f"Z-Score:  {[f'{s:.4f}' for s in zscore]}")


if __name__ == "__main__":
    test_rrf()
    test_normalization()
    evaluate_fusion()
    
    print("\n" + "=" * 70)
    print("Lab 3 Complete!")
    print("=" * 70)

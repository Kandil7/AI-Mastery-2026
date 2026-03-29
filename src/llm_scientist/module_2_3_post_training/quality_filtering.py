"""
Quality Filtering - Module 2.3.4

Production-ready quality filtering implementations:
- Reward model filtering
- Perplexity filtering
- Diversity scoring
- Quality filter pipeline

References:
- "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for an example."""
    overall: float = 0.0
    reward_score: float = 0.0
    perplexity_score: float = 0.0
    diversity_score: float = 0.0
    length_score: float = 0.0
    coherence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall': self.overall,
            'reward_score': self.reward_score,
            'perplexity_score': self.perplexity_score,
            'diversity_score': self.diversity_score,
            'length_score': self.length_score,
            'coherence_score': self.coherence_score,
            'metadata': self.metadata,
        }


class RewardModelFilter:
    """
    Reward Model-based Quality Filter.
    
    Uses a trained reward model to score and filter examples
    based on quality.
    
    Args:
        reward_model: Trained reward model
        tokenizer: Tokenizer
        threshold: Minimum score threshold
        
    Example:
        >>> filter = RewardModelFilter(reward_model, tokenizer)
        >>> filtered = filter.filter(examples, threshold=0.5)
    """
    
    def __init__(
        self,
        reward_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.batch_size = batch_size
        
        if reward_model:
            self.reward_model.eval()
    
    def _score_single(
        self,
        instruction: str,
        response: str,
        input_text: str = "",
    ) -> float:
        """Score a single example."""
        if not self.reward_model or not self.tokenizer:
            return 0.5  # Default score
        
        # Format input for reward model
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {response}"
        else:
            text = f"Instruction: {instruction}\nResponse: {response}"
        
        # Tokenize
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        # Get reward score
        with torch.no_grad():
            outputs = self.reward_model(inputs)
            
            if isinstance(outputs, tuple):
                score = outputs[0]
            else:
                score = outputs
            
            # Handle different output formats
            if isinstance(score, torch.Tensor):
                score = score.item()
        
        # Normalize to 0-1 range (assuming sigmoid output)
        score = 1 / (1 + math.exp(-score))
        
        return score
    
    def _score_batch(
        self,
        examples: List[Dict[str, str]],
    ) -> List[float]:
        """Score a batch of examples."""
        if not self.reward_model or not self.tokenizer:
            return [0.5] * len(examples)
        
        scores = []
        
        for i in range(0, len(examples), self.batch_size):
            batch = examples[i:i + self.batch_size]
            
            # Tokenize batch
            texts = []
            for ex in batch:
                instruction = ex.get('instruction', '')
                response = ex.get('output', '')
                input_text = ex.get('input', '')
                
                if input_text:
                    text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {response}"
                else:
                    text = f"Instruction: {instruction}\nResponse: {response}"
                
                texts.append(text)
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Get scores
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                
                if isinstance(outputs, tuple):
                    batch_scores = outputs[0]
                else:
                    batch_scores = outputs
                
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.cpu().tolist()
                
                # Normalize
                batch_scores = [1 / (1 + math.exp(-s)) for s in batch_scores]
                scores.extend(batch_scores)
        
        return scores
    
    def filter(
        self,
        examples: List[Dict[str, str]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """
        Filter examples by reward score.
        
        Args:
            examples: Examples to filter
            threshold: Score threshold (uses instance threshold if not provided)
        
        Returns:
            Filtered examples
        """
        threshold = threshold if threshold is not None else self.threshold
        
        scores = self._score_batch(examples)
        
        filtered = []
        for ex, score in zip(examples, scores):
            if score >= threshold:
                ex['_reward_score'] = score
                filtered.append(ex)
        
        logger.info(f"Reward filter: {len(examples)} -> {len(filtered)} examples")
        return filtered
    
    def score(
        self,
        examples: List[Dict[str, str]],
    ) -> List[QualityScore]:
        """
        Score examples without filtering.
        
        Args:
            examples: Examples to score
        
        Returns:
            List of QualityScore
        """
        scores = self._score_batch(examples)
        
        return [
            QualityScore(
                overall=score,
                reward_score=score,
                metadata={'source': 'reward_model'},
            )
            for score in scores
        ]


class PerplexityFilter:
    """
    Perplexity-based Quality Filter.
    
    Filters examples based on language model perplexity.
    Lower perplexity indicates more fluent, natural text.
    
    Args:
        model: Language model for perplexity calculation
        tokenizer: Tokenizer
        max_perplexity: Maximum acceptable perplexity
        
    Example:
        >>> filter = PerplexityFilter(model, tokenizer)
        >>> filtered = filter.filter(examples)
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        max_perplexity: float = 100.0,
        batch_size: int = 16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_perplexity = max_perplexity
        self.batch_size = batch_size
        
        if model:
            self.model.eval()
    
    def _calculate_perplexity(
        self,
        text: str,
    ) -> float:
        """Calculate perplexity for a single text."""
        if not self.model or not self.tokenizer:
            return float('inf')
        
        # Tokenize
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        # Calculate loss
        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
            loss = outputs.loss
        
        # Convert to perplexity
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def _calculate_batch_perplexity(
        self,
        texts: List[str],
    ) -> List[float]:
        """Calculate perplexity for a batch of texts."""
        if not self.model or not self.tokenizer:
            return [float('inf')] * len(texts)
        
        perplexities = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(**inputs, labels=input_ids)
                loss = outputs.loss
            
            # Handle batch loss
            if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                batch_ppl = [torch.exp(loss).item()] * len(batch)
            else:
                batch_ppl = [torch.exp(l).item() for l in loss]
            
            perplexities.extend(batch_ppl)
        
        return perplexities
    
    def filter(
        self,
        examples: List[Dict[str, str]],
        max_perplexity: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """
        Filter examples by perplexity.
        
        Args:
            examples: Examples to filter
            max_perplexity: Maximum perplexity threshold
        
        Returns:
            Filtered examples
        """
        max_perplexity = max_perplexity or self.max_perplexity
        
        # Calculate perplexity for responses
        texts = [ex.get('output', '') for ex in examples]
        perplexities = self._calculate_batch_perplexity(texts)
        
        filtered = []
        for ex, ppl in zip(examples, perplexities):
            if ppl <= max_perplexity:
                ex['_perplexity'] = ppl
                filtered.append(ex)
        
        logger.info(f"Perplexity filter: {len(examples)} -> {len(filtered)} examples")
        return filtered
    
    def score(
        self,
        examples: List[Dict[str, str]],
    ) -> List[QualityScore]:
        """
        Score examples by perplexity.
        
        Args:
            examples: Examples to score
        
        Returns:
            List of QualityScore
        """
        texts = [ex.get('output', '') for ex in examples]
        perplexities = self._calculate_batch_perplexity(texts)
        
        # Convert perplexity to score (lower is better)
        scores = []
        for ppl in perplexities:
            # Normalize: assume max_perplexity is worst case
            score = max(0, 1 - (ppl / self.max_perplexity))
            scores.append(
                QualityScore(
                    overall=score,
                    perplexity_score=score,
                    metadata={'perplexity': ppl},
                )
            )
        
        return scores


class DiversityScorer:
    """
    Diversity Scorer for measuring example uniqueness.
    
    Measures how diverse an example is compared to others
    in the dataset.
    
    Args:
        embedding_model: Model for generating embeddings
        tokenizer: Tokenizer
        similarity_threshold: Threshold for considering duplicates
        
    Example:
        >>> scorer = DiversityScorer(embedding_model, tokenizer)
        >>> scores = scorer.score(examples)
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        similarity_threshold: float = 0.9,
    ):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.similarity_threshold = similarity_threshold
        
        self._embeddings_cache: Dict[str, torch.Tensor] = {}
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        
        if not self.embedding_model or not self.tokenizer:
            # Fallback: use simple hash-based embedding
            return self._hash_embedding(text)
        
        # Tokenize
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        # Get embedding
        with torch.no_grad():
            outputs = self.embedding_model(inputs)
            
            if isinstance(outputs, tuple):
                embedding = outputs[0]
            else:
                embedding = outputs
            
            # Pool if needed
            if embedding.dim() == 3:
                embedding = embedding.mean(dim=1)
            
            # Normalize
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        if torch.cuda.is_available():
            embedding = embedding.cpu()
        
        self._embeddings_cache[text] = embedding
        return embedding
    
    def _hash_embedding(self, text: str) -> torch.Tensor:
        """Generate simple hash-based embedding."""
        # Simple hash-based embedding for fallback
        import hashlib
        
        hash_bytes = hashlib.md5(text.encode()).digest()
        embedding = torch.tensor([b / 255.0 for b in hash_bytes])
        return F.normalize(embedding.unsqueeze(0), p=2, dim=-1)
    
    def _cosine_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        return torch.dot(emb1.flatten(), emb2.flatten()).item()
    
    def score(
        self,
        examples: List[Dict[str, str]],
    ) -> List[QualityScore]:
        """
        Score examples by diversity.
        
        Args:
            examples: Examples to score
        
        Returns:
            List of QualityScore with diversity scores
        """
        # Get embeddings for all examples
        texts = [ex.get('output', '') for ex in examples]
        embeddings = [self._get_embedding(t) for t in texts]
        
        scores = []
        
        for i, (ex, emb) in enumerate(zip(examples, embeddings)):
            # Calculate average similarity to other examples
            similarities = []
            for j, other_emb in enumerate(embeddings):
                if i != j:
                    sim = self._cosine_similarity(emb, other_emb)
                    similarities.append(sim)
            
            # Diversity score: lower similarity = higher diversity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            diversity_score = 1 - avg_similarity
            
            scores.append(
                QualityScore(
                    overall=diversity_score,
                    diversity_score=diversity_score,
                    metadata={'avg_similarity': avg_similarity},
                )
            )
        
        return scores
    
    def filter_duplicates(
        self,
        examples: List[Dict[str, str]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """
        Filter out near-duplicate examples.
        
        Args:
            examples: Examples to filter
            threshold: Similarity threshold for duplicates
        
        Returns:
            Deduplicated examples
        """
        threshold = threshold or self.similarity_threshold
        
        # Get embeddings
        texts = [ex.get('output', '') for ex in examples]
        embeddings = [self._get_embedding(t) for t in texts]
        
        # Track which examples to keep
        keep = [True] * len(examples)
        
        for i in range(len(examples)):
            if not keep[i]:
                continue
            
            for j in range(i + 1, len(examples)):
                if not keep[j]:
                    continue
                
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim > threshold:
                    # Mark the later one as duplicate
                    keep[j] = False
        
        filtered = [ex for ex, k in zip(examples, keep) if k]
        
        logger.info(f"Diversity filter: {len(examples)} -> {len(filtered)} examples")
        return filtered


class QualityFilterPipeline:
    """
    Complete Quality Filtering Pipeline.
    
    Combines multiple filtering strategies:
    1. Reward model scoring
    2. Perplexity filtering
    3. Diversity scoring
    4. Length filtering
    5. Rule-based filtering
    
    Example:
        >>> pipeline = QualityFilterPipeline(reward_model, lm_model, tokenizer)
        >>> filtered = pipeline.filter(examples)
    """
    
    def __init__(
        self,
        reward_model: Optional[nn.Module] = None,
        language_model: Optional[nn.Module] = None,
        embedding_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        
        # Initialize filters
        self.reward_filter = RewardModelFilter(reward_model, tokenizer)
        self.perplexity_filter = PerplexityFilter(language_model, tokenizer)
        self.diversity_scorer = DiversityScorer(embedding_model, tokenizer)
        
        # Default thresholds
        self.thresholds = {
            'reward': 0.5,
            'perplexity': 100.0,
            'diversity': 0.3,
            'min_length': 10,
            'max_length': 2000,
        }
    
    def _rule_based_filter(
        self,
        examples: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Apply rule-based filtering."""
        filtered = []
        
        for ex in examples:
            instruction = ex.get('instruction', '')
            output = ex.get('output', '')
            
            # Length checks
            if len(output) < self.thresholds['min_length']:
                continue
            if len(output) > self.thresholds['max_length']:
                continue
            
            # Basic quality checks
            if not output.strip():
                continue
            
            # Check for repetition
            words = output.split()
            if len(words) > 10:
                # Check for excessive repetition
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    continue
            
            filtered.append(ex)
        
        logger.info(f"Rule-based filter: {len(examples)} -> {len(filtered)} examples")
        return filtered
    
    def filter(
        self,
        examples: List[Dict[str, str]],
        apply_reward: bool = True,
        apply_perplexity: bool = True,
        apply_diversity: bool = True,
        apply_rules: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Filter examples through the pipeline.
        
        Args:
            examples: Examples to filter
            apply_reward: Whether to apply reward filtering
            apply_perplexity: Whether to apply perplexity filtering
            apply_diversity: Whether to apply diversity filtering
            apply_rules: Whether to apply rule-based filtering
        
        Returns:
            Filtered examples
        """
        current = examples
        
        # Step 1: Rule-based filtering
        if apply_rules:
            current = self._rule_based_filter(current)
        
        # Step 2: Reward model filtering
        if apply_reward and self.reward_filter.reward_model:
            current = self.reward_filter.filter(
                current,
                threshold=self.thresholds['reward'],
            )
        
        # Step 3: Perplexity filtering
        if apply_perplexity and self.perplexity_filter.model:
            current = self.perplexity_filter.filter(
                current,
                max_perplexity=self.thresholds['perplexity'],
            )
        
        # Step 4: Diversity filtering
        if apply_diversity:
            current = self.diversity_scorer.filter_duplicates(
                current,
                threshold=self.thresholds['diversity'],
            )
        
        return current
    
    def score(
        self,
        examples: List[Dict[str, str]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[QualityScore]:
        """
        Score examples without filtering.
        
        Args:
            examples: Examples to score
            weights: Weights for combining scores
        
        Returns:
            List of combined QualityScore
        """
        weights = weights or {
            'reward': 0.4,
            'perplexity': 0.3,
            'diversity': 0.2,
            'length': 0.1,
        }
        
        # Get scores from each filter
        reward_scores = self.reward_filter.score(examples)
        perplexity_scores = self.perplexity_filter.score(examples)
        diversity_scores = self.diversity_scorer.score(examples)
        
        combined = []
        
        for i, ex in enumerate(examples):
            # Calculate length score
            output = ex.get('output', '')
            length = len(output)
            length_score = min(1.0, length / 100)  # Normalize
            
            # Combine scores
            overall = (
                weights['reward'] * reward_scores[i].reward_score +
                weights['perplexity'] * perplexity_scores[i].perplexity_score +
                weights['diversity'] * diversity_scores[i].diversity_score +
                weights['length'] * length_score
            )
            
            combined.append(
                QualityScore(
                    overall=overall,
                    reward_score=reward_scores[i].reward_score,
                    perplexity_score=perplexity_scores[i].perplexity_score,
                    diversity_score=diversity_scores[i].diversity_score,
                    length_score=length_score,
                    metadata={
                        'weights': weights,
                        'output_length': length,
                    },
                )
            )
        
        return combined
    
    def filter_and_score(
        self,
        examples: List[Dict[str, str]],
        min_score: float = 0.5,
    ) -> Tuple[List[Dict[str, str]], List[QualityScore]]:
        """
        Filter examples and return scores.
        
        Args:
            examples: Examples to process
            min_score: Minimum score threshold
        
        Returns:
            Tuple of (filtered_examples, scores)
        """
        # Score all examples
        scores = self.score(examples)
        
        # Filter by score
        filtered = []
        filtered_scores = []
        
        for ex, score in zip(examples, scores):
            if score.overall >= min_score:
                ex['_quality_score'] = score.overall
                filtered.append(ex)
                filtered_scores.append(score)
        
        return filtered, filtered_scores
    
    def save_scores(
        self,
        scores: List[QualityScore],
        path: Union[str, Path],
    ) -> None:
        """Save scores to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [s.to_dict() for s in scores]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(scores)} scores to {path}")


class QualityDataset(Dataset):
    """
    PyTorch Dataset with quality filtering.
    
    Args:
        examples: List of examples
        tokenizer: Tokenizer
        quality_pipeline: Quality filter pipeline
        min_score: Minimum quality score
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: Any,
        quality_pipeline: Optional[QualityFilterPipeline] = None,
        min_score: float = 0.5,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Apply quality filtering
        if quality_pipeline:
            examples, scores = quality_pipeline.filter_and_score(
                examples,
                min_score=min_score,
            )
        
        self.examples = examples
        self._preprocess()
    
    def _preprocess(self) -> None:
        """Preprocess examples for training."""
        self.processed = []
        
        for ex in self.examples:
            instruction = ex.get('instruction', '')
            input_text = ex.get('input', '')
            output = ex.get('output', '')
            
            # Format as instruction-response pair
            if input_text:
                text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
            else:
                text = f"Instruction: {instruction}\nResponse: {output}"
            
            # Tokenize
            encoded = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
            )
            
            self.processed.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': encoded['input_ids'].copy(),
            })
    
    def __len__(self) -> int:
        return len(self.processed)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
        }

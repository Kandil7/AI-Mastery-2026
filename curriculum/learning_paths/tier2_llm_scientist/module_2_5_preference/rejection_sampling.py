"""
Rejection Sampling - Module 2.5.1

Production-ready rejection sampling for preference data:
- Multi-response generation
- Response scoring
- Preference pair creation

References:
- "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class Response:
    """A generated response."""
    text: str
    token_ids: List[int]
    log_probs: List[float]
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'token_ids': self.token_ids,
            'log_probs': self.log_probs,
            'score': self.score,
        }


@dataclass
class PreferencePair:
    """A preference pair (chosen, rejected)."""
    prompt: str
    chosen: Response
    rejected: Response
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'prompt': self.prompt,
            'chosen': self.chosen.to_dict(),
            'rejected': self.rejected.to_dict(),
            'metadata': self.metadata,
        }
    
    def to_dpo_format(self) -> Dict[str, str]:
        """Convert to DPO training format."""
        return {
            'prompt': self.prompt,
            'chosen': self.chosen.text,
            'rejected': self.rejected.text,
        }


class MultiResponseGenerator:
    """
    Multi-Response Generator for rejection sampling.
    
    Generates multiple responses for each prompt using
    different sampling strategies.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        num_responses: Number of responses per prompt
        
    Example:
        >>> generator = MultiResponseGenerator(model, tokenizer)
        >>> responses = generator.generate(prompts)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        num_responses: int = 4,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_responses = num_responses
        self.max_new_tokens = max_new_tokens
        
        self.device = next(model.parameters()).device
    
    def _generate_single(
        self,
        prompt_ids: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Tuple[List[int], List[float]]:
        """Generate a single response."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            sequences = outputs.sequences
            scores = outputs.scores
            
            # Extract generated tokens (excluding prompt)
            prompt_len = prompt_ids.shape[-1]
            generated_ids = sequences[0, prompt_len:].tolist()
            
            # Compute log probs
            log_probs = []
            for score in scores:
                log_prob = F.log_softmax(score[0], dim=-1)
                log_probs.append(log_prob.item())
        
        return generated_ids, log_probs
    
    def generate(
        self,
        prompts: List[str],
        temperatures: Optional[List[float]] = None,
    ) -> Dict[str, List[Response]]:
        """
        Generate multiple responses for each prompt.
        
        Args:
            prompts: List of prompts
            temperatures: Optional list of temperatures for diversity
        
        Returns:
            Dictionary mapping prompts to lists of responses
        """
        if temperatures is None:
            temperatures = [0.7, 0.8, 0.9, 1.0][:self.num_responses]
        
        all_responses = {}
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                add_special_tokens=False,
            ).to(self.device)
            
            responses = []
            
            for i, temp in enumerate(temperatures[:self.num_responses]):
                # Generate response
                token_ids, log_probs = self._generate_single(
                    inputs,
                    temperature=temp,
                )
                
                # Decode
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                
                responses.append(Response(
                    text=text,
                    token_ids=token_ids,
                    log_probs=log_probs,
                ))
            
            all_responses[prompt] = responses
            logger.info(f"Generated {len(responses)} responses for prompt")
        
        return all_responses


class ResponseScorer:
    """
    Response Scorer for ranking responses.
    
    Scores responses using a reward model or heuristic metrics.
    
    Args:
        reward_model: Optional reward model for scoring
        tokenizer: Tokenizer
        
    Example:
        >>> scorer = ResponseScorer(reward_model, tokenizer)
        >>> scored = scorer.score(responses)
    """
    
    def __init__(
        self,
        reward_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        
        if reward_model:
            self.reward_model.eval()
        
        self.device = None
        if reward_model:
            self.device = next(reward_model.parameters()).device
    
    def _compute_heuristic_score(
        self,
        response: Response,
    ) -> float:
        """Compute heuristic score based on response properties."""
        text = response.text
        
        # Length score (prefer moderate length)
        length = len(text.split())
        length_score = 1.0 / (1.0 + abs(length - 50) / 50)
        
        # Repetition penalty
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
        else:
            unique_ratio = 0
        repetition_score = unique_ratio
        
        # Average log prob
        if response.log_probs:
            avg_log_prob = sum(response.log_probs) / len(response.log_probs)
            log_prob_score = 1.0 / (1.0 + abs(avg_log_prob))
        else:
            log_prob_score = 0.5
        
        # Combined score
        score = 0.4 * length_score + 0.3 * repetition_score + 0.3 * log_prob_score
        
        return score
    
    def _compute_reward_score(
        self,
        prompt: str,
        response: Response,
    ) -> float:
        """Compute score using reward model."""
        if not self.reward_model or not self.tokenizer:
            return self._compute_heuristic_score(response)
        
        # Format input
        text = f"{prompt}\n\n{response.text}"
        
        # Tokenize
        inputs = self.tokenizer.encode(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        # Get reward score
        with torch.no_grad():
            outputs = self.reward_model(inputs)
            
            if isinstance(outputs, tuple):
                score = outputs[0]
            else:
                score = outputs
            
            if isinstance(score, torch.Tensor):
                score = score.item()
        
        # Normalize to 0-1
        score = 1 / (1 + abs(-score))
        
        return score
    
    def score(
        self,
        prompt: str,
        responses: List[Response],
    ) -> List[Response]:
        """
        Score a list of responses.
        
        Args:
            prompt: Original prompt
            responses: List of responses to score
        
        Returns:
            Scored responses
        """
        for response in responses:
            if self.reward_model:
                response.score = self._compute_reward_score(prompt, response)
            else:
                response.score = self._compute_heuristic_score(response)
        
        return responses
    
    def rank(
        self,
        prompt: str,
        responses: List[Response],
    ) -> List[Response]:
        """
        Rank responses by score.
        
        Args:
            prompt: Original prompt
            responses: List of responses
        
        Returns:
            Ranked responses (highest score first)
        """
        scored = self.score(prompt, responses)
        return sorted(scored, key=lambda r: r.score, reverse=True)


class RejectionSampler:
    """
    Rejection Sampler for preference data creation.
    
    Generates multiple responses and selects the best
    (chosen) and worst (rejected) based on scores.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        reward_model: Optional reward model
        num_responses: Number of responses per prompt
        
    Example:
        >>> sampler = RejectionSampler(model, tokenizer, reward_model)
        >>> pairs = sampler.sample(prompts)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_model: Optional[nn.Module] = None,
        num_responses: int = 4,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_responses = num_responses
        
        self.generator = MultiResponseGenerator(
            model,
            tokenizer,
            num_responses=num_responses,
            max_new_tokens=max_new_tokens,
        )
        
        self.scorer = ResponseScorer(reward_model, tokenizer)
    
    def sample(
        self,
        prompts: List[str],
    ) -> List[PreferencePair]:
        """
        Generate preference pairs from prompts.
        
        Args:
            prompts: List of prompts
        
        Returns:
            List of PreferencePair
        """
        logger.info(f"Generating preference pairs for {len(prompts)} prompts...")
        
        # Generate responses
        all_responses = self.generator.generate(prompts)
        
        # Create preference pairs
        pairs = []
        
        for prompt, responses in all_responses.items():
            if len(responses) < 2:
                continue
            
            # Score and rank
            ranked = self.scorer.rank(prompt, responses)
            
            # Select best and worst
            chosen = ranked[0]
            rejected = ranked[-1]
            
            # Only create pair if there's a score difference
            if chosen.score > rejected.score:
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    metadata={
                        'chosen_score': chosen.score,
                        'rejected_score': rejected.score,
                        'score_diff': chosen.score - rejected.score,
                    },
                ))
        
        logger.info(f"Created {len(pairs)} preference pairs")
        
        return pairs
    
    def sample_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
    ) -> List[PreferencePair]:
        """
        Generate preference pairs in batches.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size
        
        Returns:
            List of PreferencePair
        """
        all_pairs = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            pairs = self.sample(batch)
            all_pairs.extend(pairs)
            
            logger.info(f"Processed batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        
        return all_pairs
    
    def save(
        self,
        pairs: List[PreferencePair],
        path: Union[str, Path],
        format: str = 'jsonl',
    ) -> None:
        """
        Save preference pairs to file.
        
        Args:
            pairs: List of preference pairs
            path: Output path
            format: Output format ('jsonl' or 'json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(path, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + '\n')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([p.to_dict() for p in pairs], f, indent=2)
        
        logger.info(f"Saved {len(pairs)} preference pairs to {path}")
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
    ) -> List[PreferencePair]:
        """Load preference pairs from file."""
        path = Path(path)
        
        pairs = []
        
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    pairs.append(cls._dict_to_pair(data))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    pairs.append(cls._dict_to_pair(item))
        
        return pairs
    
    @staticmethod
    def _dict_to_pair(data: Dict) -> PreferencePair:
        """Convert dictionary to PreferencePair."""
        chosen = Response(
            text=data['chosen']['text'],
            token_ids=data['chosen'].get('token_ids', []),
            log_probs=data['chosen'].get('log_probs', []),
            score=data['chosen'].get('score', 0),
        )
        
        rejected = Response(
            text=data['rejected']['text'],
            token_ids=data['rejected'].get('token_ids', []),
            log_probs=data['rejected'].get('log_probs', []),
            score=data['rejected'].get('score', 0),
        )
        
        return PreferencePair(
            prompt=data['prompt'],
            chosen=chosen,
            rejected=rejected,
            metadata=data.get('metadata', {}),
        )


class PreferencePairCreator:
    """
    Preference Pair Creator from existing data.
    
    Creates preference pairs from existing datasets
    with human annotations or model scores.
    
    Example:
        >>> creator = PreferencePairCreator()
        >>> pairs = creator.create_from_ratings(data)
    """
    
    def create_from_ratings(
        self,
        data: List[Dict[str, Any]],
        rating_key: str = 'rating',
        min_rating_diff: float = 0.5,
    ) -> List[PreferencePair]:
        """
        Create preference pairs from rated responses.
        
        Args:
            data: List of examples with ratings
            rating_key: Key for rating value
            min_rating_diff: Minimum rating difference for pair
        
        Returns:
            List of PreferencePair
        """
        pairs = []
        
        # Group by prompt
        by_prompt = {}
        for item in data:
            prompt = item.get('prompt', '')
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(item)
        
        # Create pairs
        for prompt, items in by_prompt.items():
            if len(items) < 2:
                continue
            
            # Sort by rating
            sorted_items = sorted(
                items,
                key=lambda x: x.get(rating_key, 0),
                reverse=True,
            )
            
            # Create pairs with sufficient rating difference
            for i in range(len(sorted_items)):
                for j in range(i + 1, len(sorted_items)):
                    rating_diff = (
                        sorted_items[i].get(rating_key, 0) -
                        sorted_items[j].get(rating_key, 0)
                    )
                    
                    if rating_diff >= min_rating_diff:
                        pairs.append(PreferencePair(
                            prompt=prompt,
                            chosen=Response(
                                text=sorted_items[i].get('response', ''),
                                token_ids=[],
                                log_probs=[],
                                score=sorted_items[i].get(rating_key, 0),
                            ),
                            rejected=Response(
                                text=sorted_items[j].get('response', ''),
                                token_ids=[],
                                log_probs=[],
                                score=sorted_items[j].get(rating_key, 0),
                            ),
                            metadata={
                                'rating_diff': rating_diff,
                            },
                        ))
                        break  # Only create one pair per prompt
        
        return pairs
    
    def create_from_scores(
        self,
        data: List[Dict[str, Any]],
        score_key: str = 'score',
    ) -> List[PreferencePair]:
        """
        Create preference pairs from scored responses.
        
        Args:
            data: List of examples with scores
            score_key: Key for score value
        
        Returns:
            List of PreferencePair
        """
        return self.create_from_ratings(data, rating_key=score_key)
    
    def filter_pairs(
        self,
        pairs: List[PreferencePair],
        min_score_diff: float = 0.1,
        max_chosen_length: int = 2000,
        min_chosen_length: int = 10,
    ) -> List[PreferencePair]:
        """
        Filter preference pairs by quality.
        
        Args:
            pairs: List of preference pairs
            min_score_diff: Minimum score difference
            max_chosen_length: Maximum chosen response length
            min_chosen_length: Minimum chosen response length
        
        Returns:
            Filtered pairs
        """
        filtered = []
        
        for pair in pairs:
            # Check score difference
            if pair.metadata.get('score_diff', 0) < min_score_diff:
                continue
            
            # Check length
            chosen_length = len(pair.chosen.text.split())
            if chosen_length < min_chosen_length or chosen_length > max_chosen_length:
                continue
            
            # Check for empty responses
            if not pair.chosen.text.strip() or not pair.rejected.text.strip():
                continue
            
            filtered.append(pair)
        
        logger.info(f"Filtered {len(pairs)} -> {len(filtered)} pairs")
        
        return filtered


class PreferenceDataset(Dataset):
    """
    PyTorch Dataset for preference pairs.
    
    Args:
        pairs: List of preference pairs
        tokenizer: Tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        pairs: List[PreferencePair],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self._preprocess()
    
    def _preprocess(self) -> None:
        """Preprocess pairs for training."""
        self.processed = []
        
        for pair in self.pairs:
            # Tokenize chosen
            chosen_encoded = self.tokenizer(
                pair.prompt + pair.chosen.text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            
            # Tokenize rejected
            rejected_encoded = self.tokenizer(
                pair.prompt + pair.rejected.text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            
            self.processed.append({
                'prompt': pair.prompt,
                'chosen_input_ids': chosen_encoded['input_ids'],
                'chosen_attention_mask': chosen_encoded['attention_mask'],
                'rejected_input_ids': rejected_encoded['input_ids'],
                'rejected_attention_mask': rejected_encoded['attention_mask'],
            })
    
    def __len__(self) -> int:
        return len(self.processed)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.processed[idx]
        return {
            'prompt': item['prompt'],
            'chosen_input_ids': torch.tensor(item['chosen_input_ids'], dtype=torch.long),
            'chosen_attention_mask': torch.tensor(item['chosen_attention_mask'], dtype=torch.long),
            'rejected_input_ids': torch.tensor(item['rejected_input_ids'], dtype=torch.long),
            'rejected_attention_mask': torch.tensor(item['rejected_attention_mask'], dtype=torch.long),
        }

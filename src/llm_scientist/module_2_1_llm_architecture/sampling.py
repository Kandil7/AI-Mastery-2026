"""
Sampling Strategies - Module 2.1.4

Production-ready implementations of text sampling strategies:
- Greedy Search
- Beam Search
- Temperature Sampling
- Top-p (Nucleus) Sampling
- Top-k Sampling
- Contrastive Search

References:
- "The Curious Case of Neural Text Generation" (Holtzman et al., 2019)
- "Contrastive Search: What's Needed for Better Text Generation" (Su et al., 2022)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SampleOutput:
    """Output from sampling."""
    sequences: Tensor
    scores: Optional[Tensor] = None
    sequences_scores: Optional[Tensor] = None
    attentions: Optional[List[Tensor]] = field(default_factory=list)
    hidden_states: Optional[List[Tensor]] = field(default_factory=list)


class BaseSampler(ABC):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample next token from logits."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset sampler state."""
        pass


class GreedySampler(BaseSampler):
    """
    Greedy Search Sampler.
    
    Selects the token with the highest probability at each step.
    Simple but can lead to repetitive and generic text.
    
    Example:
        >>> sampler = GreedySampler()
        >>> logits = torch.randn(1, 50257)
        >>> next_token = sampler.sample(logits)
    """
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using greedy decoding.
        
        Args:
            logits: Logits from model (batch, vocab_size)
            previous_tokens: Previously generated tokens (optional)
        
        Returns:
            Next token IDs (batch, 1)
        """
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    def reset(self) -> None:
        """Reset sampler state (no state for greedy)."""
        pass
    
    @staticmethod
    def generate(
        model: nn.Module,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> SampleOutput:
        """
        Generate text using greedy search.
        
        Args:
            model: Language model
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            attention_mask: Attention mask
        
        Returns:
            SampleOutput with generated sequences
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids
        scores = []
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(generated, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :]
            
            # Greedy selection
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Store score
            scores.append(F.log_softmax(next_token_logits, dim=-1).gather(1, next_token))
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)],
                    dim=1,
                )
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        # Compute sequence scores
        sequence_scores = torch.cat(scores, dim=1).sum(dim=1) if scores else None
        
        return SampleOutput(
            sequences=generated,
            scores=torch.cat(scores, dim=1) if scores else None,
            sequences_scores=sequence_scores,
        )


class BeamSearchSampler(BaseSampler):
    """
    Beam Search Sampler.
    
    Maintains multiple hypotheses (beams) and selects the best
    sequence at the end. Better than greedy but more computationally
    expensive.
    
    Args:
        num_beams: Number of beams to maintain
        length_penalty: Length penalty factor
        early_stopping: Whether to stop when all beams finish
        no_repeat_ngram_size: Size of n-grams to avoid repeating
        num_return_sequences: Number of sequences to return per input
        
    Example:
        >>> sampler = BeamSearchSampler(num_beams=5, length_penalty=0.6)
        >>> outputs = sampler.generate(model, input_ids, max_new_tokens=100)
    """
    
    def __init__(
        self,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        no_repeat_ngram_size: int = 0,
        num_return_sequences: int = 1,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_return_sequences = num_return_sequences
        
        # State for n-gram tracking
        self._generated_ngrams: Dict[int, set] = {}
    
    def _get_ngrams(
        self,
        tokens: List[int],
        n: int,
    ) -> set:
        """Extract n-grams from token sequence."""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.add(ngram)
        return ngrams
    
    def _is_ngram_allowed(
        self,
        tokens: List[int],
        new_token: int,
    ) -> bool:
        """Check if adding new_token would create a repeated n-gram."""
        if self.no_repeat_ngram_size <= 1:
            return True
        
        n = self.no_repeat_ngram_size
        if len(tokens) < n - 1:
            return True
        
        potential_ngram = tuple(tokens[-(n - 1):] + [new_token])
        beam_id = len(tokens)
        
        if beam_id not in self._generated_ngrams:
            self._generated_ngrams[beam_id] = self._get_ngrams(tokens, n)
        
        return potential_ngram not in self._generated_ngrams[beam_id]
    
    def _calculate_length_penalty(
        self,
        score: float,
        length: int,
    ) -> float:
        """Apply length penalty to score."""
        if self.length_penalty == 1.0:
            return score
        return score / (length ** self.length_penalty)
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using beam search (internal step).
        
        Note: Beam search is typically implemented in the generate method
        rather than as a per-step sampler.
        """
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    def reset(self) -> None:
        """Reset sampler state."""
        self._generated_ngrams = {}
    
    def generate(
        self,
        model: nn.Module,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> SampleOutput:
        """
        Generate text using beam search.
        
        Args:
            model: Language model
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            attention_mask: Attention mask
        
        Returns:
            SampleOutput with generated sequences
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0
        
        # Expand input for beams
        # (batch, seq) -> (batch * num_beams, seq)
        input_ids = input_ids.repeat_interleave(self.num_beams, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.num_beams, dim=0)
        
        # Initialize beam scores
        beam_scores = torch.zeros(
            batch_size * self.num_beams,
            dtype=torch.float,
            device=device,
        )
        
        # Track finished beams
        done = [False] * (batch_size * self.num_beams)
        
        # Store hypotheses
        hypotheses = [[] for _ in range(batch_size * self.num_beams)]
        
        self.reset()
        
        for step in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :]
            
            # Apply log softmax
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            
            # Reshape for beam selection
            # (batch * beams, vocab) -> (batch, beams * vocab)
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(
                batch_size, self.num_beams * vocab_size,
            )
            
            # Select top 2*num_beams candidates (for diversity)
            candidates = torch.topk(
                next_token_scores,
                2 * self.num_beams,
                dim=-1,
            )
            candidate_scores = candidates.values
            candidate_indices = candidates.indices
            
            # Select best beams
            next_beam_scores = candidate_scores[:, :self.num_beams]
            next_beam_indices = candidate_indices[:, :self.num_beams]
            
            # Convert flat indices to beam and token indices
            next_beams = next_beam_indices // vocab_size
            next_tokens = next_beam_indices % vocab_size
            
            # Update hypotheses and check for EOS
            new_hypotheses = []
            new_done = []
            
            for batch_idx in range(batch_size):
                for beam_idx in range(self.num_beams):
                    global_beam_idx = batch_idx * self.num_beams + beam_idx
                    
                    if done[global_beam_idx]:
                        new_hypotheses.append(hypotheses[global_beam_idx])
                        new_done.append(True)
                        continue
                    
                    # Get the beam this hypothesis came from
                    source_beam = batch_idx * self.num_beams + next_beams[batch_idx, beam_idx].item()
                    token = next_tokens[batch_idx, beam_idx].item()
                    
                    # Check n-gram constraint
                    if not self._is_ngram_allowed(hypotheses[source_beam], token):
                        token = pad_token_id
                    
                    new_hyp = hypotheses[source_beam] + [token]
                    new_hypotheses.append(new_hyp)
                    
                    # Check if finished
                    is_finished = (token == eos_token_id) if eos_token_id is not None else False
                    new_done.append(is_finished)
            
            hypotheses = new_hypotheses
            done = new_done
            
            # Reshape for next iteration
            input_ids = next_tokens.view(-1, 1)
            beam_scores = next_beam_scores.view(-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(input_ids)],
                    dim=1,
                )
            
            # Check if all beams are done
            if self.early_stopping and all(done):
                break
        
        # Select best hypotheses per batch
        final_sequences = []
        final_scores = []
        
        for batch_idx in range(batch_size):
            batch_hypotheses = []
            batch_scores = []
            
            for beam_idx in range(self.num_beams):
                global_idx = batch_idx * self.num_beams + beam_idx
                hyp = hypotheses[global_idx]
                
                # Calculate score with length penalty
                score = beam_scores[global_idx].item()
                score = self._calculate_length_penalty(score, len(hyp))
                
                batch_hypotheses.append((hyp, score))
            
            # Sort by score
            batch_hypotheses.sort(key=lambda x: x[1], reverse=True)
            
            # Take top sequences
            for i in range(min(self.num_return_sequences, len(batch_hypotheses))):
                hyp, score = batch_hypotheses[i]
                final_sequences.append(hyp)
                final_scores.append(score)
        
        # Convert to tensor
        if not final_sequences:
            final_sequences = [input_ids[0].tolist()]
        
        max_len = max(len(seq) for seq in final_sequences)
        
        # Pad sequences
        padded_sequences = []
        for seq in final_sequences:
            padded = seq + [pad_token_id] * (max_len - len(seq))
            padded_sequences.append(padded)
        
        sequences = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        sequence_scores = torch.tensor(final_scores, dtype=torch.float, device=device)
        
        return SampleOutput(
            sequences=sequences,
            sequences_scores=sequence_scores,
        )


class TemperatureSampler(BaseSampler):
    """
    Temperature Sampling.
    
    Scales logits by temperature before sampling:
    - T < 1: More confident/conservative
    - T > 1: More diverse/random
    - T = 1: Standard sampling
    
    Args:
        temperature: Sampling temperature
        
    Example:
        >>> sampler = TemperatureSampler(temperature=0.8)
        >>> logits = torch.randn(1, 50257)
        >>> next_token = sampler.sample(logits)
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using temperature scaling.
        
        Args:
            logits: Logits from model (batch, vocab_size)
            previous_tokens: Previously generated tokens
        
        Returns:
            Next token IDs (batch, 1)
        """
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def reset(self) -> None:
        """Reset sampler state."""
        pass
    
    @staticmethod
    def generate(
        model: nn.Module,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> SampleOutput:
        """
        Generate text using temperature sampling.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            attention_mask: Attention mask
        
        Returns:
            SampleOutput with generated sequences
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids
        scores = []
        
        for _ in range(max_new_tokens):
            outputs = model(generated, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            scores.append(F.log_softmax(next_token_logits, dim=-1).gather(1, next_token))
            generated = torch.cat([generated, next_token], dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)],
                    dim=1,
                )
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return SampleOutput(
            sequences=generated,
            scores=torch.cat(scores, dim=1) if scores else None,
        )


class TopKSampler(BaseSampler):
    """
    Top-k Sampling.
    
    Samples from the k most likely tokens. Filters out tokens
    with probability below the top-k threshold.
    
    Args:
        k: Number of top tokens to consider
        temperature: Sampling temperature
        
    Example:
        >>> sampler = TopKSampler(k=50, temperature=0.8)
        >>> logits = torch.randn(1, 50257)
        >>> next_token = sampler.sample(logits)
    """
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature
    
    @staticmethod
    def filter_top_k(
        logits: Tensor,
        k: int,
        min_tokens_to_keep: int = 1,
    ) -> Tensor:
        """
        Filter logits to keep only top-k tokens.
        
        Args:
            logits: Input logits
            k: Number of tokens to keep
            min_tokens_to_keep: Minimum tokens to keep
        
        Returns:
            Filtered logits (others set to -inf)
        """
        if k < 1:
            return logits
        
        # Get top-k indices
        top_k_indices = torch.topk(logits, max(k, min_tokens_to_keep), dim=-1).indices
        
        # Create mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # Set non-top-k to -inf
        filtered = logits.masked_fill(~mask, float('-inf'))
        
        return filtered
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using top-k filtering.
        
        Args:
            logits: Logits from model
            previous_tokens: Previously generated tokens
        
        Returns:
            Next token IDs
        """
        # Filter to top-k
        filtered_logits = self.filter_top_k(logits, self.k)
        
        # Apply temperature
        filtered_logits = filtered_logits / self.temperature
        
        # Convert to probabilities
        probs = F.softmax(filtered_logits, dim=-1)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def reset(self) -> None:
        """Reset sampler state."""
        pass


class TopPSampler(BaseSampler):
    """
    Top-p (Nucleus) Sampling.
    
    Samples from the smallest set of tokens whose cumulative
    probability exceeds p. More adaptive than top-k.
    
    Args:
        p: Cumulative probability threshold
        temperature: Sampling temperature
        min_tokens_to_keep: Minimum tokens to keep
        
    Example:
        >>> sampler = TopPSampler(p=0.9, temperature=0.8)
        >>> logits = torch.randn(1, 50257)
        >>> next_token = sampler.sample(logits)
    
    Reference:
        "The Curious Case of Neural Text Generation" (Holtzman et al., 2019)
    """
    
    def __init__(
        self,
        p: float = 0.9,
        temperature: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.p = p
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
    
    @staticmethod
    def filter_top_p(
        logits: Tensor,
        p: float,
        min_tokens_to_keep: int = 1,
    ) -> Tensor:
        """
        Filter logits using nucleus sampling.
        
        Args:
            logits: Input logits
            p: Cumulative probability threshold
            min_tokens_to_keep: Minimum tokens to keep
        
        Returns:
            Filtered logits
        """
        if p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1),
            dim=-1,
        )
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        
        # Shift to also remove the token just above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Create mask for original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove,
        )
        
        # Set removed tokens to -inf
        filtered = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return filtered
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using top-p filtering.
        
        Args:
            logits: Logits from model
            previous_tokens: Previously generated tokens
        
        Returns:
            Next token IDs
        """
        # Filter using nucleus
        filtered_logits = self.filter_top_p(logits, self.p, self.min_tokens_to_keep)
        
        # Apply temperature
        filtered_logits = filtered_logits / self.temperature
        
        # Convert to probabilities
        probs = F.softmax(filtered_logits, dim=-1)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def reset(self) -> None:
        """Reset sampler state."""
        pass


class ContrastiveSampler(BaseSampler):
    """
    Contrastive Search Sampler.
    
    Uses contrastive learning principles to select tokens that are
    both confident and diverse from previous context.
    
    Args:
        k: Number of candidate tokens to consider
        alpha: Degeneration penalty factor (0-1)
        temperature: Sampling temperature
        
    Example:
        >>> sampler = ContrastiveSampler(k=5, alpha=0.6)
        >>> logits = torch.randn(1, 50257)
        >>> hidden = torch.randn(1, 10, 768)
        >>> next_token = sampler.sample(logits, hidden)
    
    Reference:
        "Contrastive Search: What's Needed for Better Text Generation" (Su et al., 2022)
    """
    
    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.6,
        temperature: float = 1.0,
    ):
        self.k = k
        self.alpha = alpha
        self.temperature = temperature
        
        # Cache for previous hidden states
        self._hidden_cache: Optional[Tensor] = None
    
    def _compute_degeneration_penalty(
        self,
        candidate_hidden: Tensor,
        previous_hidden: Tensor,
    ) -> Tensor:
        """
        Compute degeneration penalty based on similarity to previous context.
        
        Args:
            candidate_hidden: Hidden states for candidate tokens
            previous_hidden: Previous hidden states
        
        Returns:
            Penalty scores
        """
        # Normalize
        candidate_norm = F.normalize(candidate_hidden, dim=-1)
        previous_norm = F.normalize(previous_hidden, dim=-1)
        
        # Compute max similarity
        similarity = torch.matmul(candidate_norm, previous_norm.transpose(-2, -1))
        max_similarity = similarity.max(dim=-1).values
        
        return max_similarity
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using contrastive search.
        
        Args:
            logits: Logits from model
            previous_tokens: Previously generated tokens
            hidden_states: Hidden states for degeneration penalty
        
        Returns:
            Next token IDs
        """
        # Get top-k candidates
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Apply temperature
        top_k_probs = F.softmax(top_k_logits / self.temperature, dim=-1)
        
        # Get hidden states for candidates
        if hidden_states is not None:
            # Get last hidden state
            last_hidden = hidden_states[:, -1, :]  # (batch, dim)
            
            # Compute candidate hidden states (approximation)
            # In practice, you'd need to run forward pass for each candidate
            candidate_hidden = last_hidden.unsqueeze(1).expand(-1, self.k, -1)
            
            # Compute degeneration penalty
            if self._hidden_cache is not None:
                penalty = self._compute_degeneration_penalty(
                    candidate_hidden,
                    self._hidden_cache,
                )
                
                # Combine probability and penalty
                scores = top_k_probs - self.alpha * penalty
            else:
                scores = top_k_probs
            
            # Select best token
            best_idx = torch.argmax(scores, dim=-1, keepdim=True)
            next_token = torch.gather(top_k_indices, -1, best_idx)
            
            # Update cache
            self._hidden_cache = torch.cat(
                [self._hidden_cache, last_hidden.unsqueeze(1)],
                dim=1,
            ) if self._hidden_cache is not None else last_hidden.unsqueeze(1)
        else:
            # Fallback to top-k sampling
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token)
        
        return next_token
    
    def reset(self) -> None:
        """Reset sampler state."""
        self._hidden_cache = None
    
    @staticmethod
    def generate(
        model: nn.Module,
        input_ids: Tensor,
        max_new_tokens: int,
        k: int = 5,
        alpha: float = 0.6,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> SampleOutput:
        """
        Generate text using contrastive search.
        
        Args:
            model: Language model with hidden states output
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens
            k: Number of candidates
            alpha: Degeneration penalty
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            attention_mask: Attention mask
        
        Returns:
            SampleOutput with generated sequences
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids
        hidden_cache = None
        
        for _ in range(max_new_tokens):
            # Forward pass with hidden states
            outputs = model(generated, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            
            next_token_logits = logits[:, -1, :]
            
            # Get top-k candidates
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            if hidden_states is not None:
                last_hidden = hidden_states[:, -1, :]  # (batch, dim)
                
                # Expand for candidates
                candidate_hidden = last_hidden.unsqueeze(1).expand(-1, k, -1)
                
                if hidden_cache is not None:
                    # Compute similarity penalty
                    candidate_norm = F.normalize(candidate_hidden, dim=-1)
                    cache_norm = F.normalize(hidden_cache, dim=-1)
                    similarity = torch.matmul(candidate_norm, cache_norm.transpose(-2, -1))
                    penalty = similarity.max(dim=-1).values
                    
                    # Combined score
                    scores = top_k_probs - alpha * penalty
                    best_idx = torch.argmax(scores, dim=-1, keepdim=True)
                else:
                    best_idx = torch.argmax(top_k_probs, dim=-1, keepdim=True)
                
                next_token = torch.gather(top_k_indices, -1, best_idx)
                
                # Update cache
                hidden_cache = torch.cat(
                    [hidden_cache, last_hidden.unsqueeze(1)],
                    dim=1,
                ) if hidden_cache is not None else last_hidden.unsqueeze(1)
            else:
                # Fallback to greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)],
                    dim=1,
                )
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return SampleOutput(sequences=generated)


class TypicalSampler(BaseSampler):
    """
    Typical Sampling.
    
    Filters tokens based on information content (surprisal).
    Keeps tokens whose information content is close to the entropy.
    
    Args:
        mass: Probability mass to retain
        temperature: Sampling temperature
        
    Reference:
        "Locally Typical Sampling" (Meister et al., 2022)
    """
    
    def __init__(self, mass: float = 0.9, temperature: float = 1.0):
        self.mass = mass
        self.temperature = temperature
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using typical sampling.
        
        Args:
            logits: Logits from model
            previous_tokens: Previously generated tokens
        
        Returns:
            Next token IDs
        """
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Compute entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        
        # Compute surprisal (information content)
        surprisal = -log_probs
        
        # Compute distance from entropy
        distance = torch.abs(surprisal - entropy)
        
        # Sort by distance
        sorted_distance, sorted_indices = torch.sort(distance, dim=-1)
        
        # Compute cumulative probability
        sorted_probs = torch.gather(probs, -1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff
        cutoff = cumulative_probs < self.mass
        cutoff[..., 0] = True  # Keep at least one token
        
        # Create mask
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, cutoff)
        
        # Filter logits
        filtered_logits = scaled_logits.masked_fill(~mask, float('-inf'))
        
        # Sample
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        
        return next_token
    
    def reset(self) -> None:
        """Reset sampler state."""
        pass


class EtaSampler(BaseSampler):
    """
    Eta (η) Sampling.
    
    Combines temperature and top-p with dynamic thresholding
    based on entropy.
    
    Args:
        eta: Eta parameter for threshold
        epsilon: Minimum probability threshold
        
    Reference:
        "Eta Sampling: Efficient and Effective Text Generation" (Hewitt et al., 2022)
    """
    
    def __init__(self, eta: float = 0.1, epsilon: float = 1e-5):
        self.eta = eta
        self.epsilon = epsilon
    
    def sample(
        self,
        logits: Tensor,
        previous_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample using eta sampling.
        
        Args:
            logits: Logits from model
            previous_tokens: Previously generated tokens
        
        Returns:
            Next token IDs
        """
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        
        # Compute threshold
        threshold = torch.clamp(
            self.epsilon + self.eta * torch.exp(-entropy),
            min=self.epsilon,
            max=1.0,
        )
        
        # Filter by threshold
        mask = probs >= threshold
        
        # Ensure at least one token
        if not mask.any():
            mask = probs == probs.max(dim=-1, keepdim=True).values
        
        # Renormalize
        filtered_probs = probs.masked_fill(~mask, 0)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # Sample
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        
        return next_token
    
    def reset(self) -> None:
        """Reset sampler state."""
        pass


class SamplerRegistry:
    """Registry for sampling strategies."""
    
    _samplers: Dict[str, type] = {
        'greedy': GreedySampler,
        'beam': BeamSearchSampler,
        'temperature': TemperatureSampler,
        'top_k': TopKSampler,
        'top_p': TopPSampler,
        'contrastive': ContrastiveSampler,
        'typical': TypicalSampler,
        'eta': EtaSampler,
    }
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseSampler:
        """Get a sampler by name."""
        if name not in cls._samplers:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(cls._samplers.keys())}")
        return cls._samplers[name](**kwargs)
    
    @classmethod
    def register(cls, name: str, sampler_class: type) -> None:
        """Register a new sampler."""
        cls._samplers[name] = sampler_class
    
    @classmethod
    def list_samplers(cls) -> List[str]:
        """List available samplers."""
        return list(cls._samplers.keys())

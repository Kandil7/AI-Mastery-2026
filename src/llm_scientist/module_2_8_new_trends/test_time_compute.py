"""
Test-Time Compute - Module 2.8.4

Test-time computation techniques: Chain-of-Thought, Self-Consistency, Majority Voting, Verification.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TestTimeComputeConfig:
    """Configuration for test-time compute."""
    num_samples: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    use_cot: bool = True
    use_self_consistency: bool = True
    use_verification: bool = True


class ChainOfThoughtGenerator:
    """
    Chain-of-Thought Generator.
    
    Generates step-by-step reasoning for problems.
    
    Reference: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
    """
    
    COT_PROMPT = """Solve the following problem step by step.

Problem: {problem}

Let's think step by step:
"""
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(
        self,
        problem: str,
        max_tokens: int = 512,
    ) -> str:
        """Generate chain-of-thought reasoning."""
        prompt = self.COT_PROMPT.format(problem=problem)
        
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors='pt',
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated[len(prompt):].strip()
    
    def generate_batch(
        self,
        problems: List[str],
    ) -> List[str]:
        """Generate CoT for multiple problems."""
        return [self.generate(p) for p in problems]


class SelfConsistency:
    """
    Self-Consistency for improved accuracy.
    
    Samples multiple reasoning paths and selects the most consistent answer.
    
    Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TestTimeComputeConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
    
    def sample_reasoning_paths(
        self,
        problem: str,
        num_samples: Optional[int] = None,
    ) -> List[str]:
        """Sample multiple reasoning paths."""
        num_samples = num_samples or self.config.num_samples
        
        prompt = ChainOfThoughtGenerator.COT_PROMPT.format(problem=problem)
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        paths = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                paths.append(generated[len(prompt):].strip())
        
        return paths
    
    def extract_answer(self, reasoning: str) -> str:
        """Extract final answer from reasoning."""
        import re
        
        # Look for answer patterns
        patterns = [
            r'(?:answer is|therefore|thus|so)\s*[:\-]?\s*(.+?)(?:\.|$)',
            r'####\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:is the answer|is correct)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return last sentence
        sentences = reasoning.split('.')
        if sentences:
            return sentences[-1].strip()
        
        return reasoning
    
    def compute_consistency(
        self,
        problem: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compute self-consistent answer.
        
        Returns:
            Tuple of (answer, metadata)
        """
        # Sample reasoning paths
        paths = self.sample_reasoning_paths(problem)
        
        # Extract answers
        answers = [self.extract_answer(path) for path in paths]
        
        # Count votes
        vote_counts = Counter(answers)
        
        # Get most common answer
        most_common = vote_counts.most_common(1)[0]
        
        metadata = {
            'num_samples': len(paths),
            'unique_answers': len(vote_counts),
            'vote_distribution': dict(vote_counts),
            'consensus_ratio': most_common[1] / len(answers),
            'reasoning_paths': paths,
        }
        
        return most_common[0], metadata


class MajorityVoting:
    """
    Majority Voting for answer aggregation.
    
    Aggregates multiple model outputs using voting.
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any, config: TestTimeComputeConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
    
    def sample_answers(
        self,
        prompt: str,
        num_samples: Optional[int] = None,
    ) -> List[str]:
        """Sample multiple answers."""
        num_samples = num_samples or self.config.num_samples
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        answers = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=self.config.temperature,
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answers.append(generated[len(prompt):].strip())
        
        return answers
    
    def vote(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Perform majority voting.
        
        Returns:
            Tuple of (answer, metadata)
        """
        answers = self.sample_answers(prompt)
        
        # Count votes
        vote_counts = Counter(answers)
        
        # Get most common
        most_common = vote_counts.most_common(1)[0]
        
        metadata = {
            'all_answers': answers,
            'vote_counts': dict(vote_counts),
            'consensus_ratio': most_common[1] / len(answers),
        }
        
        return most_common[0], metadata


class VerificationModule:
    """
    Verification Module for answer checking.
    
    Verifies answers using various methods.
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    VERIFICATION_PROMPT = """Verify if the following answer to the problem is correct.

Problem: {problem}
Proposed Answer: {answer}
Reasoning: {reasoning}

Is this answer correct? Respond with just "Yes" or "No".

Verification:"""
    
    def verify(
        self,
        problem: str,
        answer: str,
        reasoning: str = "",
    ) -> Tuple[bool, float]:
        """
        Verify an answer.
        
        Returns:
            Tuple of (is_correct, confidence)
        """
        prompt = self.VERIFICATION_PROMPT.format(
            problem=problem,
            answer=answer,
            reasoning=reasoning,
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip().lower()
        
        # Parse response
        if 'yes' in response:
            return True, 0.9
        elif 'no' in response:
            return False, 0.9
        else:
            # Use probability from logits
            logits = self.model(inputs).logits[0, -1, :]
            yes_id = self.tokenizer.encode('Yes')[0]
            no_id = self.tokenizer.encode('No')[0]
            
            probs = torch.softmax(logits, dim=-1)
            confidence = max(probs[yes_id], probs[no_id]).item()
            
            return probs[yes_id] > probs[no_id], confidence
    
    def verify_with_sampling(
        self,
        problem: str,
        answer: str,
        reasoning: str = "",
        num_samples: int = 5,
    ) -> Tuple[bool, float]:
        """Verify with multiple samples for robustness."""
        verifications = []
        
        for _ in range(num_samples):
            is_correct, confidence = self.verify(problem, answer, reasoning)
            verifications.append((is_correct, confidence))
        
        # Aggregate
        correct_count = sum(1 for v in verifications if v[0])
        avg_confidence = sum(v[1] for v in verifications) / len(verifications)
        
        return correct_count > num_samples / 2, avg_confidence


class TestTimeCompute:
    """
    Complete Test-Time Compute System.
    
    Combines CoT, self-consistency, and verification.
    
    Example:
        >>> ttc = TestTimeCompute(model, tokenizer, config)
        >>> result = ttc.solve(problem)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TestTimeComputeConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.cot_generator = ChainOfThoughtGenerator(model, tokenizer)
        self.self_consistency = SelfConsistency(model, tokenizer, config)
        self.majority_voting = MajorityVoting(model, tokenizer, config)
        self.verifier = VerificationModule(model, tokenizer)
    
    def solve(
        self,
        problem: str,
        use_cot: Optional[bool] = None,
        use_self_consistency: Optional[bool] = None,
        use_verification: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Solve a problem using test-time compute.
        
        Args:
            problem: Problem to solve
            use_cot: Use chain-of-thought
            use_self_consistency: Use self-consistency
            use_verification: Use verification
        
        Returns:
            Solution with metadata
        """
        use_cot = use_cot if use_cot is not None else self.config.use_cot
        use_sc = use_self_consistency if use_self_consistency is not None else self.config.use_self_consistency
        use_ver = use_verification if use_verification is not None else self.config.use_verification
        
        result = {
            'problem': problem,
            'answer': None,
            'reasoning': None,
            'verified': False,
            'metadata': {},
        }
        
        if use_sc:
            # Self-consistency with CoT
            answer, sc_metadata = self.self_consistency.compute_consistency(problem)
            result['answer'] = answer
            result['reasoning'] = sc_metadata['reasoning_paths'][0]
            result['metadata']['self_consistency'] = sc_metadata
        
        elif use_cot:
            # Simple CoT
            reasoning = self.cot_generator.generate(problem)
            answer = self.self_consistency.extract_answer(reasoning)
            result['answer'] = answer
            result['reasoning'] = reasoning
        
        else:
            # Direct answer with majority voting
            prompt = f"Problem: {problem}\n\nAnswer:"
            answer, mv_metadata = self.majority_voting.vote(prompt)
            result['answer'] = answer
            result['metadata']['majority_voting'] = mv_metadata
        
        # Verification
        if use_ver and result['answer']:
            is_correct, confidence = self.verifier.verify(
                problem,
                result['answer'],
                result['reasoning'] or "",
            )
            result['verified'] = is_correct
            result['metadata']['verification_confidence'] = confidence
        
        return result
    
    def solve_batch(
        self,
        problems: List[str],
    ) -> List[Dict[str, Any]]:
        """Solve multiple problems."""
        return [self.solve(p) for p in problems]
    
    def compute_accuracy(
        self,
        problems: List[Tuple[str, str]],  # (problem, correct_answer)
    ) -> float:
        """Compute accuracy on a set of problems."""
        correct = 0
        
        for problem, correct_answer in problems:
            result = self.solve(problem)
            
            # Compare answers (simple string comparison)
            if result['answer'].lower().strip() == correct_answer.lower().strip():
                correct += 1
        
        return correct / len(problems) if problems else 0

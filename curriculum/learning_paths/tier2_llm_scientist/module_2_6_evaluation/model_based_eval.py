"""
Model-Based Evaluation - Module 2.6.3

LLM-as-judge evaluation framework.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ScoringRubric:
    """Scoring rubric for evaluation."""
    criteria: Dict[str, str]
    scale: Dict[int, str]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Format rubric as prompt."""
        text = "## Evaluation Rubric\n\n"
        text += "### Criteria:\n"
        for name, desc in self.criteria.items():
            text += f"- {name}: {desc}\n"
        text += "\n### Scale:\n"
        for score, desc in self.scale.items():
            text += f"{score}: {desc}\n"
        return text


@dataclass
class EvaluationOutput:
    """Output from model-based evaluation."""
    scores: Dict[str, float]
    rationale: str
    confidence: float = 0.0


class LLMJudge:
    """
    LLM-as-Judge for evaluating model outputs.
    
    Uses an LLM to score responses based on defined criteria.
    
    Args:
        model: Judge model
        tokenizer: Tokenizer
        rubric: Scoring rubric
        
    Example:
        >>> judge = LLMJudge(model, tokenizer, rubric)
        >>> output = judge.evaluate(prompt, response)
    """
    
    EVALUATION_PROMPT = """You are an expert evaluator. Evaluate the following response based on the criteria.

{rubric}

---
Prompt: {prompt}
Response: {response}
---

Provide your evaluation in the following JSON format:
{{
    "scores": {{"criterion1": score1, "criterion2": score2}},
    "rationale": "Your detailed rationale",
    "confidence": confidence_score
}}

Evaluation:"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        rubric: Optional[ScoringRubric] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rubric = rubric or self._default_rubric()
        
        self.device = next(model.parameters()).device
    
    def _default_rubric(self) -> ScoringRubric:
        """Get default rubric."""
        return ScoringRubric(
            criteria={
                'helpfulness': 'How helpful is the response?',
                'accuracy': 'How accurate is the information?',
                'clarity': 'How clear and well-organized is the response?',
            },
            scale={
                1: 'Poor',
                2: 'Below Average',
                3: 'Average',
                4: 'Good',
                5: 'Excellent',
            },
        )
    
    def _format_prompt(
        self,
        prompt: str,
        response: str,
    ) -> str:
        """Format evaluation prompt."""
        return self.EVALUATION_PROMPT.format(
            rubric=self.rubric.to_prompt(),
            prompt=prompt,
            response=response,
        )
    
    def _parse_output(self, text: str) -> EvaluationOutput:
        """Parse model output."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                
                return EvaluationOutput(
                    scores=data.get('scores', {}),
                    rationale=data.get('rationale', ''),
                    confidence=data.get('confidence', 0.5),
                )
        except Exception:
            pass
        
        # Fallback: simple parsing
        return EvaluationOutput(
            scores={'overall': 3.0},
            rationale=text,
        )
    
    def evaluate(
        self,
        prompt: str,
        response: str,
    ) -> EvaluationOutput:
        """Evaluate a response."""
        eval_prompt = self._format_prompt(prompt, response)
        
        inputs = self.tokenizer.encode(
            eval_prompt,
            return_tensors='pt',
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self._parse_output(generated[len(eval_prompt):])
    
    def evaluate_batch(
        self,
        prompt_response_pairs: List[Tuple[str, str]],
    ) -> List[EvaluationOutput]:
        """Evaluate multiple responses."""
        return [
            self.evaluate(prompt, response)
            for prompt, response in prompt_response_pairs
        ]


class PairwiseEvaluator:
    """
    Pairwise Comparison Evaluator.
    
    Compares two responses and determines which is better.
    
    Args:
        model: Judge model
        tokenizer: Tokenizer
        
    Example:
        >>> evaluator = PairwiseEvaluator(model, tokenizer)
        >>> winner = evaluator.compare(prompt, response_a, response_b)
    """
    
    COMPARISON_PROMPT = """Compare the two responses to the following prompt.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider: helpfulness, accuracy, clarity, and completeness.

Respond with just the letter: A or B

Winner:"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Tuple[str, str]:
        """
        Compare two responses.
        
        Returns:
            Tuple of (winner, rationale)
        """
        eval_prompt = self.COMPARISON_PROMPT.format(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
        )
        
        inputs = self.tokenizer.encode(
            eval_prompt,
            return_tensors='pt',
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated[len(eval_prompt):].strip()
        
        # Extract winner
        winner = 'A' if 'A' in result.upper() else 'B'
        
        return winner, result
    
    def compare_batch(
        self,
        comparisons: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str]]:
        """Compare multiple pairs."""
        return [
            self.compare(prompt, resp_a, resp_b)
            for prompt, resp_a, resp_b in comparisons
        ]


class ModelBasedEvaluator:
    """
    Complete Model-Based Evaluation System.
    
    Combines multiple evaluation methods for comprehensive assessment.
    
    Args:
        judge_model: Model for evaluation
        tokenizer: Tokenizer
        output_dir: Directory for results
        
    Example:
        >>> evaluator = ModelBasedEvaluator(model, tokenizer)
        >>> results = evaluator.evaluate_all(prompts, responses)
    """
    
    def __init__(
        self,
        judge_model: Any,
        tokenizer: Any,
        output_dir: str = './eval_results',
    ):
        self.tokenizer = tokenizer
        
        self.judge = LLMJudge(judge_model, tokenizer)
        self.pairwise = PairwiseEvaluator(judge_model, tokenizer)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
    
    def score_response(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """Score a single response."""
        output = self.judge.evaluate(prompt, response)
        
        result = {
            'prompt': prompt,
            'response': response,
            'scores': output.scores,
            'rationale': output.rationale,
            'confidence': output.confidence,
        }
        
        self.results.append(result)
        
        return result
    
    def compare_responses(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Dict[str, Any]:
        """Compare two responses."""
        winner, rationale = self.pairwise.compare(prompt, response_a, response_b)
        
        result = {
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
            'winner': winner,
            'rationale': rationale,
        }
        
        self.results.append(result)
        
        return result
    
    def evaluate_all(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Dict[str, Any]:
        """Evaluate all prompt-response pairs."""
        logger.info(f"Evaluating {len(prompts)} examples...")
        
        for prompt, response in zip(prompts, responses):
            self.score_response(prompt, response)
        
        # Compute aggregate metrics
        metrics = self._compute_metrics()
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if not self.results:
            return {}
        
        # Collect all scores
        all_scores = {}
        
        for result in self.results:
            if 'scores' not in result:
                continue
            
            for criterion, score in result['scores'].items():
                if criterion not in all_scores:
                    all_scores[criterion] = []
                all_scores[criterion].append(score)
        
        # Compute averages
        avg_scores = {
            criterion: sum(scores) / len(scores)
            for criterion, scores in all_scores.items()
        }
        
        return {
            'num_evaluations': len(self.results),
            'avg_scores': avg_scores,
            'detailed_results': self.results,
        }
    
    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """Save results to file."""
        path = self.output_dir / 'model_eval_results.json'
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {path}")

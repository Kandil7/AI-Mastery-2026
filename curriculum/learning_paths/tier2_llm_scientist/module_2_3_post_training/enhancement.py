"""
Data Enhancement - Module 2.3.3

Production-ready data enhancement techniques:
- Chain-of-Thought generation
- Branch-Solve-Merge
- Self-Reflection
- Self-Correction

References:
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- "Branch-Solve-Merge Improves Large Language Model Evaluation and Generation" (Wu et al., 2023)
- "Self-Refinement: Language Models Can Self-Improve" (Madaan et al., 2023)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .formats import Conversation, Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class EnhancedExample:
    """An enhanced training example with reasoning."""
    instruction: str
    input: str = ""
    output: str = ""
    reasoning: str = ""
    steps: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    reflection: str = ""
    corrections: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_conversation(self, include_reasoning: bool = True) -> Conversation:
        """Convert to Conversation format."""
        messages = []
        
        # User message
        user_content = self.instruction
        if self.input:
            user_content += f"\n\n{self.input}"
        messages.append(Message(role=MessageRole.USER, content=user_content))
        
        # Assistant message with reasoning
        if include_reasoning and self.reasoning:
            assistant_content = f"Let me think through this step by step.\n\n{self.reasoning}\n\nTherefore, the answer is: {self.output}"
        else:
            assistant_content = self.output
        
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_content))
        
        return Conversation(
            messages=messages,
            metadata={
                'steps': self.steps,
                'alternatives': self.alternatives,
                'reflection': self.reflection,
                **self.metadata,
            },
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output,
            'reasoning': self.reasoning,
            'steps': self.steps,
            'alternatives': self.alternatives,
            'reflection': self.reflection,
            'corrections': self.corrections,
            'metadata': self.metadata,
        }


class ChainOfThoughtGenerator:
    """
    Chain-of-Thought (CoT) Generator.
    
    Generates step-by-step reasoning for instructions to improve
    model reasoning capabilities.
    
    Args:
        model: LLM model for generation
        tokenizer: Tokenizer
        
    Reference:
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
    
    Example:
        >>> cot_gen = ChainOfThoughtGenerator(model, tokenizer)
        >>> enhanced = cot_gen.enhance(example)
    """
    
    # CoT prompt templates
    COT_PROMPTS = {
        'general': """Let's think step by step.

Instruction: {instruction}
Input: {input}

Step-by-step reasoning:
""",
        'math': """Let's solve this math problem step by step.

Problem: {instruction}

Step 1: Understand what we're asked to find.
Step 2: Identify the relevant information.
Step 3: Apply the appropriate mathematical operations.
Step 4: Verify the result.

Solution:
""",
        'logic': """Let's analyze this logically.

Problem: {instruction}

Step 1: Identify the premises.
Step 2: Apply logical reasoning.
Step 3: Draw conclusions.
Step 4: Check for consistency.

Answer:
""",
        'code': """Let's solve this coding problem step by step.

Task: {instruction}

Step 1: Understand the requirements.
Step 2: Plan the solution approach.
Step 3: Write the code.
Step 4: Test and verify.

Code:
""",
    }
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
    
    def _generate_reasoning(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Generate reasoning using the model."""
        if not self.model or not self.tokenizer:
            return ""
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    def _extract_steps(self, reasoning: str) -> List[str]:
        """Extract individual steps from reasoning."""
        steps = []
        
        # Try to find numbered steps
        import re
        pattern = r'(?:Step\s*\d*[:\.]|\d+\.)\s*(.+?)(?=(?:Step\s*\d*[:\.]|\d+\.|$))'
        matches = re.findall(pattern, reasoning, re.IGNORECASE | re.DOTALL)
        
        if matches:
            steps = [m.strip() for m in matches]
        else:
            # Split by newlines as fallback
            steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
        
        return steps
    
    def enhance(
        self,
        instruction: str,
        input_text: str = "",
        output: str = "",
        category: str = 'general',
    ) -> EnhancedExample:
        """
        Enhance an example with chain-of-thought reasoning.
        
        Args:
            instruction: The instruction
            input_text: Optional input text
            output: Expected output
            category: Category for prompt selection
        
        Returns:
            EnhancedExample with reasoning
        """
        # Select appropriate prompt
        prompt_template = self.COT_PROMPTS.get(category, self.COT_PROMPTS['general'])
        
        prompt = prompt_template.format(
            instruction=instruction,
            input=input_text,
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(prompt)
        
        # Extract steps
        steps = self._extract_steps(reasoning)
        
        return EnhancedExample(
            instruction=instruction,
            input=input_text,
            output=output,
            reasoning=reasoning,
            steps=steps,
            metadata={'category': category, 'cot_generated': True},
        )
    
    def enhance_batch(
        self,
        examples: List[Dict[str, str]],
        category: str = 'general',
    ) -> List[EnhancedExample]:
        """Enhance multiple examples with CoT."""
        return [
            self.enhance(
                ex.get('instruction', ''),
                ex.get('input', ''),
                ex.get('output', ''),
                category,
            )
            for ex in examples
        ]


class BranchSolveMerge:
    """
    Branch-Solve-Merge (BSM) for improved generation.
    
    Generates multiple solution branches and merges them
    for higher quality outputs.
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        num_branches: Number of solution branches
        
    Reference:
        "Branch-Solve-Merge Improves Large Language Model Evaluation and Generation" (Wu et al., 2023)
    
    Example:
        >>> bsm = BranchSolveMerge(model, tokenizer, num_branches=3)
        >>> result = bsm.generate(instruction, input_text)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        num_branches: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_branches = num_branches
    
    def _generate_branch(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate a single solution branch."""
        if not self.model or not self.tokenizer:
            return ""
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    def _merge_branches(
        self,
        branches: List[str],
        instruction: str,
    ) -> str:
        """Merge multiple branches into a final answer."""
        if not branches:
            return ""
        
        if len(branches) == 1:
            return branches[0]
        
        # Create merge prompt
        merge_prompt = f"""Given the following different approaches to solving a problem, synthesize them into a single, comprehensive answer.

Instruction: {instruction}

Approach 1:
{branches[0]}

Approach 2:
{branches[1] if len(branches) > 1 else ''}

Approach 3:
{branches[2] if len(branches) > 2 else ''}

Synthesized answer:
"""
        
        return self._generate_branch(merge_prompt, temperature=0.3)
    
    def generate(
        self,
        instruction: str,
        input_text: str = "",
    ) -> EnhancedExample:
        """
        Generate using Branch-Solve-Merge.
        
        Args:
            instruction: The instruction
            input_text: Optional input text
        
        Returns:
            EnhancedExample with branches and merged result
        """
        # Create base prompt
        base_prompt = f"""Instruction: {instruction}
Input: {input_text if input_text else 'N/A'}

Solution:
"""
        
        # Generate multiple branches with different temperatures
        branches = []
        temperatures = [0.5, 0.7, 0.9][:self.num_branches]
        
        for temp in temperatures:
            branch = self._generate_branch(base_prompt, temperature=temp)
            if branch:
                branches.append(branch)
        
        # Merge branches
        merged = self._merge_branches(branches, instruction)
        
        return EnhancedExample(
            instruction=instruction,
            input=input_text,
            output=merged,
            alternatives=branches,
            metadata={'bpm_generated': True, 'num_branches': len(branches)},
        )


class SelfReflection:
    """
    Self-Reflection for response improvement.
    
    Generates reflections on model outputs to identify
    areas for improvement.
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        
    Reference:
        "Self-Refinement: Language Models Can Self-Improve" (Madaan et al., 2023)
    
    Example:
        >>> reflector = SelfReflection(model, tokenizer)
        >>> reflection = reflector.reflect(instruction, response)
    """
    
    REFLECTION_PROMPT = """Analyze the following response critically.

Instruction: {instruction}
Input: {input}
Response: {response}

Please evaluate the response on these criteria:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the instruction?
3. Clarity: Is it easy to understand?
4. Helpfulness: Would this be useful to the user?
5. Safety: Is the response appropriate and safe?

For each criterion, identify any issues and suggest improvements.

Reflection:
"""
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
    
    def _generate_reflection(
        self,
        instruction: str,
        input_text: str,
        response: str,
    ) -> str:
        """Generate reflection on a response."""
        if not self.model or not self.tokenizer:
            return ""
        
        prompt = self.REFLECTION_PROMPT.format(
            instruction=instruction,
            input=input_text,
            response=response,
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.5,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    def reflect(
        self,
        instruction: str,
        response: str,
        input_text: str = "",
    ) -> str:
        """
        Generate reflection on a response.
        
        Args:
            instruction: The instruction
            response: The response to reflect on
            input_text: Optional input text
        
        Returns:
            Reflection text
        """
        return self._generate_reflection(instruction, input_text, response)
    
    def enhance(
        self,
        instruction: str,
        output: str,
        input_text: str = "",
    ) -> EnhancedExample:
        """
        Enhance an example with self-reflection.
        
        Args:
            instruction: The instruction
            output: The output
            input_text: Optional input text
        
        Returns:
            EnhancedExample with reflection
        """
        reflection = self.reflect(instruction, output, input_text)
        
        return EnhancedExample(
            instruction=instruction,
            input=input_text,
            output=output,
            reflection=reflection,
            metadata={'reflection_generated': True},
        )


class SelfCorrection:
    """
    Self-Correction for iterative improvement.
    
    Uses reflection to identify issues and generate
    corrected responses.
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        max_iterations: Maximum correction iterations
        
    Example:
        >>> corrector = SelfCorrection(model, tokenizer)
        >>> corrected = corrector.correct(instruction, initial_response)
    """
    
    CORRECTION_PROMPT = """Based on the reflection, generate an improved response.

Instruction: {instruction}
Input: {input}
Original Response: {original_response}
Reflection: {reflection}

Improved Response:
"""
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_iterations: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        
        self.reflector = SelfReflection(model, tokenizer)
    
    def _generate_correction(
        self,
        instruction: str,
        input_text: str,
        original_response: str,
        reflection: str,
    ) -> str:
        """Generate a corrected response."""
        if not self.model or not self.tokenizer:
            return original_response
        
        prompt = self.CORRECTION_PROMPT.format(
            instruction=instruction,
            input=input_text,
            original_response=original_response,
            reflection=reflection,
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.5,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    def correct(
        self,
        instruction: str,
        initial_response: str,
        input_text: str = "",
    ) -> EnhancedExample:
        """
        Generate a corrected response through self-reflection.
        
        Args:
            instruction: The instruction
            initial_response: Initial response
            input_text: Optional input text
        
        Returns:
            EnhancedExample with corrections
        """
        current_response = initial_response
        corrections = []
        
        for i in range(self.max_iterations):
            # Generate reflection
            reflection = self.reflector.reflect(
                instruction,
                current_response,
                input_text,
            )
            
            # Generate correction
            corrected = self._generate_correction(
                instruction,
                input_text,
                current_response,
                reflection,
            )
            
            # Record correction
            corrections.append({
                'iteration': i + 1,
                'reflection': reflection,
                'corrected_response': corrected,
            })
            
            # Check if response changed significantly
            if corrected == current_response:
                break
            
            current_response = corrected
        
        return EnhancedExample(
            instruction=instruction,
            input=input_text,
            output=current_response,
            corrections=corrections,
            metadata={
                'correction_iterations': len(corrections),
                'self_corrected': True,
            },
        )
    
    def correct_batch(
        self,
        examples: List[Dict[str, str]],
    ) -> List[EnhancedExample]:
        """Correct multiple examples."""
        return [
            self.correct(
                ex.get('instruction', ''),
                ex.get('output', ''),
                ex.get('input', ''),
            )
            for ex in examples
        ]


class DataEnhancementPipeline:
    """
    Complete Data Enhancement Pipeline.
    
    Combines multiple enhancement techniques:
    1. Chain-of-Thought
    2. Branch-Solve-Merge
    3. Self-Reflection
    4. Self-Correction
    
    Example:
        >>> pipeline = DataEnhancementPipeline(model, tokenizer)
        >>> enhanced = pipeline.enhance_dataset(dataset)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        self.cot_generator = ChainOfThoughtGenerator(model, tokenizer)
        self.bsm = BranchSolveMerge(model, tokenizer)
        self.reflector = SelfReflection(model, tokenizer)
        self.corrector = SelfCorrection(model, tokenizer)
    
    def enhance_with_cot(
        self,
        examples: List[Dict[str, str]],
        category: str = 'general',
    ) -> List[EnhancedExample]:
        """Enhance examples with Chain-of-Thought."""
        return self.cot_generator.enhance_batch(examples, category)
    
    def enhance_with_bsm(
        self,
        examples: List[Dict[str, str]],
    ) -> List[EnhancedExample]:
        """Enhance examples with Branch-Solve-Merge."""
        return [
            self.bsm.generate(
                ex.get('instruction', ''),
                ex.get('input', ''),
            )
            for ex in examples
        ]
    
    def enhance_with_reflection(
        self,
        examples: List[Dict[str, str]],
    ) -> List[EnhancedExample]:
        """Enhance examples with Self-Reflection."""
        return [
            self.reflector.enhance(
                ex.get('instruction', ''),
                ex.get('output', ''),
                ex.get('input', ''),
            )
            for ex in examples
        ]
    
    def enhance_with_correction(
        self,
        examples: List[Dict[str, str]],
    ) -> List[EnhancedExample]:
        """Enhance examples with Self-Correction."""
        return self.corrector.correct_batch(examples)
    
    def enhance_dataset(
        self,
        examples: List[Dict[str, str]],
        methods: Optional[List[str]] = None,
    ) -> List[EnhancedExample]:
        """
        Enhance dataset using specified methods.
        
        Args:
            examples: List of examples to enhance
            methods: Enhancement methods to apply
        
        Returns:
            List of EnhancedExample
        """
        methods = methods or ['cot']
        enhanced = []
        
        for ex in examples:
            result = EnhancedExample(
                instruction=ex.get('instruction', ''),
                input=ex.get('input', ''),
                output=ex.get('output', ''),
            )
            
            if 'cot' in methods:
                cot_result = self.cot_generator.enhance(
                    result.instruction,
                    result.input,
                    result.output,
                )
                result.reasoning = cot_result.reasoning
                result.steps = cot_result.steps
            
            if 'bsm' in methods:
                bsm_result = self.bsm.generate(
                    result.instruction,
                    result.input,
                )
                result.output = bsm_result.output
                result.alternatives = bsm_result.alternatives
            
            if 'reflection' in methods:
                refl_result = self.reflector.enhance(
                    result.instruction,
                    result.output,
                    result.input,
                )
                result.reflection = refl_result.reflection
            
            if 'correction' in methods:
                corr_result = self.corrector.correct(
                    result.instruction,
                    result.output,
                    result.input,
                )
                result.output = corr_result.output
                result.corrections = corr_result.corrections
            
            enhanced.append(result)
        
        return enhanced
    
    def save(
        self,
        enhanced: List[EnhancedExample],
        path: str,
    ) -> None:
        """Save enhanced examples to file."""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [ex.to_dict() for ex in enhanced]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(enhanced)} enhanced examples to {path}")

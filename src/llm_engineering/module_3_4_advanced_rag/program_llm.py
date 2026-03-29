"""
Program LLM Module

Production-ready DSPy integration and prompt optimization:
- DSPy wrapper for programmatic LLM usage
- Prompt optimization
- Bootstrapping demonstrations
- Compiled programs

Features:
- Signature-based programming
- Automatic prompt optimization
- Demonstration collection
- Program compilation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class Signature:
    """
    Defines input/output signature for a program step.

    Similar to DSPy's Signature class.
    """

    inputs: Dict[str, str]  # field_name -> description
    outputs: Dict[str, str]  # field_name -> description
    instructions: str = ""

    def to_prompt(self) -> str:
        """Convert signature to prompt format."""
        lines = []

        if self.instructions:
            lines.append(self.instructions)
            lines.append("")

        lines.append("Inputs:")
        for name, desc in self.inputs.items():
            lines.append(f"  - {name}: {desc}")

        lines.append("")
        lines.append("Outputs:")
        for name, desc in self.outputs.items():
            lines.append(f"  - {name}: {desc}")

        return "\n".join(lines)

    @classmethod
    def from_string(cls, signature_str: str) -> "Signature":
        """Parse signature from string format."""
        # Format: "input1, input2 -> output1, output2"
        parts = signature_str.split("->")

        inputs = {}
        outputs = {}

        if len(parts) >= 1:
            for inp in parts[0].split(","):
                inp = inp.strip()
                if inp:
                    inputs[inp] = f"Input {inp}"

        if len(parts) >= 2:
            for out in parts[1].split(","):
                out = out.strip()
                if out:
                    outputs[out] = f"Output {out}"

        return cls(inputs=inputs, outputs=outputs)


@dataclass
class Demonstration:
    """A demonstration example for few-shot learning."""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Demonstration":
        return cls(
            inputs=data["inputs"],
            outputs=data["outputs"],
            weight=data.get("weight", 1.0),
        )


@dataclass
class ProgramState:
    """State of a compiled program."""

    signature: Signature
    demonstrations: List[Demonstration] = field(default_factory=list)
    prompt_template: str = ""
    compiled: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": {
                "inputs": self.signature.inputs,
                "outputs": self.signature.outputs,
                "instructions": self.signature.instructions,
            },
            "demonstrations": [d.to_dict() for d in self.demonstrations],
            "prompt_template": self.prompt_template,
            "compiled": self.compiled,
            "metrics": self.metrics,
        }


class ProgramLLM(ABC):
    """Abstract base class for programmatic LLM usage."""

    def __init__(
        self,
        llm_client: Any,
        signature: Signature,
    ) -> None:
        self.llm_client = llm_client
        self.signature = signature

        self._demonstrations: List[Demonstration] = []
        self._prompt_template = self._build_default_template()
        self._stats = {
            "total_calls": 0,
            "total_tokens": 0,
        }

    def _build_default_template(self) -> str:
        """Build default prompt template."""
        return """{instructions}

{demonstrations}

Inputs:
{inputs}

Outputs:"""

    def add_demonstration(self, demonstration: Demonstration) -> None:
        """Add a demonstration example."""
        self._demonstrations.append(demonstration)

    def add_demonstrations(self, demonstrations: List[Demonstration]) -> None:
        """Add multiple demonstrations."""
        self._demonstrations.extend(demonstrations)

    def clear_demonstrations(self) -> None:
        """Clear all demonstrations."""
        self._demonstrations.clear()

    @abstractmethod
    async def forward(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the program."""
        pass

    def _format_prompt(self, **kwargs: Any) -> str:
        """Format prompt with inputs and demonstrations."""
        # Format demonstrations
        demos_text = ""
        if self._demonstrations:
            demos_text = "\n\nExamples:\n" + "\n\n".join(
                self._format_demonstration(d) for d in self._demonstrations[-5:]  # Last 5
            )

        # Format inputs
        inputs_text = "\n".join(
            f"  - {k}: {v}" for k, v in kwargs.items() if k in self.signature.inputs
        )

        return self._prompt_template.format(
            instructions=self.signature.instructions,
            demonstrations=demos_text,
            inputs=inputs_text,
        )

    def _format_demonstration(self, demo: Demonstration) -> str:
        """Format a demonstration for prompt."""
        lines = ["Example:"]

        lines.append("Inputs:")
        for k, v in demo.inputs.items():
            lines.append(f"  - {k}: {v}")

        lines.append("Outputs:")
        for k, v in demo.outputs.items():
            lines.append(f"  - {k}: {v}")

        return "\n".join(lines)

    def _parse_outputs(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into output fields."""
        outputs = {}

        # Try to extract each output field
        for field_name in self.signature.outputs:
            # Look for field in response
            patterns = [
                f"{field_name}: (.+?)(?:\n|$)",
                f"{field_name}=(.+?)(?:\n|$)",
                f"\"{field_name}\":\s*\"(.+?)\"",
            ]

            import re
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    outputs[field_name] = match.group(1).strip()
                    break

        # If no structured output found, use entire response
        if not outputs and self.signature.outputs:
            first_output = list(self.signature.outputs.keys())[0]
            outputs[first_output] = response_text.strip()

        return outputs


class DSPyWrapper:
    """
    Wrapper for DSPy library integration.

    Provides DSPy-style programming when DSPy is available,
    with fallback to custom implementation.
    """

    def __init__(
        self,
        llm_client: Any,
        model: Optional[str] = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model

        self._dspy = None
        self._lm = None

        self._try_import_dspy()

    def _try_import_dspy(self) -> None:
        """Try to import DSPy."""
        try:
            import dspy
            self._dspy = dspy

            # Configure LM
            class DSPyLM:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model

                def __call__(self, prompt, **kwargs):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self.client.generate(
                            messages=[{"role": "user", "content": prompt}],
                            **kwargs,
                        )
                    )
                    return response.content if hasattr(response, 'content') else str(response)

            self._lm = DSPyLM(self.llm_client, self.model)
            logger.info("DSPy wrapper initialized")
        except ImportError:
            logger.warning("DSPy not installed. Using fallback implementation.")

    def create_module(
        self,
        signature: Signature,
        **kwargs: Any,
    ) -> "DSPyModule":
        """Create a DSPy-style module."""
        if self._dspy:
            return self._create_dspy_module(signature, **kwargs)
        else:
            return self._create_fallback_module(signature, **kwargs)

    def _create_dspy_module(
        self,
        signature: Signature,
        **kwargs: Any,
    ) -> "DSPyModule":
        """Create module using DSPy."""
        # Convert signature to DSPy format
        dspy_signature = self._dspy.Signature(
            signature.inputs,
            signature.outputs,
            signature.instructions,
        )

        class DSPyModule(self._dspy.Module):
            def __init__(self, signature, lm):
                super().__init__()
                self.signature = signature
                self.lm = lm
                self.predictor = self._dspy.Predict(signature)

            def forward(self, **kwargs):
                with self._dspy.context(lm=self.lm):
                    return self.predictor(**kwargs)

        return DSPyModule(dspy_signature, self._lm)

    def _create_fallback_module(
        self,
        signature: Signature,
        **kwargs: Any,
    ) -> "DSPyModule":
        """Create fallback module without DSPy."""
        return FallbackModule(self.llm_client, signature)

    def compile(
        self,
        program: "DSPyModule",
        trainset: List[Demonstration],
        metric: Callable[[Any, Any], float],
        **kwargs: Any,
    ) -> "DSPyModule":
        """Compile program with training data."""
        if self._dspy:
            return self._compile_dspy(program, trainset, metric, **kwargs)
        else:
            return self._compile_fallback(program, trainset, metric, **kwargs)

    def _compile_dspy(
        self,
        program: "DSPyModule",
        trainset: List[Demonstration],
        metric: Callable,
        **kwargs: Any,
    ) -> "DSPyModule":
        """Compile using DSPy."""
        # Convert trainset to DSPy format
        dspy_trainset = []
        for demo in trainset:
            example = self._dspy.Example(
                **demo.inputs,
                **demo.outputs,
            )
            dspy_trainset.append(example)

        # Use DSPy optimizer
        optimizer = self._dspy.MIPRO(
            metric=metric,
            num_candidates=kwargs.get("num_candidates", 3),
            init_temperature=kwargs.get("init_temperature", 1.0),
        )

        compiled = optimizer.compile(
            program,
            trainset=dspy_trainset,
            **kwargs,
        )

        return compiled

    def _compile_fallback(
        self,
        program: "DSPyModule",
        trainset: List[Demonstration],
        metric: Callable,
        **kwargs: Any,
    ) -> "DSPyModule":
        """Fallback compilation without DSPy."""
        # Simple demonstration-based optimization
        program.add_demonstrations(trainset)
        program.compiled = True
        return program


class DSPyModule:
    """Base class for DSPy-style modules."""

    def __init__(
        self,
        llm_client: Any,
        signature: Signature,
    ) -> None:
        self.llm_client = llm_client
        self.signature = signature
        self.demonstrations: List[Demonstration] = []
        self.compiled = False

    def add_demonstrations(self, demonstrations: List[Demonstration]) -> None:
        """Add demonstrations."""
        self.demonstrations.extend(demonstrations)

    async def forward(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute module."""
        raise NotImplementedError


class FallbackModule(DSPyModule):
    """Fallback module without DSPy."""

    async def forward(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute with fallback implementation."""
        # Build prompt
        prompt_parts = []

        if self.signature.instructions:
            prompt_parts.append(self.signature.instructions)

        # Add demonstrations
        if self.demonstrations:
            prompt_parts.append("\nExamples:")
            for demo in self.demonstrations[-5:]:
                prompt_parts.append("\nInput:")
                for k, v in demo.inputs.items():
                    prompt_parts.append(f"  {k}: {v}")
                prompt_parts.append("\nOutput:")
                for k, v in demo.outputs.items():
                    prompt_parts.append(f"  {k}: {v}")

        # Add current input
        prompt_parts.append("\n\nNow solve:")
        for k, v in kwargs.items():
            prompt_parts.append(f"  {k}: {v}")

        prompt_parts.append("\n\nOutput:")

        prompt = "\n".join(prompt_parts)

        # Call LLM
        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse outputs
        outputs = {}
        for field in self.signature.outputs:
            outputs[field] = response_text

        return outputs


class PromptOptimizer:
    """
    Optimizes prompts through iterative refinement.

    Features:
    - Automatic prompt variation
    - Performance-based selection
    - Demonstration optimization
    """

    def __init__(
        self,
        llm_client: Any,
        metric: Callable[[Dict[str, Any], Dict[str, Any]], float],
    ) -> None:
        self.llm_client = llm_client
        self.metric = metric

        self._best_prompt: str = ""
        self._best_score: float = 0.0
        self._history: List[Dict[str, Any]] = []

    async def optimize(
        self,
        base_prompt: str,
        trainset: List[Demonstration],
        num_iterations: int = 5,
        num_variants: int = 3,
    ) -> str:
        """
        Optimize prompt through iterations.

        Args:
            base_prompt: Starting prompt
            trainset: Training demonstrations
            num_iterations: Number of optimization iterations
            num_variants: Variants to generate per iteration

        Returns:
            Optimized prompt
        """
        self._best_prompt = base_prompt
        self._best_score = await self._evaluate(base_prompt, trainset)

        logger.info(f"Initial score: {self._best_score:.4f}")

        for iteration in range(num_iterations):
            # Generate variants
            variants = await self._generate_variants(
                self._best_prompt,
                num_variants,
            )

            # Evaluate variants
            for variant in variants:
                score = await self._evaluate(variant, trainset)

                self._history.append({
                    "iteration": iteration,
                    "prompt": variant[:200],
                    "score": score,
                })

                if score > self._best_score:
                    self._best_score = score
                    self._best_prompt = variant
                    logger.info(f"New best score: {self._best_score:.4f}")

        return self._best_prompt

    async def _generate_variants(
        self,
        prompt: str,
        num_variants: int,
    ) -> List[str]:
        """Generate prompt variants."""
        generation_prompt = f"""Generate {num_variants} variations of this prompt.
Each variation should express the same task but with different wording.
Keep the core instructions intact.

Original Prompt:
{prompt}

Variations (separate with '---'):"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": generation_prompt}],
            temperature=0.7,
        )

        content = response.content if hasattr(response, 'content') else str(response)
        variants = [v.strip() for v in content.split("---") if v.strip()]

        return variants[:num_variants]

    async def _evaluate(
        self,
        prompt: str,
        trainset: List[Demonstration],
    ) -> float:
        """Evaluate prompt on trainset."""
        scores = []

        for demo in trainset:
            # Run with prompt
            messages = [{"role": "user", "content": prompt + "\n\n" + str(demo.inputs)}]
            response = await self.llm_client.generate(
                messages=messages,
                temperature=0.1,
            )

            predicted = response.content if hasattr(response, 'content') else str(response)

            # Calculate metric
            score = self.metric(demo.outputs, {"output": predicted})
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0

    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._history.copy()


class Bootstrapper:
    """
    Bootstraps demonstrations for few-shot learning.

    Generates high-quality demonstrations automatically.
    """

    def __init__(
        self,
        llm_client: Any,
        signature: Signature,
    ) -> None:
        self.llm_client = llm_client
        self.signature = signature

    async def bootstrap(
        self,
        inputs: List[Dict[str, Any]],
        num_demos: int = 5,
    ) -> List[Demonstration]:
        """
        Bootstrap demonstrations from inputs.

        Args:
            inputs: List of input examples
            num_demos: Number of demonstrations to create

        Returns:
            List of demonstrations
        """
        demonstrations = []

        for input_example in inputs[:num_demos]:
            # Generate output using LLM
            output = await self._generate_output(input_example)

            demo = Demonstration(
                inputs=input_example,
                outputs=output,
            )
            demonstrations.append(demo)

        return demonstrations

    async def _generate_output(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate output for given inputs."""
        prompt = f"""{self.signature.to_prompt()}

Inputs:
{json.dumps(inputs, indent=2)}

Outputs:"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse outputs
        outputs = {}
        for field in self.signature.outputs:
            outputs[field] = response_text

        return outputs

    async def bootstrap_with_refinement(
        self,
        inputs: List[Dict[str, Any]],
        num_demos: int = 5,
        refinement_iterations: int = 2,
    ) -> List[Demonstration]:
        """Bootstrap with self-refinement."""
        demonstrations = await self.bootstrap(inputs, num_demos)

        for _ in range(refinement_iterations):
            demonstrations = await self._refine_demonstrations(demonstrations)

        return demonstrations

    async def _refine_demonstrations(
        self,
        demonstrations: List[Demonstration],
    ) -> List[Demonstration]:
        """Refine demonstrations for quality."""
        refined = []

        for demo in demonstrations:
            # Ask LLM to improve the output
            refinement_prompt = f"""Improve this output to be more accurate and helpful.

Input: {json.dumps(demo.inputs)}
Current Output: {json.dumps(demo.outputs)}

Improved Output:"""

            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.2,
            )

            improved_text = response.content if hasattr(response, 'content') else str(response)

            refined_demo = Demonstration(
                inputs=demo.inputs,
                outputs={"output": improved_text},
                weight=demo.weight + 0.1,  # Increase weight for refined demos
            )
            refined.append(refined_demo)

        return refined


class CompiledProgram:
    """
    A compiled and optimized program.

    Can be saved, loaded, and executed efficiently.
    """

    def __init__(
        self,
        llm_client: Any,
        state: ProgramState,
    ) -> None:
        self.llm_client = llm_client
        self.state = state

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the compiled program."""
        # Build prompt
        prompt_parts = []

        if self.state.signature.instructions:
            prompt_parts.append(self.state.signature.instructions)

        # Add demonstrations
        if self.state.demonstrations:
            prompt_parts.append("\nExamples:")
            for demo in self.state.demonstrations:
                prompt_parts.append("\nInput:")
                for k, v in demo.inputs.items():
                    prompt_parts.append(f"  {k}: {v}")
                prompt_parts.append("\nOutput:")
                for k, v in demo.outputs.items():
                    prompt_parts.append(f"  {k}: {v}")

        # Add current input
        prompt_parts.append("\n\nNow solve:")
        for k, v in kwargs.items():
            prompt_parts.append(f"  {k}: {v}")

        prompt_parts.append("\n\nOutput:")

        prompt = "\n".join(prompt_parts)

        # Call LLM
        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse outputs
        outputs = {}
        for field in self.state.signature.outputs:
            outputs[field] = response_text

        return outputs

    def save(self, path: Union[str, Path]) -> None:
        """Save program to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        logger.info(f"Saved compiled program to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        llm_client: Any,
    ) -> "CompiledProgram":
        """Load program from file."""
        with open(path, "r") as f:
            data = json.load(f)

        signature = Signature(
            inputs=data["signature"]["inputs"],
            outputs=data["signature"]["outputs"],
            instructions=data["signature"]["instructions"],
        )

        state = ProgramState(
            signature=signature,
            demonstrations=[
                Demonstration.from_dict(d) for d in data.get("demonstrations", [])
            ],
            prompt_template=data.get("prompt_template", ""),
            compiled=data.get("compiled", False),
            metrics=data.get("metrics", {}),
        )

        return cls(llm_client, state)

    def get_stats(self) -> Dict[str, Any]:
        """Get program statistics."""
        return {
            "compiled": self.state.compiled,
            "num_demonstrations": len(self.state.demonstrations),
            "metrics": self.state.metrics,
        }


class ProgramCompiler:
    """
    Compiles programs with optimization.

    Orchestrates the full compilation pipeline.
    """

    def __init__(
        self,
        llm_client: Any,
        signature: Signature,
    ) -> None:
        self.llm_client = llm_client
        self.signature = signature

        self.dspy_wrapper = DSPyWrapper(llm_client)
        self.optimizer: Optional[PromptOptimizer] = None
        self.bootstrapper = Bootstrapper(llm_client, signature)

    async def compile(
        self,
        base_prompt: str,
        train_inputs: List[Dict[str, Any]],
        train_outputs: Optional[List[Dict[str, Any]]] = None,
        metric: Optional[Callable] = None,
        num_demos: int = 5,
        optimize: bool = True,
    ) -> CompiledProgram:
        """
        Compile a program.

        Args:
            base_prompt: Base prompt template
            train_inputs: Training inputs
            train_outputs: Optional training outputs
            metric: Optional evaluation metric
            num_demos: Number of demonstrations
            optimize: Whether to optimize prompt

        Returns:
            Compiled program
        """
        # Bootstrap demonstrations
        demonstrations = await self.bootstrapper.bootstrap(
            train_inputs,
            num_demos,
        )

        # If outputs provided, update demonstrations
        if train_outputs:
            for demo, output in zip(demonstrations, train_outputs):
                demo.outputs = output

        # Create program state
        state = ProgramState(
            signature=self.signature,
            demonstrations=demonstrations,
            prompt_template=base_prompt,
        )

        # Optimize if requested
        if optimize and metric:
            self.optimizer = PromptOptimizer(self.llm_client, metric)
            optimized_prompt = await self.optimizer.optimize(
                base_prompt,
                demonstrations,
            )
            state.prompt_template = optimized_prompt

        state.compiled = True

        return CompiledProgram(self.llm_client, state)

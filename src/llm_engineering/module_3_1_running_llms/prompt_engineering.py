"""
Prompt Engineering Module

Production-ready implementations of advanced prompting techniques:
- Zero-shot prompting
- Few-shot prompting with examples
- Chain-of-Thought (CoT)
- ReAct (Reasoning + Acting)
- Self-Consistency
- Tree-of-Thought (ToT)

Features:
- Prompt templating with variables
- Example management
- Strategy composition
- Token counting and optimization
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .apis import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class PromptStrategy(str, Enum):
    """Available prompt strategies."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHT = "tree_of_thought"
    SKEPTIC = "skeptic"  # Critique and refine
    MAIEUTIC = "maieutic"  # Self-explanation


@dataclass
class FewShotExample:
    """A few-shot example for prompting."""

    input: str
    output: str
    explanation: Optional[str] = None
    weight: float = 1.0  # For weighted example selection

    def format(self) -> str:
        """Format example for inclusion in prompt."""
        formatted = f"Input: {self.input}\nOutput: {self.output}"
        if self.explanation:
            formatted += f"\nExplanation: {self.explanation}"
        return formatted


@dataclass
class PromptTemplate:
    """
    Template for constructing prompts with variables.

    Supports:
    - Variable substitution with {variable_name}
    - Conditional sections
    - Example injection
    """

    template: str
    variables: Optional[Dict[str, Any]] = None
    examples: List[FewShotExample] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with provided variables.

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Rendered prompt string
        """
        variables = {**(self.variables or {}), **kwargs}

        # Handle examples section
        rendered = self.template
        if "{examples}" in rendered:
            examples_text = "\n\n".join(
                f"Example {i + 1}:\n{ex.format()}"
                for i, ex in enumerate(self.examples)
            )
            rendered = rendered.replace("{examples}", examples_text)

        # Substitute variables
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, str(value))

        return rendered

    def get_messages(
        self,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """
        Get messages in chat format.

        Returns:
            List of message dicts with role and content
        """
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        rendered = self.render(**kwargs)
        messages.append({"role": "user", "content": rendered})

        return messages

    @classmethod
    def from_file(cls, file_path: str, **kwargs: Any) -> "PromptTemplate":
        """Load template from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
        return cls(template=template, **kwargs)

    def save(self, file_path: str) -> None:
        """Save template to file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.template)


@dataclass
class ThoughtStep:
    """A single step in reasoning chain."""

    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a response."""

    question: str
    steps: List[ThoughtStep]
    final_answer: str
    total_steps: int
    time_taken: float


class BasePromptStrategy(ABC):
    """Abstract base class for prompt strategies."""

    def __init__(
        self,
        client: BaseLLMClient,
        template: Optional[PromptTemplate] = None,
    ) -> None:
        self.client = client
        self.template = template or PromptTemplate(template="{input}")

    @abstractmethod
    async def execute(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> Union[str, ReasoningTrace]:
        """Execute the strategy."""
        pass


class ZeroShotStrategy(BasePromptStrategy):
    """Zero-shot prompting strategy."""

    async def execute(
        self,
        input_text: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute zero-shot prompting.

        Args:
            input_text: The input/query
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})

        response = await self.client.generate(messages=messages, **kwargs)
        return response.content


class FewShotStrategy(BasePromptStrategy):
    """Few-shot prompting with examples."""

    def __init__(
        self,
        client: BaseLLMClient,
        examples: List[FewShotExample],
        template: Optional[PromptTemplate] = None,
        example_selection: str = "all",  # all, top_k, similar
        max_examples: int = 5,
    ) -> None:
        super().__init__(client, template)
        self.examples = examples
        self.example_selection = example_selection
        self.max_examples = max_examples

    def _select_examples(self, input_text: str) -> List[FewShotExample]:
        """Select examples based on strategy."""
        if self.example_selection == "all":
            return self.examples[:self.max_examples]
        elif self.example_selection == "top_k":
            # Could implement similarity-based selection here
            return self.examples[:self.max_examples]
        else:
            return self.examples[:self.max_examples]

    async def execute(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> str:
        """Execute few-shot prompting."""
        selected_examples = self._select_examples(input_text)

        # Build prompt with examples
        examples_text = "\n\n".join(
            f"Example {i + 1}:\n{ex.format()}"
            for i, ex in enumerate(selected_examples)
        )

        prompt = f"""{examples_text}

Now solve this:
Input: {input_text}
Output:"""

        response = await self.client.generate(
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content


class ChainOfThoughtStrategy(BasePromptStrategy):
    """
    Chain-of-Thought prompting strategy.

    Encourages the model to show its reasoning step-by-step
    before providing the final answer.
    """

    COT_PROMPT = """Let's think step by step to solve this problem.

{input}

First, let me break down the problem:
"""

    ANSWER_PATTERN = re.compile(
        r"(?:therefore|thus|hence|so|answer is|final answer:?)\s*([^\n]+)",
        re.IGNORECASE,
    )

    async def execute(
        self,
        input_text: str,
        show_reasoning: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Execute chain-of-thought prompting.

        Args:
            input_text: The problem/question
            show_reasoning: Whether to include reasoning in output
            **kwargs: Generation parameters

        Returns:
            Final answer (with or without reasoning)
        """
        prompt = self.COT_PROMPT.format(input=input_text)

        response = await self.client.generate(
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        full_response = response.content

        # Try to extract final answer
        match = self.ANSWER_PATTERN.search(full_response)
        if match:
            final_answer = match.group(1).strip()
        else:
            # Use last sentence as answer
            sentences = full_response.split(".")
            final_answer = sentences[-1].strip() if sentences else full_response

        if show_reasoning:
            return full_response
        return final_answer

    async def execute_with_trace(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute CoT and return full reasoning trace."""
        start_time = time.time()

        prompt = self.COT_PROMPT.format(input=input_text)
        response = await self.client.generate(
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        # Parse reasoning steps
        steps = self._parse_reasoning_steps(response.content)

        return ReasoningTrace(
            question=input_text,
            steps=steps,
            final_answer=steps[-1].thought if steps else response.content,
            total_steps=len(steps),
            time_taken=time.time() - start_time,
        )

    def _parse_reasoning_steps(self, text: str) -> List[ThoughtStep]:
        """Parse text into reasoning steps."""
        steps = []

        # Split by common step indicators
        step_patterns = [
            r"(?:step \d+|first|second|third|next|then|finally)[:\s]+",
            r"^\d+\.\s+",
            r"^(let's|let us)\s+",
        ]

        lines = text.split("\n")
        current_step = []

        for line in lines:
            is_new_step = any(
                re.match(pattern, line.lower())
                for pattern in step_patterns
            )

            if is_new_step and current_step:
                steps.append(ThoughtStep(thought="\n".join(current_step)))
                current_step = [line]
            else:
                current_step.append(line)

        if current_step:
            steps.append(ThoughtStep(thought="\n".join(current_step)))

        return steps


class ReActStrategy(BasePromptStrategy):
    """
    ReAct (Reasoning + Acting) strategy.

    Combines reasoning with action execution in an iterative loop.
    """

    REACT_TEMPLATE = """Solve a task by interleaving reasoning and actions.

You have access to the following tools:
{tools}

Use the following format:
Thought: Your reasoning about what to do
Action: The action to take (one of: {tool_names})
Action Input: The input to the action
Observation: The result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: Your final answer

Begin!

Question: {input}
{history}
Thought:"""

    def __init__(
        self,
        client: BaseLLMClient,
        tools: Dict[str, Callable[[str], str]],
        template: Optional[PromptTemplate] = None,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(client, template)
        self.tools = tools
        self.max_iterations = max_iterations

    async def execute(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> str:
        """Execute ReAct strategy with tool use."""
        history = ""
        tool_names = ", ".join(self.tools.keys())
        tools_description = "\n".join(
            f"- {name}: {tool.__doc__ or 'No description'}"
            for name, tool in self.tools.items()
        )

        for iteration in range(self.max_iterations):
            prompt = self.REACT_TEMPLATE.format(
                tools=tools_description,
                tool_names=tool_names,
                input=input_text,
                history=history,
            )

            response = await self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                stop=["Observation:", "observation:"],
                **kwargs,
            )

            content = response.content.strip()
            history += f"\nThought: {content}"

            # Check for final answer
            if "Final Answer:" in content or "I now know the final answer" in content:
                # Extract final answer
                if "Final Answer:" in content:
                    return content.split("Final Answer:")[-1].strip()
                return content

            # Parse action
            action_match = re.search(
                r"Action:\s*(\w+)\s*\n\s*Action Input:\s*(.+)",
                content,
                re.IGNORECASE,
            )

            if action_match:
                action_name = action_match.group(1).strip()
                action_input = action_match.group(2).strip()

                if action_name in self.tools:
                    try:
                        observation = self.tools[action_name](action_input)
                        history += f"\nObservation: {observation}"
                    except Exception as e:
                        history += f"\nObservation: Error - {e}"
                else:
                    history += f"\nObservation: Unknown tool: {action_name}"
            else:
                # No action found, continue reasoning
                pass

        return f"Failed to complete after {self.max_iterations} iterations. Best response: {history}"


class SelfConsistencyStrategy(BasePromptStrategy):
    """
    Self-Consistency strategy.

    Generates multiple reasoning paths and selects the most
    consistent answer through voting.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        n_samples: int = 5,
        template: Optional[PromptTemplate] = None,
    ) -> None:
        super().__init__(client, template)
        self.n_samples = n_samples

    async def execute(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> str:
        """
        Execute self-consistency strategy.

        Args:
            input_text: The problem/question
            **kwargs: Generation parameters (temperature will be overridden)

        Returns:
            Most consistent answer
        """
        # Generate multiple samples with higher temperature
        responses = []

        tasks = [
            self.client.generate(
                messages=[{"role": "user", "content": input_text}],
                temperature=0.7 + (i * 0.1),  # Vary temperature
                **kwargs,
            )
            for i in range(self.n_samples)
        ]

        results = await asyncio.gather(*tasks)
        responses = [r.content for r in results]

        # Extract answers and vote
        answers = self._extract_answers(responses)
        return self._majority_vote(answers)

    def _extract_answers(self, responses: List[str]) -> List[str]:
        """Extract final answers from responses."""
        answers = []
        answer_pattern = re.compile(
            r"(?:answer is|final answer:?)\s*([^\n]+)",
            re.IGNORECASE,
        )

        for response in responses:
            match = answer_pattern.search(response)
            if match:
                answers.append(match.group(1).strip().lower())
            else:
                # Use last sentence
                sentences = response.split(".")
                answers.append(sentences[-1].strip().lower() if sentences else response.lower())

        return answers

    def _majority_vote(self, answers: List[str]) -> str:
        """Select answer by majority vote."""
        from collections import Counter

        if not answers:
            return ""

        counter = Counter(answers)
        most_common = counter.most_common(1)[0]
        return most_common[0]


class TreeOfThoughtStrategy(BasePromptStrategy):
    """
    Tree-of-Thought (ToT) strategy.

    Explores multiple reasoning paths in a tree structure,
    evaluating and backtracking as needed.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        breadth: int = 3,  # Branches per node
        depth: int = 3,  # Max depth
        template: Optional[PromptTemplate] = None,
    ) -> None:
        super().__init__(client, template)
        self.breadth = breadth
        self.depth = depth

    async def execute(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> str:
        """Execute tree-of-thought strategy."""
        # Initialize root
        root = {
            "thought": input_text,
            "children": [],
            "value": 0.0,
        }

        # BFS to explore tree
        await self._explore_tree(root, depth=0)

        # Find best leaf
        best_leaf = self._find_best_leaf(root)
        return best_leaf["thought"] if best_leaf else input_text

    async def _explore_tree(self, node: Dict, depth: int) -> None:
        """Recursively explore the thought tree."""
        if depth >= self.depth:
            return

        # Generate possible next thoughts
        thoughts = await self._generate_thoughts(node["thought"], self.breadth)

        for thought in thoughts:
            child = {
                "thought": thought,
                "children": [],
                "value": await self._evaluate_thought(thought),
            }
            node["children"].append(child)

            # Recursively explore
            await self._explore_tree(child, depth + 1)

    async def _generate_thoughts(
        self,
        current_thought: str,
        n_thoughts: int,
    ) -> List[str]:
        """Generate possible next thoughts."""
        prompt = f"""Given the current thought:
"{current_thought}"

Generate {n_thoughts} different possible next steps in reasoning.
Format each as a separate line prefixed with "Thought: "

Current thought: {current_thought}

Next thoughts:
"""

        response = await self.client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        thoughts = []
        for line in response.content.split("\n"):
            if line.strip().startswith("Thought:"):
                thoughts.append(line.split("Thought:")[-1].strip())

        return thoughts[:n_thoughts] if thoughts else [current_thought]

    async def _evaluate_thought(self, thought: str) -> float:
        """Evaluate the quality of a thought."""
        prompt = f"""Evaluate the quality of this thought on a scale of 0 to 1.
Consider: logical consistency, relevance, and progress toward solution.

Thought: {thought}

Rating (just output a number between 0 and 1):"""

        response = await self.client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        try:
            rating = float(response.content.strip())
            return max(0.0, min(1.0, rating))
        except ValueError:
            return 0.5

    def _find_best_leaf(self, node: Dict) -> Optional[Dict]:
        """Find the best leaf node."""
        if not node["children"]:
            return node

        best_child = None
        best_value = -1

        for child in node["children"]:
            leaf = self._find_best_leaf(child)
            if leaf and leaf["value"] > best_value:
                best_value = leaf["value"]
                best_child = leaf

        return best_child


class PromptEngineer:
    """
    Main interface for prompt engineering.

    Provides unified access to all prompting strategies with
    configuration and composition support.
    """

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client
        self._strategies: Dict[PromptStrategy, BasePromptStrategy] = {}
        self._default_strategy = PromptStrategy.ZERO_SHOT

    def register_strategy(
        self,
        strategy: PromptStrategy,
        instance: BasePromptStrategy,
    ) -> None:
        """Register a prompt strategy."""
        self._strategies[strategy] = instance
        logger.info(f"Registered strategy: {strategy}")

    def get_strategy(self, strategy: PromptStrategy) -> BasePromptStrategy:
        """Get or create a strategy instance."""
        if strategy not in self._strategies:
            self._strategies[strategy] = self._create_default_strategy(strategy)
        return self._strategies[strategy]

    def _create_default_strategy(
        self,
        strategy: PromptStrategy,
    ) -> BasePromptStrategy:
        """Create default strategy instance."""
        strategies = {
            PromptStrategy.ZERO_SHOT: ZeroShotStrategy,
            PromptStrategy.FEW_SHOT: lambda: FewShotStrategy(self.client, []),
            PromptStrategy.CHAIN_OF_THOUGHT: lambda: ChainOfThoughtStrategy(self.client),
            PromptStrategy.REACT: lambda: ReActStrategy(self.client, {}),
            PromptStrategy.SELF_CONSISTENCY: lambda: SelfConsistencyStrategy(self.client),
            PromptStrategy.TREE_OF_THOUGHT: lambda: TreeOfThoughtStrategy(self.client),
        }

        factory = strategies.get(strategy)
        if factory:
            return factory() if callable(factory) else factory
        raise ValueError(f"Unknown strategy: {strategy}")

    async def execute(
        self,
        input_text: str,
        strategy: Optional[PromptStrategy] = None,
        **kwargs: Any,
    ) -> Union[str, ReasoningTrace]:
        """
        Execute prompting with specified strategy.

        Args:
            input_text: Input/query
            strategy: Prompting strategy to use
            **kwargs: Strategy-specific parameters

        Returns:
            Generated response or reasoning trace
        """
        strategy = strategy or self._default_strategy
        strategy_instance = self.get_strategy(strategy)

        logger.info(f"Executing strategy: {strategy}")
        return await strategy_instance.execute(input_text, **kwargs)

    def create_template(
        self,
        template: str,
        system_prompt: Optional[str] = None,
        examples: Optional[List[FewShotExample]] = None,
    ) -> PromptTemplate:
        """Create a prompt template."""
        return PromptTemplate(
            template=template,
            system_prompt=system_prompt,
            examples=examples or [],
        )

    def create_few_shot_strategy(
        self,
        examples: List[FewShotExample],
        **kwargs: Any,
    ) -> FewShotStrategy:
        """Create a few-shot strategy with examples."""
        return FewShotStrategy(self.client, examples, **kwargs)

    def create_react_strategy(
        self,
        tools: Dict[str, Callable[[str], str]],
        **kwargs: Any,
    ) -> ReActStrategy:
        """Create a ReAct strategy with tools."""
        return ReActStrategy(self.client, tools, **kwargs)


# Utility functions

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.

    Uses simple heuristic based on model.
    For production, use tiktoken library.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: ~4 chars per token
        return len(text) // 4


def optimize_prompt_length(
    prompt: str,
    max_tokens: int,
    model: str = "gpt-4",
) -> str:
    """
    Truncate prompt to fit within token limit.

    Preserves beginning and end of prompt.
    """
    current_tokens = count_tokens(prompt, model)

    if current_tokens <= max_tokens:
        return prompt

    # Calculate truncation point
    ratio = max_tokens / current_tokens
    truncate_length = int(len(prompt) * ratio * 0.95)  # 5% buffer

    # Keep beginning and end
    split_point = truncate_length // 2
    return f"{prompt[:split_point]}\n...[truncated]...\n{prompt[-split_length:]}"


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from response text."""
    # Try to find JSON in text
    json_pattern = re.compile(r"(\{[^{}]*\}|\[[^\[\]]*\])", re.DOTALL)
    matches = json_pattern.findall(text)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try parsing entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

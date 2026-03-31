"""
Orchestration Module - AI-Mastery-2026

This module provides LangChain-style orchestration patterns for building
complex AI applications with chains, agents, and tools.

Key Components:
- Chain: Base class for processing pipelines
- SequentialChain: Run steps in sequence
- ParallelChain: Run steps in parallel
- Agent: Autonomous reasoning agent (ReAct pattern)
- Tool: Base class for agent tools
- Memory: Conversation and context memory

Author: AI-Mastery-2026
License: MIT
"""

import numpy as np
from typing import (
    List, Dict, Any, Optional, Union, Callable, Awaitable,
    TypeVar, Generic
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import logging
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OrchestrationConfig:
    """
    Configuration for orchestration components.
    
    Attributes:
        max_iterations: Maximum agent reasoning iterations
        timeout_seconds: Timeout for chain execution
        retry_attempts: Number of retry attempts on failure
        verbose: Enable verbose logging
        memory_size: Maximum memory entries to keep
    """
    max_iterations: int = 10
    timeout_seconds: float = 60.0
    retry_attempts: int = 3
    verbose: bool = False
    memory_size: int = 100


# =============================================================================
# Message Types
# =============================================================================

class MessageRole(Enum):
    """Role of a message in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """
    Represents a message in a conversation.
    
    Attributes:
        role: Role of the message sender
        content: Message content
        metadata: Additional metadata
        timestamp: When the message was created
    """
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# Memory Management
# =============================================================================

class Memory(ABC):
    """
    Abstract base class for memory management.
    
    Memory stores context and history that can be used
    by chains and agents for contextual reasoning.
    """
    
    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        """Add item to memory."""
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get item from memory."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        pass
    
    @abstractmethod
    def to_context(self) -> str:
        """Convert memory to context string."""
        pass


class ConversationMemory(Memory):
    """
    Memory for storing conversation history.
    
    Maintains a sliding window of recent messages that can be
    included in prompts for context-aware responses.
    
    Example:
        >>> memory = ConversationMemory(max_messages=10)
        >>> memory.add_message(Message(role=MessageRole.USER, content="Hello"))
        >>> memory.add_message(Message(role=MessageRole.ASSISTANT, content="Hi there!"))
        >>> context = memory.to_context()
    
    Attributes:
        max_messages: Maximum number of messages to keep
    """
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to retain
        """
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self._variables: Dict[str, Any] = {}
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to conversation history.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        
        # Trim to max size
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add(self, key: str, value: Any) -> None:
        """Add a variable to memory."""
        self._variables[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """Get a variable from memory."""
        return self._variables.get(key)
    
    def clear(self) -> None:
        """Clear all messages and variables."""
        self.messages.clear()
        self._variables.clear()
    
    def to_context(self) -> str:
        """
        Convert conversation history to a context string.
        
        Returns:
            Formatted conversation history
        """
        lines = []
        for msg in self.messages:
            role = msg.role.value.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def get_last_n(self, n: int) -> List[Message]:
        """Get the last n messages."""
        return self.messages[-n:]
    
    @property
    def summary(self) -> str:
        """Get a brief summary of the conversation."""
        if not self.messages:
            return "No conversation history."
        return f"Conversation with {len(self.messages)} messages."


class BufferMemory(Memory):
    """
    Simple key-value buffer memory.
    
    Stores arbitrary key-value pairs with optional
    maximum size limit.
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize buffer memory."""
        self.max_size = max_size
        self._buffer: Dict[str, Any] = {}
    
    def add(self, key: str, value: Any) -> None:
        """Add item to buffer."""
        if len(self._buffer) >= self.max_size:
            # Remove oldest item
            oldest = next(iter(self._buffer))
            del self._buffer[oldest]
        self._buffer[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from buffer."""
        return self._buffer.get(key)
    
    def clear(self) -> None:
        """Clear buffer."""
        self._buffer.clear()
    
    def to_context(self) -> str:
        """Convert buffer to context string."""
        return json.dumps(self._buffer, indent=2)


# =============================================================================
# Tool System
# =============================================================================

@dataclass
class ToolResult:
    """
    Result from a tool execution.
    
    Attributes:
        output: The tool's output
        success: Whether execution was successful
        error: Error message if failed
    """
    output: Any
    success: bool = True
    error: Optional[str] = None


class Tool(ABC):
    """
    Abstract base class for agent tools.
    
    Tools extend agent capabilities by providing
    specialized functions for specific tasks.
    
    Example:
        >>> class SearchTool(Tool):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="search",
        ...             description="Search the web for information"
        ...         )
        ...     
        ...     def run(self, query: str) -> ToolResult:
        ...         # Implement search logic
        ...         return ToolResult(output="Search results...")
    
    Attributes:
        name: Unique tool name
        description: What the tool does (for agent reasoning)
        parameters: Expected parameters with descriptions
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, str]] = None
    ):
        """
        Initialize tool.
        
        Args:
            name: Tool name (used in agent prompts)
            description: What the tool does
            parameters: Dict of parameter name -> description
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with output or error
        """
        pass
    
    def to_schema(self) -> Dict[str, Any]:
        """
        Convert tool to JSON schema for prompting.
        
        Returns:
            Tool schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides tool discovery and lookup functionality
    for agents to find and use tools.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(SearchTool())
        >>> registry.register(CalculatorTool())
        >>> tools = registry.list_tools()
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def to_prompt(self) -> str:
        """
        Generate tool descriptions for agent prompts.
        
        Returns:
            Formatted tool descriptions
        """
        lines = ["Available Tools:"]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                for param, desc in tool.parameters.items():
                    lines.append(f"    - {param}: {desc}")
        return "\n".join(lines)


# Built-in Tools

class CalculatorTool(Tool):
    """
    Built-in calculator tool for mathematical operations.
    
    Example:
        >>> calc = CalculatorTool()
        >>> result = calc.run(expression="2 + 2 * 3")
        >>> print(result.output)  # 8
    """
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={"expression": "Mathematical expression to evaluate"}
        )
    
    def run(self, expression: str = "", **kwargs) -> ToolResult:
        """Evaluate a mathematical expression."""
        try:
            # Safe evaluation (only math operations)
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return ToolResult(
                    output=None,
                    success=False,
                    error="Invalid characters in expression"
                )
            result = eval(expression)
            return ToolResult(output=result, success=True)
        except Exception as e:
            return ToolResult(output=None, success=False, error=str(e))


class SearchTool(Tool):
    """
    Built-in search tool (mock implementation).
    
    In production, connect to a real search API.
    """
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for information on a topic",
            parameters={"query": "Search query"}
        )
    
    def run(self, query: str = "", **kwargs) -> ToolResult:
        """Execute search (mock implementation)."""
        # Mock search results
        return ToolResult(
            output=f"Search results for '{query}': [Mock results - implement real search]",
            success=True
        )


# =============================================================================
# Chain Base Classes
# =============================================================================

class Chain(ABC):
    """
    Abstract base class for processing chains.
    
    Chains are composable units of processing that can be
    connected to form complex pipelines.
    
    Key concepts:
        - Input/Output: Chains transform inputs to outputs
        - Composition: Chains can contain other chains
        - Memory: Chains can use shared memory for context
    
    Example:
        >>> chain = SequentialChain([step1, step2, step3])
        >>> result = chain.run(input="Hello")
    """
    
    def __init__(self, name: str = "chain"):
        """
        Initialize chain.
        
        Args:
            name: Chain name for logging
        """
        self.name = name
        self.memory: Optional[Memory] = None
        self.config = OrchestrationConfig()
    
    def with_memory(self, memory: Memory) -> "Chain":
        """
        Attach memory to this chain.
        
        Args:
            memory: Memory instance to attach
            
        Returns:
            Self for chaining
        """
        self.memory = memory
        return self
    
    @abstractmethod
    def run(self, input: Any, **kwargs) -> Any:
        """
        Execute the chain.
        
        Args:
            input: Input to the chain
            **kwargs: Additional arguments
            
        Returns:
            Chain output
        """
        pass
    
    async def arun(self, input: Any, **kwargs) -> Any:
        """
        Async version of run.
        
        By default, wraps sync run. Override for true async.
        """
        return self.run(input, **kwargs)


class SequentialChain(Chain):
    """
    Chain that runs steps in sequence.
    
    Output of each step becomes input to the next step.
    
    Example:
        >>> chain = SequentialChain([
        ...     LambdaChain(lambda x: x.upper()),
        ...     LambdaChain(lambda x: f"Result: {x}")
        ... ])
        >>> chain.run("hello")  # "Result: HELLO"
    """
    
    def __init__(
        self,
        steps: List[Chain],
        name: str = "sequential"
    ):
        """
        Initialize sequential chain.
        
        Args:
            steps: List of chains to run in order
            name: Chain name
        """
        super().__init__(name=name)
        self.steps = steps
    
    def run(self, input: Any, **kwargs) -> Any:
        """
        Run all steps sequentially.
        
        Args:
            input: Initial input
            **kwargs: Additional arguments passed to all steps
            
        Returns:
            Output of the last step
        """
        current = input
        
        for i, step in enumerate(self.steps):
            if self.config.verbose:
                logger.info(f"[{self.name}] Running step {i+1}/{len(self.steps)}: {step.name}")
            
            # Pass memory to step
            if self.memory:
                step.memory = self.memory
            
            current = step.run(current, **kwargs)
        
        return current
    
    async def arun(self, input: Any, **kwargs) -> Any:
        """Async sequential execution."""
        current = input
        for step in self.steps:
            current = await step.arun(current, **kwargs)
        return current


class ParallelChain(Chain):
    """
    Chain that runs steps in parallel.
    
    All steps receive the same input, outputs are collected.
    
    Example:
        >>> chain = ParallelChain([
        ...     LambdaChain(lambda x: x.upper()),
        ...     LambdaChain(lambda x: len(x))
        ... ])
        >>> chain.run("hello")  # ["HELLO", 5]
    """
    
    def __init__(
        self,
        steps: List[Chain],
        name: str = "parallel"
    ):
        """
        Initialize parallel chain.
        
        Args:
            steps: List of chains to run in parallel
            name: Chain name
        """
        super().__init__(name=name)
        self.steps = steps
    
    def run(self, input: Any, **kwargs) -> List[Any]:
        """
        Run all steps in parallel (sync version runs sequentially).
        
        Args:
            input: Input passed to all steps
            **kwargs: Additional arguments
            
        Returns:
            List of outputs from all steps
        """
        results = []
        for step in self.steps:
            if self.memory:
                step.memory = self.memory
            results.append(step.run(input, **kwargs))
        return results
    
    async def arun(self, input: Any, **kwargs) -> List[Any]:
        """Async parallel execution."""
        tasks = [step.arun(input, **kwargs) for step in self.steps]
        return await asyncio.gather(*tasks)


class LambdaChain(Chain):
    """
    Simple chain that wraps a function.
    
    Example:
        >>> chain = LambdaChain(lambda x: x * 2)
        >>> chain.run(5)  # 10
    """
    
    def __init__(
        self,
        func: Callable[[Any], Any],
        name: str = "lambda"
    ):
        """
        Initialize lambda chain.
        
        Args:
            func: Function to execute
            name: Chain name
        """
        super().__init__(name=name)
        self.func = func
    
    def run(self, input: Any, **kwargs) -> Any:
        """Execute the function."""
        return self.func(input)


class ConditionalChain(Chain):
    """
    Chain that conditionally executes different branches.
    
    Example:
        >>> chain = ConditionalChain(
        ...     condition=lambda x: x > 0,
        ...     if_true=positive_chain,
        ...     if_false=negative_chain
        ... )
    """
    
    def __init__(
        self,
        condition: Callable[[Any], bool],
        if_true: Chain,
        if_false: Chain,
        name: str = "conditional"
    ):
        """
        Initialize conditional chain.
        
        Args:
            condition: Function that returns True/False
            if_true: Chain to run if condition is True
            if_false: Chain to run if condition is False
            name: Chain name
        """
        super().__init__(name=name)
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
    
    def run(self, input: Any, **kwargs) -> Any:
        """Execute appropriate branch based on condition."""
        if self.condition(input):
            return self.if_true.run(input, **kwargs)
        else:
            return self.if_false.run(input, **kwargs)


# =============================================================================
# Agent System
# =============================================================================

class AgentState(Enum):
    """Agent execution state."""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"


@dataclass
class AgentStep:
    """
    Represents one step in agent reasoning.
    
    Attributes:
        thought: Agent's reasoning
        action: Tool to use (or None if done)
        action_input: Input to the tool
        observation: Result from the tool
    """
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class Agent(ABC):
    """
    Abstract base class for autonomous agents.
    
    Agents can reason, plan, and use tools to accomplish tasks.
    They implement a reasoning loop: Think → Act → Observe → Repeat.
    
    Attributes:
        tools: Tool registry for available tools
        memory: Agent memory
    """
    
    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None,
        config: Optional[OrchestrationConfig] = None
    ):
        """
        Initialize agent.
        
        Args:
            tools: Tool registry
            memory: Agent memory
            config: Agent configuration
        """
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory()
        self.config = config or OrchestrationConfig()
        self.steps: List[AgentStep] = []
    
    @abstractmethod
    def think(self, input: str) -> AgentStep:
        """
        Generate thought and decide on action.
        
        Args:
            input: Current input/context
            
        Returns:
            AgentStep with thought and action
        """
        pass
    
    def act(self, action: str, action_input: str) -> str:
        """
        Execute an action using a tool.
        
        Args:
            action: Tool name
            action_input: Input to the tool
            
        Returns:
            Tool output as string
        """
        tool = self.tools.get(action)
        if tool is None:
            return f"Error: Unknown tool '{action}'"
        
        result = tool.run(**{"input": action_input})
        if result.success:
            return str(result.output)
        else:
            return f"Error: {result.error}"
    
    def run(self, input: str) -> str:
        """
        Run the agent on input.
        
        Args:
            input: Task or question
            
        Returns:
            Final answer/response
        """
        self.steps = []
        current_input = input
        
        for i in range(self.config.max_iterations):
            # Think
            step = self.think(current_input)
            self.steps.append(step)
            
            # Check if done
            if step.action is None or step.action.lower() == "finish":
                return step.thought
            
            # Act
            observation = self.act(step.action, step.action_input or "")
            step.observation = observation
            
            # Update input for next iteration
            current_input = f"{current_input}\n\nObservation: {observation}"
        
        return "Max iterations reached. Final thoughts: " + self.steps[-1].thought


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) agent implementation.
    
    Uses the ReAct prompting pattern for interleaved reasoning
    and acting. The agent alternates between:
        1. Thought: Reason about the current state
        2. Action: Decide what tool to use
        3. Observation: See the result
    
    Example:
        >>> agent = ReActAgent()
        >>> agent.tools.register(SearchTool())
        >>> agent.tools.register(CalculatorTool())
        >>> answer = agent.run("What is the population of France times 2?")
    
    Reference:
        Yao et al. (2022) - ReAct: Synergizing Reasoning and Acting in Language Models
    """
    
    REACT_PROMPT = """You are a helpful AI assistant that can use tools to answer questions.

{tools}

Use the following format:

Thought: Think about what to do
Action: tool_name
Action Input: input to the tool
Observation: result from the tool
... (repeat Thought/Action/Observation as needed)
Thought: I have the answer
Action: finish
Action Input: final answer

Begin!

Question: {question}
{history}
Thought:"""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None,
        config: Optional[OrchestrationConfig] = None
    ):
        """
        Initialize ReAct agent.
        
        Args:
            llm: Language model for reasoning (optional)
            tools: Tool registry
            memory: Agent memory
            config: Configuration
        """
        super().__init__(tools=tools, memory=memory, config=config)
        self.llm = llm
        self._model = None
    
    def _load_model(self):
        """Load language model for reasoning."""
        if self._model is None and self.llm is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._model = "loaded"  # Placeholder
            except ImportError:
                self._model = "fallback"
    
    def _generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        self._load_model()
        
        if self._model == "fallback":
            # Simple fallback: extract first tool name and return mock response
            tools = self.tools.list_tools()
            if tools:
                return f"I should use the {tools[0].name} tool.\nAction: {tools[0].name}\nAction Input: example"
            return "I have the answer.\nAction: finish\nAction Input: Unable to process"
        
        # Use actual LLM if available
        if self.llm:
            return self.llm(prompt)
        
        return "Action: finish\nAction Input: LLM not configured"
    
    def think(self, input: str) -> AgentStep:
        """
        Generate thought and action using ReAct pattern.
        
        Args:
            input: Current question/context
            
        Returns:
            AgentStep with reasoning
        """
        # Build history from previous steps
        history = ""
        for step in self.steps:
            history += f"\nThought: {step.thought}"
            if step.action:
                history += f"\nAction: {step.action}"
                history += f"\nAction Input: {step.action_input}"
            if step.observation:
                history += f"\nObservation: {step.observation}"
        
        # Create prompt
        prompt = self.REACT_PROMPT.format(
            tools=self.tools.to_prompt(),
            question=input,
            history=history
        )
        
        # Generate response
        response = self._generate(prompt)
        
        # Parse response
        thought = ""
        action = None
        action_input = None
        
        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()
        
        return AgentStep(
            thought=thought or response,
            action=action,
            action_input=action_input
        )


# =============================================================================
# Utility Functions
# =============================================================================

def create_simple_chain(*funcs: Callable) -> SequentialChain:
    """
    Create a sequential chain from functions.
    
    Args:
        *funcs: Functions to chain together
        
    Returns:
        SequentialChain
        
    Example:
        >>> chain = create_simple_chain(
        ...     lambda x: x.upper(),
        ...     lambda x: f"[{x}]"
        ... )
        >>> chain.run("hello")  # "[HELLO]"
    """
    steps = [LambdaChain(f, name=f.__name__) for f in funcs]
    return SequentialChain(steps)


def run_with_retry(
    chain: Chain,
    input: Any,
    max_attempts: int = 3
) -> Any:
    """
    Run a chain with retry on failure.
    
    Args:
        chain: Chain to run
        input: Input to the chain
        max_attempts: Maximum retry attempts
        
    Returns:
        Chain output
        
    Raises:
        Exception: If all attempts fail
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return chain.run(input)
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
    
    raise last_error


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Orchestration Module Demo")
    print("=" * 60)
    
    # 1. Simple Chain
    print("\n1. Sequential Chain")
    print("-" * 40)
    
    chain = SequentialChain([
        LambdaChain(lambda x: x.upper(), name="uppercase"),
        LambdaChain(lambda x: f"[{x}]", name="bracket"),
        LambdaChain(lambda x: x * 2, name="double")
    ])
    
    result = chain.run("hello")
    print(f"  Input: 'hello'")
    print(f"  Output: '{result}'")
    
    # 2. Parallel Chain
    print("\n2. Parallel Chain")
    print("-" * 40)
    
    parallel = ParallelChain([
        LambdaChain(lambda x: x.upper()),
        LambdaChain(lambda x: len(x)),
        LambdaChain(lambda x: x[::-1])
    ])
    
    results = parallel.run("hello")
    print(f"  Input: 'hello'")
    print(f"  Outputs: {results}")
    
    # 3. Tools
    print("\n3. Tool Registry")
    print("-" * 40)
    
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    
    calc = registry.get("calculator")
    result = calc.run(expression="2 + 3 * 4")
    print(f"  Calculator: 2 + 3 * 4 = {result.output}")
    
    # 4. Memory
    print("\n4. Conversation Memory")
    print("-" * 40)
    
    memory = ConversationMemory()
    memory.add_message(Message(role=MessageRole.USER, content="What is AI?"))
    memory.add_message(Message(role=MessageRole.ASSISTANT, content="AI is artificial intelligence."))
    
    print(f"  Context:\n{memory.to_context()}")
    
    # 5. Agent
    print("\n5. ReAct Agent")
    print("-" * 40)
    
    agent = ReActAgent(tools=registry)
    print(f"  Agent initialized with tools: {[t.name for t in registry.list_tools()]}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")

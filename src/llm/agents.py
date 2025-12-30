"""
LLM Agent Design Patterns Module
================================
Agent architectures, tool integration, and orchestration patterns
following the White-Box Approach.

Mathematical Foundations:
- ReAct (Reason + Act) framework
- Chain-of-Thought prompting
- Function calling and tool use
- Multi-agent coordination

Author: AI-Mastery-2026
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


# ============================================================
# CORE DATA STRUCTURES
# ============================================================

class AgentRole(Enum):
    """Predefined agent roles for multi-agent systems."""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    PLANNER = "planner"


@dataclass
class Message:
    """
    Message structure for agent communication.
    
    Follows OpenAI chat format for compatibility.
    """
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class Tool:
    """
    Tool definition for agent function calling.
    
    Based on OpenAI function calling schema.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        try:
            result = self.function(**kwargs)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return json.dumps({"error": str(e)})


@dataclass
class AgentState:
    """
    Tracks agent's internal state.
    
    Used for persistence, debugging, and resumption.
    """
    messages: List[Message] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    
    def add_message(self, message: Message):
        """Add message to history."""
        self.messages.append(message)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get message history as list of dicts."""
        return [m.to_dict() for m in self.messages]


# ============================================================
# MEMORY SYSTEMS
# ============================================================

class Memory(ABC):
    """Abstract base class for agent memory."""
    
    @abstractmethod
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add content to memory."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant memories."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all memories."""
        pass


class ConversationMemory(Memory):
    """
    Simple conversation buffer memory.
    
    Stores recent conversation turns.
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, Any]] = []
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a turn to conversation history."""
        self.history.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        
        # Trim to max turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve last k turns (ignores query for simple buffer)."""
        return [h["content"] for h in self.history[-k:]]
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
    
    def get_formatted_history(self) -> str:
        """Get history as formatted string."""
        return "\n".join([
            f"Turn {i+1}: {h['content']}" 
            for i, h in enumerate(self.history)
        ])


class SummaryMemory(Memory):
    """
    Summary-based memory for long conversations.
    
    Periodically summarizes old conversations to save context.
    """
    
    def __init__(self, summarize_fn: Callable[[str], str], 
                 buffer_size: int = 5, summary_size: int = 500):
        """
        Args:
            summarize_fn: Function to summarize text
            buffer_size: Number of turns before summarizing
            summary_size: Max characters for summary
        """
        self.summarize_fn = summarize_fn
        self.buffer_size = buffer_size
        self.summary_size = summary_size
        self.buffer: List[str] = []
        self.summaries: List[str] = []
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add content, summarizing when buffer is full."""
        self.buffer.append(content)
        
        if len(self.buffer) >= self.buffer_size:
            # Summarize buffer
            buffer_text = "\n".join(self.buffer)
            summary = self.summarize_fn(buffer_text)
            self.summaries.append(summary[:self.summary_size])
            self.buffer = []
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve summaries and recent buffer content."""
        results = self.summaries[-k:] + self.buffer
        return results[-k:]
    
    def clear(self):
        """Clear all memory."""
        self.buffer = []
        self.summaries = []


class VectorMemory(Memory):
    """
    Vector-based semantic memory using embeddings.
    
    Uses cosine similarity for retrieval.
    """
    
    def __init__(self, embed_fn: Callable[[str], List[float]]):
        """
        Args:
            embed_fn: Function to embed text to vector
        """
        self.embed_fn = embed_fn
        self.memories: List[Dict[str, Any]] = []
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add content with its embedding."""
        embedding = self.embed_fn(content)
        self.memories.append({
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {}
        })
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve most similar memories."""
        if not self.memories:
            return []
        
        import numpy as np
        
        query_embedding = np.array(self.embed_fn(query))
        
        # Compute similarities
        similarities = []
        for mem in self.memories:
            mem_embedding = np.array(mem["embedding"])
            sim = np.dot(query_embedding, mem_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(mem_embedding) + 1e-10
            )
            similarities.append((sim, mem["content"]))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in similarities[:k]]
    
    def clear(self):
        """Clear all memories."""
        self.memories = []


# ============================================================
# BASE AGENT CLASS
# ============================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common interface for different agent patterns.
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_fn: Callable[[List[Dict[str, Any]]], str],
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        max_iterations: int = 10
    ):
        """
        Args:
            name: Agent name for identification
            system_prompt: System instructions
            llm_fn: Function to call LLM (takes messages, returns response)
            tools: List of available tools
            memory: Agent memory system
            max_iterations: Maximum reasoning iterations
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm_fn = llm_fn
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory
        self.max_iterations = max_iterations
        self.state = AgentState()
    
    def _get_system_message(self) -> Message:
        """Get system message with agent instructions."""
        return Message(role="system", content=self.system_prompt)
    
    def _call_llm(self, messages: List[Message]) -> str:
        """Call LLM with messages."""
        msg_dicts = [m.to_dict() for m in messages]
        return self.llm_fn(msg_dicts)
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response.
        
        Supports both JSON format and custom markup.
        """
        tool_calls = []
        
        # Try to find JSON tool calls
        try:
            # Look for ```json blocks
            json_pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                parsed = json.loads(match)
                if "tool" in parsed or "function" in parsed:
                    tool_calls.append(parsed)
        except json.JSONDecodeError:
            pass
        
        # Look for action/input pattern (ReAct style)
        action_pattern = r'Action:\s*(\w+)\nAction Input:\s*(.+?)(?=\n(?:Action:|Observation:|$))'
        matches = re.findall(action_pattern, response, re.DOTALL)
        
        for action, action_input in matches:
            try:
                args = json.loads(action_input.strip())
            except json.JSONDecodeError:
                args = {"input": action_input.strip()}
            
            tool_calls.append({
                "tool": action,
                "args": args
            })
        
        return tool_calls
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return result."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.tools[tool_name]
        return tool.execute(**args)
    
    @abstractmethod
    def run(self, user_input: str) -> str:
        """
        Run the agent on user input.
        
        Must be implemented by subclasses.
        """
        pass


# ============================================================
# REACT AGENT
# ============================================================

class ReActAgent(BaseAgent):
    """
    ReAct (Reason + Act) Agent.
    
    Implementation of the ReAct framework that interleaves reasoning
    traces with actions for improved decision making.
    
    Pattern:
        Thought -> Action -> Observation -> ... -> Final Answer
    
    Reference:
        Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
    
    Example:
        >>> agent = ReActAgent(
        ...     name="assistant",
        ...     system_prompt="You are a helpful assistant.",
        ...     llm_fn=call_llm,
        ...     tools=[search_tool, calculator_tool]
        ... )
        >>> result = agent.run("What is the population of France?")
    """
    
    REACT_PROMPT_TEMPLATE = """You are an AI assistant that follows the ReAct pattern.

Available Tools:
{tool_descriptions}

To use a tool, respond in this format:
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [JSON arguments for the tool]

After receiving an Observation, continue reasoning until you have a final answer.

When you have the final answer, respond:
Thought: [final reasoning]
Final Answer: [your answer to the user]

Remember:
- Always think before acting
- Use tools when you need external information
- Provide clear, helpful final answers
"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enhance_system_prompt()
    
    def _enhance_system_prompt(self):
        """Add ReAct instructions to system prompt."""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        react_prompt = self.REACT_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions or "No tools available."
        )
        
        self.system_prompt = f"{self.system_prompt}\n\n{react_prompt}"
    
    def run(self, user_input: str) -> str:
        """
        Run ReAct loop on user input.
        
        Returns:
            Final answer from the agent
        """
        # Initialize conversation
        messages = [
            self._get_system_message(),
            Message(role="user", content=user_input)
        ]
        
        if self.memory:
            # Add relevant memories
            memories = self.memory.retrieve(user_input)
            if memories:
                memory_content = "Relevant context:\n" + "\n".join(memories)
                messages.insert(1, Message(role="system", content=memory_content))
        
        for iteration in range(self.max_iterations):
            self.state.step_count = iteration + 1
            
            # Get LLM response
            response = self._call_llm(messages)
            messages.append(Message(role="assistant", content=response))
            
            logger.debug(f"Iteration {iteration + 1}: {response[:200]}...")
            
            # Check for final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                
                # Save to memory
                if self.memory:
                    self.memory.add(f"Q: {user_input}\nA: {final_answer}")
                
                return final_answer
            
            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls and no final answer - prompt for continuation
                messages.append(Message(
                    role="user",
                    content="Please continue your reasoning or provide a final answer."
                ))
                continue
            
            # Execute tools and add observations
            for tc in tool_calls:
                tool_name = tc.get("tool") or tc.get("function", {}).get("name")
                args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {"input": args}
                
                result = self._execute_tool(tool_name, args)
                
                observation = f"Observation: {result}"
                messages.append(Message(role="user", content=observation))
                
                self.state.tool_results.append({
                    "tool": tool_name,
                    "args": args,
                    "result": result
                })
        
        return "I was unable to complete the task within the iteration limit."


# ============================================================
# FUNCTION CALLING AGENT
# ============================================================

class FunctionCallingAgent(BaseAgent):
    """
    Agent using OpenAI-style function calling.
    
    Cleaner interface for structured tool use without
    explicit ReAct prompting.
    
    Example:
        >>> agent = FunctionCallingAgent(
        ...     name="assistant",
        ...     system_prompt="You are a helpful assistant.",
        ...     llm_fn=call_llm_with_tools,
        ...     tools=[search_tool]
        ... )
    """
    
    def __init__(self, *args, parse_fn: Optional[Callable] = None, **kwargs):
        """
        Args:
            parse_fn: Optional custom function to parse tool calls from response
        """
        super().__init__(*args, **kwargs)
        self.parse_fn = parse_fn
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for API call."""
        return [tool.to_schema() for tool in self.tools.values()]
    
    def run(self, user_input: str) -> str:
        """Run the function calling agent."""
        messages = [
            self._get_system_message(),
            Message(role="user", content=user_input)
        ]
        
        for iteration in range(self.max_iterations):
            # Call LLM (should support tool use)
            response = self._call_llm(messages)
            
            # Parse response for tool calls
            if self.parse_fn:
                tool_calls = self.parse_fn(response)
            else:
                tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls - this is the final response
                return response
            
            # Add assistant response
            messages.append(Message(role="assistant", content=response))
            
            # Execute tools
            for tc in tool_calls:
                tool_name = tc.get("tool") or tc.get("name")
                args = tc.get("args") or tc.get("arguments", {})
                
                result = self._execute_tool(tool_name, args)
                
                # Add tool result
                messages.append(Message(
                    role="tool",
                    content=result,
                    name=tool_name
                ))
        
        return "Maximum iterations reached."


# ============================================================
# CHAIN OF THOUGHT AGENT
# ============================================================

class ChainOfThoughtAgent(BaseAgent):
    """
    Chain-of-Thought (CoT) prompting agent.
    
    Encourages step-by-step reasoning for complex problems.
    
    Reference:
        Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    """
    
    COT_PROMPT = """Let's approach this step-by-step:

1. First, I'll analyze the problem
2. Then, I'll break it down into sub-problems
3. I'll solve each sub-problem
4. Finally, I'll combine the results

Let me think through this carefully..."""
    
    def run(self, user_input: str) -> str:
        """Run with chain-of-thought prompting."""
        # Enhance input with CoT prompt
        enhanced_input = f"{user_input}\n\n{self.COT_PROMPT}"
        
        messages = [
            self._get_system_message(),
            Message(role="user", content=enhanced_input)
        ]
        
        response = self._call_llm(messages)
        
        # Save to memory
        if self.memory:
            self.memory.add(f"Q: {user_input}\nA: {response}")
        
        return response


# ============================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================

class AgentOrchestrator:
    """
    Orchestrates multiple agents for complex tasks.
    
    Patterns supported:
        - Sequential: Agents run in sequence, passing results
        - Parallel: Agents run concurrently on same input
        - Hierarchical: Coordinator agent delegates to specialists
    
    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> orchestrator.add_agent(researcher, role=AgentRole.RESEARCHER)
        >>> orchestrator.add_agent(writer, role=AgentRole.EXECUTOR)
        >>> result = orchestrator.run_sequential("Research and summarize AI trends")
    """
    
    def __init__(self):
        self.agents: Dict[str, Tuple[BaseAgent, AgentRole]] = {}
    
    def add_agent(self, agent: BaseAgent, role: AgentRole = AgentRole.EXECUTOR):
        """Add an agent to the orchestrator."""
        self.agents[agent.name] = (agent, role)
    
    def run_sequential(self, task: str) -> str:
        """
        Run agents sequentially, passing output to next agent.
        """
        current_input = task
        results = []
        
        for name, (agent, role) in self.agents.items():
            logger.info(f"Running agent: {name} (role: {role.value})")
            
            # Format input based on role
            if results:
                agent_input = f"Previous output:\n{results[-1]}\n\nOriginal task: {task}"
            else:
                agent_input = current_input
            
            result = agent.run(agent_input)
            results.append(result)
        
        return results[-1] if results else ""
    
    def run_parallel(self, task: str) -> Dict[str, str]:
        """
        Run all agents in parallel on same input.
        
        Returns dict mapping agent name to result.
        """
        results = {}
        
        # In production, use asyncio or threading
        for name, (agent, _) in self.agents.items():
            logger.info(f"Running agent: {name}")
            results[name] = agent.run(task)
        
        return results
    
    def run_hierarchical(self, task: str, coordinator_name: str) -> str:
        """
        Run with coordinator delegating to specialists.
        
        Coordinator decides which agents to use.
        """
        if coordinator_name not in self.agents:
            raise ValueError(f"Coordinator '{coordinator_name}' not found")
        
        coordinator, _ = self.agents[coordinator_name]
        specialists = {
            name: agent for name, (agent, role) in self.agents.items()
            if name != coordinator_name
        }
        
        # Create delegation prompt
        specialist_list = "\n".join([
            f"- {name}: {agent.system_prompt[:100]}..."
            for name, agent in specialists.items()
        ])
        
        delegation_prompt = f"""Task: {task}

Available specialists:
{specialist_list}

Decide which specialist(s) to delegate to and what subtasks to assign.
Format your response as:
DELEGATE: [agent_name]
SUBTASK: [subtask description]
"""
        
        # Get coordinator's plan
        plan = coordinator.run(delegation_prompt)
        
        # Parse and execute delegations
        results = []
        delegation_pattern = r'DELEGATE:\s*(\w+)\nSUBTASK:\s*(.+?)(?=DELEGATE:|$)'
        delegations = re.findall(delegation_pattern, plan, re.DOTALL)
        
        for agent_name, subtask in delegations:
            if agent_name in specialists:
                result = specialists[agent_name].run(subtask.strip())
                results.append(f"{agent_name}: {result}")
        
        # Final synthesis by coordinator
        synthesis_prompt = f"""Original task: {task}

Specialist results:
{chr(10).join(results)}

Please synthesize these results into a final response."""
        
        return coordinator.run(synthesis_prompt)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_tool(
    name: str,
    description: str,
    function: Callable,
    parameters: Optional[Dict[str, Any]] = None
) -> Tool:
    """
    Helper to create a tool from a function.
    
    Args:
        name: Tool name
        description: What the tool does
        function: The function to execute
        parameters: JSON schema for parameters (auto-generated if None)
    
    Returns:
        Tool instance
    """
    if parameters is None:
        # Auto-generate basic schema from function signature
        import inspect
        sig = inspect.signature(function)
        
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            parameters["properties"][param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
    
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        function=function
    )


# ============================================================
# EXAMPLE TOOLS
# ============================================================

def calculator(expression: str) -> Dict[str, Any]:
    """
    Simple calculator tool.
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Result dictionary
    """
    try:
        # Safe evaluation (production should use proper parser)
        allowed = set('0123456789+-*/().^ ')
        if all(c in allowed for c in expression):
            result = eval(expression.replace('^', '**'))
            return {"result": result, "expression": expression}
        else:
            return {"error": "Invalid characters in expression"}
    except Exception as e:
        return {"error": str(e)}


def web_search(query: str) -> Dict[str, Any]:
    """
    Simulated web search tool.
    
    In production, integrate with real search API.
    """
    # Placeholder - integrate with actual search API
    return {
        "query": query,
        "results": [
            {"title": f"Result for: {query}", "snippet": "Simulated search result..."}
        ],
        "note": "This is a placeholder. Integrate with real search API."
    }


# Pre-built tools
CALCULATOR_TOOL = create_tool(
    name="calculator",
    description="Perform mathematical calculations. Use for any arithmetic.",
    function=calculator,
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)

SEARCH_TOOL = create_tool(
    name="web_search",
    description="Search the web for information. Use when you need current or external information.",
    function=web_search,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Data structures
    'Message', 'Tool', 'AgentState', 'AgentRole',
    # Memory
    'Memory', 'ConversationMemory', 'SummaryMemory', 'VectorMemory',
    # Agents
    'BaseAgent', 'ReActAgent', 'FunctionCallingAgent', 'ChainOfThoughtAgent',
    # Orchestration
    'AgentOrchestrator',
    # Utilities
    'create_tool', 'CALCULATOR_TOOL', 'SEARCH_TOOL',
]

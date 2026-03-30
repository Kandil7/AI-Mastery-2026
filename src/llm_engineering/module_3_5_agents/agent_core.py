"""
Agent Core Module

Production-ready agent implementations:
- Base agent class
- ReAct agent (Reasoning + Acting)
- Planning agent
- Reflective agent

Features:
- State management
- Action execution
- Step tracking
- Memory integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentAction:
    """An action to be executed by the agent."""

    name: str
    input: Dict[str, Any]
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "tool_call_id": self.tool_call_id,
        }


@dataclass
class AgentObservation:
    """Observation from action execution."""

    action_name: str
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class AgentStep:
    """A single step in agent execution."""

    step_number: int
    thought: str
    action: Optional[AgentAction] = None
    observation: Optional[AgentObservation] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action.to_dict() if self.action else None,
            "observation": self.observation.to_dict() if self.observation else None,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """State of the agent during execution."""

    task: str
    status: AgentStatus = AgentStatus.IDLE
    steps: List[AgentStep] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_step(self, step: AgentStep) -> None:
        """Add a step to the state."""
        self.steps.append(step)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class AgentResult:
    """Result of agent execution."""

    success: bool
    result: str
    state: AgentState
    total_steps: int
    total_time_ms: float
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "total_steps": self.total_steps,
            "total_time_ms": self.total_time_ms,
            "tokens_used": self.tokens_used,
        }


class Agent(ABC):
    """Abstract base class for agents."""

    def __init__(
        self,
        name: str,
        llm_client: Any,
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.llm_client = llm_client
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.verbose = verbose

        self._state: Optional[AgentState] = None
        self._stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_steps": 0,
        }

    @abstractmethod
    async def run(
        self,
        task: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent on a task."""
        pass

    @abstractmethod
    async def stream(
        self,
        task: str,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent execution steps."""
        pass

    def get_state(self) -> Optional[AgentState]:
        """Get current agent state."""
        return self._state

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self._stats,
            "avg_steps": (
                self._stats["total_steps"] / self._stats["total_runs"]
                if self._stats["total_runs"] > 0 else 0
            ),
        }

    def _create_state(self, task: str) -> AgentState:
        """Create new agent state."""
        self._state = AgentState(task=task)
        return self._state

    def _update_stats(self, success: bool, steps: int) -> None:
        """Update execution statistics."""
        self._stats["total_runs"] += 1
        if success:
            self._stats["successful_runs"] += 1
        else:
            self._stats["failed_runs"] += 1
        self._stats["total_steps"] += steps


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) agent.

    Interleaves reasoning with action execution.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that solves tasks by thinking step by step and using tools.

You have access to these tools:
{tools}

Use the following format:
Thought: Your reasoning about what to do
Action: The tool name
Action Input: JSON input to the tool
Observation: The tool's output
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: Your complete answer

Begin!"""

    def __init__(
        self,
        name: str = "react_agent",
        llm_client: Any = None,
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, llm_client, tools, max_iterations, **kwargs)

    async def run(
        self,
        task: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run ReAct agent."""
        start_time = time.time()
        state = self._create_state(task)
        state.status = AgentStatus.RUNNING

        history = []
        tools_description = self._format_tools()

        system_prompt = self.SYSTEM_PROMPT.format(tools=tools_description)

        if context:
            history.append({"role": "system", "content": f"Context: {context}"})

        history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": f"Task: {task}"})

        for iteration in range(self.max_iterations):
            # Get LLM response
            response = await self.llm_client.generate(
                messages=history,
                temperature=0.1,
            )

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            thought, action, action_input, is_final = self._parse_response(content)

            # Add step
            step = AgentStep(
                step_number=iteration + 1,
                thought=thought,
            )

            if is_final:
                state.result = action or content
                state.status = AgentStatus.COMPLETED
                state.add_step(step)

                total_time = (time.time() - start_time) * 1000
                self._update_stats(True, iteration + 1)

                return AgentResult(
                    success=True,
                    result=state.result,
                    state=state,
                    total_steps=iteration + 1,
                    total_time_ms=total_time,
                )

            # Execute action
            if action and action in self.tools:
                try:
                    observation = await self._execute_action(action, action_input)
                    step.action = AgentAction(name=action, input=action_input)
                    step.observation = AgentObservation(
                        action_name=action,
                        output=observation,
                    )

                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": f"Observation: {observation}"})
                except Exception as e:
                    step.observation = AgentObservation(
                        action_name=action,
                        output=None,
                        error=str(e),
                    )
                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": f"Error: {e}"})
            else:
                # Invalid action
                step.observation = AgentObservation(
                    action_name=action or "unknown",
                    output=None,
                    error=f"Unknown tool: {action}",
                )
                history.append({"role": "assistant", "content": content})
                history.append({"role": "user", "content": f"Error: Unknown tool '{action}'"})

            state.add_step(step)

        # Max iterations reached
        state.status = AgentStatus.FAILED
        state.error = f"Max iterations ({self.max_iterations}) reached"

        total_time = (time.time() - start_time) * 1000
        self._update_stats(False, self.max_iterations)

        return AgentResult(
            success=False,
            result=state.error,
            state=state,
            total_steps=self.max_iterations,
            total_time_ms=total_time,
        )

    async def stream(
        self,
        task: str,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream ReAct execution."""
        start_time = time.time()
        state = self._create_state(task)
        state.status = AgentStatus.RUNNING

        yield {"type": "start", "task": task, "timestamp": start_time}

        history = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(tools=self._format_tools())},
            {"role": "user", "content": f"Task: {task}"},
        ]

        for iteration in range(self.max_iterations):
            response = await self.llm_client.generate(messages=history, temperature=0.1)
            content = response.content if hasattr(response, 'content') else str(response)

            thought, action, action_input, is_final = self._parse_response(content)

            step_data = {
                "type": "step",
                "step_number": iteration + 1,
                "thought": thought,
                "timestamp": time.time(),
            }

            if is_final:
                step_data["final_answer"] = action or content
                state.status = AgentStatus.COMPLETED
                yield step_data
                yield {"type": "complete", "result": step_data["final_answer"]}
                return

            step_data["action"] = action
            step_data["action_input"] = action_input
            yield step_data

            if action and action in self.tools:
                try:
                    observation = await self._execute_action(action, action_input)
                    step_data["observation"] = observation
                    yield step_data

                    history.append({"role": "assistant", "content": content})
                    history.append({"role": "user", "content": f"Observation: {observation}"})
                except Exception as e:
                    step_data["error"] = str(e)
                    yield step_data
            else:
                step_data["error"] = f"Unknown tool: {action}"
                yield step_data

        yield {"type": "error", "message": "Max iterations reached"}

    def _parse_response(
        self,
        content: str,
    ) -> Tuple[str, Optional[str], Optional[Dict], bool]:
        """Parse LLM response into components."""
        import re

        thought = ""
        action = None
        action_input = None
        is_final = False

        # Check for final answer
        final_match = re.search(r"Final Answer:\s*(.+)", content, re.IGNORECASE)
        if final_match:
            thought = content[:final_match.start()].strip()
            return thought, None, None, True

        # Check for "I now know the final answer"
        if "I now know the final answer" in content.lower():
            is_final = True
            thought = content
            return thought, None, None, True

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", content, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(r"Action Input:\s*(.+)", content, re.IGNORECASE)
        if input_match:
            input_text = input_match.group(1).strip()
            try:
                action_input = json.loads(input_text)
            except json.JSONDecodeError:
                action_input = {"input": input_text}

        return thought, action, action_input, is_final

    async def _execute_action(
        self,
        action_name: str,
        action_input: Dict[str, Any],
    ) -> Any:
        """Execute a tool action."""
        tool = self.tools.get(action_name)
        if not tool:
            raise ValueError(f"Unknown tool: {action_name}")

        if asyncio.iscoroutinefunction(tool):
            return await tool(**action_input)
        else:
            return tool(**action_input)

    def _format_tools(self) -> str:
        """Format tools description."""
        if not self.tools:
            return "No tools available."

        lines = []
        for name, tool in self.tools.items():
            doc = tool.__doc__ or "No description"
            lines.append(f"- {name}: {doc.strip().split(chr(10))[0]}")

        return "\n".join(lines)


class PlanningAgent(Agent):
    """
    Planning agent that creates and executes plans.

    Breaks complex tasks into sub-tasks and executes them sequentially.
    """

    PLANNING_PROMPT = """Break down the following task into a step-by-step plan.
Each step should be clear and actionable.

Task: {task}

Create a plan with numbered steps. Each step should be something that can be completed independently.

Plan:"""

    EXECUTION_PROMPT = """Execute the following step of the plan.

Current Step: {step}
Progress: {progress}

Complete this step and provide the result:"""

    def __init__(
        self,
        name: str = "planning_agent",
        llm_client: Any = None,
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, llm_client, tools, max_iterations, **kwargs)

    async def run(
        self,
        task: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run planning agent."""
        start_time = time.time()
        state = self._create_state(task)
        state.status = AgentStatus.RUNNING

        # Create plan
        plan = await self._create_plan(task)
        state.metadata["plan"] = plan

        if self.verbose:
            logger.info(f"Created plan with {len(plan)} steps")

        # Execute plan
        results = []
        for i, step in enumerate(plan):
            step_result = await self._execute_step(step, i + 1, plan, results)

            step_obj = AgentStep(
                step_number=i + 1,
                thought=f"Executing: {step}",
            )
            step_obj.observation = AgentObservation(
                action_name="plan_step",
                output=step_result,
            )
            state.add_step(step_obj)

            results.append(step_result)

            # Check if we should continue
            if not await self._should_continue(step_result, i, plan):
                state.status = AgentStatus.FAILED
                state.error = "Plan execution failed"
                break

        state.status = AgentStatus.COMPLETED
        state.result = await self._synthesize_results(task, plan, results)

        total_time = (time.time() - start_time) * 1000
        self._update_stats(state.status == AgentStatus.COMPLETED, len(plan))

        return AgentResult(
            success=state.status == AgentStatus.COMPLETED,
            result=state.result,
            state=state,
            total_steps=len(plan),
            total_time_ms=total_time,
        )

    async def _create_plan(self, task: str) -> List[str]:
        """Create execution plan."""
        prompt = self.PLANNING_PROMPT.format(task=task)

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.content if hasattr(response, 'content') else str(response)

        # Parse plan steps
        steps = []
        for line in content.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering
                step = line.lstrip("0123456789.- ").strip()
                if step:
                    steps.append(step)

        return steps if steps else [task]

    async def _execute_step(
        self,
        step: str,
        step_num: int,
        plan: List[str],
        previous_results: List[str],
    ) -> str:
        """Execute a single plan step."""
        progress = f"Step {step_num}/{len(plan)}"

        prompt = self.EXECUTION_PROMPT.format(
            step=step,
            progress=progress,
        )

        if previous_results:
            prompt += f"\n\nPrevious results:\n" + "\n".join(previous_results)

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def _should_continue(
        self,
        result: str,
        step_index: int,
        plan: List[str],
    ) -> bool:
        """Determine if execution should continue."""
        # Simple implementation - always continue unless error
        if "error" in result.lower() or "failed" in result.lower():
            return False
        return True

    async def _synthesize_results(
        self,
        task: str,
        plan: List[str],
        results: List[str],
    ) -> str:
        """Synthesize final result from step results."""
        synthesis_prompt = f"""Synthesize the final answer from these step results.

Task: {task}

Step Results:
{chr(10).join(f"Step {i+1}: {r}" for i, r in enumerate(results))}

Final Answer:"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.2,
        )

        return response.content if hasattr(response, 'content') else str(response)


class ReflectiveAgent(Agent):
    """
    Reflective agent that improves through self-reflection.

    Reviews its own outputs and iteratively improves them.
    """

    REFLECTION_PROMPT = """Review the following answer and identify areas for improvement.

Task: {task}
Current Answer: {answer}

Critique the answer:
1. Is it complete?
2. Is it accurate?
3. Is it clear?
4. What's missing?

Critique:"""

    IMPROVEMENT_PROMPT = """Improve the following answer based on the critique.

Task: {task}
Current Answer: {answer}
Critique: {critique}

Improved Answer:"""

    def __init__(
        self,
        name: str = "reflective_agent",
        llm_client: Any = None,
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10,
        reflection_iterations: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, llm_client, tools, max_iterations, **kwargs)
        self.reflection_iterations = reflection_iterations

    async def run(
        self,
        task: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run reflective agent."""
        start_time = time.time()
        state = self._create_state(task)
        state.status = AgentStatus.RUNNING

        # Initial answer
        answer = await self._generate_initial_answer(task, context)

        state.add_step(AgentStep(
            step_number=1,
            thought="Generated initial answer",
            observation=AgentObservation(
                action_name="initial_generation",
                output=answer[:500],
            ),
        ))

        # Reflection iterations
        for i in range(self.reflection_iterations):
            # Reflect
            critique = await self._reflect(task, answer)

            state.add_step(AgentStep(
                step_number=i + 2,
                thought=f"Reflection iteration {i + 1}",
                observation=AgentObservation(
                    action_name="reflection",
                    output=critique[:500],
                ),
            ))

            # Improve
            new_answer = await self._improve(task, answer, critique)

            if new_answer == answer:
                # No improvement
                break

            answer = new_answer

        state.result = answer
        state.status = AgentStatus.COMPLETED

        total_time = (time.time() - start_time) * 1000
        total_steps = 1 + self.reflection_iterations
        self._update_stats(True, total_steps)

        return AgentResult(
            success=True,
            result=answer,
            state=state,
            total_steps=total_steps,
            total_time_ms=total_time,
        )

    async def _generate_initial_answer(
        self,
        task: str,
        context: Optional[str],
    ) -> str:
        """Generate initial answer."""
        messages = [{"role": "user", "content": f"Task: {task}"}]

        if context:
            messages.insert(0, {"role": "system", "content": context})

        response = await self.llm_client.generate(messages=messages, temperature=0.3)
        return response.content if hasattr(response, 'content') else str(response)

    async def _reflect(self, task: str, answer: str) -> str:
        """Reflect on answer quality."""
        prompt = self.REFLECTION_PROMPT.format(task=task, answer=answer)

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def _improve(self, task: str, answer: str, critique: str) -> str:
        """Improve answer based on critique."""
        prompt = self.IMPROVEMENT_PROMPT.format(
            task=task,
            answer=answer,
            critique=critique,
        )

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.content if hasattr(response, 'content') else str(response)


class AgentFactory:
    """Factory for creating agents."""

    @staticmethod
    def create(
        agent_type: str,
        llm_client: Any,
        tools: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> Agent:
        """
        Create an agent.

        Args:
            agent_type: Type of agent
            llm_client: LLM client
            tools: Optional tools
            **kwargs: Additional arguments

        Returns:
            Configured agent
        """
        agents = {
            "react": ReActAgent,
            "planning": PlanningAgent,
            "reflective": ReflectiveAgent,
        }

        agent_class = agents.get(agent_type.lower())
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_class(
            llm_client=llm_client,
            tools=tools,
            **kwargs,
        )

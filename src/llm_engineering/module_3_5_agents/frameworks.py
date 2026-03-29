"""
Agent Frameworks Module

Production-ready integrations with agent frameworks:
- LangGraph workflows
- CrewAI integration
- AutoGen multi-agent

Features:
- Framework abstraction
- Workflow definition
- Multi-agent coordination
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class WorkflowState:
    """State of a workflow execution."""

    current_node: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentDefinition:
    """Definition of an agent in a multi-agent system."""

    name: str
    role: str
    goal: str
    backstory: str = ""
    tools: Optional[List[str]] = None
    llm_config: Optional[Dict[str, Any]] = None
    verbose: bool = True


@dataclass
class TaskDefinition:
    """Definition of a task."""

    description: str
    expected_output: str
    agent: Optional[str] = None
    context: Optional[List[str]] = None
    async_execution: bool = False
    output_json: Optional[Dict[str, Any]] = None
    output_pydantic: Optional[Any] = None


class BaseFrameworkAgent(ABC):
    """Abstract base class for framework agents."""

    @abstractmethod
    async def run(self, input_text: str, **kwargs: Any) -> str:
        """Run the agent."""
        pass

    @abstractmethod
    async def stream(self, input_text: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream agent output."""
        pass


class LangGraphAgent(BaseFrameworkAgent):
    """
    LangGraph-based agent.

    Uses LangGraph for workflow definition and execution.
    """

    def __init__(
        self,
        llm_client: Any,
        tools: Optional[List[Callable]] = None,
        graph_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm_client = llm_client
        self.tools = tools or []
        self.graph_config = graph_config or {}

        self._graph = None
        self._langgraph = None

        self._try_import_langgraph()

    def _try_import_langgraph(self) -> None:
        """Try to import LangGraph."""
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.prebuilt import ToolNode
            self._langgraph = {
                "StateGraph": StateGraph,
                "END": END,
                "ToolNode": ToolNode,
            }
            logger.info("LangGraph imported successfully")
        except ImportError:
            logger.warning("LangGraph not installed. Using fallback implementation.")

    def _create_graph(self) -> Any:
        """Create LangGraph workflow."""
        if not self._langgraph:
            return None

        StateGraph = self._langgraph["StateGraph"]
        END = self._langgraph["END"]

        # Define state
        class AgentState(dict):
            messages: list
            next: str

        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._langgraph["ToolNode"](self.tools))

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def _agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent node in graph."""
        messages = state.get("messages", [])

        response = await self.llm_client.generate(
            messages=messages,
            tools=[self._tool_to_dict(t) for t in self.tools] if self.tools else None,
        )

        return {"messages": [response.raw_response] if hasattr(response, 'raw_response') else [response]}

    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Determine if execution should continue."""
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"

    def _tool_to_dict(self, tool: Callable) -> Dict[str, Any]:
        """Convert tool to dict format."""
        import inspect
        sig = inspect.signature(tool)

        return {
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": tool.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string", "description": f"Parameter {name}"}
                        for name in sig.parameters
                    },
                    "required": list(sig.parameters.keys()),
                },
            },
        }

    async def run(self, input_text: str, **kwargs: Any) -> str:
        """Run LangGraph agent."""
        if not self._graph:
            self._graph = self._create_graph()

        if not self._graph:
            # Fallback without LangGraph
            return await self._fallback_run(input_text, **kwargs)

        # Initialize state
        initial_state = {
            "messages": [{"role": "user", "content": input_text}],
        }

        # Run graph
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._graph.invoke(initial_state),
        )

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)

        return ""

    async def _fallback_run(self, input_text: str, **kwargs: Any) -> str:
        """Fallback run without LangGraph."""
        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": input_text}],
            **kwargs,
        )
        return response.content if hasattr(response, 'content') else str(response)

    async def stream(self, input_text: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream LangGraph agent output."""
        # Simple streaming fallback
        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": input_text}],
            stream=True,
            **kwargs,
        )

        if hasattr(response, '__aiter__'):
            async for chunk in response:
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
        else:
            yield response.content if hasattr(response, 'content') else str(response)

    def add_node(
        self,
        name: str,
        action: Callable,
        condition: Optional[Callable] = None,
    ) -> "LangGraphAgent":
        """Add node to graph."""
        if self.graph_config is None:
            self.graph_config = {"nodes": [], "edges": []}

        self.graph_config["nodes"].append({
            "name": name,
            "action": action,
            "condition": condition,
        })

        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable] = None,
    ) -> "LangGraphAgent":
        """Add edge to graph."""
        if self.graph_config is None:
            self.graph_config = {"nodes": [], "edges": []}

        self.graph_config["edges"].append({
            "source": source,
            "target": target,
            "condition": condition,
        })

        return self


class CrewAIAgent(BaseFrameworkAgent):
    """
    CrewAI-based agent.

    Uses CrewAI for role-based multi-agent collaboration.
    """

    def __init__(
        self,
        agents: Optional[List[AgentDefinition]] = None,
        tasks: Optional[List[TaskDefinition]] = None,
        verbose: bool = True,
    ) -> None:
        self.agents = agents or []
        self.tasks = tasks or []
        self.verbose = verbose

        self._crew = None
        self._crewai = None

        self._try_import_crewai()

    def _try_import_crewai(self) -> None:
        """Try to import CrewAI."""
        try:
            from crewai import Agent, Task, Crew
            self._crewai = {
                "Agent": Agent,
                "Task": Task,
                "Crew": Crew,
            }
            logger.info("CrewAI imported successfully")
        except ImportError:
            logger.warning("CrewAI not installed. Using fallback implementation.")

    def _create_crew(self) -> Any:
        """Create CrewAI crew."""
        if not self._crewai:
            return None

        Agent = self._crewai["Agent"]
        Task = self._crewai["Task"]
        Crew = self._crewai["Crew"]

        # Create agents
        crewai_agents = []
        for agent_def in self.agents:
            agent = Agent(
                role=agent_def.role,
                goal=agent_def.goal,
                backstory=agent_def.backstory,
                verbose=agent_def.verbose,
            )
            crewai_agents.append(agent)

        # Create tasks
        crewai_tasks = []
        for task_def in self.tasks:
            task = Task(
                description=task_def.description,
                expected_output=task_def.expected_output,
                agent=crewai_agents[0] if crewai_agents else None,
            )
            crewai_tasks.append(task)

        # Create crew
        crew = Crew(
            agents=crewai_agents,
            tasks=crewai_tasks,
            verbose=self.verbose,
        )

        return crew

    async def run(self, input_text: str, **kwargs: Any) -> str:
        """Run CrewAI crew."""
        if not self._crew:
            self._crew = self._create_crew()

        if not self._crew:
            # Fallback without CrewAI
            return await self._fallback_run(input_text, **kwargs)

        # Run crew
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._crew.kickoff(inputs={"input": input_text}),
        )

        return str(result)

    async def _fallback_run(self, input_text: str, **kwargs: Any) -> str:
        """Fallback run without CrewAI."""
        # Simulate multi-agent processing
        result = f"Processed: {input_text}"

        for agent in self.agents:
            result += f"\n\n[{agent.role}]: {agent.goal}"

        return result

    async def stream(self, input_text: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream CrewAI output."""
        result = await self.run(input_text, **kwargs)

        # Stream word by word
        for word in result.split():
            yield word + " "
            await asyncio.sleep(0.01)

    def add_agent(self, agent: AgentDefinition) -> "CrewAIAgent":
        """Add agent to crew."""
        self.agents.append(agent)
        return self

    def add_task(self, task: TaskDefinition) -> "CrewAIAgent":
        """Add task to crew."""
        self.tasks.append(task)
        return self


class AutoGenAgent(BaseFrameworkAgent):
    """
    AutoGen-based agent.

    Uses AutoGen for conversational multi-agent systems.
    """

    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        agents: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.llm_config = llm_config or {}
        self.agents = agents or []

        self._assistant = None
        self._user_proxy = None
        self._autogen = None

        self._try_import_autogen()

    def _try_import_autogen(self) -> None:
        """Try to import AutoGen."""
        try:
            from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
            self._autogen = {
                "AssistantAgent": AssistantAgent,
                "UserProxyAgent": UserProxyAgent,
                "config_list_from_json": config_list_from_json,
            }
            logger.info("AutoGen imported successfully")
        except ImportError:
            logger.warning("AutoGen not installed. Using fallback implementation.")

    def _create_agents(self) -> None:
        """Create AutoGen agents."""
        if not self._autogen:
            return

        AssistantAgent = self._autogen["AssistantAgent"]
        UserProxyAgent = self._autogen["UserProxyAgent"]

        # Create assistant agent
        self._assistant = AssistantAgent(
            name="assistant",
            llm_config=self.llm_config,
            system_message="You are a helpful AI assistant.",
        )

        # Create user proxy
        self._user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "coding"},
        )

    async def run(self, input_text: str, **kwargs: Any) -> str:
        """Run AutoGen conversation."""
        if not self._assistant:
            self._create_agents()

        if not self._assistant:
            # Fallback without AutoGen
            return await self._fallback_run(input_text, **kwargs)

        # Start conversation
        loop = asyncio.get_event_loop()

        def initiate_chat():
            self._user_proxy.initiate_chat(
                self._assistant,
                message=input_text,
            )
            return self._assistant.last_message()

        result = await loop.run_in_executor(None, initiate_chat)

        return result.get("content", "") if result else ""

    async def _fallback_run(self, input_text: str, **kwargs: Any) -> str:
        """Fallback run without AutoGen."""
        return f"AutoGen fallback response to: {input_text}"

    async def stream(self, input_text: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream AutoGen output."""
        result = await self.run(input_text, **kwargs)

        for char in result:
            yield char
            await asyncio.sleep(0.01)

    def register_function(
        self,
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "AutoGenAgent":
        """Register function for agent to use."""
        if self._user_proxy:
            self._user_proxy.register_function(
                function=function,
                name=name,
                description=description,
            )

        return self


class MultiAgentSystem:
    """
    Multi-agent system coordinator.

    Manages multiple agents and their interactions.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, BaseFrameworkAgent] = {}
        self._orchestrator: Optional[str] = None

    def add_agent(
        self,
        name: str,
        agent: BaseFrameworkAgent,
    ) -> "MultiAgentSystem":
        """Add agent to system."""
        self._agents[name] = agent
        logger.info(f"Added agent: {name}")
        return self

    def set_orchestrator(self, name: str) -> "MultiAgentSystem":
        """Set orchestrator agent."""
        if name in self._agents:
            self._orchestrator = name
        return self

    async def run(
        self,
        input_text: str,
        agent_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Run specific agent or orchestrator."""
        if agent_name:
            if agent_name not in self._agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            return await self._agents[agent_name].run(input_text, **kwargs)

        if self._orchestrator:
            return await self._agents[self._orchestrator].run(input_text, **kwargs)

        # Run all agents and combine results
        results = await asyncio.gather(
            *[agent.run(input_text, **kwargs) for agent in self._agents.values()]
        )

        return "\n\n".join(results)

    async def chain(
        self,
        input_text: str,
        agent_order: List[str],
        **kwargs: Any,
    ) -> str:
        """Chain agents in sequence."""
        current_input = input_text

        for agent_name in agent_order:
            if agent_name not in self._agents:
                raise ValueError(f"Unknown agent: {agent_name}")

            output = await self._agents[agent_name].run(current_input, **kwargs)
            current_input = output

        return current_input

    async def parallel(
        self,
        input_text: str,
        agent_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Run agents in parallel."""
        agents_to_run = agent_names or list(self._agents.keys())

        async def run_agent(name: str) -> tuple:
            result = await self._agents[name].run(input_text, **kwargs)
            return name, result

        results = await asyncio.gather(*[run_agent(name) for name in agents_to_run])

        return dict(results)

    async def debate(
        self,
        topic: str,
        agent_names: List[str],
        rounds: int = 3,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Simulate debate between agents."""
        if len(agent_names) < 2:
            raise ValueError("Debate requires at least 2 agents")

        history = []
        current_topic = topic

        for round_num in range(rounds):
            round_results = []

            for agent_name in agent_names:
                if agent_name not in self._agents:
                    continue

                prompt = f"Debate topic: {current_topic}\n\nRound {round_num + 1}. Present your argument:"
                response = await self._agents[agent_name].run(prompt, **kwargs)

                round_results.append({
                    "agent": agent_name,
                    "response": response,
                    "round": round_num + 1,
                })

            history.extend(round_results)

        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_agents": len(self._agents),
            "agents": list(self._agents.keys()),
            "orchestrator": self._orchestrator,
        }


class WorkflowBuilder:
    """
    Builder for agent workflows.

    Fluent interface for defining complex workflows.
    """

    def __init__(self) -> None:
        self._steps: List[Dict[str, Any]] = []
        self._conditions: List[Dict[str, Any]] = []

    def step(
        self,
        name: str,
        agent: BaseFrameworkAgent,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ) -> "WorkflowBuilder":
        """Add workflow step."""
        self._steps.append({
            "name": name,
            "agent": agent,
            "input_transform": input_transform,
            "output_transform": output_transform,
        })
        return self

    def condition(
        self,
        name: str,
        check: Callable[[str], bool],
        on_true: str,
        on_false: str,
    ) -> "WorkflowBuilder":
        """Add conditional branch."""
        self._conditions.append({
            "name": name,
            "check": check,
            "on_true": on_true,
            "on_false": on_false,
        })
        return self

    async def execute(self, input_text: str, **kwargs: Any) -> str:
        """Execute workflow."""
        current_input = input_text

        for step in self._steps:
            # Transform input
            if step["input_transform"]:
                current_input = step["input_transform"](current_input)

            # Run agent
            output = await step["agent"].run(current_input, **kwargs)

            # Transform output
            if step["output_transform"]:
                output = step["output_transform"](output)

            current_input = output

        return current_input

    def build(self) -> Callable:
        """Build workflow function."""
        async def workflow(input_text: str, **kwargs: Any) -> str:
            return await self.execute(input_text, **kwargs)

        return workflow

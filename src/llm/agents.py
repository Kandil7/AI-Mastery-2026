"""
AI Agents Module

This module implements various AI agent architectures and patterns,
including reactive agents, deliberative agents, and multi-agent systems.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import openai
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of agent types."""
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HIERARCHICAL = "hierarchical"
    MULTI_AGENT = "multi_agent"
    FUNCTION_CALLING = "function_calling"


@dataclass
class AgentState:
    """Represents the state of an agent."""
    beliefs: Dict[str, Any]
    goals: List[str]
    actions: List[str]
    context: Dict[str, Any]
    timestamp: float


@dataclass
class AgentMessage:
    """Represents a message between agents."""
    sender: str
    receiver: str
    content: str
    message_type: str  # "request", "response", "notification", "action"
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.state = AgentState(
            beliefs={},
            goals=[],
            actions=[],
            context={},
            timestamp=time.time()
        )
        self.message_queue = asyncio.Queue()
        self.is_running = False
    
    @abstractmethod
    async def perceive(self, environment_state: Dict[str, Any]) -> None:
        """Perceive the environment and update internal state."""
        pass
    
    @abstractmethod
    async def decide(self) -> str:
        """Decide on the next action based on current state."""
        pass
    
    @abstractmethod
    async def act(self, action: str) -> Dict[str, Any]:
        """Execute an action and return the result."""
        pass
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message."""
        # Default implementation - can be overridden
        logger.info(f"Agent {self.name} received message: {message.content}")
        return None
    
    async def run(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution loop for the agent."""
        self.is_running = True
        
        while self.is_running:
            # Perceive environment
            await self.perceive(environment_state)
            
            # Decide on action
            action = await self.decide()
            
            # Execute action
            result = await self.act(action)
            
            # Update state
            self.state.timestamp = time.time()
            
            # Process any messages
            while not self.message_queue.empty():
                message = await self.message_queue.get()
                response = await self.process_message(message)
                if response:
                    # Handle response if needed
                    pass
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
            
            return result
    
    def stop(self):
        """Stop the agent."""
        self.is_running = False


class ReactiveAgent(BaseAgent):
    """
    Reactive agent that responds directly to environmental stimuli
    without maintaining complex internal state.
    """
    
    def __init__(self, agent_id: str, name: str, rules: Dict[str, str]):
        super().__init__(agent_id, name)
        self.rules = rules  # Mapping of conditions to actions
    
    async def perceive(self, environment_state: Dict[str, Any]) -> None:
        """Perceive the environment and update beliefs."""
        # Update context with environment state
        self.state.context.update(environment_state)
        
        # Update beliefs based on environment
        for key, value in environment_state.items():
            self.state.beliefs[f"env_{key}"] = value
    
    async def decide(self) -> str:
        """Decide on action based on rules and current state."""
        # Apply rules to determine action
        for condition, action in self.rules.items():
            if self._evaluate_condition(condition):
                return action
        
        # Default action if no rules match
        return "idle"
    
    async def act(self, action: str) -> Dict[str, Any]:
        """Execute the action."""
        logger.info(f"Reactive agent {self.name} performing action: {action}")
        
        # Record action
        self.state.actions.append(action)
        
        # Simulate action execution
        if action == "move_forward":
            result = {"status": "success", "position_change": (0, 1)}
        elif action == "turn_left":
            result = {"status": "success", "direction_change": -90}
        elif action == "turn_right":
            result = {"status": "success", "direction_change": 90}
        elif action == "sense":
            result = {"status": "success", "environment_data": self.state.context}
        else:
            result = {"status": "idle", "message": "No action performed"}
        
        return result
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition against current state."""
        # Simple condition evaluation
        # In a real implementation, this would be more sophisticated
        if condition == "object_detected":
            return self.state.context.get("object_detected", False)
        elif condition == "battery_low":
            return self.state.context.get("battery_level", 100) < 20
        elif condition == "goal_reached":
            return self.state.context.get("goal_reached", False)
        else:
            return False


class DeliberativeAgent(BaseAgent):
    """
    Deliberative agent that maintains beliefs, goals, and plans
    before taking actions.
    """
    
    def __init__(self, agent_id: str, name: str, llm_model: str = "gpt-3.5-turbo"):
        super().__init__(agent_id, name)
        self.llm_model = llm_model
        self.plans: List[List[str]] = []  # List of action sequences
        self.current_plan_index = 0
    
    async def perceive(self, environment_state: Dict[str, Any]) -> None:
        """Perceive the environment and update internal state."""
        # Update context
        self.state.context.update(environment_state)
        
        # Update beliefs based on environment
        for key, value in environment_state.items():
            self.state.beliefs[f"env_{key}"] = value
        
        # Check if goals are still valid
        self._validate_goals()
    
    async def decide(self) -> str:
        """Decide on the next action, potentially creating a plan."""
        # If we have a current plan, continue executing it
        if self.plans and self.current_plan_index < len(self.plans[-1]):
            next_action = self.plans[-1][self.current_plan_index]
            self.current_plan_index += 1
            return next_action
        
        # Otherwise, create a new plan based on goals
        if self.state.goals:
            await self._create_plan()
            if self.plans and len(self.plans[-1]) > 0:
                next_action = self.plans[-1][0]
                self.current_plan_index = 1
                return next_action
        
        # If no goals or plan, return idle
        return "idle"
    
    async def act(self, action: str) -> Dict[str, Any]:
        """Execute the action."""
        logger.info(f"Deliberative agent {self.name} performing action: {action}")
        
        # Record action
        self.state.actions.append(action)
        
        # Simulate action execution
        result = {
            "action": action,
            "status": "executed",
            "timestamp": time.time()
        }
        
        # Check if action achieved any goals
        await self._check_goal_achievement(action)
        
        return result
    
    async def _create_plan(self):
        """Create a plan to achieve the current goals."""
        # In a real implementation, this would use more sophisticated planning
        # For now, we'll use a simple approach
        
        goal = self.state.goals[0] if self.state.goals else "idle"
        
        # Simple planning based on common goals
        if "navigate" in goal.lower() or "go to" in goal.lower():
            plan = ["plan_route", "move_forward", "check_position", "adjust_direction"] * 3
        elif "pick up" in goal.lower() or "collect" in goal.lower():
            plan = ["approach_object", "verify_grasp", "pick_up", "verify_success"]
        elif "analyze" in goal.lower() or "examine" in goal.lower():
            plan = ["scan_area", "collect_data", "process_data", "generate_report"]
        else:
            plan = ["idle"]
        
        self.plans.append(plan)
        self.current_plan_index = 0
    
    async def _check_goal_achievement(self, action: str):
        """Check if the action achieved any goals."""
        # Simple goal achievement checking
        for goal in self.state.goals[:]:  # Use slice to avoid modification during iteration
            if self._goal_achieved(goal, action):
                self.state.goals.remove(goal)
                logger.info(f"Goal achieved: {goal}")
    
    def _goal_achieved(self, goal: str, action: str) -> bool:
        """Check if a specific goal has been achieved."""
        # Simple implementation - in reality, this would be more complex
        if "navigate" in goal.lower() and "move" in action.lower():
            return self.state.context.get("at_destination", False)
        elif "pick up" in goal.lower() and "pick" in action.lower():
            return self.state.context.get("object_grasped", False)
        return False
    
    def _validate_goals(self):
        """Validate that current goals are still relevant."""
        # Remove goals that are no longer relevant
        valid_goals = []
        for goal in self.state.goals:
            if self._goal_is_valid(goal):
                valid_goals.append(goal)
            else:
                logger.info(f"Removing invalid goal: {goal}")
        
        self.state.goals = valid_goals
    
    def _goal_is_valid(self, goal: str) -> bool:
        """Check if a goal is still valid."""
        # Simple validation - in reality, this would check environment conditions
        return True


class FunctionCallingAgent(BaseAgent):
    """
    Agent that uses function calling capabilities of LLMs.
    """
    
    def __init__(self, agent_id: str, name: str, tools: List[BaseTool]):
        super().__init__(agent_id, name)
        self.tools = {tool.name: tool for tool in tools}
        self.llm = OpenAI(temperature=0)
        
        # Initialize LangChain agent
        self.langchain_agent = initialize_agent(
            tools=list(self.tools.values()),
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    async def perceive(self, environment_state: Dict[str, Any]) -> None:
        """Perceive the environment and update internal state."""
        self.state.context.update(environment_state)
        
        # Update beliefs based on environment
        for key, value in environment_state.items():
            self.state.beliefs[f"env_{key}"] = value
    
    async def decide(self) -> str:
        """Decide on the next action using LLM and tools."""
        # Formulate a query based on current state and goals
        query = self._formulate_query()
        
        try:
            # Use the LangChain agent to decide on action
            result = self.langchain_agent.run(query)
            return result
        except Exception as e:
            logger.error(f"Error in agent decision: {e}")
            return "error"
    
    async def act(self, action: str) -> Dict[str, Any]:
        """Execute the action."""
        logger.info(f"Function calling agent {self.name} performing action: {action}")
        
        # Record action
        self.state.actions.append(action)
        
        return {
            "action": action,
            "status": "executed",
            "timestamp": time.time()
        }
    
    def _formulate_query(self) -> str:
        """Formulate a query for the LLM based on current state."""
        context_str = json.dumps(self.state.context, indent=2)
        goals_str = json.dumps(self.state.goals, indent=2)
        
        query = f"""
        Current context: {context_str}
        
        Current goals: {goals_str}
        
        Based on the current context and goals, what should be the next action?
        Use the available tools to gather information or perform actions.
        """
        
        return query


class MultiAgentSystem:
    """
    System that coordinates multiple agents.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_router = MessageRouter()
        self.is_running = False
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the system."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.name} to multi-agent system")
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the system."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.stop()
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id} from multi-agent system")
    
    async def send_message(self, message: AgentMessage):
        """Send a message to an agent."""
        await self.message_router.route_message(message)
    
    async def run(self):
        """Run the multi-agent system."""
        self.is_running = True
        
        # Run all agents concurrently
        agent_tasks = []
        for agent in self.agents.values():
            # Each agent runs in its own task
            task = asyncio.create_task(agent.run({}))
            agent_tasks.append(task)
        
        # Wait for all agents to complete (they run indefinitely)
        await asyncio.gather(*agent_tasks, return_exceptions=True)
    
    def stop(self):
        """Stop the multi-agent system."""
        self.is_running = False
        for agent in self.agents.values():
            agent.stop()


class MessageRouter:
    """Routes messages between agents."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the router."""
        self.agents[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the router."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    async def route_message(self, message: AgentMessage):
        """Route a message to the appropriate agent."""
        if message.receiver in self.agents:
            receiver = self.agents[message.receiver]
            await receiver.message_queue.put(message)
        else:
            logger.warning(f"Message receiver {message.receiver} not found")


class AgentOrchestrator:
    """
    Orchestrates complex multi-agent workflows.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows = {}
        self.current_workflow = None
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
    
    def define_workflow(self, workflow_id: str, steps: List[Dict[str, Any]]):
        """Define a workflow with steps."""
        self.workflows[workflow_id] = steps
    
    async def execute_workflow(self, workflow_id: str, initial_context: Dict[str, Any] = None):
        """Execute a defined workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if initial_context is None:
            initial_context = {}
        
        self.current_workflow = workflow_id
        steps = self.workflows[workflow_id]
        
        context = initial_context.copy()
        
        for step in steps:
            agent_id = step["agent_id"]
            action = step["action"]
            params = step.get("params", {})
            
            if agent_id not in self.agents:
                logger.error(f"Agent {agent_id} not found in orchestrator")
                continue
            
            agent = self.agents[agent_id]
            
            # Update agent's context with current workflow context
            agent.state.context.update(context)
            
            # Execute the action
            result = await agent.act(action)
            
            # Update context with results
            context.update(result)
            
            # Add any specific outputs to context
            if "output_key" in step:
                context[step["output_key"]] = result
        
        return context


# Tool implementations for function calling agents
class CalculatorTool(BaseTool):
    """A simple calculator tool."""
    
    name = "calculator"
    description = "Useful for performing mathematical calculations"
    
    def _run(self, query: str) -> str:
        """Run the calculator tool."""
        try:
            # Simple evaluation - in production, use a safer method
            result = eval(query)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Asynchronous version of the calculator tool."""
        return self._run(query)


class SearchTool(BaseTool):
    """A search tool (simulated)."""
    
    name = "search"
    description = "Useful for searching information on the internet"
    
    def _run(self, query: str) -> str:
        """Run the search tool."""
        # Simulate search results
        return f"Simulated search results for: {query}"
    
    async def _arun(self, query: str) -> str:
        """Asynchronous version of the search tool."""
        return self._run(query)


class DatabaseTool(BaseTool):
    """A database query tool (simulated)."""
    
    name = "database"
    description = "Useful for querying a database"
    
    def _run(self, query: str) -> str:
        """Run the database tool."""
        # Simulate database query
        return f"Simulated database results for query: {query}"
    
    async def _arun(self, query: str) -> str:
        """Asynchronous version of the database tool."""
        return self._run(query)


def create_agent(agent_type: AgentType, agent_id: str, name: str, **kwargs) -> BaseAgent:
    """
    Factory function to create different types of agents.
    
    Args:
        agent_type: Type of agent to create
        agent_id: Unique identifier for the agent
        name: Name of the agent
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Agent instance
    """
    if agent_type == AgentType.REACTIVE:
        rules = kwargs.get("rules", {})
        return ReactiveAgent(agent_id, name, rules)
    elif agent_type == AgentType.DELIBERATIVE:
        llm_model = kwargs.get("llm_model", "gpt-3.5-turbo")
        return DeliberativeAgent(agent_id, name, llm_model)
    elif agent_type == AgentType.FUNCTION_CALLING:
        tools = kwargs.get("tools", [])
        return FunctionCallingAgent(agent_id, name, tools)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def create_simple_reactive_agent(agent_id: str, name: str) -> ReactiveAgent:
    """Create a simple reactive agent with basic rules."""
    rules = {
        "object_detected": "move_towards_object",
        "battery_low": "return_to_charger",
        "goal_reached": "report_success",
        "default": "explore"
    }
    return ReactiveAgent(agent_id, name, rules)


def create_simple_deliberative_agent(agent_id: str, name: str) -> DeliberativeAgent:
    """Create a simple deliberative agent."""
    return DeliberativeAgent(agent_id, name)


def create_function_calling_agent(agent_id: str, name: str) -> FunctionCallingAgent:
    """Create a function calling agent with common tools."""
    tools = [
        CalculatorTool(),
        SearchTool(),
        DatabaseTool()
    ]
    return FunctionCallingAgent(agent_id, name, tools)


# Example usage and testing
async def main():
    """Example usage of the agents module."""
    logger.info("Starting agent examples...")
    
    # Create a reactive agent
    reactive_agent = create_simple_reactive_agent("reactive_001", "ReactiveBot")
    reactive_agent.state.goals = ["explore_environment"]
    
    # Create a deliberative agent
    deliberative_agent = create_simple_deliberative_agent("delib_001", "DeliberativeBot")
    deliberative_agent.state.goals = ["navigate_to_location", "pick_up_object"]
    
    # Create a function calling agent
    func_agent = create_function_calling_agent("func_001", "FunctionBot")
    func_agent.state.goals = ["calculate_complex_equation", "search_for_information"]
    
    # Example environment state
    env_state = {
        "object_detected": True,
        "battery_level": 80,
        "goal_reached": False,
        "position": (10, 20)
    }
    
    # Run agents
    print("Reactive Agent:")
    result = await reactive_agent.act("move_towards_object")
    print(f"Result: {result}")
    
    print("\nDeliberative Agent:")
    await deliberative_agent.perceive(env_state)
    action = await deliberative_agent.decide()
    result = await deliberative_agent.act(action)
    print(f"Action: {action}, Result: {result}")
    
    print("\nFunction Calling Agent:")
    await func_agent.perceive(env_state)
    action = await func_agent.decide()
    result = await func_agent.act(action)
    print(f"Action: {action}, Result: {result}")
    
    # Create and run a multi-agent system
    print("\nMulti-Agent System:")
    mas = MultiAgentSystem()
    mas.add_agent(reactive_agent)
    mas.add_agent(deliberative_agent)
    mas.add_agent(func_agent)
    
    # Send a message between agents
    message = AgentMessage(
        sender="reactive_001",
        receiver="delib_001",
        content="Object detected at coordinates (15, 25)",
        message_type="notification",
        timestamp=time.time()
    )
    
    await mas.send_message(message)
    
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
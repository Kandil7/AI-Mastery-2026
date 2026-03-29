"""
Module 3.5: Agents

Production-ready agent implementations:
- Agent Core: Base agent, ReAct, planning, reflection
- Protocols: MCP, A2A protocol
- Vendor SDKs: OpenAI Agents, Google ADK, Claude Agent
- Frameworks: LangGraph, CrewAI, AutoGen
"""

from .agent_core import (
    Agent,
    ReActAgent,
    PlanningAgent,
    ReflectiveAgent,
    AgentState,
    AgentAction,
    AgentStep,
)
from .protocols import (
    ModelContextProtocol,
    Agent2AgentProtocol,
    Message,
    ToolCall,
    ProtocolHandler,
)
from .vendor_sdks import (
    OpenAIAgentsSDK,
    GoogleADK,
    ClaudeAgentSDK,
    VendorSDKWrapper,
)
from .frameworks import (
    LangGraphAgent,
    CrewAIAgent,
    AutoGenAgent,
    MultiAgentSystem,
)

__all__ = [
    # Agent Core
    "Agent",
    "ReActAgent",
    "PlanningAgent",
    "ReflectiveAgent",
    "AgentState",
    "AgentAction",
    "AgentStep",
    # Protocols
    "ModelContextProtocol",
    "Agent2AgentProtocol",
    "Message",
    "ToolCall",
    "ProtocolHandler",
    # Vendor SDKs
    "OpenAIAgentsSDK",
    "GoogleADK",
    "ClaudeAgentSDK",
    "VendorSDKWrapper",
    # Frameworks
    "LangGraphAgent",
    "CrewAIAgent",
    "AutoGenAgent",
    "MultiAgentSystem",
]

__version__ = "1.0.0"

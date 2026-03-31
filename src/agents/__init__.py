"""
Agents Module
=============

AI Agent orchestration, multi-agent systems, and specialized support agents.
Includes LangChain-style chains, tools, memory systems, and ReAct agents.
"""

from .orchestration import (
    OrchestrationConfig,
    Memory,
    ConversationMemory,
    Tool,
    ToolRegistry,
    Chain,
    SequentialChain,
    ParallelChain,
    Agent,
    ReActAgent,
)

from .multi_agent_systems import (
    AgentRole,
    Task,
    AgentState as MultiAgentState,
    MultiAgent,
    ResearchTeam,
    ArabicContentTeam,
    CodeGenerationTeam,
)

from .support_agent import (
    ConversationState,
    SentimentType,
    SupportArticle,
    Message as SupportMessage,
    Conversation,
    RetrievedSource,
    ContentGuardrail,
    SourceCitationEngine,
    ConfidenceScorer,
    CXScoreAnalyzer,
    SupportAgent,
)

__all__ = [
    # Orchestration
    "OrchestrationConfig",
    "Memory",
    "ConversationMemory",
    "Tool",
    "ToolRegistry",
    "Chain",
    "SequentialChain",
    "ParallelChain",
    "Agent",
    "ReActAgent",
    # Multi-Agent Systems
    "AgentRole",
    "Task",
    "MultiAgentState",
    "MultiAgent",
    "ResearchTeam",
    "ArabicContentTeam",
    "CodeGenerationTeam",
    # Support Agent
    "ConversationState",
    "SentimentType",
    "SupportArticle",
    "SupportMessage",
    "Conversation",
    "RetrievedSource",
    "ContentGuardrail",
    "SourceCitationEngine",
    "ConfidenceScorer",
    "CXScoreAnalyzer",
    "SupportAgent",
]

"""
Agent System for Islamic Literature RAG

Specialized agents that can use the RAG system for different tasks.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Available agent roles for Islamic literature."""

    RESEARCHER = "researcher"
    STUDENT = "student"
    TEACHER = "teacher"
    FATWA_REQUESTER = "fatwa_requester"
    COMPARATOR = "comparator"
    TRANSLATOR = "translator"
    HISTORIAN = "historian"
    LINGUIST = "linguist"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    role: AgentRole
    name: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


# Agent configurations
AGENT_CONFIGS = {
    AgentRole.RESEARCHER: AgentConfig(
        role=AgentRole.RESEARCHER,
        name="الباحث الإسلامي",
        description="Deep research on Islamic topics with evidence-based answers",
        system_prompt="""You are an Islamic researcher specialized in finding and analyzing evidence from classical sources.

Your approach:
1. Find relevant evidence from Quran, Hadith, and classical texts
2. Present different scholarly opinions
3. Analyze the strength of evidence
4. Provide balanced conclusions

Always cite sources using traditional Islamic format.""",
        tools=["rag_query", "cross_reference", "authority_ranker"],
        capabilities=["deep_research", "evidence_analysis", "scholarly_opinions"],
    ),
    AgentRole.STUDENT: AgentConfig(
        role=AgentRole.STUDENT,
        name="الطالب المسلم",
        description="Learning companion for Islamic studies students",
        system_prompt="""You are a helpful Islamic studies tutor.

Your approach:
1. Explain concepts clearly in simple language
2. Use examples from everyday life
3. Break down complex topics
4. Test understanding with questions

Be patient and encouraging. Adjust your explanations based on the student's level.""",
        tools=["rag_query", "concept_explainer"],
        capabilities=["explanation", "examples", "interactive_learning"],
    ),
    AgentRole.TEACHER: AgentConfig(
        role=AgentRole.TEACHER,
        name="المعلم الإسلامي",
        description="Teaching assistant for Islamic education",
        system_prompt="""You are an experienced Islamic teacher.

Your approach:
1. Structure lessons logically
2. Start with foundations
3. Build progressively to advanced topics
4. Include practical applications
5. Test understanding

Create comprehensive lessons with clear objectives.""",
        tools=["rag_query", "lesson_builder", "quiz_generator"],
        capabilities=["lesson_planning", "curriculum_design", "assessment"],
    ),
    AgentRole.FATWA_REQUESTER: AgentConfig(
        role=AgentRole.FATWA_REQUESTER,
        name="السائل الفقهي",
        description="Ask fiqh questions and get scholarly answers",
        system_prompt="""You are a Fatwa research system.

Your approach:
1. Understand the question clearly
2. Find relevant rulings from classical sources
3. Present the evidence
4. Mention different opinions if any
5. Conclude with the preferred view
6. Always note "consult a scholar for personal fatwas"

Important: This is research, not a real fatwa. Always include disclaimers.""",
        tools=["rag_query", "fiqh_specialist", "authority_ranker"],
        capabilities=["fiqh_research", "evidence_presentation", "scholarly_rulings"],
    ),
    AgentRole.COMPARATOR: AgentConfig(
        role=AgentRole.COMPARATOR,
        name="المقارن",
        description="Compare Islamic scholarly opinions across madhhabs",
        system_prompt="""You are a comparative Islamic scholar.

Your approach:
1. Present all four madhhab positions on an issue
2. Compare the evidence each uses
3. Identify areas of consensus and disagreement
4. Present the reasoning behind different views
5. Note which view has stronger evidence

Always be fair and balanced. Never favor one madhhab unfairly.""",
        tools=["rag_query", "comparative_fiqh", "cross_reference"],
        capabilities=["madhhab_comparison", "evidence_analysis", "ijma_identification"],
    ),
    AgentRole.TRANSLATOR: AgentConfig(
        role=AgentRole.TRANSLATOR,
        name="المترجم",
        description="Translate and explain Islamic texts",
        system_prompt="""You are an Islamic texts translator and explainer.

Your approach:
1. Translate accurately, preserving meaning
2. Explain difficult vocabulary
3. Provide contextual background
4. Note classical interpretations
5. Keep explanations accessible

Focus on accuracy while making classical texts understandable.""",
        tools=["rag_query", "text_explainer"],
        capabilities=["translation", "explanation", "contextualization"],
    ),
    AgentRole.HISTORIAN: AgentConfig(
        role=AgentRole.HISTORIAN,
        name="المؤرخ الإسلامي",
        description="Research Islamic history and biographies",
        system_prompt="""You are an Islamic historian.

Your approach:
1. Present events in chronological context
2. Identify causes and effects
3. Present multiple historical sources
4. Note historical debates
5. Connect past to present

Be objective and cite sources for historical claims.""",
        tools=["rag_query", "timeline_reconstructor", "cross_reference"],
        capabilities=["historical_analysis", "timeline_building", "biography"],
    ),
    AgentRole.LINGUIST: AgentConfig(
        role=AgentRole.LINGUIST,
        name="اللغوي العربي",
        description="Analyze Arabic language in Islamic texts",
        system_prompt="""You are an Arabic linguist specializing in Islamic texts.

Your approach:
1. Analyze Arabic grammar and morphology
2. Explain lexical meanings
3. Discuss rhetorical devices
4. Note stylistic features
5. Connect language to meaning

Make Arabic linguistics accessible to students.""",
        tools=["rag_query", "concept_extractor"],
        capabilities=["grammar_analysis", "rhetoric_explanation", "lexical_study"],
    ),
}


class IslamicRAGAgent:
    """
    An agent that uses the Islamic literature RAG system.

    Each agent has a specific role and can use various tools
    to accomplish tasks related to Islamic scholarship.
    """

    def __init__(
        self,
        role: AgentRole,
        pipeline: Any = None,
        tools: Optional[Dict[str, Any]] = None,
    ):
        self.role = role
        self.config = AGENT_CONFIGS.get(role)
        self.pipeline = pipeline
        self.tools = tools or {}

        # Initialize default tools if not provided
        if pipeline and not self.tools:
            self._init_default_tools()

    def _init_default_tools(self):
        """Initialize default tools from pipeline."""

        from ..specialists.islamic_scholars import create_islamic_scholar
        from .advanced_features import (
            create_authority_ranker,
            create_concept_extractor,
            create_multi_hop_reasoning,
        )

        self.tools = {
            "rag_query": self._rag_query,
            "authority_ranker": create_authority_ranker(),
            "concept_explainer": create_concept_extractor(),
            "fiqh_specialist": create_islamic_scholar("fiqh", self.pipeline),
            "comparative_fiqh": create_islamic_scholar(
                "fiqh_comparative", self.pipeline
            ),
            "multi_hop_reasoning": create_multi_hop_reasoning(self.pipeline),
        }

    async def _rag_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query the RAG pipeline."""

        if not self.pipeline:
            return {"error": "Pipeline not initialized"}

        return await self.pipeline.query(query, top_k=top_k)

    async def execute(
        self,
        task: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task using the agent's capabilities.

        Args:
            task: The task to execute
            params: Optional parameters

        Returns:
            Task result
        """

        params = params or {}

        # Route to appropriate handler
        if task == "research":
            return await self._research(params.get("question", ""))
        elif task == "teach":
            return await self._teach(params.get("topic", ""))
        elif task == "get_fatwa":
            return await self._get_fatwa(params.get("question", ""))
        elif task == "compare":
            return await self._compare(params.get("topic", ""))
        elif task == "translate":
            return await self._translate(params.get("text", ""))
        elif task == "history":
            return await self._research_history(params.get("topic", ""))
        elif task == "analyze_language":
            return await self._analyze_language(params.get("text", ""))
        else:
            return {"error": f"Unknown task: {task}"}

    async def _research(self, question: str) -> Dict[str, Any]:
        """Conduct research on an Islamic topic."""

        # Use multi-hop reasoning for deep research
        if "multi_hop_reasoning" in self.tools:
            result = await self.tools["multi_hop_reasoning"].reason(question)
            return result

        # Fallback to simple query
        result = await self._rag_query(question)

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        }

    async def _teach(self, topic: str) -> Dict[str, Any]:
        """Teach a topic with structured lesson."""

        # Get foundational content
        result = await self._rag_query(f"أساسيات {topic}", top_k=5)

        lesson = {
            "topic": topic,
            "objectives": [
                f"Understand the basics of {topic}",
                f"Identify key concepts in {topic}",
                f"Apply knowledge practically",
            ],
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
            "practice_questions": [
                f"What is {topic}?",
                f"Explain the importance of {topic}",
                f"How is {topic} applied in practice?",
            ],
        }

        return lesson

    async def _get_fatwa(self, question: str) -> Dict[str, Any]:
        """Get a fiqh-style answer (research, not real fatwa)."""

        # Use fiqh specialist
        if "fiqh_specialist" in self.tools:
            result = await self.tools["fiqh_specialist"].query(question)
        else:
            result = await self._rag_query(question)

        # Add disclaimer
        disclaimer = """
⚠️ IMPORTANT DISCLAIMER:
This is AI-generated research for educational purposes only.
It is NOT a real fatwa (religious ruling).
For actual religious rulings, please consult a qualified Islamic scholar.

This response should be used as a starting point for further research with authentic sources.
"""

        return {
            "question": question,
            "answer": result.get("answer", "") + disclaimer,
            "sources": result.get("sources", []),
            "type": "fiqh_research",
        }

    async def _compare(self, topic: str) -> Dict[str, Any]:
        """Compare madhhab opinions on a topic."""

        # Use comparative fiqh
        if "comparative_fiqh" in self.tools:
            result = await self.tools["comparative_fiqh"].query_with_comparison(topic)
            return result

        # Fallback
        result = await self._rag_query(f"آراء المذاهب في {topic}", top_k=10)

        return {
            "topic": topic,
            "madhhab_views": result.get("sources", []),
        }

    async def _translate(self, text: str) -> Dict[str, Any]:
        """Translate and explain Arabic text."""

        result = await self._rag_query(f"شرح وتفسير: {text}", top_k=3)

        return {
            "original": text,
            "explanation": result.get("answer", ""),
            "sources": result.get("sources", []),
        }

    async def _research_history(self, topic: str) -> Dict[str, Any]:
        """Research Islamic historical topics."""

        result = await self._rag_query(f"تاريخ {topic}", top_k=5)

        return {
            "topic": topic,
            "history": result.get("answer", ""),
            "sources": result.get("sources", []),
        }

    async def _analyze_language(self, text: str) -> Dict[str, Any]:
        """Analyze Arabic language features."""

        # Use concept extractor
        if "concept_explainer" in self.tools:
            concepts = self.tools["concept_explainer"].extract(text)
        else:
            concepts = []

        # Get linguistic analysis from RAG
        result = await self._rag_query(f"تحليل لغوي: {text}", top_k=3)

        return {
            "text": text,
            "concepts_found": concepts,
            "analysis": result.get("answer", ""),
            "sources": result.get("sources", []),
        }


# Agent Factory
def create_agent(
    role: str,
    pipeline: Any = None,
    tools: Optional[Dict[str, Any]] = None,
) -> IslamicRAGAgent:
    """Create an agent with the specified role."""

    role_map = {
        "researcher": AgentRole.RESEARCHER,
        "student": AgentRole.STUDENT,
        "teacher": AgentRole.TEACHER,
        "fatwa": AgentRole.FATWA_REQUESTER,
        "comparator": AgentRole.COMPARATOR,
        "translator": AgentRole.TRANSLATOR,
        "historian": AgentRole.HISTORIAN,
        "linguist": AgentRole.LINGUIST,
    }

    role_enum = role_map.get(role.lower())

    if not role_enum:
        raise ValueError(f"Unknown role: {role}")

    return IslamicRAGAgent(role_enum, pipeline, tools)


# Multi-Agent System
class AgentTeam:
    """
    A team of agents that can collaborate on complex tasks.
    """

    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline
        self.agents: Dict[AgentRole, IslamicRAGAgent] = {}

        # Initialize all agent types
        for role in AgentRole:
            self.agents[role] = create_agent(role.value, pipeline)

    async def collaborate(
        self,
        task: str,
        requires_roles: List[AgentRole],
    ) -> Dict[str, Any]:
        """
        Have multiple agents collaborate on a task.

        Args:
            task: The task description
            required_roles: Which agent roles are needed

        Returns:
            Combined results from all agents
        """

        results = {}

        for role in required_roles:
            agent = self.agents.get(role)
            if agent:
                # Determine appropriate action based on role
                action = self._determine_action(role, task)
                result = await agent.execute(action, {"question": task})
                results[role.value] = result

        # Synthesize results
        synthesis = self._synthesize_results(task, results)

        return {
            "task": task,
            "agent_results": results,
            "synthesis": synthesis,
        }

    def _determine_action(self, role: AgentRole, task: str) -> str:
        """Determine what action an agent should take for a task."""

        if role == AgentRole.RESEARCHER:
            return "research"
        elif role == AgentRole.FATWA_REQUESTER:
            return "get_fatwa"
        elif role == AgentRole.COMPARATOR:
            return "compare"
        elif role == AgentRole.TEACHER:
            return "teach"
        elif role == AgentRole.HISTORIAN:
            return "history"
        elif role == AgentRole.LINGUIST:
            return "analyze_language"
        else:
            return "research"

    def _synthesize_results(
        self,
        task: str,
        results: Dict[str, Any],
    ) -> str:
        """Synthesize results from multiple agents into a unified response."""

        synthesis = f"## Task: {task}\n\n"

        synthesis += "### Analysis from Multiple Perspectives:\n\n"

        for role, result in results.items():
            synthesis += f"**{role}:**\n"
            synthesis += (
                f"{result.get('answer', result.get('analysis', ''))[:300]}...\n\n"
            )

        synthesis += (
            "---\n*This response combines insights from multiple specialized agents.*"
        )

        return synthesis

"""
Complete Islamic Literature RAG System - Integration Module

This module integrates all the specialized components:
1. Base RAG Pipeline
2. Domain Specialists
3. Advanced Features
4. Agents
5. Evaluation

Usage:
    from rag_system import IslamicRAG

    rag = IslamicRAG()
    await rag.initialize()

    # Basic query
    result = await rag.query("ما هو التوحيد؟")

    # Specialized queries
    result = await rag.query_as_scholar("fiqh", "ما حكم الزكاة؟")
    result = await rag.compare_madhhabs("ما حكم الصيام؟")
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Import all components
from .pipeline.complete_pipeline import (
    CompleteRAGPipeline,
    RAGConfig,
    create_rag_pipeline,
    QueryResult,
)
from .specialists.islamic_scholars import (
    IslamicScholar,
    ComparativeFiqhScholar,
    ChainOfScholarship,
    create_islamic_scholar,
    create_comparative_fiqh_scholar,
    IslamicDomain,
)
from .specialists.advanced_features import (
    AuthorityRanker,
    CrossReferenceSystem,
    MultiHopReasoning,
    TimelineReconstructor,
    ProgressiveRetrieval,
    ConceptExtractor,
    create_authority_ranker,
    create_cross_reference_system,
    create_multi_hop_reasoning,
    create_timeline_reconstructor,
    create_progressive_retrieval,
    create_concept_extractor,
)
from .agents.agent_system import (
    IslamicRAGAgent,
    AgentTeam,
    AgentRole,
    create_agent,
)
from .evaluation.islamic_metrics import (
    IslamicRAGEvaluator,
    create_islamic_evaluator,
)


@dataclass
class IslamicRAGConfig:
    """Configuration for the complete Islamic RAG system."""

    # Base RAG config
    datasets_path: str = "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
    output_path: str = "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data"

    # Embedding
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_device: str = "cpu"

    # LLM
    llm_provider: str = "mock"
    llm_model: str = "gpt-4o"

    # Processing
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 50
    rerank_top_k: int = 5

    # Advanced features
    enable_authority_ranking: bool = True
    enable_cross_references: bool = True
    enable_multi_hop: bool = True


class IslamicRAG:
    """
    Complete Islamic Literature RAG System.

    Provides a unified interface for:
    - Basic RAG queries
    - Domain-specific queries (Tafsir, Hadith, Fiqh, etc.)
    - Comparative analysis
    - Multi-hop reasoning
    - Specialized agent interactions
    """

    def __init__(self, config: Optional[IslamicRAGConfig] = None):
        self.config = config or IslamicRAGConfig()

        # Base pipeline
        self._pipeline: Optional[CompleteRAGPipeline] = None

        # Specialized components
        self._scholars: Dict[str, IslamicScholar] = {}
        self._comparative_fiqh: Optional[ComparativeFiqhScholar] = None
        self._chain_reasoning: Optional[ChainOfScholarship] = None

        # Advanced features
        self._authority_ranker: Optional[AuthorityRanker] = None
        self._cross_refs: Optional[CrossReferenceSystem] = None
        self._timeline: Optional[TimelineReconstructor] = None
        self._concept_extractor: Optional[ConceptExtractor] = None

        # Agents
        self._agents: Dict[str, IslamicRAGAgent] = {}
        self._agent_team: Optional[AgentTeam] = None

        # Evaluation
        self._evaluator: Optional[IslamicRAGEvaluator] = None

        # State
        self._initialized = False

    async def initialize(
        self,
        load_existing: bool = True,
    ):
        """
        Initialize all components.

        Args:
            load_existing: Whether to load existing indexes
        """

        logger.info("Initializing Islamic RAG System...")

        # Create base pipeline config
        pipeline_config = RAGConfig(
            datasets_path=self.config.datasets_path,
            output_path=self.config.output_path,
            embedding_model=self.config.embedding_model,
            embedding_device=self.config.embedding_device,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            retrieval_top_k=self.config.retrieval_top_k,
            rerank_top_k=self.config.rerank_top_k,
        )

        # Create pipeline
        self._pipeline = create_rag_pipeline(pipeline_config)

        # Try loading existing index
        if load_existing:
            try:
                self._pipeline.load_indexes()
                logger.info("Loaded existing index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

        # Initialize specialized scholars
        self._init_scholars()

        # Initialize advanced features
        self._init_advanced_features()

        # Initialize agents
        self._init_agents()

        # Initialize evaluator
        self._evaluator = create_islamic_evaluator(self._pipeline)

        self._initialized = True

        logger.info("Islamic RAG System initialized successfully")

    def _init_scholars(self):
        """Initialize domain specialists."""

        if not self._pipeline:
            return

        # Create scholars for each domain
        domains = [
            "quran",
            "hadith",
            "fiqh",
            "fiqh_hanafi",
            "fiqh_maliki",
            "fiqh_shafii",
            "fiqh_hanbali",
            "aqeedah",
            "arabic",
            "history",
            "spirituality",
            "literature",
        ]

        for domain in domains:
            self._scholars[domain] = create_islamic_scholar(domain, self._pipeline)

        # Comparative fiqh scholar
        self._comparative_fiqh = create_comparative_fiqh_scholar(self._pipeline)

        # Chain of scholarship reasoning
        self._chain_reasoning = ChainOfScholarship(self._pipeline)

        logger.info(f"Initialized {len(self._scholars)} domain specialists")

    def _init_advanced_features(self):
        """Initialize advanced features."""

        self._authority_ranker = create_authority_ranker()
        self._cross_refs = create_cross_reference_system()
        self._timeline = create_timeline_reconstructor()
        self._concept_extractor = create_concept_extractor()

        if self._pipeline:
            self._multi_hop = create_multi_hop_reasoning(self._pipeline)
            self._progressive = create_progressive_retrieval(self._pipeline)

        logger.info("Initialized advanced features")

    def _init_agents(self):
        """Initialize agent system."""

        roles = [
            "researcher",
            "student",
            "teacher",
            "fatwa",
            "comparator",
            "historian",
            "linguist",
        ]

        for role in roles:
            self._agents[role] = create_agent(role, self._pipeline)

        self._agent_team = AgentTeam(self._pipeline)

        logger.info(f"Initialized {len(self._agents)} agents")

    # ==================== Basic Query ====================

    async def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Basic RAG query.

        Args:
            question: User question
            top_k: Number of results

        Returns:
            Query result with answer and sources
        """

        self._ensure_initialized()

        result = await self._pipeline.query(question, top_k=top_k)

        # Apply authority ranking
        if self._authority_ranker and result.sources:
            result.sources = self._authority_ranker.rerank_by_authority(result.sources)

        return {
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
            "latency_ms": result.latency_ms,
        }

    # ==================== Domain-Specific Queries ====================

    async def query_as_scholar(
        self,
        domain: str,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Query using a domain specialist.

        Args:
            domain: Domain name (quran, hadith, fiqh, aqeedah, etc.)
            question: User question
            top_k: Number of results

        Returns:
            Domain-specific result
        """

        self._ensure_initialized()

        scholar = self._scholars.get(domain.lower())

        if not scholar:
            raise ValueError(f"Unknown domain: {domain}")

        result = await scholar.query(question, top_k=top_k)

        return {
            "domain": domain,
            "domain_name": scholar.config.name_ar,
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
        }

    async def query_tafsir(self, question: str) -> Dict[str, Any]:
        """Query specifically about Quranic exegesis."""
        return await self.query_as_scholar("quran", question)

    async def query_hadith(self, question: str) -> Dict[str, Any]:
        """Query specifically about Hadith."""
        return await self.query_as_scholar("hadith", question)

    async def query_fiqh(self, question: str) -> Dict[str, Any]:
        """Query specifically about jurisprudence."""
        return await self.query_as_scholar("fiqh", question)

    # ==================== Comparative Queries ====================

    async def compare_madhhabs(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Compare madhhab opinions on a topic.

        Args:
            question: Fiard's question

        Returns:
            Comparative analysis from all four madhhabs
        """

        self._ensure_initialized()

        if not self._comparative_fiqh:
            raise RuntimeError("Comparative fiqh not initialized")

        result = await self._comparative_fiqh.query_with_comparison(question)

        return result

    # ==================== Advanced Reasoning ====================

    async def reason_with_chain(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning on a question.

        Args:
            question: Complex question requiring reasoning

        Returns:
            Reasoning chain with conclusion
        """

        self._ensure_initialized()

        if not self._chain_reasoning:
            raise RuntimeError("Chain reasoning not initialized")

        result = await self._chain_reasoning.reason(question)

        return result

    async def reason_multihop(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning.

        Args:
            question: Complex question

        Returns:
            Multi-hop result
        """

        self._ensure_initialized()

        if not hasattr(self, "_multi_hop") or not self._multi_hop:
            # Fallback to basic query
            return await self.query(question)

        result = await self._multi_hop.reason(question)

        return result

    async def retrieve_progressive(
        self,
        question: str,
        target_size: int = 2000,
    ) -> Dict[str, Any]:
        """
        Progressively retrieve context until target size.

        Args:
            question: Question to retrieve context for
            target_size: Target context size in characters

        Returns:
            Progressive retrieval result
        """

        self._ensure_initialized()

        if not hasattr(self, "_progressive") or not self._progressive:
            # Fallback to basic query
            result = await self.query(question)
            return {
                "question": question,
                "context": result.get("answer", ""),
                "sources": result.get("sources", []),
            }

        result = await self._progressive.retrieve_progressive(question, target_size)

        return result

    # ==================== Agent Interactions ====================

    async def ask_as_researcher(self, question: str) -> Dict[str, Any]:
        """Ask as a research specialist."""

        self._ensure_initialized()

        agent = self._agents.get("researcher")
        if not agent:
            return await self.query(question)

        return await agent.execute("research", {"question": question})

    async def ask_as_student(self, topic: str) -> Dict[str, Any]:
        """Ask as a student for learning."""

        self._ensure_initialized()

        agent = self._agents.get("student")
        if not agent:
            return await self.query(topic)

        return await agent.execute("teach", {"topic": topic})

    async def ask_fatwa(self, question: str) -> Dict[str, Any]:
        """
        Ask a fiqh question (research, not real fatwa).

        Note: Always includes disclaimer that this is research.
        """

        self._ensure_initialized()

        agent = self._agents.get("fatwa")
        if not agent:
            return await self.query(question)

        return await agent.execute("get_fatwa", {"question": question})

    async def ask_agent(
        self,
        role: str,
        task: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Interact with a specific agent.

        Args:
            role: Agent role (researcher, student, teacher, etc.)
            task: Task to execute
            params: Task parameters

        Returns:
            Agent response
        """

        self._ensure_initialized()

        agent = self._agents.get(role.lower())

        if not agent:
            raise ValueError(f"Unknown agent role: {role}")

        return await agent.execute(task, params or {})

    async def collaborate_agents(
        self,
        task: str,
        roles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Have multiple agents collaborate on a task.

        Args:
            task: Task description
            roles: List of agent roles to involve

        Returns:
            Collaborative result
        """

        self._ensure_initialized()

        if not self._agent_team:
            raise RuntimeError("Agent team not initialized")

        # Convert role strings to enums
        role_enums = []
        if roles:
            role_map = {
                "researcher": AgentRole.RESEARCHER,
                "student": AgentRole.STUDENT,
                "teacher": AgentRole.TEACHER,
                "fatwa": AgentRole.FATWA_REQUESTER,
                "comparator": AgentRole.COMPARATOR,
                "historian": AgentRole.HISTORIAN,
                "linguist": AgentRole.LINGUIST,
            }

            for role in roles:
                if role.lower() in role_map:
                    role_enums.append(role_map[role.lower()])

        result = await self._agent_team.collaborate(task, role_enums)

        return result

    # ==================== Utility Methods ====================

    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract Islamic concepts from text."""

        if not self._concept_extractor:
            return []

        return self._concept_extractor.extract(text)

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> IslamicRAGEvaluator:
        """
        Evaluate an answer using Islamic-specific metrics.

        Args:
            question: Original question
            answer: Generated answer
            sources: Retrieved sources

        Returns:
            Evaluation metrics
        """

        self._ensure_initialized()

        if not self._evaluator:
            raise RuntimeError("Evaluator not initialized")

        return await self._evaluator.evaluate_full(question, answer, sources)

    async def index_documents(
        self,
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ):
        """Index documents from the dataset."""

        self._ensure_initialized()

        await self._pipeline.index_documents(
            limit=limit,
            categories=categories,
        )

        logger.info("Document indexing complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""

        if not self._pipeline:
            return {"initialized": False}

        base_stats = self._pipeline.get_stats()

        return {
            **base_stats,
            "initialized": self._initialized,
            "domains_available": list(self._scholars.keys()),
            "agents_available": list(self._agents.keys()),
        }

    def _ensure_initialized(self):
        """Ensure system is initialized."""

        if not self._initialized:
            raise RuntimeError("System not initialized. Call await initialize() first.")

    async def close(self):
        """Clean up resources."""

        logger.info("Shutting down Islamic RAG System...")
        # Add cleanup logic here if needed
        self._initialized = False


# ==================== Factory Function ====================


def create_islamic_rag(
    config: Optional[IslamicRAGConfig] = None,
    **kwargs,
) -> IslamicRAG:
    """
    Factory function to create Islamic RAG system.

    Args:
        config: Optional configuration
        **kwargs: Configuration overrides

    Returns:
        IslamicRAG instance
    """

    if config is None:
        config = IslamicRAGConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return IslamicRAG(config)


# ==================== Convenience Functions ====================


async def quick_query(question: str) -> Dict[str, Any]:
    """
    Quick query without full initialization.

    For testing purposes only.
    """

    rag = create_islamic_rag()
    await rag.initialize(load_existing=False)

    try:
        result = await rag.query(question)
        return result
    finally:
        await rag.close()


# ==================== Main Entry ====================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("Islamic RAG System - Demo")
        print("=" * 50)

        # Create system
        rag = create_islamic_rag()
        await rag.initialize()

        # Get stats
        stats = rag.get_stats()
        print(f"System Stats: {stats}")

        # Basic query
        print("\n--- Basic Query ---")
        result = await rag.query("ما هو التوحيد؟")
        print(f"Answer: {result['answer'][:200]}...")

        # Domain query
        print("\n--- Tafsir Query ---")
        result = await rag.query_tafsir("ما تفسير آية الكرسي؟")
        print(f"Domain: {result['domain_name']}")

        # Close
        await rag.close()

    asyncio.run(main())

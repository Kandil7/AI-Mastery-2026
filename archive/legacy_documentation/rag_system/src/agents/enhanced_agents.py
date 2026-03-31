"""
Enhanced Agents for Islamic Literature RAG - 2026

Advanced multi-agent system with:
- Specialized domain agents
- Agent collaboration protocols
- Tool use and function calling
- Memory and context management
- Islamic scholarly methodology
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import asyncio
import hashlib

logger = logging.getLogger(__name__)


# ==================== Enhanced Agent Roles ====================


class EnhancedAgentRole(Enum):
    """Enhanced agent roles with Islamic specialization."""

    # Core Research Roles
    MUHAQQIQ = "muhaqqiq"  # محقق - Deep researcher
    MUFTI = "mufti"  # مفتي - Fiqh researcher (not real fatwa)
    MUFASSIR = "mufassir"  # مفسر - Quranic exegesis specialist
    MUHADDITH = "muhaddith"  # محدث - Hadith specialist
    LUGHAWI = "lughawi"  # لغوي - Arabic linguist
    MUARRIKH = "muarrikh"  # مؤرخ - Islamic historian

    # Teaching Roles
    MURABBI = "murabbi"  # مربّي - Educator
    MUDARRIS = "mudarris"  # مدرّس - Teacher
    MURSHID = "murshid"  # مرشد - Guide

    # Analysis Roles
    MUQARIN = "muqarin"  # مقارن - Comparative scholar
    MUHAQIQ = "muhaqiq"  # محقق - Verifier
    MUNAQID = "munaqid"  # مناقش - Critical reviewer

    # Support Roles
    AMIN_AL_MAKTABA = "amin_maktaba"  # أمين المكتبة - Librarian
    MURAJJIH = "murajjih"  # مرجح - Preference giver (tarjeeh)


# ==================== Enhanced Agent Configuration ====================


@dataclass
class EnhancedAgentConfig:
    """Enhanced agent configuration."""

    role: EnhancedAgentRole
    name_ar: str
    name_en: str
    description_ar: str
    description_en: str

    # Methodology
    methodology: str = ""
    principles: List[str] = field(default_factory=list)

    # Tools
    available_tools: List[str] = field(default_factory=list)
    specialized_tools: List[str] = field(default_factory=list)

    # Prompts
    system_prompt_ar: str = ""
    system_prompt_en: str = ""

    # Constraints
    max_context_length: int = 4000
    max_sources: int = 10
    require_evidence: bool = True
    require_citations: bool = True

    # Islamic adab
    include_disclaimer: bool = False
    disclaimer_text: str = ""


# Enhanced agent configurations
ENHANCED_AGENT_CONFIGS = {
    EnhancedAgentRole.MUHAQQIQ: EnhancedAgentConfig(
        role=EnhancedAgentRole.MUHAQQIQ,
        name_ar="المحقق الإسلامي",
        name_en="Islamic Researcher",
        description_ar="باحث متخصص في التحقيق والاستدلال",
        description_en="Deep research specialist with evidence-based methodology",
        methodology="المنهج الاستدلالي التحقيقي",
        principles=[
            "البحث عن الأدلة من المصادر الأصيلة",
            "تقييم قوة الإسناد",
            "مقارنة الأقوال",
            "الترجيح بناء على قوة الدليل",
        ],
        available_tools=["rag_query", "multi_hop", "cross_reference"],
        specialized_tools=["authority_ranker", "source_verifier"],
        system_prompt_ar="""أنت محقق إسلامي متخصص في البحث العميق والاستدلال.

منهجك:
1. البحث عن الأدلة من المصادر الأصيلة (القرآن، السنة، آثار السلف)
2. تقييم قوة الأدلة والإسناد
3. مقارنة الأقوال المختلفة
4. الترجيح بناء على قوة الدليل

التزم بالدقة العلمية والموضوعية.""",
        require_evidence=True,
        require_citations=True,
    ),
    EnhancedAgentRole.MUFTI: EnhancedAgentConfig(
        role=EnhancedAgentRole.MUFTI,
        name_ar="الباحث الفقهي",
        name_en="Fiqh Researcher",
        description_ar="باحث متخصص في الاستنباط الفقهي",
        description_en="Fiqh research specialist (not a real mufti)",
        methodology="المنهج الاستنباطي الفقهي",
        principles=[
            "فهم السؤال بدقة",
            "البحث عن النصوص الشرعية",
            "فهم أقوال العلماء",
            "مراعاة المقاصد والضوابط",
        ],
        available_tools=["rag_query", "fiqh_specialist", "comparative_fiqh"],
        specialized_tools=["madhhab_finder", "ruling_extractor"],
        system_prompt_ar="""أنت باحث فقهي متخصص في الاستنباط من المصادر الشرعية.

منهجك:
1. فهم السؤال وتحليله
2. البحث عن النصوص (القرآن، السنة)
3. جمع أقوال العلماء
4. ذكر الأدلة والترجيحات

⚠️ تنبيه: هذا بحث علمي وليس فتوى.
للأسئلة الشخصية استشر عالماً متخصصاً.""",
        require_evidence=True,
        require_citations=True,
        include_disclaimer=True,
        disclaimer_text="""
⚠️ تنبيه مهم:
هذا البحث لأغراض علمية وتعليمية فقط، وليس فتوى شرعية.
للأسئلة الشخصية والاستفتاءات، يرجى مراجعة عالم متخصص أو جهة إفتاء معتمدة.""",
    ),
    EnhancedAgentRole.MUFASSIR: EnhancedAgentConfig(
        role=EnhancedAgentRole.MUFASSIR,
        name_ar="المفسر المتخصص",
        name_en="Tafsir Specialist",
        description_ar="متخصص في تفسير القرآن وعلومه",
        description_en="Quranic exegesis and sciences specialist",
        methodology="المنهج التفسيري",
        principles=[
            "تفسير القرآن بالقرآن",
            "تفسير القرآن بالسنة",
            "أقوال الصحابة والتابعين",
            "مراعاة اللغة العربية",
        ],
        available_tools=["rag_query", "quran_search", "tafsir_finder"],
        specialized_tools=["verse_finder", "qiraat_checker"],
        system_prompt_ar="""أنت مفسر متخصص في علوم القرآن والتفسير.

منهجك:
1. تفسير القرآن بالقرآن
2. تفسير القرآن بالسنة الصحيحة
3 أقوال الصحابة والتابعين
4. مراعاة قواعد اللغة العربية
5. ذكر أسباب النزول عند وجودها

التزم بمنهج السلف في التفسير.""",
        require_evidence=True,
        require_citations=True,
    ),
    EnhancedAgentRole.MUHADDITH: EnhancedAgentConfig(
        role=EnhancedAgentRole.MUHADDITH,
        name_ar="المحدث المتخصص",
        name_en="Hadith Specialist",
        description_ar="متخصص في علوم الحديث وتخريجه",
        description_en="Hadith sciences and verification specialist",
        methodology="المنهج الحديثي",
        principles=[
            "التحقق من صحة الإسناد",
            "معرفة درجات الحديث",
            "تخريج الأحاديث",
            "فهم متن الحديث",
        ],
        available_tools=["rag_query", "hadith_search", "grading_checker"],
        specialized_tools=["isnad_analyzer", "narrator_finder"],
        system_prompt_ar="""أنت محدث متخصص في علوم الحديث وتخريجه.

منهجك:
1. التحقق من صحة الإسناد
2. ذكر درجة الحديث (صحيح، حسن، ضعيف)
3. تخريج الحديث من مصادره
4. شرح معاني المفردات
5. استخراج الفوائد

التزم بالدقة في التخريج والتقييم.""",
        require_evidence=True,
        require_citations=True,
    ),
    EnhancedAgentRole.LUGHAWI: EnhancedAgentConfig(
        role=EnhancedAgentRole.LUGHAWI,
        name_ar="اللغوي المتخصص",
        name_en="Arabic Linguist",
        description_ar="متخصص في اللغة العربية وتحليل النصوص",
        description_en="Arabic language and text analysis specialist",
        methodology="المنهج اللغوي التحليلي",
        principles=[
            "تحليل الصرف والنحو",
            "شرح المعاني اللغوية",
            "ذكر الشواهد الشعرية",
            "تتبع الاستعمال",
        ],
        available_tools=["rag_query", "grammar_analyzer", "lexicon_search"],
        specialized_tools=["root_finder", "rhetoric_analyzer"],
        system_prompt_ar="""أنت لغوي متخصص في اللغة العربية وتحليل النصوص.

منهجك:
1. تحليل الصرف والنحو
2. شرح المعاني اللغوية
3. ذكر الشواهد من القرآن والشعر
4. تتبع الاستعمال عند العرب
5. شرح البلاغة والأساليب

اجعل التحليل واضحاً للطلاب.""",
        require_evidence=True,
        require_citations=False,
    ),
    EnhancedAgentRole.MUARRIKH: EnhancedAgentConfig(
        role=EnhancedAgentRole.MUARRIKH,
        name_ar="المؤرخ الإسلامي",
        name_en="Islamic Historian",
        description_ar="متخصص في التاريخ الإسلامي والتراجم",
        description_en="Islamic history and biography specialist",
        methodology="المنهج التاريخي النقدي",
        principles=[
            "التسلسل الزمني",
            "نقد الأسانيد التاريخية",
            "مقارنة المصادر",
            "السياق التاريخي",
        ],
        available_tools=["rag_query", "timeline_builder", "biography_search"],
        specialized_tools=["event_dater", "chain_verifier"],
        system_prompt_ar="""أنت مؤرخ متخصص في التاريخ الإسلامي.

منهجك:
1. ترتيب الأحداث زمنياً
2. نقد الأسانيد التاريخية
3. مقارنة المصادر المختلفة
4. ذكر السياق التاريخي
5. الربط بين الأحداث

كن موضوعياً ودقيقاً في النسب.""",
        require_evidence=True,
        require_citations=True,
    ),
    EnhancedAgentRole.MURABBI: EnhancedAgentConfig(
        role=EnhancedAgentRole.MURABBI,
        name_ar="المربي الإسلامي",
        name_en="Islamic Educator",
        description_ar="مربي ومتخصص في التربية الإسلامية",
        description_en="Islamic education and tarbiyah specialist",
        methodology="المنهج التربوي",
        principles=[
            "التدرج في التعليم",
            "الربط بين العلم والعمل",
            "التربية بالقدوة",
            "مراعاة الفروق الفردية",
        ],
        available_tools=["rag_query", "lesson_builder", "quiz_generator"],
        specialized_tools=["curriculum_planner", "progress_tracker"],
        system_prompt_ar="""أنت مربٍ إسلامي متخصص في التربية والتعليم.

منهجك:
1. التدرج من السهل إلى الصعب
2. الربط بين العلم والعمل
3. استخدام الأمثلة العملية
4. التشجيع والتحفيز
5. تقييم الفهم

كن حكيماً ومرفقاً في التربية.""",
        require_evidence=False,
        require_citations=False,
    ),
    EnhancedAgentRole.MUQARIN: EnhancedAgentConfig(
        role=EnhancedAgentRole.MUQARIN,
        name_ar="المقارن المتخصص",
        name_en="Comparative Scholar",
        description_ar="متخصص في الدراسات المقارنة",
        description_en="Comparative Islamic studies specialist",
        methodology="المنهج المقارن",
        principles=[
            "العدالة في العرض",
            "فهم أدلة كل قول",
            "بيان مواطن الاتفاق",
            "توضيح مواطن الخلاف",
        ],
        available_tools=["rag_query", "comparative_fiqh", "madhhab_finder"],
        specialized_tools=["consensus_checker", "tarjeeh_analyzer"],
        system_prompt_ar="""أنت باحث متخصص في الدراسات المقارنة.

منهجك:
1. العرض العادل لجميع الأقوال
2. فهم أدلة كل مذهب
3. بيان مواطن الإجماع
4. توضيح أسباب الخلاف
5. الترجيح بدون تعصب

كن منصفاً وموضوعياً.""",
        require_evidence=True,
        require_citations=True,
    ),
}


# ==================== Enhanced Islamic RAG Agent ====================


class EnhancedIslamicRAGAgent:
    """
    Enhanced Islamic RAG Agent with advanced capabilities.

    Features:
    - Role-specific methodology
    - Tool use and composition
    - Memory management
    - Islamic scholarly adab
    - Multi-step reasoning
    """

    def __init__(
        self,
        role: EnhancedAgentRole,
        pipeline: Any = None,
        config: Optional[EnhancedAgentConfig] = None,
    ):
        self.role = role
        self.pipeline = pipeline
        self.config = config or ENHANCED_AGENT_CONFIGS.get(role)

        if not self.config:
            raise ValueError(f"Unknown role: {role}")

        # Initialize tools
        self.tools: Dict[str, Any] = {}
        self._init_tools()

        # Memory
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = 10

    def _init_tools(self):
        """Initialize agent tools."""

        # Base tools
        self.tools = {
            "rag_query": self._rag_query,
            "multi_hop": self._multi_hop_reasoning,
            "cross_reference": self._cross_reference,
        }

        # Specialized tools based on role
        if self.role in [
            EnhancedAgentRole.MUHAQQIQ,
            EnhancedAgentRole.MUHAQIQ,
        ]:
            from ..specialists.advanced_features import create_authority_ranker

            self.tools["authority_ranker"] = create_authority_ranker()

        if self.role == EnhancedAgentRole.MUFTI:
            from ..specialists.islamic_scholars import create_islamic_scholar

            self.tools["fiqh_specialist"] = create_islamic_scholar(
                "fiqh", self.pipeline
            )
            self.tools["comparative_fiqh"] = create_islamic_scholar(
                "fiqh_comparative", self.pipeline
            )

        if self.role == EnhancedAgentRole.MUFASSIR:
            from ..specialists.islamic_scholars import create_islamic_scholar

            self.tools["quran_specialist"] = create_islamic_scholar(
                "quran", self.pipeline
            )

        if self.role == EnhancedAgentRole.MUHADDITH:
            from ..specialists.islamic_scholars import create_islamic_scholar

            self.tools["hadith_specialist"] = create_islamic_scholar(
                "hadith", self.pipeline
            )

    async def _rag_query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the RAG pipeline."""

        if not self.pipeline:
            return {"error": "Pipeline not available"}

        try:
            result = await self.pipeline.query(query, top_k=top_k, filters=filters)
            return {
                "query": query,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "latency_ms": result.get("latency_ms", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _multi_hop_reasoning(self, question: str) -> Dict[str, Any]:
        """Perform multi-hop reasoning."""

        if not self.pipeline:
            return {"error": "Pipeline not available"}

        try:
            from ..specialists.advanced_features import create_multi_hop_reasoning

            reasoner = create_multi_hop_reasoning(self.pipeline)
            result = await reasoner.reason(question)

            return result
        except Exception as e:
            return {"error": str(e)}

    async def _cross_reference(self, topic: str) -> Dict[str, Any]:
        """Find cross-references on a topic."""

        if not self.pipeline:
            return {"error": "Pipeline not available"}

        try:
            from ..specialists.advanced_features import create_cross_reference_system

            cross_refs = create_cross_reference_system()
            result = await cross_refs.find_cross_references(topic, self.pipeline)

            return result
        except Exception as e:
            return {"error": str(e)}

    async def execute(
        self,
        task: str,
        params: Optional[Dict[str, Any]] = None,
        use_methodology: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a task using agent's methodology.

        Args:
            task: Task description
            params: Task parameters
            use_methodology: Apply role-specific methodology

        Returns:
            Task result with scholarly approach
        """

        params = params or {}

        # Add to conversation history
        self._add_to_history("user", task, params)

        # Execute based on role
        result = await self._execute_by_role(task, params)

        # Apply methodology if requested
        if use_methodology and self.config.methodology:
            result = await self._apply_methodology(result, task)

        # Add disclaimer if required
        if self.config.include_disclaimer:
            result["disclaimer"] = self.config.disclaimer_text

        # Add to history
        self._add_to_history("assistant", result)

        return result

    async def _execute_by_role(
        self,
        task: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task based on role specialization."""

        role_handlers = {
            EnhancedAgentRole.MUHAQQIQ: self._research_task,
            EnhancedAgentRole.MUFTI: self._fiqh_task,
            EnhancedAgentRole.MUFASSIR: self._tafsir_task,
            EnhancedAgentRole.MUHADDITH: self._hadith_task,
            EnhancedAgentRole.LUGHAWI: self._linguistic_task,
            EnhancedAgentRole.MUARRIKH: self._history_task,
            EnhancedAgentRole.MURABBI: self._education_task,
            EnhancedAgentRole.MUQARIN: self._comparative_task,
        }

        handler = role_handlers.get(self.role)

        if handler:
            return await handler(task, params)
        else:
            return await self._research_task(task, params)

    async def _research_task(
        self,
        question: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep research task."""

        # Use multi-hop reasoning
        if "multi_hop" in self.tools:
            result = await self.tools["multi_hop"](question)
        else:
            result = await self._rag_query(question)

        # Rank by authority
        if "authority_ranker" in self.tools and "sources" in result:
            ranked = self.tools["authority_ranker"].rerank_by_authority(
                result["sources"]
            )
            result["ranked_sources"] = ranked

        return result

    async def _fiqh_task(
        self,
        question: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fiqh research task."""

        # Use fiqh specialist
        if "fiqh_specialist" in self.tools:
            result = await self.tools["fiqh_specialist"].query(question)
        else:
            result = await self._rag_query(question)

        # Get comparative view
        if "comparative_fiqh" in self.tools:
            comparative = await self.tools["comparative_fiqh"].query_with_comparison(
                question
            )
            result["comparative"] = comparative

        return result

    async def _tafsir_task(
        self,
        question: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tafsir task."""

        # Search for verse
        if "quran_specialist" in self.tools:
            result = await self.tools["quran_specialist"].query(question)
        else:
            result = await self._rag_query(f"تفسير {question}")

        return result

    async def _hadith_task(
        self,
        question: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hadith task."""

        if "hadith_specialist" in self.tools:
            result = await self.tools["hadith_specialist"].query(question)
        else:
            result = await self._rag_query(f"حديث {question}")

        return result

    async def _linguistic_task(
        self,
        text: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Linguistic analysis task."""

        # Analyze grammar and meaning
        result = await self._rag_query(f"تحليل لغوي: {text}")

        return result

    async def _history_task(
        self,
        topic: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Historical research task."""

        result = await self._rag_query(f"تاريخ {topic}")

        # Build timeline if available
        if "timeline_builder" in self.tools:
            timeline = await self.tools["timeline_builder"](topic)
            result["timeline"] = timeline

        return result

    async def _education_task(
        self,
        topic: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Educational task."""

        # Build lesson plan
        result = await self._rag_query(f"أساسيات {topic}")

        lesson = {
            "topic": topic,
            "objectives": [
                f"فهم أساسيات {topic}",
                f"معرفة المفاهيم الرئيسية",
                f"التطبيق العملي",
            ],
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
            "questions": [
                f"ما هو {topic}؟",
                f"ما أهمية {topic}؟",
                f"كيف نطبق {topic}؟",
            ],
        }

        return lesson

    async def _comparative_task(
        self,
        topic: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Comparative research task."""

        if "comparative_fiqh" in self.tools:
            result = await self.tools["comparative_fiqh"].query_with_comparison(topic)
        else:
            result = await self._rag_query(f"آراء المذاهب في {topic}")

        return result

    async def _apply_methodology(
        self,
        result: Dict[str, Any],
        task: str,
    ) -> Dict[str, Any]:
        """Apply role-specific methodology to result."""

        # Add methodology information
        result["methodology"] = self.config.methodology
        result["principles"] = self.config.principles

        return result

    def _add_to_history(
        self,
        role: str,
        content: Any,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Add to conversation history."""

        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "params": params,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Trim if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def get_context(self) -> str:
        """Get conversation context for LLM."""

        if not self.conversation_history:
            return ""

        context_parts = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}")
            else:
                context_parts.append(f"Assistant: {json.dumps(msg['content'], ensure_ascii=False)}")

        return "\n".join(context_parts)


# ==================== Agent Team with Collaboration ====================


class EnhancedAgentTeam:
    """
    Enhanced agent team with collaboration protocols.

    Supports:
    - Sequential collaboration
    - Parallel consultation
    - Consensus building
    - Conflict resolution
    """

    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline
        self.agents: Dict[EnhancedAgentRole, EnhancedIslamicRAGAgent] = {}

        # Initialize all agents
        for role in EnhancedAgentRole:
            self.agents[role] = EnhancedIslamicRAGAgent(role, pipeline)

    async def collaborate_sequential(
        self,
        task: str,
        roles: List[EnhancedAgentRole],
    ) -> Dict[str, Any]:
        """
        Sequential collaboration: each agent builds on previous.

        Args:
            task: Task description
            roles: Ordered list of agent roles

        Returns:
            Combined result
        """

        current_result = None

        for role in roles:
            agent = self.agents.get(role)
            if not agent:
                continue

            # Build on previous result
            if current_result:
                enhanced_task = f"{task}\n\nPrevious analysis: {json.dumps(current_result, ensure_ascii=False)[:500]}"
            else:
                enhanced_task = task

            result = await agent.execute(enhanced_task)
            current_result = result

        return current_result or {"error": "No agents available"}

    async def consult_parallel(
        self,
        task: str,
        roles: List[EnhancedAgentRole],
    ) -> Dict[str, Any]:
        """
        Parallel consultation: all agents respond independently.

        Args:
            task: Task description
            roles: List of agent roles to consult

        Returns:
            Combined opinions
        """

        # Run all agents in parallel
        tasks = []
        for role in roles:
            agent = self.agents.get(role)
            if agent:
                tasks.append(agent.execute(task))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined = {
            "task": task,
            "consultations": {},
        }

        for role, result in zip(roles, results):
            if isinstance(result, Exception):
                combined["consultations"][role.value] = {"error": str(result)}
            else:
                combined["consultations"][role.value] = result

        # Synthesize
        combined["synthesis"] = self._synthesize_opinions(combined["consultations"])

        return combined

    async def build_consensus(
        self,
        task: str,
        roles: Optional[List[EnhancedAgentRole]] = None,
    ) -> Dict[str, Any]:
        """
        Build consensus among agents.

        Args:
            task: Task description
            roles: Roles to include (default: all research roles)

        Returns:
            Consensus result with areas of agreement/disagreement
        """

        if not roles:
            roles = [
                EnhancedAgentRole.MUHAQQIQ,
                EnhancedAgentRole.MUFTI,
                EnhancedAgentRole.MUQARIN,
            ]

        # Get all opinions
        consultations = await self.consult_parallel(task, roles)

        # Find areas of agreement
        consensus = self._find_consensus(consultations["consultations"])

        return {
            "task": task,
            "consensus": consensus,
            "disagreements": self._find_disagreements(consultations["consultations"]),
            "consultations": consultations["consultations"],
        }

    def _synthesize_opinions(
        self,
        opinions: Dict[str, Any],
    ) -> str:
        """Synthesize multiple opinions into unified response."""

        synthesis = "##综合意见\n\n"

        for role, opinion in opinions.items():
            if "error" not in opinion:
                synthesis += f"**{role}:**\n"
                answer = opinion.get("answer", "")
                synthesis += f"{answer[:300]}...\n\n"

        return synthesis

    def _find_consensus(
        self,
        opinions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Find areas of consensus among opinions."""

        # Placeholder implementation
        return {
            "has_consensus": True,
            "consensus_points": ["Point 1", "Point 2"],
            "confidence": 0.8,
        }

    def _find_disagreements(
        self,
        opinions: Dict[str, Any],
    ) -> List[str]:
        """Find areas of disagreement."""

        # Placeholder implementation
        return []


# ==================== Factory Functions ====================


def create_enhanced_agent(
    role: str,
    pipeline: Any = None,
) -> EnhancedIslamicRAGAgent:
    """Create an enhanced agent."""

    role_map = {
        "muhaqqiq": EnhancedAgentRole.MUHAQQIQ,
        "mufti": EnhancedAgentRole.MUFTI,
        "mufassir": EnhancedAgentRole.MUFASSIR,
        "muhaddith": EnhancedAgentRole.MUHADDITH,
        "lughawi": EnhancedAgentRole.LUGHAWI,
        "muarrikh": EnhancedAgentRole.MUARRIKH,
        "murabbi": EnhancedAgentRole.MURABBI,
        "muqarin": EnhancedAgentRole.MUQARIN,
    }

    role_enum = role_map.get(role.lower())

    if not role_enum:
        raise ValueError(f"Unknown role: {role}")

    return EnhancedIslamicRAGAgent(role_enum, pipeline)


def create_enhanced_agent_team(pipeline: Any = None) -> EnhancedAgentTeam:
    """Create an enhanced agent team."""
    return EnhancedAgentTeam(pipeline)


# Import datetime
from datetime import datetime

# Import Any from typing
from typing import Any

"""
Semantic Router Service
=======================
Classifies user queries to optimize cost and performance.

خدمة التوجيه الدلالي لتصنيف الاستفسارات وتحسين الأداء والتكلفة
"""

import structlog
from enum import Enum
from src.application.ports.llm import LLMPort

log = structlog.get_logger()

class QueryIntent(Enum):
    CHITCHAT = "chitchat"           # Greetings, thanks, small talk
    DOCUMENT_QNA = "document_qna"   # Question about uploaded documents
    LIVE_INFO = "live_info"         # Questions needing current events/web search
    REASONING = "reasoning"         # Multi-hop questions (GraphRAG)

class SemanticRouterService:
    """
    Intelligent router to decide which pipeline to execute.
    
    قرار التصميم: تقليل هدر المصادر عبر توجيه السؤال للمسار الصحيح فوراً
    """

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def route(self, question: str) -> QueryIntent:
        """
        Determine the intent of the user question.
        """
        prompt = (
            "Classify the user question into one of these categories:\n"
            "- 'chitchat': Greetings, personal questions, or polite remarks.\n"
            "- 'document_qna': Factual questions about specific documents or data.\n"
            "- 'live_info': Questions about today's news, current prices, or live data.\n"
            "- 'reasoning': Complex questions requiring connecting multiple pieces of info.\n\n"
            f"Question: {question}\n\n"
            "Response (output ONLY the category name):"
        )
        
        try:
            response = self._llm.generate(prompt, temperature=0.0).lower().strip()
            
            # Map response to enum
            for intent in QueryIntent:
                if intent.value in response:
                    log.info("query_routed", intent=intent.value)
                    return intent
            
            # Default to QnA if unsure
            return QueryIntent.DOCUMENT_QNA
            
        except Exception as e:
            log.warning("routing_failed", error=str(e))
            return QueryIntent.DOCUMENT_QNA

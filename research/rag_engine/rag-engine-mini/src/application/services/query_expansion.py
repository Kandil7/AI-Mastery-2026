"""
Query Expansion Service
========================
Generates related queries to improve retrieval recall.

خدمة توسيع الاستعلام - توليد استعلامات متعلقة لتحسين الاستدعاء
"""

from typing import Sequence

from src.application.ports.llm import LLMPort


class QueryExpansionService:
    """
    Service for expanding a single query into multiple related queries.
    
    Helps catch synonyms and related concepts that might not match
    the original query exactly.
    
    توسيع الاستعلام للمساعدة في العثور على المرادفات والمفاهيم المتعلقة
    """
    
    EXPANSION_PROMPT = """Generate 3 related search queries for the following question to improve retrieval in a RAG system.
Focus on different aspects or synonyms.
Only output the queries, one per line, no numbering.

Question: {question}

Queries:"""
    
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm
    
    def expand(self, question: str) -> list[str]:
        """
        Generate 3 related queries for the given question.
        
        Args:
            question: Original question
            
        Returns:
            List of expanded queries (including the original)
        """
        try:
            prompt = self.EXPANSION_PROMPT.format(question=question)
            response = self._llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            expanded = [q.strip() for q in response.split("\n") if q.strip()]
            # Ensure unique and include original
            results = list(set([question] + expanded))
            return results[:4]  # Limit to 4 total
            
        except Exception:
            # Fallback to original query on error
            return [question]

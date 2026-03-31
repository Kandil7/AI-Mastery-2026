"""
Self-Critique Service (Self-RAG)
================================
Logic for evaluating retrieval quality and answer truthfulness.

خدمة التقييم الذاتي لضمان دقة الردود ومنع الهلوسة
"""

import structlog
from src.application.ports.llm import LLMPort
from src.domain.entities import Chunk

log = structlog.get_logger()

class SelfCritiqueService:
    """
    Self-RAG Evaluator based on LLM feedback.
    
    قرار التصميم: استخدام الـ LLM كمقيم (LLM-as-a-Judge) لاتخاذ قرارات المسار
    """

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def grade_retrieval(self, question: str, chunks: list[Chunk]) -> str:
        """
        Grade if the retrieved chunks contain enough info to answer.
        Returns: "relevant" | "irrelevant"
        """
        if not chunks:
            return "irrelevant"

        context = "\n\n".join([c.text for c in chunks])
        prompt = (
            "You are a grader evaluating the relevance of retrieved documents to a user question.\n"
            f"User Question: {question}\n\n"
            f"Retrieved Documents:\n{context[:6000]}\n\n"
            "If the documents contain information that can answer the question, respond with 'relevant'.\n"
            "Otherwise, respond with 'irrelevant'. respond with ONLY one word."
        )
        
        response = self._llm.generate(prompt, temperature=0.0).lower().strip()
        result = "relevant" if "relevant" in response else "irrelevant"
        
        log.info("retrieval_graded", result=result, chunk_count=len(chunks))
        return result

    def grade_answer(self, question: str, answer: str, chunks: list[Chunk]) -> str:
        """
        Grade if the answer is grounded in the context (no hallucination).
        Returns: "grounded" | "hallucination"
        """
        context = "\n\n".join([c.text for c in chunks])
        prompt = (
            "You are a grader evaluating if an answer is grounded in / supported by a set of facts.\n"
            f"Facts:\n{context[:6000]}\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Is the answer fully supported by the facts? Respond with 'grounded'.\n"
            "If there is ANY info in the answer not in the facts, respond with 'hallucination'.\n"
            "Respond with ONLY one word."
        )
        
        response = self._llm.generate(prompt, temperature=0.0).lower().strip()
        result = "grounded" if "grounded" in response else "hallucination"
        
        log.info("answer_graded", result=result)
        return result

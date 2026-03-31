"""
LLM Reranker Adapter
=====================
Reranking using LLM as fallback when Cross-Encoder is unavailable.

محول إعادة الترتيب باستخدام نموذج اللغة
"""

from typing import Sequence

from src.application.ports.llm import LLMPort
from src.domain.entities import Chunk


class LLMReranker:
    """
    LLM-based reranker using prompt-based scoring.
    
    Uses LLM to score relevance of each chunk to query.
    Slower and more expensive than Cross-Encoder but works with any LLM.
    
    أبطأ وأكثر تكلفة من Cross-Encoder لكن يعمل مع أي نموذج لغة
    """
    
    SCORING_PROMPT = """Rate the relevance of the following passage to the query on a scale of 1-10.
Only respond with a single number.

Query: {query}

Passage: {passage}

Relevance score (1-10):"""
    
    def __init__(self, llm: LLMPort) -> None:
        """
        Initialize with LLM.
        
        Args:
            llm: LLM port for scoring
        """
        self._llm = llm
    
    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[Chunk],
        top_n: int,
    ) -> Sequence[Chunk]:
        """
        Rerank chunks by LLM-scored relevance.
        
        Note: Makes one LLM call per chunk, can be slow/expensive.
        Consider batching or using Cross-Encoder instead.
        """
        if not chunks:
            return []
        
        scored: list[tuple[Chunk, float]] = []
        
        for chunk in chunks:
            try:
                prompt = self.SCORING_PROMPT.format(
                    query=query,
                    passage=chunk.text[:500],  # Truncate for efficiency
                )
                
                response = self._llm.generate(
                    prompt,
                    temperature=0.0,
                    max_tokens=5,
                )
                
                # Parse score from response
                score = self._parse_score(response)
                scored.append((chunk, score))
                
            except Exception:
                # On error, give middle score
                scored.append((chunk, 5.0))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in scored[:top_n]]
    
    def _parse_score(self, response: str) -> float:
        """Parse numeric score from LLM response."""
        try:
            # Extract first number from response
            text = response.strip()
            for word in text.split():
                try:
                    score = float(word.strip(".,"))
                    if 1 <= score <= 10:
                        return score
                except ValueError:
                    continue
            return 5.0  # Default middle score
        except Exception:
            return 5.0

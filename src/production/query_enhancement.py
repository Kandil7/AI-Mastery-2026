"""
Query Enhancement for Production RAG Systems
=============================================

Implements advanced query understanding and enhancement techniques
to bridge the vocabulary gap between user queries and indexed documents.

Key Components:
- QueryRewriter: LLM-based query optimization
- HyDEGenerator: Hypothetical Document Embeddings
- MultiQueryGenerator: Parallel query variants
- SynonymExpander: Term expansion

Reference: "From Prototype to Production: Enterprise RAG Systems"
"""

import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import numpy as np


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class EnhancedQuery:
    """
    An enhanced query with variants and metadata.
    
    Production systems don't just pass raw queries - they enrich them.
    """
    original: str
    rewritten: Optional[str] = None
    variants: List[str] = field(default_factory=list)
    hypothetical_doc: Optional[str] = None
    expanded_terms: Dict[str, List[str]] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_queries(self) -> List[str]:
        """Get all query variants for retrieval."""
        queries = [self.original]
        if self.rewritten:
            queries.append(self.rewritten)
        queries.extend(self.variants)
        return list(set(queries))


# ============================================================
# QUERY ENHANCEMENT STRATEGIES
# ============================================================

class BaseQueryEnhancer(ABC):
    """Base class for query enhancement strategies."""
    
    @abstractmethod
    def enhance(self, query: str) -> EnhancedQuery:
        """Enhance a raw query."""
        pass


class QueryRewriter:
    """
    LLM-based query rewriting.
    
    Production Pattern:
    Transform conversational queries into optimized search queries.
    
    Example:
        Input: "Can you tell me about our sales last quarter?"
        Output: ["Q3 sales report", "quarterly revenue analysis", 
                 "financial performance last quarter"]
    
    Benefits:
    - Bridges vocabulary gap
    - Removes conversational filler
    - Generates keyword-rich variants
    """
    
    # System prompt for query rewriting (would use LLM in production)
    REWRITE_PROMPT = """
    Rewrite this conversational query into 3 optimized search queries.
    Focus on:
    1. Key terms and concepts
    2. Domain-specific terminology
    3. Removing filler words
    
    Query: {query}
    
    Optimized queries:
    """
    
    def __init__(self, llm_caller: Optional[Callable] = None):
        """
        Args:
            llm_caller: Function that calls LLM with a prompt
                       In production: openai.chat, anthropic.messages, etc.
        """
        self.llm_caller = llm_caller
    
    def rewrite(self, query: str) -> List[str]:
        """
        Rewrite query into optimized variants.
        
        Falls back to rule-based rewriting if no LLM.
        """
        if self.llm_caller:
            return self._llm_rewrite(query)
        return self._rule_based_rewrite(query)
    
    def _llm_rewrite(self, query: str) -> List[str]:
        """Use LLM to generate query variants."""
        prompt = self.REWRITE_PROMPT.format(query=query)
        response = self.llm_caller(prompt)
        
        # Parse response into list of queries
        lines = response.strip().split('\n')
        variants = []
        for line in lines:
            # Remove numbering and clean
            clean = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            if clean and len(clean) > 3:
                variants.append(clean)
        
        return variants[:5]  # Limit to 5 variants
    
    def _rule_based_rewrite(self, query: str) -> List[str]:
        """Rule-based query rewriting (fallback)."""
        variants = []
        
        # Remove question words
        cleaned = re.sub(
            r'^(what|where|when|why|how|can you|could you|please|tell me|show me)\s+',
            '', query.lower(), flags=re.IGNORECASE
        )
        cleaned = cleaned.strip('?').strip()
        
        if cleaned != query.lower():
            variants.append(cleaned)
        
        # Extract noun phrases (simplified)
        words = query.split()
        if len(words) >= 3:
            # Last 3-4 words often contain the key concept
            variants.append(' '.join(words[-3:]))
        
        # Remove common filler
        fillers = ['about', 'regarding', 'concerning', 'related to']
        for filler in fillers:
            if filler in query.lower():
                parts = query.lower().split(filler)
                if len(parts) > 1 and parts[1].strip():
                    variants.append(parts[1].strip())
        
        return list(set(variants))


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE).
    
    Production Pattern:
    Instead of embedding the query directly, generate a hypothetical
    answer/document and embed that. This creates embeddings that are
    more similar to actual documents.
    
    Example:
        Query: "What are the risks of AI adoption?"
        
        Hypothetical Doc: "AI adoption carries several risks including
        data privacy concerns, algorithmic bias, security vulnerabilities,
        workforce displacement, and regulatory compliance challenges..."
        
        Then embed the hypothetical doc for retrieval.
    
    Benefits:
    - Better semantic alignment with documents
    - Improved recall for conceptual queries
    - Especially effective for question-answering
    """
    
    HYDE_PROMPT = """
    Write a detailed paragraph that would be a good answer to this question.
    The paragraph should contain relevant facts and terminology that would
    appear in an actual document answering this question.
    
    Question: {query}
    
    Hypothetical document:
    """
    
    def __init__(
        self, 
        llm_caller: Optional[Callable] = None,
        embedder: Optional[Callable] = None
    ):
        self.llm_caller = llm_caller
        self.embedder = embedder
    
    def generate(self, query: str) -> str:
        """Generate hypothetical document for query."""
        if self.llm_caller:
            return self._llm_generate(query)
        return self._template_generate(query)
    
    def _llm_generate(self, query: str) -> str:
        """Use LLM to generate hypothetical document."""
        prompt = self.HYDE_PROMPT.format(query=query)
        return self.llm_caller(prompt)
    
    def _template_generate(self, query: str) -> str:
        """Template-based generation (fallback)."""
        # Simple template expansion
        templates = [
            "This document discusses {topic}. The key aspects include...",
            "Regarding {topic}, the important considerations are...",
            "The {topic} involves several components: first...",
        ]
        
        # Extract likely topic from query
        topic = re.sub(r'^(what|how|why|when|where|who)\s+(is|are|was|were|do|does)\s+',
                      '', query.lower())
        topic = topic.strip('?').strip()
        
        template = templates[hash(query) % len(templates)]
        return template.format(topic=topic)
    
    def get_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for hypothetical document."""
        if not self.embedder:
            return None
        
        hypothetical = self.generate(query)
        return self.embedder(hypothetical)


class MultiQueryGenerator:
    """
    Multi-Query Retrieval.
    
    Production Pattern:
    Generate diverse query variants, retrieve for each, then merge results.
    
    This increases recall by approaching the question from multiple angles:
    - Synonyms and paraphrases
    - Different levels of specificity
    - Alternative phrasings
    
    Example:
        Original: "machine learning best practices"
        
        Variants:
        1. "ML engineering guidelines"
        2. "how to train models effectively"  
        3. "machine learning tips and techniques"
        4. "deep learning best practices"
    """
    
    MULTI_QUERY_PROMPT = """
    Generate 4 different search queries that would help find documents 
    answering this question. Use different angles and phrasings.
    
    Original query: {query}
    
    Variant queries:
    """
    
    def __init__(self, llm_caller: Optional[Callable] = None):
        self.llm_caller = llm_caller
    
    def generate(self, query: str, n: int = 4) -> List[str]:
        """Generate multiple query variants."""
        if self.llm_caller:
            return self._llm_generate(query, n)
        return self._rule_based_generate(query, n)
    
    def _llm_generate(self, query: str, n: int) -> List[str]:
        """Use LLM to generate diverse queries."""
        prompt = self.MULTI_QUERY_PROMPT.format(query=query)
        response = self.llm_caller(prompt)
        
        lines = response.strip().split('\n')
        variants = []
        for line in lines:
            clean = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            if clean and len(clean) > 5:
                variants.append(clean)
        
        return variants[:n]
    
    def _rule_based_generate(self, query: str, n: int) -> List[str]:
        """Rule-based multi-query generation."""
        variants = [query]
        words = query.lower().split()
        
        # Synonym substitution maps
        synonyms = {
            'best': ['top', 'recommended', 'optimal'],
            'practices': ['methods', 'techniques', 'approaches'],
            'how': ['ways', 'methods', 'steps'],
            'machine learning': ['ML', 'deep learning', 'AI'],
            'problem': ['issue', 'challenge', 'difficulty'],
            'solution': ['fix', 'resolution', 'answer'],
        }
        
        # Generate variants with synonyms
        for original, subs in synonyms.items():
            if original in query.lower():
                for sub in subs[:2]:
                    variant = re.sub(original, sub, query, flags=re.IGNORECASE)
                    if variant != query:
                        variants.append(variant)
        
        # Add a more specific variant
        if len(words) >= 3:
            variants.append(query + " examples")
        
        # Add a broader variant
        if len(words) >= 4:
            variants.append(' '.join(words[:3]))
        
        return list(set(variants))[:n]


class SynonymExpander:
    """
    Term expansion with synonyms.
    
    Production Pattern:
    Automatically expand query terms with synonyms to improve recall.
    
    Example:
        "annual revenue" → also search for:
        - "yearly sales"
        - "fiscal income"
        - "yearly revenue"
    """
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        self.synonym_dict = synonym_dict or self._default_synonyms()
    
    def _default_synonyms(self) -> Dict[str, List[str]]:
        """Default domain-agnostic synonyms."""
        return {
            # Business
            "revenue": ["sales", "income", "earnings"],
            "cost": ["expense", "spending", "expenditure"],
            "profit": ["earnings", "margin", "gain"],
            "annual": ["yearly", "fiscal year"],
            "quarterly": ["Q1 Q2 Q3 Q4", "three-month"],
            
            # Tech
            "api": ["endpoint", "interface", "service"],
            "error": ["bug", "issue", "failure", "exception"],
            "performance": ["speed", "efficiency", "throughput"],
            "security": ["safety", "protection", "authentication"],
            
            # ML
            "model": ["algorithm", "classifier", "predictor"],
            "training": ["learning", "fitting", "optimization"],
            "accuracy": ["precision", "performance", "quality"],
            "data": ["dataset", "information", "records"],
        }
    
    def expand(self, query: str) -> Dict[str, List[str]]:
        """
        Expand query terms with synonyms.
        
        Returns:
            Dict mapping original terms to their synonyms
        """
        expansions = {}
        query_lower = query.lower()
        
        for term, synonyms in self.synonym_dict.items():
            if term in query_lower:
                expansions[term] = synonyms
        
        return expansions
    
    def get_expanded_query(self, query: str) -> str:
        """Get query with synonyms appended."""
        expansions = self.expand(query)
        
        if not expansions:
            return query
        
        # Append synonyms in parentheses
        extra_terms = []
        for term, synonyms in expansions.items():
            extra_terms.extend(synonyms[:2])  # Limit per term
        
        return f"{query} ({' '.join(extra_terms)})"


# ============================================================
# UNIFIED QUERY ENHANCEMENT PIPELINE
# ============================================================

class QueryEnhancementPipeline:
    """
    Complete query enhancement pipeline.
    
    Orchestrates multiple enhancement strategies:
    1. Query Rewriting: Clean and optimize
    2. Multi-Query: Generate variants
    3. Synonym Expansion: Broaden coverage
    4. HyDE: Hypothetical document (optional)
    
    Example:
        pipeline = QueryEnhancementPipeline()
        enhanced = pipeline.enhance("What are ML best practices?")
        
        # Use enhanced.get_all_queries() for retrieval
    """
    
    def __init__(
        self,
        rewriter: Optional[QueryRewriter] = None,
        multi_query: Optional[MultiQueryGenerator] = None,
        synonym_expander: Optional[SynonymExpander] = None,
        hyde: Optional[HyDEGenerator] = None,
        use_hyde: bool = False
    ):
        self.rewriter = rewriter or QueryRewriter()
        self.multi_query = multi_query or MultiQueryGenerator()
        self.synonym_expander = synonym_expander or SynonymExpander()
        self.hyde = hyde or HyDEGenerator()
        self.use_hyde = use_hyde
    
    def enhance(self, query: str) -> EnhancedQuery:
        """
        Enhance a query with all strategies.
        
        Args:
            query: Raw user query
            
        Returns:
            EnhancedQuery with variants and metadata
        """
        enhanced = EnhancedQuery(original=query)
        
        # Step 1: Rewrite for optimization
        rewrites = self.rewriter.rewrite(query)
        if rewrites:
            enhanced.rewritten = rewrites[0]
            enhanced.variants.extend(rewrites[1:])
        
        # Step 2: Generate multi-query variants
        multi_variants = self.multi_query.generate(query, n=3)
        enhanced.variants.extend(multi_variants)
        
        # Step 3: Synonym expansion
        enhanced.expanded_terms = self.synonym_expander.expand(query)
        
        # Step 4: HyDE (optional, expensive)
        if self.use_hyde:
            enhanced.hypothetical_doc = self.hyde.generate(query)
        
        # Dedupe variants
        enhanced.variants = list(set(enhanced.variants))
        
        # Add metadata
        enhanced.metadata = {
            "enhancement_count": len(enhanced.variants),
            "has_hyde": self.use_hyde,
            "expanded_terms_count": len(enhanced.expanded_terms)
        }
        
        return enhanced


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Models
    "EnhancedQuery",
    # Enhancers
    "BaseQueryEnhancer",
    "QueryRewriter",
    "HyDEGenerator",
    "MultiQueryGenerator",
    "SynonymExpander",
    # Pipeline
    "QueryEnhancementPipeline",
]

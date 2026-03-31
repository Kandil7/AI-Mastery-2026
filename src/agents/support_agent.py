"""
Support Agent Module
====================

Production support agent with hallucination guardrails.

Inspired by Intercom Fin architecture:
- Content Scope Enforcement: Only answer from approved sources
- Hallucination Prevention: Confidence thresholds + fallback
- Source Citation: Every answer includes references
- CX Score: Automated conversation quality scoring

Features:
- Strict content boundary enforcement
- Multi-level confidence scoring
- Automatic human handoff for low confidence
- Sentiment and resolution analysis

References:
- Intercom Fin architecture case study
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
import re
import logging
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class ConversationState(Enum):
    """State of a support conversation."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


class SentimentType(Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"


@dataclass
class SupportArticle:
    """A support knowledge base article."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    
@dataclass
class Message:
    """A message in a conversation."""
    id: str
    role: str  # "user" or "agent"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)  # Article IDs
    confidence: float = 1.0


@dataclass
class Conversation:
    """A support conversation."""
    id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    state: ConversationState = ConversationState.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedSource:
    """A retrieved support article with similarity score."""
    article: SupportArticle
    score: float
    snippet: str  # Relevant snippet from the article


# ============================================================
# CONTENT GUARDRAIL
# ============================================================

class ContentGuardrail:
    """
    Enforce content boundaries for the support agent.
    
    Ensures the agent only uses approved content sources
    and doesn't generate information outside its knowledge base.
    """
    
    def __init__(self, 
                 min_similarity_threshold: float = 0.5,
                 min_sources_required: int = 1):
        self.min_similarity_threshold = min_similarity_threshold
        self.min_sources_required = min_sources_required
        self.approved_articles: Dict[str, SupportArticle] = {}
        self.blocked_topics: Set[str] = set()
        
    def add_approved_article(self, article: SupportArticle) -> None:
        """Add an article to the approved knowledge base."""
        self.approved_articles[article.id] = article
        
    def add_blocked_topic(self, topic: str) -> None:
        """Add a topic that the agent should not discuss."""
        self.blocked_topics.add(topic.lower())
        
    def check_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a query is within scope.
        
        Returns:
            (is_allowed, rejection_reason)
        """
        query_lower = query.lower()
        
        # Check for blocked topics
        for topic in self.blocked_topics:
            if topic in query_lower:
                return False, f"Query contains blocked topic: {topic}"
                
        return True, None
    
    def validate_sources(self, 
                        sources: List[RetrievedSource]) -> Tuple[bool, List[RetrievedSource]]:
        """
        Validate retrieved sources meet quality thresholds.
        
        Returns:
            (is_valid, filtered_sources)
        """
        # Filter by similarity threshold
        valid_sources = [
            s for s in sources 
            if s.score >= self.min_similarity_threshold
        ]
        
        # Check minimum sources requirement
        if len(valid_sources) < self.min_sources_required:
            return False, valid_sources
            
        return True, valid_sources
    
    def validate_response(self, 
                         response: str,
                         sources: List[RetrievedSource]) -> Tuple[bool, float, str]:
        """
        Validate that a response is grounded in sources.
        
        Returns:
            (is_grounded, grounding_score, explanation)
        """
        if not sources:
            return False, 0.0, "No sources provided"
            
        # Check response keywords against source content
        response_words = set(response.lower().split())
        source_words = set()
        for source in sources:
            source_words.update(source.article.content.lower().split())
            
        # Calculate overlap
        overlap = len(response_words & source_words)
        response_unique = len(response_words - {"the", "a", "an", "is", "are", "was", "were"})
        
        if response_unique == 0:
            return False, 0.0, "Empty response"
            
        grounding_score = overlap / response_unique
        
        # Threshold check
        if grounding_score < 0.3:
            return False, grounding_score, "Response may contain hallucinated content"
            
        return True, grounding_score, "Response is grounded in sources"


# ============================================================
# SOURCE CITATION ENGINE
# ============================================================

class SourceCitationEngine:
    """
    Engine for generating source citations in responses.
    
    Ensures transparency by linking responses to source articles.
    """
    
    def __init__(self, citation_format: str = "inline"):
        self.citation_format = citation_format  # "inline", "footnote", "links"
        
    def format_citation(self, article: SupportArticle) -> str:
        """Format a single citation."""
        if self.citation_format == "inline":
            return f"[Source: {article.title}]"
        elif self.citation_format == "footnote":
            return f"[^{article.id}]"
        else:
            return f"• {article.title}"
            
    def add_citations(self, 
                      response: str,
                      sources: List[RetrievedSource]) -> str:
        """
        Add citations to a response.
        
        Args:
            response: Original response text
            sources: Sources used
            
        Returns:
            Response with citations added
        """
        if not sources:
            return response
            
        if self.citation_format == "inline":
            # Add inline citations at the end
            citations = " | ".join([
                self.format_citation(s.article) 
                for s in sources[:3]  # Limit to top 3
            ])
            return f"{response}\n\n📚 Sources: {citations}"
            
        elif self.citation_format == "footnote":
            # Add footnote-style citations
            citation_text = response
            references = "\n\n---\n**References:**\n"
            
            for i, source in enumerate(sources[:5]):
                references += f"\n[^{i+1}]: {source.article.title}"
                
            return citation_text + references
            
        else:  # links
            links = "\n\n**Learn more:**\n"
            for source in sources[:3]:
                links += f"\n• {source.article.title}"
            return response + links
    
    def extract_relevant_snippet(self, 
                                 article: SupportArticle,
                                 query: str,
                                 max_length: int = 200) -> str:
        """
        Extract the most relevant snippet from an article.
        
        Args:
            article: Source article
            query: User query
            max_length: Maximum snippet length
            
        Returns:
            Relevant snippet text
        """
        # Simple approach: find paragraph with most keyword overlap
        paragraphs = article.content.split('\n\n')
        query_words = set(query.lower().split())
        
        best_para = ""
        best_score = 0
        
        for para in paragraphs:
            if len(para.strip()) < 20:
                continue
                
            para_words = set(para.lower().split())
            overlap = len(query_words & para_words)
            
            if overlap > best_score:
                best_score = overlap
                best_para = para
                
        # Truncate if needed
        if len(best_para) > max_length:
            best_para = best_para[:max_length-3] + "..."
            
        return best_para if best_para else article.content[:max_length]


# ============================================================
# CONFIDENCE SCORER
# ============================================================

class ConfidenceScorer:
    """
    Multi-factor confidence scoring for agent responses.
    
    Combines:
    - Retrieval quality
    - Source relevance
    - Response grounding
    - Query clarity
    """
    
    def __init__(self,
                 retrieval_weight: float = 0.4,
                 grounding_weight: float = 0.3,
                 clarity_weight: float = 0.2,
                 source_count_weight: float = 0.1):
        self.retrieval_weight = retrieval_weight
        self.grounding_weight = grounding_weight
        self.clarity_weight = clarity_weight
        self.source_count_weight = source_count_weight
        
        # Confidence thresholds
        self.auto_respond_threshold = 0.7
        self.human_review_threshold = 0.5
        self.reject_threshold = 0.3
        
    def score(self,
              retrieval_scores: List[float],
              grounding_score: float,
              query: str,
              num_sources: int) -> Tuple[float, str]:
        """
        Calculate overall confidence score.
        
        Returns:
            (confidence_score, recommendation)
        """
        # Retrieval component
        if retrieval_scores:
            retrieval_component = np.mean(sorted(retrieval_scores, reverse=True)[:3])
        else:
            retrieval_component = 0.0
            
        # Grounding component (already normalized)
        grounding_component = min(1.0, grounding_score)
        
        # Query clarity component
        clarity_component = self._score_query_clarity(query)
        
        # Source count component
        source_component = min(1.0, num_sources / 3)
        
        # Weighted combination
        total_score = (
            self.retrieval_weight * retrieval_component +
            self.grounding_weight * grounding_component +
            self.clarity_weight * clarity_component +
            self.source_count_weight * source_component
        )
        
        # Determine recommendation
        if total_score >= self.auto_respond_threshold:
            recommendation = "auto_respond"
        elif total_score >= self.human_review_threshold:
            recommendation = "respond_with_disclaimer"
        elif total_score >= self.reject_threshold:
            recommendation = "human_review"
        else:
            recommendation = "reject"
            
        return float(total_score), recommendation
    
    def _score_query_clarity(self, query: str) -> float:
        """Score how clear/answerable a query is."""
        score = 0.5  # Base score
        
        # Positive signals
        if '?' in query:
            score += 0.1
        if len(query.split()) >= 3:
            score += 0.1
        if len(query.split()) <= 50:
            score += 0.1
            
        # Negative signals
        if query.isupper():
            score -= 0.2  # Angry caps
        if len(query.split()) < 2:
            score -= 0.2  # Too vague
            
        return max(0.0, min(1.0, score))


# ============================================================
# CX SCORE ANALYZER
# ============================================================

class CXScoreAnalyzer:
    """
    Customer Experience (CX) Score analyzer.
    
    Automated analysis of 100% of conversations,
    replacing traditional CSAT surveys.
    
    Analyzes:
    - Sentiment throughout conversation
    - Resolution status
    - Agent tone and helpfulness
    - Customer effort
    """
    
    POSITIVE_WORDS = {
        "thank", "thanks", "great", "perfect", "awesome", "excellent",
        "helpful", "solved", "fixed", "works", "appreciate"
    }
    
    NEGATIVE_WORDS = {
        "frustrated", "angry", "terrible", "awful", "worst", "useless",
        "waste", "stupid", "hate", "disappointed", "unacceptable"
    }
    
    EFFORT_INDICATORS = {
        "high": ["again", "still", "already told", "how many times", "repeated"],
        "low": ["easy", "quick", "simple", "immediately", "right away"]
    }
    
    def __init__(self):
        self.conversation_scores: Dict[str, Dict[str, Any]] = {}
        
    def analyze_conversation(self, 
                            conversation: Conversation) -> Dict[str, Any]:
        """
        Analyze a complete conversation.
        
        Returns:
            CX Score report with sentiment, resolution, and recommendations
        """
        analysis = {
            "conversation_id": conversation.id,
            "user_id": conversation.user_id,
            "message_count": len(conversation.messages),
            "duration_minutes": self._calculate_duration(conversation),
            "sentiment_journey": [],
            "overall_sentiment": SentimentType.NEUTRAL.value,
            "resolution_status": conversation.state.value,
            "customer_effort": "medium",
            "agent_helpfulness": 0.5,
            "cx_score": 0.0,
            "recommendations": []
        }
        
        # Analyze each message
        sentiments = []
        for msg in conversation.messages:
            if msg.role == "user":
                sentiment = self._analyze_sentiment(msg.content)
                sentiments.append(sentiment)
                analysis["sentiment_journey"].append({
                    "message_id": msg.id,
                    "sentiment": sentiment.value,
                    "timestamp": msg.timestamp.isoformat()
                })
                
        # Overall sentiment (weighted toward end)
        if sentiments:
            analysis["overall_sentiment"] = self._aggregate_sentiment(sentiments).value
            
        # Customer effort analysis
        analysis["customer_effort"] = self._analyze_effort(conversation)
        
        # Agent helpfulness
        analysis["agent_helpfulness"] = self._analyze_agent_helpfulness(conversation)
        
        # Calculate CX Score (0-100)
        analysis["cx_score"] = self._calculate_cx_score(analysis)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Store for trend analysis
        self.conversation_scores[conversation.id] = analysis
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> SentimentType:
        """Analyze sentiment of a single message."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        # Check for frustration indicators
        if any(ind in text_lower for ind in ["!", "?!", "!!!"]):
            negative_count += 1
            
        if negative_count > positive_count + 1:
            if any(w in text_lower for w in ["angry", "frustrated", "!!!"]):
                return SentimentType.FRUSTRATED
            return SentimentType.NEGATIVE
        elif positive_count > negative_count:
            return SentimentType.POSITIVE
        else:
            return SentimentType.NEUTRAL
            
    def _aggregate_sentiment(self, 
                            sentiments: List[SentimentType]) -> SentimentType:
        """Aggregate sentiments with recency weighting."""
        if not sentiments:
            return SentimentType.NEUTRAL
            
        # Weight recent sentiments more heavily
        weights = np.linspace(0.5, 1.5, len(sentiments))
        
        scores = {
            SentimentType.POSITIVE: 1.0,
            SentimentType.NEUTRAL: 0.5,
            SentimentType.NEGATIVE: 0.0,
            SentimentType.FRUSTRATED: -0.5
        }
        
        weighted_score = sum(
            scores[s] * w for s, w in zip(sentiments, weights)
        ) / sum(weights)
        
        if weighted_score >= 0.7:
            return SentimentType.POSITIVE
        elif weighted_score >= 0.3:
            return SentimentType.NEUTRAL
        elif weighted_score >= 0.0:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.FRUSTRATED
            
    def _analyze_effort(self, conversation: Conversation) -> str:
        """Analyze customer effort level."""
        all_text = " ".join([m.content for m in conversation.messages if m.role == "user"])
        text_lower = all_text.lower()
        
        high_effort_score = sum(
            1 for ind in self.EFFORT_INDICATORS["high"] if ind in text_lower
        )
        low_effort_score = sum(
            1 for ind in self.EFFORT_INDICATORS["low"] if ind in text_lower
        )
        
        # Also consider message count
        if len(conversation.messages) > 10:
            high_effort_score += 2
        elif len(conversation.messages) > 5:
            high_effort_score += 1
            
        if high_effort_score > low_effort_score + 1:
            return "high"
        elif low_effort_score > high_effort_score:
            return "low"
        else:
            return "medium"
            
    def _analyze_agent_helpfulness(self, conversation: Conversation) -> float:
        """Analyze agent helpfulness (0-1 scale)."""
        agent_messages = [m for m in conversation.messages if m.role == "agent"]
        
        if not agent_messages:
            return 0.5
            
        score = 0.5
        
        for msg in agent_messages:
            # Positive: message has sources
            if msg.sources:
                score += 0.1
            # Positive: reasonable length
            if 50 < len(msg.content) < 500:
                score += 0.05
            # Positive: high confidence
            if msg.confidence > 0.7:
                score += 0.1
            # Negative: very short responses
            if len(msg.content) < 20:
                score -= 0.1
                
        return max(0.0, min(1.0, score))
    
    def _calculate_cx_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall CX Score (0-100)."""
        score = 50.0  # Base score
        
        # Sentiment impact (+/- 20)
        sentiment = SentimentType(analysis["overall_sentiment"])
        sentiment_scores = {
            SentimentType.POSITIVE: 20,
            SentimentType.NEUTRAL: 0,
            SentimentType.NEGATIVE: -15,
            SentimentType.FRUSTRATED: -25
        }
        score += sentiment_scores.get(sentiment, 0)
        
        # Resolution impact (+/- 20)
        if analysis["resolution_status"] == "resolved":
            score += 20
        elif analysis["resolution_status"] == "escalated":
            score += 5  # At least they got help
        elif analysis["resolution_status"] == "abandoned":
            score -= 20
            
        # Effort impact (+/- 15)
        effort_scores = {"low": 15, "medium": 0, "high": -15}
        score += effort_scores.get(analysis["customer_effort"], 0)
        
        # Agent helpfulness impact (+/- 10)
        score += (analysis["agent_helpfulness"] - 0.5) * 20
        
        return max(0.0, min(100.0, score))
    
    def _calculate_duration(self, conversation: Conversation) -> float:
        """Calculate conversation duration in minutes."""
        if len(conversation.messages) < 2:
            return 0.0
            
        first = conversation.messages[0].timestamp
        last = conversation.messages[-1].timestamp
        
        return (last - first).total_seconds() / 60
    
    def _generate_recommendations(self, 
                                   analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recs = []
        
        if analysis["cx_score"] < 50:
            recs.append("Review conversation for training opportunities")
            
        if analysis["customer_effort"] == "high":
            recs.append("Investigate why multiple interactions were needed")
            
        if analysis["overall_sentiment"] in ["negative", "frustrated"]:
            recs.append("Follow up with customer for feedback")
            
        if analysis["agent_helpfulness"] < 0.5:
            recs.append("Review agent responses for completeness")
            
        if analysis["resolution_status"] == "abandoned":
            recs.append("Send follow-up to attempt resolution")
            
        return recs
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all analyzed conversations."""
        if not self.conversation_scores:
            return {}
            
        scores = list(self.conversation_scores.values())
        
        return {
            "total_conversations": len(scores),
            "avg_cx_score": np.mean([s["cx_score"] for s in scores]),
            "resolution_rate": sum(
                1 for s in scores if s["resolution_status"] == "resolved"
            ) / len(scores),
            "positive_sentiment_rate": sum(
                1 for s in scores if s["overall_sentiment"] == "positive"
            ) / len(scores),
            "high_effort_rate": sum(
                1 for s in scores if s["customer_effort"] == "high"
            ) / len(scores)
        }


# ============================================================
# SUPPORT AGENT
# ============================================================

class SupportAgent:
    """
    Production support agent with guardrails.
    
    Combines:
    - Content guardrails for hallucination prevention
    - Source citation for transparency
    - Confidence scoring for human handoff
    - CX Score analysis for quality monitoring
    
    Example:
        >>> agent = SupportAgent()
        >>> agent.add_article(SupportArticle(id="1", title="...", content="..."))
        >>> response = agent.respond(conversation, "How do I reset my password?")
        >>> if response.should_escalate:
        ...     # Hand off to human
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.5,
                 max_sources: int = 3,
                 enable_citations: bool = True):
        self.guardrail = ContentGuardrail()
        self.citation_engine = SourceCitationEngine()
        self.confidence_scorer = ConfidenceScorer()
        self.cx_analyzer = CXScoreAnalyzer()
        
        self.confidence_threshold = confidence_threshold
        self.max_sources = max_sources
        self.enable_citations = enable_citations
        
        # Knowledge base
        self.articles: Dict[str, SupportArticle] = {}
        self.embeddings: Optional[np.ndarray] = None
        
        # Fallback responses
        self.fallback_responses = {
            "low_confidence": "I'm not certain about this. Let me connect you with a team member who can help.",
            "no_sources": "I don't have information about that in my knowledge base. Would you like to speak with a support agent?",
            "blocked_topic": "I'm not able to help with that topic. Please contact our support team directly.",
            "general_error": "I encountered an issue processing your request. Let me get a team member to help you."
        }
        
        # Generation function (placeholder)
        self.generate_fn: Optional[Callable] = None
        
    def add_article(self, article: SupportArticle) -> None:
        """Add an article to the knowledge base."""
        self.articles[article.id] = article
        self.guardrail.add_approved_article(article)
        
        # Update embeddings
        if article.embedding is not None:
            if self.embeddings is None:
                self.embeddings = article.embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([
                    self.embeddings, 
                    article.embedding.reshape(1, -1)
                ])
                
    def add_articles(self, articles: List[SupportArticle]) -> None:
        """Add multiple articles."""
        for article in articles:
            self.add_article(article)
            
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
        
    def retrieve_sources(self, 
                        query: str,
                        query_embedding: np.ndarray,
                        k: int = 5) -> List[RetrievedSource]:
        """Retrieve relevant sources for a query."""
        if self.embeddings is None or len(self.articles) == 0:
            return []
            
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Compute similarities
        similarities = emb_norms @ query_norm
        
        # Get top-k
        article_list = list(self.articles.values())
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        sources = []
        for idx in top_k_indices:
            article = article_list[idx]
            snippet = self.citation_engine.extract_relevant_snippet(article, query)
            sources.append(RetrievedSource(
                article=article,
                score=float(similarities[idx]),
                snippet=snippet
            ))
            
        return sources
    
    def respond(self,
                conversation: Conversation,
                query: str,
                query_embedding: np.ndarray = None) -> Message:
        """
        Generate a response to a user query.
        
        Args:
            conversation: Current conversation
            query: User's question
            query_embedding: Optional query embedding (random if not provided)
            
        Returns:
            Agent message with response
        """
        import uuid
        
        # Default embedding if not provided
        if query_embedding is None:
            query_embedding = np.random.randn(384)
            
        # Check query scope
        is_allowed, rejection_reason = self.guardrail.check_query(query)
        if not is_allowed:
            return self._create_fallback_message(
                "blocked_topic", 
                conversation.id,
                reason=rejection_reason
            )
            
        # Retrieve sources
        sources = self.retrieve_sources(query, query_embedding, k=5)
        
        # Validate sources
        is_valid, valid_sources = self.guardrail.validate_sources(sources)
        if not is_valid or not valid_sources:
            return self._create_fallback_message("no_sources", conversation.id)
            
        # Generate response
        response_text = self._generate_response(query, valid_sources)
        
        # Validate response grounding
        is_grounded, grounding_score, _ = self.guardrail.validate_response(
            response_text, valid_sources
        )
        
        # Calculate confidence
        retrieval_scores = [s.score for s in valid_sources]
        confidence, recommendation = self.confidence_scorer.score(
            retrieval_scores=retrieval_scores,
            grounding_score=grounding_score,
            query=query,
            num_sources=len(valid_sources)
        )
        
        # Handle low confidence
        if recommendation == "reject":
            return self._create_fallback_message("low_confidence", conversation.id)
            
        # Add citations if enabled
        if self.enable_citations:
            response_text = self.citation_engine.add_citations(
                response_text, 
                valid_sources[:self.max_sources]
            )
            
        # Add disclaimer for uncertain responses
        if recommendation == "respond_with_disclaimer":
            response_text = (
                "Based on my knowledge, " + response_text + 
                "\n\n_If this doesn't fully answer your question, "
                "I can connect you with a team member._"
            )
            
        # Create message
        message = Message(
            id=str(uuid.uuid4()),
            role="agent",
            content=response_text,
            timestamp=datetime.now(),
            sources=[s.article.id for s in valid_sources[:self.max_sources]],
            confidence=confidence,
            metadata={
                "recommendation": recommendation,
                "grounding_score": grounding_score,
                "is_grounded": is_grounded
            }
        )
        
        # Add to conversation
        conversation.messages.append(message)
        
        return message
    
    def _generate_response(self, 
                          query: str,
                          sources: List[RetrievedSource]) -> str:
        """Generate a response using the LLM."""
        # Build context from sources
        context_parts = []
        for source in sources:
            context_parts.append(
                f"**{source.article.title}**\n{source.snippet}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful support agent. Answer the customer's question 
using ONLY the information provided in the context below.

If you cannot answer based on the context, say "I don't have enough information."
Be concise and helpful.

Context:
{context}

Customer Question: {query}

Answer:"""

        if self.generate_fn:
            return self.generate_fn(prompt)
        else:
            # Placeholder response using source content
            if sources:
                return f"Based on our documentation: {sources[0].snippet}"
            return "I'm here to help. Could you provide more details?"
            
    def _create_fallback_message(self, 
                                  fallback_type: str,
                                  conversation_id: str,
                                  reason: str = None) -> Message:
        """Create a fallback message."""
        import uuid
        
        content = self.fallback_responses.get(
            fallback_type, 
            self.fallback_responses["general_error"]
        )
        
        return Message(
            id=str(uuid.uuid4()),
            role="agent",
            content=content,
            timestamp=datetime.now(),
            confidence=0.0,
            metadata={
                "fallback_type": fallback_type,
                "reason": reason,
                "requires_escalation": True
            }
        )
    
    def analyze_conversation(self, 
                            conversation: Conversation) -> Dict[str, Any]:
        """Analyze conversation quality using CX Score."""
        return self.cx_analyzer.analyze_conversation(conversation)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate agent metrics."""
        return self.cx_analyzer.get_aggregate_metrics()


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """Demonstrate Support Agent usage."""
    import uuid
    
    # Create agent
    agent = SupportAgent(confidence_threshold=0.5)
    
    # Add knowledge base articles
    articles = [
        SupportArticle(
            id="article_1",
            title="How to Reset Your Password",
            content="""To reset your password, follow these steps:
            
1. Go to the login page
2. Click "Forgot Password"
3. Enter your email address
4. Check your email for a reset link
5. Click the link and create a new password

Your password must be at least 8 characters and include a number.""",
            category="account",
            tags=["password", "login", "security"],
            embedding=np.random.randn(384)
        ),
        SupportArticle(
            id="article_2",
            title="Billing and Subscription FAQ",
            content="""Common billing questions:

**How do I update my payment method?**
Go to Settings > Billing > Payment Methods to add or update your card.

**When am I charged?**
You're charged on the same date each month as your original signup.

**How do I cancel?**
Go to Settings > Subscription > Cancel. Your access continues until the end of your billing period.""",
            category="billing",
            tags=["billing", "payment", "subscription"],
            embedding=np.random.randn(384)
        )
    ]
    
    agent.add_articles(articles)
    print(f"Added {len(articles)} articles to knowledge base")
    
    # Create a conversation
    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id="user_123"
    )
    
    # Add user message
    user_message = Message(
        id=str(uuid.uuid4()),
        role="user",
        content="How do I reset my password? I forgot it.",
        timestamp=datetime.now()
    )
    conversation.messages.append(user_message)
    
    # Get response
    query_embedding = np.random.randn(384)
    response = agent.respond(
        conversation, 
        user_message.content,
        query_embedding
    )
    
    print(f"\nUser: {user_message.content}")
    print(f"\nAgent (confidence: {response.confidence:.2f}):")
    print(response.content)
    print(f"\nSources used: {response.sources}")
    
    # Simulate resolution
    thanks_message = Message(
        id=str(uuid.uuid4()),
        role="user",
        content="Thanks! That worked perfectly.",
        timestamp=datetime.now()
    )
    conversation.messages.append(thanks_message)
    conversation.state = ConversationState.RESOLVED
    
    # Analyze conversation
    analysis = agent.analyze_conversation(conversation)
    print(f"\n--- CX Analysis ---")
    print(f"CX Score: {analysis['cx_score']:.1f}/100")
    print(f"Sentiment: {analysis['overall_sentiment']}")
    print(f"Resolution: {analysis['resolution_status']}")
    print(f"Customer Effort: {analysis['customer_effort']}")
    
    return agent


if __name__ == "__main__":
    example_usage()

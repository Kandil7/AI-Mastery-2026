"""
Lab 2 Solution: Adaptive Router Implementation

This module implements a complete adaptive routing system for multimodal RAG.
"""

import re
import asyncio
from enum import Enum
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class QueryIntent(Enum):
    """Query intent types."""
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    EXPLORATORY = "exploratory"


class ModalityType(Enum):
    """Content modalities."""
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class IntentPrediction:
    """Result of intent classification."""
    intent: QueryIntent
    confidence: float
    signals: List[str]


@dataclass
class ComplexityMetrics:
    """Query complexity metrics."""
    length_score: float
    vocabulary_score: float
    structure_score: float
    entity_count: int
    constraint_count: int
    overall_complexity: float


@dataclass
class DomainPrediction:
    """Domain identification result."""
    primary_domain: str
    confidence: float
    subdomain: str
    keywords: List[str]


@dataclass
class RoutingRule:
    """Routing rule definition."""
    name: str
    condition: Callable[[dict], bool]
    action: Dict[str, Any]
    priority: int
    description: str = ""


@dataclass
class RoutingDecision:
    """Routing decision result."""
    target_indexes: List[str]
    parameters: Dict[str, Any]
    applied_rules: List[str]
    fallback: bool = False


@dataclass
class RoutingFeatures:
    """Features for ML-based routing."""
    query_length: int
    modality_scores: Dict[str, float]
    intent_scores: Dict[str, float]
    domain_scores: Dict[str, float]
    complexity: float
    entity_count: int


# ============================================================================
# QUERY ANALYSIS COMPONENTS
# ============================================================================

class IntentClassifier:
    """
    Classifies query intent using pattern matching.
    """
    
    INTENT_PATTERNS = {
        QueryIntent.INFORMATIONAL: [
            r"what is\s+", r"explain\s+", r"describe\s+", r"how does\s+",
            r"tell me about\s+", r"define\s+", r"overview of\s+",
            r"what are\s+", r"how to\s+"
        ],
        QueryIntent.NAVIGATIONAL: [
            r"find\s+", r"locate\s+", r"show me the\s+", r"where is\s+",
            r"navigate to\s+", r"access\s+", r"open\s+", r"go to\s+"
        ],
        QueryIntent.TRANSACTIONAL: [
            r"create\s+", r"generate\s+", r"write\s+", r"build\s+",
            r"implement\s+", r"deploy\s+", r"configure\s+", r"set up\s+",
            r"make\s+", r"produce\s+"
        ],
        QueryIntent.COMPARISON: [
            r"compare\s+", r"vs\s+", r"versus\s+", r"difference between\s+",
            r"better than\s+", r"which is more\s+", r"contrast\s+"
        ],
        QueryIntent.TROUBLESHOOTING: [
            r"error\s+", r"fix\s+", r"problem with\s+", r"issue\s+",
            r"not working\s+", r"failed to\s+", r"bug\s+", r"broken\s+",
            r"why isn't\s+", r"can't\s+"
        ],
        QueryIntent.EXPLORATORY: [
            r"explore\s+", r"browse\s+", r"show all\s+", r"list all\s+",
            r"what are the\s+", r"available\s+", r"options\s+"
        ]
    }
    
    def classify(self, query: str) -> IntentPrediction:
        """
        Classify query intent using pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            IntentPrediction with intent and confidence
        """
        query_lower = query.lower()
        scores = {intent: 0 for intent in QueryIntent}
        signals = {intent: [] for intent in QueryIntent}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    scores[intent] += 1
                    signals[intent].append(match.group().strip())
        
        # Find best match
        best_intent = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_intent] / max(total_score, 1)
        
        # Boost confidence if strong signal
        if scores[best_intent] >= 2:
            confidence = min(confidence * 1.3, 1.0)
        
        return IntentPrediction(
            intent=best_intent,
            confidence=round(confidence, 3),
            signals=signals[best_intent]
        )


class ComplexityEstimator:
    """
    Estimates query complexity for adaptive retrieval tuning.
    """
    
    COMPLEXITY_INDICATORS = {
        'constraints': [
            'only', 'exactly', 'specific', 'particular', 'precise',
            'must', 'should', 'required', 'mandatory', 'strictly'
        ],
        'multi_part': [
            'and also', 'plus', 'additionally', 'furthermore',
            'moreover', 'in addition', 'as well as', 'also'
        ],
        'conditional': [
            'if', 'when', 'unless', 'provided that', 'assuming',
            'given that', 'in case', 'whether'
        ],
        'temporal': [
            'before', 'after', 'during', 'while', 'until',
            'since', 'recently', 'latest', 'current', 'previous'
        ]
    }
    
    def estimate(self, query: str) -> ComplexityMetrics:
        """
        Estimate query complexity.
        
        Args:
            query: User query string
            
        Returns:
            ComplexityMetrics with scores
        """
        words = query.split()
        query_lower = query.lower()
        
        # Length score (normalized to 0-1)
        length_score = min(len(words) / 30, 1.0)
        
        # Vocabulary score (unique words / total words)
        unique_words = set(w.lower() for w in words)
        vocabulary_score = len(unique_words) / max(len(words), 1)
        
        # Structure score (based on complexity indicators)
        structure_score = 0
        for category, indicators in self.COMPLEXITY_INDICATORS.items():
            matches = sum(1 for ind in indicators if ind in query_lower)
            structure_score += matches * 0.15
        structure_score = min(structure_score, 1.0)
        
        # Entity count (capitalized words, excluding first word)
        entity_count = sum(1 for i, word in enumerate(words) 
                          if i > 0 and word and word[0].isupper())
        
        # Constraint count
        constraint_count = sum(
            1 for ind in self.COMPLEXITY_INDICATORS['constraints']
            if ind in query_lower
        )
        
        # Overall complexity (weighted combination)
        overall_complexity = (
            0.2 * length_score +
            0.2 * vocabulary_score +
            0.3 * structure_score +
            0.15 * min(entity_count / 5, 1.0) +
            0.15 * min(constraint_count / 3, 1.0)
        )
        
        return ComplexityMetrics(
            length_score=round(length_score, 3),
            vocabulary_score=round(vocabulary_score, 3),
            structure_score=round(structure_score, 3),
            entity_count=entity_count,
            constraint_count=constraint_count,
            overall_complexity=round(overall_complexity, 3)
        )
    
    def get_retrieval_params(self, complexity: ComplexityMetrics) -> Dict[str, Any]:
        """
        Derive retrieval parameters from complexity metrics.
        
        Args:
            complexity: ComplexityMetrics instance
            
        Returns:
            Dictionary with retrieval parameters
        """
        c = complexity.overall_complexity
        
        if c < 0.3:
            return {
                'top_k': 5,
                'search_depth': 'shallow',
                'rerank': False,
                'expand_query': False,
                'timeout_ms': 100
            }
        elif c < 0.6:
            return {
                'top_k': 10,
                'search_depth': 'medium',
                'rerank': True,
                'expand_query': False,
                'timeout_ms': 200
            }
        else:
            return {
                'top_k': 20,
                'search_depth': 'deep',
                'rerank': True,
                'expand_query': True,
                'timeout_ms': 500
            }


class DomainIdentifier:
    """
    Identifies query domain for specialized retrieval routing.
    """
    
    DOMAIN_KEYWORDS = {
        'engineering': [
            'architecture', 'system', 'component', 'module', 'interface',
            'api', 'microservice', 'deployment', 'infrastructure', 'scalability',
            'load balancer', 'container', 'kubernetes', 'docker'
        ],
        'finance': [
            'revenue', 'profit', 'loss', 'budget', 'forecast',
            'investment', 'roi', 'expense', 'financial', 'quarterly',
            'earnings', 'stock', 'market', 'fiscal'
        ],
        'legal': [
            'contract', 'agreement', 'compliance', 'regulation', 'policy',
            'liability', 'terms', 'clause', 'legal', 'jurisdiction',
            'gdpr', 'privacy', 'terms of service'
        ],
        'healthcare': [
            'patient', 'diagnosis', 'treatment', 'clinical', 'medical',
            'prescription', 'symptom', 'therapy', 'healthcare', 'hospital',
            'drug', 'dosage', 'procedure'
        ],
        'software': [
            'code', 'function', 'class', 'method', 'algorithm',
            'debugging', 'testing', 'repository', 'version', 'commit',
            'git', 'pull request', 'merge', 'branch'
        ],
        'data': [
            'database', 'query', 'table', 'schema', 'etl',
            'pipeline', 'analytics', 'dashboard', 'metrics', 'kpi',
            'data warehouse', 'sql', 'nosql', 'index'
        ]
    }
    
    SUBDOMAIN_MAPPING = {
        'engineering': {
            frozenset(['api', 'interface', 'microservice']): 'backend',
            frozenset(['deployment', 'infrastructure', 'scalability', 'kubernetes', 'docker']): 'devops',
            frozenset(['architecture', 'system', 'component']): 'architecture'
        },
        'software': {
            frozenset(['code', 'function', 'class']): 'development',
            frozenset(['debugging', 'testing']): 'qa',
            frozenset(['repository', 'version', 'commit', 'git']): 'version_control'
        },
        'data': {
            frozenset(['database', 'query', 'table', 'sql']): 'database',
            frozenset(['pipeline', 'etl']): 'data_engineering',
            frozenset(['analytics', 'dashboard', 'metrics', 'kpi']): 'analytics'
        }
    }
    
    def identify(self, query: str) -> DomainPrediction:
        """
        Identify query domain.
        
        Args:
            query: User query string
            
        Returns:
            DomainPrediction with domain info
        """
        query_lower = query.lower()
        scores = {domain: 0 for domain in self.DOMAIN_KEYWORDS}
        keywords = {domain: [] for domain in self.DOMAIN_KEYWORDS}
        
        for domain, domain_keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in domain_keywords:
                if keyword in query_lower:
                    scores[domain] += 1
                    keywords[domain].append(keyword)
        
        # Find primary domain
        primary_domain = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[primary_domain] / max(total_score, 1) if scores[primary_domain] > 0 else 0
        
        # Determine subdomain
        subdomain = self._determine_subdomain(primary_domain, keywords[primary_domain])
        
        return DomainPrediction(
            primary_domain=primary_domain,
            confidence=round(min(confidence * 1.3, 1.0), 3),
            subdomain=subdomain,
            keywords=keywords[primary_domain]
        )
    
    def _determine_subdomain(self, domain: str, keywords: List[str]) -> str:
        """Determine subdomain based on keyword patterns."""
        if domain in self.SUBDOMAIN_MAPPING:
            keyword_set = frozenset(keywords)
            for key_subdomain, key_keywords in self.SUBDOMAIN_MAPPING[domain].items():
                if key_keywords.issubset(keyword_set):
                    return key_subdomain
        return 'general'


# ============================================================================
# RULE-BASED ROUTER
# ============================================================================

class RuleBasedRouter:
    """
    Routes queries to appropriate indexes based on configurable rules.
    """
    
    def __init__(self):
        self.rules: List[RoutingRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Initialize with default routing rules."""
        
        # Rule 1: Image queries → Image index (high priority)
        self.add_rule(RoutingRule(
            name="image_modality",
            condition=lambda ctx: ctx.get('modality') == 'image',
            action={
                'indexes': ['image_index'],
                'top_k': 10,
                'rerank': True,
                'rerank_threshold': 0.5
            },
            priority=10,
            description="Route image queries to image index"
        ))
        
        # Rule 2: Code queries → Code index (high priority)
        self.add_rule(RoutingRule(
            name="code_modality",
            condition=lambda ctx: ctx.get('modality') == 'code',
            action={
                'indexes': ['code_index'],
                'top_k': 5,
                'rerank': True,
                'rerank_threshold': 0.7
            },
            priority=10,
            description="Route code queries to code index"
        ))
        
        # Rule 3: Table queries → Table index
        self.add_rule(RoutingRule(
            name="table_modality",
            condition=lambda ctx: ctx.get('modality') == 'table',
            action={
                'indexes': ['table_index'],
                'top_k': 8,
                'rerank': False
            },
            priority=10,
            description="Route table queries to table index"
        ))
        
        # Rule 4: Complex queries → Hybrid search
        self.add_rule(RoutingRule(
            name="complex_query",
            condition=lambda ctx: ctx.get('complexity', 0) > 0.6,
            action={
                'indexes': ['text_index', 'code_index', 'table_index'],
                'top_k': 20,
                'rerank': True,
                'fusion': 'rrf',
                'query_expansion': True
            },
            priority=8,
            description="Use hybrid search for complex queries"
        ))
        
        # Rule 5: Troubleshooting → Recent content prioritized
        self.add_rule(RoutingRule(
            name="troubleshooting_intent",
            condition=lambda ctx: ctx.get('intent') == 'troubleshooting',
            action={
                'indexes': ['text_index', 'code_index'],
                'top_k': 15,
                'recency_boost': 0.3,
                'rerank': True
            },
            priority=7,
            description="Prioritize recent content for troubleshooting"
        ))
        
        # Rule 6: Comparison queries → Multiple modalities
        self.add_rule(RoutingRule(
            name="comparison_intent",
            condition=lambda ctx: ctx.get('intent') == 'comparison',
            action={
                'indexes': ['text_index', 'table_index'],
                'top_k': 15,
                'diversity_boost': 0.2,
                'rerank': True
            },
            priority=7,
            description="Retrieve diverse content for comparisons"
        ))
        
        # Rule 7: Engineering domain → Specialized indexes
        self.add_rule(RoutingRule(
            name="engineering_domain",
            condition=lambda ctx: ctx.get('domain') == 'engineering',
            action={
                'indexes': ['text_index', 'diagram_index', 'code_index'],
                'top_k': 15,
                'domain_boost': {'engineering': 0.2}
            },
            priority=5,
            description="Engineering domain specialized routing"
        ))
        
        # Rule 8: Mixed modality → All indexes
        self.add_rule(RoutingRule(
            name="mixed_modality",
            condition=lambda ctx: ctx.get('modality') == 'mixed',
            action={
                'indexes': ['text_index', 'image_index', 'code_index', 'table_index'],
                'top_k': 25,
                'rerank': True,
                'fusion': 'rrf'
            },
            priority=6,
            description="Search all indexes for mixed modality queries"
        ))
        
        # Default fallback rule (lowest priority)
        self.add_rule(RoutingRule(
            name="default",
            condition=lambda ctx: True,
            action={
                'indexes': ['text_index'],
                'top_k': 10,
                'rerank': False
            },
            priority=0,
            description="Default fallback routing"
        ))
    
    def add_rule(self, rule: RoutingRule):
        """Add a routing rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def route(self, context: Dict[str, Any]) -> RoutingDecision:
        """
        Determine routing based on query context.
        
        Args:
            context: Query context with modality, intent, complexity, etc.
            
        Returns:
            RoutingDecision with target indexes and parameters
        """
        applied_rules = []
        merged_action = {'indexes': [], 'top_k': 10}
        
        for rule in self.rules:
            try:
                if rule.condition(context):
                    applied_rules.append(rule.name)
                    
                    # Merge actions
                    for key, value in rule.action.items():
                        if key == 'indexes':
                            # Combine indexes without duplicates
                            existing = merged_action.get(key, [])
                            merged_action[key] = list(set(existing + value))
                        else:
                            # Later rules override earlier ones
                            merged_action[key] = value
                    
                    # Stop at first high-priority match (priority >= 10)
                    if rule.priority >= 10:
                        break
            except Exception as e:
                # Log error but continue with other rules
                print(f"Error evaluating rule {rule.name}: {e}")
                continue
        
        # Ensure at least one index
        if not merged_action.get('indexes'):
            merged_action['indexes'] = ['text_index']
            if 'fallback' not in applied_rules:
                applied_rules.append('fallback')
        
        return RoutingDecision(
            target_indexes=merged_action['indexes'],
            parameters={k: v for k, v in merged_action.items() if k != 'indexes'},
            applied_rules=applied_rules,
            fallback='fallback' in applied_rules or len(applied_rules) == 0
        )
    
    def export_rules(self) -> str:
        """Export rules as JSON-like string for inspection."""
        lines = ["Routing Rules:", "=" * 50]
        for rule in self.rules:
            lines.append(f"Rule: {rule.name} (priority: {rule.priority})")
            lines.append(f"  Description: {rule.description}")
            lines.append(f"  Action: {rule.action}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# ML-BASED ROUTER
# ============================================================================

class MLRouter:
    """
    Machine learning-based routing using historical performance data.
    For this lab, we use heuristic routing as fallback.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.index_names = ['text_index', 'image_index', 'code_index', 'table_index']
        
        if model_path:
            try:
                self.load_model(model_path)
            except FileNotFoundError:
                print(f"Model not found at {model_path}, using heuristic routing")
    
    def extract_features(self, context: Dict[str, Any]) -> RoutingFeatures:
        """Extract features from query context for ML model."""
        return RoutingFeatures(
            query_length=len(context.get('query', '').split()),
            modality_scores=self._get_modality_scores(context),
            intent_scores=self._get_intent_scores(context),
            domain_scores=self._get_domain_scores(context),
            complexity=context.get('complexity', 0.5),
            entity_count=context.get('entity_count', 0)
        )
    
    def _get_modality_scores(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract modality scores from context."""
        modality = context.get('modality', 'text')
        confidence = context.get('modality_confidence', 0.5)
        
        scores = {m: 0.1 for m in ['text', 'image', 'code', 'table']}
        scores[modality] = confidence
        return scores
    
    def _get_intent_scores(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract intent scores from context."""
        intent = context.get('intent', 'informational')
        scores = {i.value: 0.1 for i in QueryIntent}
        scores[intent] = 0.8
        return scores
    
    def _get_domain_scores(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract domain scores from context."""
        domain = context.get('domain', 'general')
        scores = {d: 0.1 for d in ['engineering', 'finance', 'software', 'data', 'general']}
        scores[domain] = 0.8
        return scores
    
    def predict_routing(self, features: RoutingFeatures) -> Dict[str, float]:
        """
        Predict routing scores for each index.
        
        Args:
            features: RoutingFeatures instance
            
        Returns:
            Probability distribution over indexes
        """
        if self.model is not None:
            # Would use trained model here
            pass
        
        # Use heuristic routing
        return self._heuristic_routing(features)
    
    def _heuristic_routing(self, features: RoutingFeatures) -> Dict[str, float]:
        """Fallback heuristic routing when model unavailable."""
        scores = {index: 0.1 for index in self.index_names}
        
        # Boost based on modality
        modality_scores = features.modality_scores
        if modality_scores.get('image', 0) > 0.5:
            scores['image_index'] += modality_scores['image'] * 0.5
        if modality_scores.get('code', 0) > 0.5:
            scores['code_index'] += modality_scores['code'] * 0.5
        if modality_scores.get('table', 0) > 0.5:
            scores['table_index'] += modality_scores['table'] * 0.5
        
        # Adjust based on complexity
        if features.complexity > 0.6:
            # Complex queries benefit from multiple indexes
            for index in self.index_names:
                scores[index] += 0.1
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def load_model(self, path: str):
        """Load trained model."""
        import joblib
        self.model = joblib.load(path)
    
    def save_model(self, path: str):
        """Save trained model."""
        import joblib
        if self.model:
            joblib.dump(self.model, path)


# ============================================================================
# HYBRID ROUTER
# ============================================================================

class HybridRouter:
    """
    Combines rule-based and ML-based routing for optimal decisions.
    """
    
    def __init__(self, rule_router: RuleBasedRouter, ml_router: MLRouter, 
                 ml_threshold: float = 0.7):
        self.rule_router = rule_router
        self.ml_router = ml_router
        self.ml_threshold = ml_threshold
    
    def route(self, context: Dict[str, Any]) -> RoutingDecision:
        """
        Use hybrid routing strategy.
        
        Args:
            context: Query context
            
        Returns:
            RoutingDecision
        """
        # Get rule-based decision
        rule_decision = self.rule_router.route(context)
        
        # Get ML scores
        features = self.ml_router.extract_features(context)
        ml_scores = self.ml_router.predict_routing(features)
        
        # Check if ML is confident
        ml_top_index = max(ml_scores, key=ml_scores.get)
        ml_confidence = ml_scores[ml_top_index]
        
        # Use ML if confident and disagrees with rules
        if ml_confidence >= self.ml_threshold:
            return self._ml_decision(ml_scores, context)
        
        # Fall back to rule-based
        return rule_decision
    
    def _ml_decision(self, scores: Dict[str, float], context: Dict[str, Any]) -> RoutingDecision:
        """Create routing decision from ML scores."""
        # Select indexes above threshold
        threshold = 0.2
        selected_indexes = [idx for idx, score in scores.items() if score >= threshold]
        
        # Calculate top_k based on number of indexes
        num_indexes = len(selected_indexes) or 1
        top_k = max(5, 15 // num_indexes)
        
        return RoutingDecision(
            target_indexes=selected_indexes or ['text_index'],
            parameters={
                'top_k': top_k,
                'rerank': True,
                'ml_scores': scores
            },
            applied_rules=['ml_routing'],
            fallback=False
        )


# ============================================================================
# ADAPTIVE RETRIEVAL PIPELINE
# ============================================================================

class AdaptiveRetrievalPipeline:
    """
    Complete adaptive retrieval pipeline integrating all components.
    """
    
    def __init__(self):
        # Import from Lab 1
        from lab_1_modality_detection.solution import QueryModalityClassifier
        
        self.modality_classifier = QueryModalityClassifier()
        self.intent_classifier = IntentClassifier()
        self.complexity_estimator = ComplexityEstimator()
        self.domain_identifier = DomainIdentifier()
        
        # Create routers
        rule_router = RuleBasedRouter()
        ml_router = MLRouter()
        self.router = HybridRouter(rule_router, ml_router)
    
    async def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Full adaptive retrieval pipeline.
        
        Args:
            query: User query
            
        Returns:
            Retrieval results with routing info
        """
        # Step 1: Query Analysis
        modality = self.modality_classifier.classify(query)
        intent = self.intent_classifier.classify(query)
        complexity = self.complexity_estimator.estimate(query)
        domain = self.domain_identifier.identify(query)
        
        # Step 2: Build context
        context = {
            'query': query,
            'modality': modality.primary.value,
            'modality_confidence': modality.confidence,
            'intent': intent.intent.value,
            'intent_confidence': intent.confidence,
            'complexity': complexity.overall_complexity,
            'domain': domain.primary_domain,
            'domain_confidence': domain.confidence,
            'entity_count': complexity.entity_count,
        }
        
        # Step 3: Route
        decision = self.router.route(context)
        
        # Step 4: Execute retrieval (mock for lab)
        results = await self._execute_retrieval(query, decision)
        
        return {
            'query': query,
            'context': context,
            'routing': {
                'indexes': decision.target_indexes,
                'parameters': decision.parameters,
                'applied_rules': decision.applied_rules,
                'fallback': decision.fallback
            },
            'results': results,
        }
    
    async def _execute_retrieval(self, query: str, decision: RoutingDecision) -> List[Dict]:
        """
        Mock retrieval for lab purposes.
        In production, this would query actual vector indexes.
        """
        # Simulate retrieval from each index
        results = []
        for index in decision.target_indexes:
            results.append({
                'index': index,
                'content': f"Mock result from {index} for query: {query[:50]}",
                'score': 0.8,
                'modality': index.split('_')[0]
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:decision.parameters.get('top_k', 10)]


# ============================================================================
# TESTING AND EVALUATION
# ============================================================================

def build_context(query: str) -> Dict[str, Any]:
    """Build query context from raw query string."""
    pipeline = AdaptiveRetrievalPipeline()
    
    modality = pipeline.modality_classifier.classify(query)
    intent = pipeline.intent_classifier.classify(query)
    complexity = pipeline.complexity_estimator.estimate(query)
    domain = pipeline.domain_identifier.identify(query)
    
    return {
        'query': query,
        'modality': modality.primary.value,
        'modality_confidence': modality.confidence,
        'intent': intent.intent.value,
        'complexity': complexity.overall_complexity,
        'domain': domain.primary_domain,
        'entity_count': complexity.entity_count,
    }


def test_adaptive_router():
    """Test the adaptive router with sample queries."""
    print("=" * 70)
    print("ADAPTIVE ROUTER TEST")
    print("=" * 70)
    print()
    
    # Create router components
    rule_router = RuleBasedRouter()
    ml_router = MLRouter()
    hybrid_router = HybridRouter(rule_router, ml_router)
    
    # Test queries
    test_cases = [
        {
            'query': 'Show me the architecture diagram',
            'expected_indexes': ['image_index'],
        },
        {
            'query': 'Write a function to calculate sum',
            'expected_indexes': ['code_index'],
        },
        {
            'query': 'Compare the revenue of Q1 and Q2',
            'expected_indexes': ['text_index', 'table_index'],
        },
        {
            'query': 'Fix the error in the database connection',
            'expected_indexes': ['text_index', 'code_index'],
        },
        {
            'query': 'Explain how the system works',
            'expected_indexes': ['text_index'],
        },
        {
            'query': 'Show me the code and the flow chart',
            'expected_indexes': ['code_index', 'image_index'],
        },
    ]
    
    correct = 0
    for test in test_cases:
        # Build context from query
        context = build_context(test['query'])
        
        # Get routing decision
        decision = hybrid_router.route(context)
        
        # Check if expected indexes are included
        expected_set = set(test['expected_indexes'])
        actual_set = set(decision.target_indexes)
        is_correct = expected_set.issubset(actual_set) or actual_set.issubset(expected_set)
        
        if is_correct:
            correct += 1
        
        print(f"Query: {test['query']}")
        print(f"  Context: modality={context['modality']}, intent={context['intent']}")
        print(f"  Indexes: {decision.target_indexes}")
        print(f"  Expected: {test['expected_indexes']}")
        print(f"  Rules: {decision.applied_rules}")
        print(f"  ✓ Correct" if is_correct else f"  ✗ Incorrect")
        print()
    
    accuracy = correct / len(test_cases)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
    print()
    
    # Print routing rules
    print("=" * 70)
    print("ROUTING RULES")
    print("=" * 70)
    print(rule_router.export_rules())


async def test_full_pipeline():
    """Test the full adaptive retrieval pipeline."""
    print()
    print("=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)
    print()
    
    pipeline = AdaptiveRetrievalPipeline()
    
    test_queries = [
        "Show me the architecture diagram",
        "Write a function to calculate sum",
        "Compare Q1 and Q2 revenue",
        "Fix the database connection error",
    ]
    
    for query in test_queries:
        result = await pipeline.retrieve(query)
        
        print(f"Query: {query}")
        print(f"  Modality: {result['context']['modality']}")
        print(f"  Intent: {result['context']['intent']}")
        print(f"  Complexity: {result['context']['complexity']}")
        print(f"  Indexes: {result['routing']['indexes']}")
        print(f"  Top K: {result['routing']['parameters'].get('top_k', 'N/A')}")
        print()


if __name__ == "__main__":
    test_adaptive_router()
    asyncio.run(test_full_pipeline())
    
    print()
    print("=" * 70)
    print("Lab 2 Complete!")
    print("=" * 70)

# Theory 2: Adaptive Retrieval Patterns

## 2.1 Introduction to Adaptive Retrieval

### 2.1.1 The Case for Adaptivity

Static retrieval systems apply the same strategy to every query, regardless of:
- Query complexity and intent
- Expected response modality
- Domain-specific requirements
- Latency constraints
- Available context window

**The Problem with Static Retrieval:**

```
Query 1: "What is the company's revenue?"
Query 2: "Show me the quarterly revenue trend chart"
Query 3: "Write SQL to calculate revenue growth"

Static RAG: Same retrieval path for all three queries
Result: Suboptimal results for queries 2 and 3

Adaptive RAG: Different paths based on query analysis
- Query 1 → Text retrieval (numerical data in documents)
- Query 2 → Image retrieval (charts and visualizations)
- Query 3 → Code retrieval (SQL examples)
Result: Optimal results for each query type
```

### 2.1.2 Adaptive Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE RETRIEVAL ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                           USER QUERY                                    │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    QUERY ANALYSIS LAYER                         │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │   Intent     │  │   Modality   │  │   Domain     │          │   │
│  │  │   Classifier │  │   Detector   │  │   Identifier │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  Complexity  │  │   Entity     │  │   Temporal   │          │   │
│  │  │   Estimator  │  │   Extractor  │  │   Analyzer   │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ROUTING DECISION ENGINE                      │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                 │   │
│  │   ┌─────────────────────────────────────────────────────────┐  │   │
│  │   │              Routing Rules Configuration                │  │   │
│  │   │                                                         │  │   │
│  │   │  IF modality == IMAGE AND complexity == LOW            │  │   │
│  │   │     THEN route → image_index, top_k=10                 │  │   │
│  │   │                                                         │  │   │
│  │   │  IF modality == CODE AND domain == DATABASE            │  │   │
│  │   │     THEN route → code_index, top_k=5, rerank=true      │  │   │
│  │   │                                                         │  │   │
│  │   │  IF modality == MIXED AND entities > 3                 │  │   │
│  │   │     THEN route → hybrid_search, top_k=20               │  │   │
│  │   │                                                         │  │   │
│  │   └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│              ┌────────────────┼────────────────┐                       │
│              ▼                ▼                ▼                       │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐       │
│  │   Text Index     │ │   Image Index    │ │   Code Index     │       │
│  │   (Dense + BM25) │ │   (CLIP vectors) │ │   (CodeBERT)     │       │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘       │
│           │                    │                    │                  │
│           └────────────────────┼────────────────────┘                  │
│                                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FUSION & RERANKING LAYER                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  • Reciprocal Rank Fusion (RRF)                                 │   │
│  │  • Score normalization across modalities                        │   │
│  │  • Cross-encoder reranking                                      │   │
│  │  • Diversity optimization                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│                         FINAL RESULTS                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1.3 Key Adaptive Dimensions

| Dimension | Description | Impact on Retrieval |
|-----------|-------------|---------------------|
| Modality | Expected content type | Index selection |
| Complexity | Query difficulty | Search depth, top_k |
| Domain | Subject area | Specialized indexes |
| Intent | Informational/Navigational/Transactional | Ranking strategy |
| Temporal | Time sensitivity | Freshness weighting |
| Entities | Named entities count | Entity-aware retrieval |

---

## 2.2 Query Analysis Components

### 2.2.1 Intent Classification

Understanding user intent is crucial for adaptive retrieval:

```python
from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass
import re

class QueryIntent(Enum):
    INFORMATIONAL = "informational"  # Seeking knowledge
    NAVIGATIONAL = "navigational"    # Seeking specific item
    TRANSACTIONAL = "transactional"  # Wanting to perform action
    COMPARISON = "comparison"        # Comparing options
    TROUBLESHOOTING = "troubleshooting"  # Solving problem
    EXPLORATORY = "exploratory"      # Browsing/discovering

@dataclass
class IntentPrediction:
    intent: QueryIntent
    confidence: float
    signals: List[str]

class IntentClassifier:
    """
    Classifies query intent using pattern matching and ML.
    """
    
    INTENT_PATTERNS = {
        QueryIntent.INFORMATIONAL: [
            r"what is\s+", r"explain\s+", r"describe\s+", r"how does\s+",
            r"tell me about\s+", r"define\s+", r"overview of\s+"
        ],
        QueryIntent.NAVIGATIONAL: [
            r"find\s+", r"locate\s+", r"show me the\s+", r"where is\s+",
            r"navigate to\s+", r"access\s+", r"open\s+"
        ],
        QueryIntent.TRANSACTIONAL: [
            r"create\s+", r"generate\s+", r"write\s+", r"build\s+",
            r"implement\s+", r"deploy\s+", r"configure\s+"
        ],
        QueryIntent.COMPARISON: [
            r"compare\s+", r"vs\s+", r"versus\s+", r"difference between\s+",
            r"better than\s+", r"which is more\s+"
        ],
        QueryIntent.TROUBLESHOOTING: [
            r"error\s+", r"fix\s+", r"problem with\s+", r"issue\s+",
            r"not working\s+", r"failed to\s+", r"bug\s+"
        ],
        QueryIntent.EXPLORATORY: [
            r"explore\s+", r"browse\s+", r"show all\s+", r"list all\s+",
            r"what are the\s+", r"available\s+"
        ]
    }
    
    def classify(self, query: str) -> IntentPrediction:
        query_lower = query.lower()
        scores = {intent: 0 for intent in QueryIntent}
        signals = {intent: [] for intent in QueryIntent}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    scores[intent] += 1
                    signals[intent].append(match.group())
        
        # Find best match
        best_intent = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_intent] / max(total_score, 1)
        
        return IntentPrediction(
            intent=best_intent,
            confidence=min(confidence * 1.5, 1.0),  # Boost confidence
            signals=signals[best_intent]
        )
    
    def classify_with_llm(self, query: str) -> IntentPrediction:
        """
        Use LLM for more nuanced intent classification.
        """
        from openai import OpenAI
        
        client = OpenAI()
        
        prompt = f"""
        Classify the following query into one of these intents:
        - INFORMATIONAL: Seeking knowledge or explanation
        - NAVIGATIONAL: Looking for specific item or location
        - TRANSACTIONAL: Wanting to create or perform action
        - COMPARISON: Comparing options or alternatives
        - TROUBLESHOOTING: Solving a problem or error
        - EXPLORATORY: Browsing or discovering options
        
        Query: "{query}"
        
        Respond with JSON: {{"intent": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return IntentPrediction(
            intent=QueryIntent(result["intent"]),
            confidence=float(result["confidence"]),
            signals=[result.get("reasoning", "")]
        )
```

### 2.2.2 Complexity Estimation

Query complexity affects retrieval depth and strategy:

```python
from dataclasses import dataclass
from typing import List
import math

@dataclass
class ComplexityMetrics:
    length_score: float
    vocabulary_score: float
    structure_score: float
    entity_count: int
    constraint_count: int
    overall_complexity: float  # 0.0 - 1.0

class ComplexityEstimator:
    """
    Estimates query complexity for adaptive retrieval tuning.
    """
    
    COMPLEXITY_INDICATORS = {
        'constraints': [
            'only', 'exactly', 'specific', 'particular', 'precise',
            'must', 'should', 'required', 'mandatory'
        ],
        'multi_part': [
            'and also', 'plus', 'additionally', 'furthermore',
            'moreover', 'in addition', 'as well as'
        ],
        'conditional': [
            'if', 'when', 'unless', 'provided that', 'assuming',
            'given that', 'in case'
        ],
        'temporal': [
            'before', 'after', 'during', 'while', 'until',
            'since', 'recently', 'latest', 'current'
        ]
    }
    
    def estimate(self, query: str) -> ComplexityMetrics:
        words = query.split()
        query_lower = query.lower()
        
        # Length score (normalized)
        length_score = min(len(words) / 30, 1.0)
        
        # Vocabulary score (unique words / total words)
        unique_words = set(words)
        vocabulary_score = len(unique_words) / max(len(words), 1)
        
        # Structure score (based on complexity indicators)
        structure_score = 0
        for category, indicators in self.COMPLEXITY_INDICATORS.items():
            matches = sum(1 for ind in indicators if ind in query_lower)
            structure_score += matches * 0.15
        structure_score = min(structure_score, 1.0)
        
        # Entity count (simple heuristic: capitalized words)
        entity_count = sum(1 for word in words if word[0].isupper())
        
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
            length_score=length_score,
            vocabulary_score=vocabulary_score,
            structure_score=structure_score,
            entity_count=entity_count,
            constraint_count=constraint_count,
            overall_complexity=round(overall_complexity, 3)
        )
    
    def get_retrieval_params(self, complexity: ComplexityMetrics) -> dict:
        """
        Derive retrieval parameters from complexity metrics.
        """
        c = complexity.overall_complexity
        
        if c < 0.3:
            return {
                'top_k': 5,
                'search_depth': 'shallow',
                'rerank': False,
                'expand_query': False
            }
        elif c < 0.6:
            return {
                'top_k': 10,
                'search_depth': 'medium',
                'rerank': True,
                'expand_query': False
            }
        else:
            return {
                'top_k': 20,
                'search_depth': 'deep',
                'rerank': True,
                'expand_query': True
            }
```

### 2.2.3 Domain Identification

Identifying the domain helps route to specialized indexes:

```python
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DomainPrediction:
    primary_domain: str
    confidence: float
    subdomain: str
    keywords: List[str]

class DomainIdentifier:
    """
    Identifies query domain for specialized retrieval routing.
    """
    
    DOMAIN_KEYWORDS = {
        'engineering': [
            'architecture', 'system', 'component', 'module', 'interface',
            'api', 'microservice', 'deployment', 'infrastructure', 'scalability'
        ],
        'finance': [
            'revenue', 'profit', 'loss', 'budget', 'forecast',
            'investment', 'roi', 'expense', 'financial', 'quarterly'
        ],
        'legal': [
            'contract', 'agreement', 'compliance', 'regulation', 'policy',
            'liability', 'terms', 'clause', 'legal', 'jurisdiction'
        ],
        'healthcare': [
            'patient', 'diagnosis', 'treatment', 'clinical', 'medical',
            'prescription', 'symptom', 'therapy', 'healthcare', 'hospital'
        ],
        'software': [
            'code', 'function', 'class', 'method', 'algorithm',
            'debugging', 'testing', 'repository', 'version', 'commit'
        ],
        'data': [
            'database', 'query', 'table', 'schema', 'etl',
            'pipeline', 'analytics', 'dashboard', 'metrics', 'kpi'
        ]
    }
    
    def identify(self, query: str) -> DomainPrediction:
        query_lower = query.lower()
        scores: Dict[str, int] = {domain: 0 for domain in self.DOMAIN_KEYWORDS}
        keywords: Dict[str, List[str]] = {domain: [] for domain in self.DOMAIN_KEYWORDS}
        
        for domain, domain_keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in domain_keywords:
                if keyword in query_lower:
                    scores[domain] += 1
                    keywords[domain].append(keyword)
        
        # Find primary domain
        primary_domain = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[primary_domain] / max(total_score, 1)
        
        # Determine subdomain based on specific keyword patterns
        subdomain = self._determine_subdomain(primary_domain, keywords[primary_domain])
        
        return DomainPrediction(
            primary_domain=primary_domain,
            confidence=min(confidence * 1.3, 1.0),
            subdomain=subdomain,
            keywords=keywords[primary_domain]
        )
    
    def _determine_subdomain(self, domain: str, keywords: List[str]) -> str:
        subdomain_mapping = {
            'engineering': {
                frozenset(['api', 'interface', 'microservice']): 'backend',
                frozenset(['deployment', 'infrastructure', 'scalability']): 'devops',
                frozenset(['architecture', 'system', 'component']): 'architecture'
            },
            'software': {
                frozenset(['code', 'function', 'class']): 'development',
                frozenset(['debugging', 'testing']): 'qa',
                frozenset(['repository', 'version', 'commit']): 'version_control'
            },
            'data': {
                frozenset(['database', 'query', 'table']): 'database',
                frozenset(['pipeline', 'etl']): 'data_engineering',
                frozenset(['analytics', 'dashboard', 'metrics']): 'analytics'
            }
        }
        
        if domain in subdomain_mapping:
            keyword_set = frozenset(keywords)
            for key_subdomain, key_keywords in subdomain_mapping[domain].items():
                if key_keywords.issubset(keyword_set):
                    return key_subdomain
        
        return 'general'
```

---

## 2.3 Routing Decision Engine

### 2.3.1 Rule-Based Routing

```python
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field
import json

@dataclass
class RoutingRule:
    name: str
    condition: Callable[[dict], bool]
    action: Dict[str, Any]
    priority: int = 0
    description: str = ""

@dataclass
class RoutingDecision:
    target_indexes: List[str]
    parameters: Dict[str, Any]
    applied_rules: List[str]
    fallback: bool = False

class RuleBasedRouter:
    """
    Routes queries to appropriate indexes based on configurable rules.
    """
    
    def __init__(self):
        self.rules: List[RoutingRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Initialize with sensible default routing rules."""
        
        # Rule 1: Image queries → Image index
        self.add_rule(RoutingRule(
            name="image_modality",
            condition=lambda ctx: ctx.get('modality') == 'image',
            action={
                'indexes': ['image_index'],
                'top_k': 10,
                'rerank': True
            },
            priority=10,
            description="Route image queries to image index"
        ))
        
        # Rule 2: Code queries → Code index
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
        
        # Rule 3: Complex queries → Hybrid search
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
        
        # Rule 4: Troubleshooting → Recent content prioritized
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
        
        # Rule 5: Comparison queries → Multiple modalities
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
        
        # Rule 6: Domain-specific routing
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
        
        # Default fallback rule
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
        """
        applied_rules = []
        merged_action = {'indexes': [], 'top_k': 10}
        
        for rule in self.rules:
            if rule.condition(context):
                applied_rules.append(rule.name)
                
                # Merge actions (later rules can override)
                for key, value in rule.action.items():
                    if key == 'indexes':
                        merged_action[key] = list(set(merged_action.get(key, []) + value))
                    else:
                        merged_action[key] = value
                
                # Stop at first high-priority match
                if rule.priority >= 10:
                    break
        
        # Ensure at least one index
        if not merged_action.get('indexes'):
            merged_action['indexes'] = ['text_index']
            applied_rules.append('fallback')
        
        return RoutingDecision(
            target_indexes=merged_action['indexes'],
            parameters={k: v for k, v in merged_action.items() if k != 'indexes'},
            applied_rules=applied_rules,
            fallback='fallback' in applied_rules
        )
    
    def export_rules(self) -> str:
        """Export rules as JSON for inspection."""
        return json.dumps([
            {
                'name': rule.name,
                'priority': rule.priority,
                'description': rule.description,
                'action': rule.action
            }
            for rule in self.rules
        ], indent=2)
```

### 2.3.2 ML-Based Routing

For more sophisticated routing, use learned models:

```python
from typing import List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class RoutingFeatures:
    query_length: int
    modality_scores: dict
    intent_scores: dict
    domain_scores: dict
    complexity: float
    entity_count: int
    historical_success: dict

class MLRouter:
    """
    Machine learning-based routing using historical performance data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.index_names = ['text_index', 'image_index', 'code_index', 'table_index']
        
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, context: Dict[str, Any]) -> RoutingFeatures:
        """Extract features from query context for ML model."""
        return RoutingFeatures(
            query_length=len(context.get('query', '').split()),
            modality_scores=context.get('modality_scores', {}),
            intent_scores=context.get('intent_scores', {}),
            domain_scores=context.get('domain_scores', {}),
            complexity=context.get('complexity', 0.5),
            entity_count=context.get('entity_count', 0),
            historical_success=context.get('historical_success', {})
        )
    
    def predict_routing(self, features: RoutingFeatures) -> Dict[str, float]:
        """
        Predict routing scores for each index.
        Returns probability distribution over indexes.
        """
        if self.model is None:
            # Fallback to heuristic routing
            return self._heuristic_routing(features)
        
        # Prepare feature vector
        feature_vector = self._featurize(features)
        
        # Get routing scores
        scores = self.model.predict_proba([feature_vector])[0]
        
        return {
            index: float(score)
            for index, score in zip(self.index_names, scores)
        }
    
    def _heuristic_routing(self, features: RoutingFeatures) -> Dict[str, float]:
        """Fallback heuristic routing when model unavailable."""
        scores = {index: 0.1 for index in self.index_names}
        
        # Boost based on modality
        if 'image' in features.modality_scores:
            scores['image_index'] += features.modality_scores['image'] * 0.5
        
        if 'code' in features.modality_scores:
            scores['code_index'] += features.modality_scores['code'] * 0.5
        
        # Normalize
        total = sum(scores.values())
        return {k: v/total for k, v in scores.items()}
    
    def _featurize(self, features: RoutingFeatures) -> np.ndarray:
        """Convert features to model input vector."""
        vector = [
            features.query_length / 50,  # Normalized
            features.complexity,
            features.entity_count / 10,
            features.modality_scores.get('text', 0),
            features.modality_scores.get('image', 0),
            features.modality_scores.get('code', 0),
            features.modality_scores.get('table', 0),
        ]
        return np.array(vector)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Train routing model on historical data.
        
        training_data format:
        [{
            'query': '...',
            'features': RoutingFeatures,
            'successful_indexes': ['text_index', 'code_index'],
            'click_scores': {'text_index': 0.8, 'code_index': 0.6}
        }]
        """
        from sklearn.ensemble import RandomForestClassifier
        
        X = []
        y = []
        
        for sample in training_data:
            features = sample['features']
            feature_vector = self._featurize(features)
            
            # Label: best performing index
            best_index = max(sample['click_scores'], key=sample['click_scores'].get)
            label = self.index_names.index(best_index)
            
            X.append(feature_vector)
            y.append(label)
        
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)
    
    def save_model(self, path: str):
        """Save trained model."""
        import joblib
        joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        """Load trained model."""
        import joblib
        self.model = joblib.load(path)
```

### 2.3.3 Hybrid Routing Strategy

Combine rule-based and ML-based approaches:

```python
class HybridRouter:
    """
    Combines rule-based and ML-based routing for optimal decisions.
    """
    
    def __init__(self, rule_router: RuleBasedRouter, ml_router: MLRouter):
        self.rule_router = rule_router
        self.ml_router = ml_router
        self.use_ml_threshold = 0.7  # Confidence threshold for ML
    
    def route(self, context: Dict[str, Any]) -> RoutingDecision:
        """
        Use hybrid routing strategy.
        """
        # Get rule-based decision
        rule_decision = self.rule_router.route(context)
        
        # Get ML scores if confidence is high enough
        features = self.ml_router.extract_features(context)
        ml_scores = self.ml_router.predict_routing(features)
        
        # Check if ML agrees with rules
        ml_top_index = max(ml_scores, key=ml_scores.get)
        ml_confidence = ml_scores[ml_top_index]
        
        if ml_confidence >= self.use_ml_threshold:
            # Use ML prediction if confident
            return self._ml_decision(ml_scores, context)
        else:
            # Fall back to rule-based
            return rule_decision
    
    def _ml_decision(self, scores: Dict[str, float], context: Dict[str, Any]) -> RoutingDecision:
        """Create routing decision from ML scores."""
        # Select indexes above threshold
        threshold = 0.2
        selected_indexes = [idx for idx, score in scores.items() if score >= threshold]
        
        # Calculate top_k based on number of indexes
        top_k = max(5, 15 // len(selected_indexes))
        
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
```

---

## 2.4 Summary

This section covered adaptive retrieval patterns:

1. **Query Analysis**: Intent classification, complexity estimation, domain identification
2. **Routing Strategies**: Rule-based, ML-based, and hybrid approaches
3. **Decision Engine**: Configurable routing with priority-based rule application
4. **Production Considerations**: Fallback strategies, confidence thresholds, monitoring

Key takeaways:
- Adaptive routing improves retrieval quality by 25-40% over static approaches
- Rule-based routing provides interpretability and control
- ML-based routing captures complex patterns from historical data
- Hybrid approaches combine the best of both strategies

In the next section, we'll explore embedding strategies for multimodal content.

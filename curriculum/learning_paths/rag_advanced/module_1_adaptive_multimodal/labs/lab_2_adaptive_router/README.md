# Lab 2: Adaptive Router Implementation

## 🎯 Lab Objectives

By completing this lab, you will:
1. Implement a rule-based routing decision engine
2. Build query analysis components (intent, complexity, domain)
3. Create hybrid routing with ML-based fallback
4. Test routing decisions on diverse queries

## 📋 Prerequisites

- Completion of Lab 1 (Modality Detection)
- Understanding of routing patterns
- Python 3.10+ with async support

## ⏱️ Time Estimate: 4-5 hours

---

## Part 1: Query Analysis Components

### Task 1.1: Intent Classification

Implement a classifier that identifies query intent:

```python
from enum import Enum

class QueryIntent(Enum):
    INFORMATIONAL = "informational"    # Seeking knowledge
    NAVIGATIONAL = "navigational"      # Seeking specific item
    TRANSACTIONAL = "transactional"    # Wanting to perform action
    COMPARISON = "comparison"          # Comparing options
    TROUBLESHOOTING = "troubleshooting" # Solving problem
    EXPLORATORY = "exploratory"        # Browsing/discovering

class IntentClassifier:
    INTENT_PATTERNS = {
        QueryIntent.INFORMATIONAL: [
            r"what is\s+", r"explain\s+", r"describe\s+", r"how does\s+",
            r"tell me about\s+", r"define\s+", r"overview of\s+"
        ],
        QueryIntent.NAVIGATIONAL: [
            r"find\s+", r"locate\s+", r"show me the\s+", r"where is\s+",
            r"navigate to\s+", r"access\s+"
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
    
    def classify(self, query: str) -> tuple:
        # Your implementation here
        pass
```

### Task 1.2: Complexity Estimation

```python
from dataclasses import dataclass

@dataclass
class ComplexityMetrics:
    length_score: float
    vocabulary_score: float
    structure_score: float
    entity_count: int
    constraint_count: int
    overall_complexity: float

class ComplexityEstimator:
    COMPLEXITY_INDICATORS = {
        'constraints': ['only', 'exactly', 'specific', 'must', 'should'],
        'multi_part': ['and also', 'plus', 'additionally', 'furthermore'],
        'conditional': ['if', 'when', 'unless', 'assuming', 'given that'],
        'temporal': ['before', 'after', 'during', 'recently', 'latest']
    }
    
    def estimate(self, query: str) -> ComplexityMetrics:
        # Your implementation here
        pass
    
    def get_retrieval_params(self, complexity: ComplexityMetrics) -> dict:
        # Return top_k, search_depth, rerank based on complexity
        pass
```

### Task 1.3: Domain Identification

```python
class DomainIdentifier:
    DOMAIN_KEYWORDS = {
        'engineering': ['architecture', 'system', 'component', 'api', 'microservice'],
        'finance': ['revenue', 'profit', 'budget', 'forecast', 'investment'],
        'software': ['code', 'function', 'class', 'algorithm', 'debugging'],
        'data': ['database', 'query', 'table', 'pipeline', 'analytics']
    }
    
    def identify(self, query: str) -> dict:
        # Your implementation here
        pass
```

---

## Part 2: Rule-Based Router

### Task 2.1: Routing Rule Definition

```python
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

@dataclass
class RoutingRule:
    name: str
    condition: Callable[[dict], bool]
    action: Dict[str, Any]
    priority: int
    description: str

@dataclass
class RoutingDecision:
    target_indexes: List[str]
    parameters: Dict[str, Any]
    applied_rules: List[str]
    fallback: bool = False

class RuleBasedRouter:
    def __init__(self):
        self.rules = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        # Define routing rules with priorities
        # Higher priority rules are evaluated first
        pass
    
    def add_rule(self, rule: RoutingRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def route(self, context: Dict[str, Any]) -> RoutingDecision:
        # Evaluate rules and return routing decision
        pass
```

### Task 2.2: Implement Default Rules

Create rules for these scenarios:

| Scenario | Condition | Action |
|----------|-----------|--------|
| Image query | modality == 'image' | Route to image_index, top_k=10 |
| Code query | modality == 'code' | Route to code_index, top_k=5, rerank=true |
| Complex query | complexity > 0.6 | Hybrid search, top_k=20 |
| Troubleshooting | intent == 'troubleshooting' | Recency boost, top_k=15 |
| Comparison | intent == 'comparison' | Multiple indexes, diversity boost |
| Default | Always true | Text index, top_k=10 |

---

## Part 3: ML-Based Router

### Task 3.1: Feature Extraction

```python
from dataclasses import dataclass

@dataclass
class RoutingFeatures:
    query_length: int
    modality_scores: dict
    intent_scores: dict
    domain_scores: dict
    complexity: float
    entity_count: int

class MLRouter:
    def __init__(self, model_path: str = None):
        self.model = None
        self.index_names = ['text_index', 'image_index', 'code_index', 'table_index']
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, context: dict) -> RoutingFeatures:
        # Extract features from query context
        pass
    
    def predict_routing(self, features: RoutingFeatures) -> Dict[str, float]:
        # Return probability distribution over indexes
        pass
    
    def _heuristic_routing(self, features: RoutingFeatures) -> Dict[str, float]:
        # Fallback when model unavailable
        pass
```

---

## Part 4: Hybrid Router Integration

### Task 4.1: Combine Rule-Based and ML Routing

```python
class HybridRouter:
    def __init__(self, rule_router: RuleBasedRouter, ml_router: MLRouter):
        self.rule_router = rule_router
        self.ml_router = ml_router
        self.use_ml_threshold = 0.7
    
    def route(self, context: Dict[str, Any]) -> RoutingDecision:
        """
        Use hybrid routing strategy:
        1. Get rule-based decision
        2. Get ML scores
        3. Use ML if confident, otherwise use rules
        """
        pass
```

### Task 4.2: End-to-End Testing

```python
def test_adaptive_router():
    # Create router components
    rule_router = RuleBasedRouter()
    ml_router = MLRouter()
    hybrid_router = HybridRouter(rule_router, ml_router)
    
    # Test queries
    test_queries = [
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
    ]
    
    for test in test_queries:
        # Build context from query
        context = build_context(test['query'])
        
        # Get routing decision
        decision = hybrid_router.route(context)
        
        # Verify
        print(f"Query: {test['query']}")
        print(f"Indexes: {decision.target_indexes}")
        print(f"Expected: {test['expected_indexes']}")
        print()
```

---

## Part 5: Integration with Modality Detection

### Task 5.1: Full Pipeline

```python
class AdaptiveRetrievalPipeline:
    def __init__(self):
        self.modality_classifier = QueryModalityClassifier()
        self.intent_classifier = IntentClassifier()
        self.complexity_estimator = ComplexityEstimator()
        self.domain_identifier = DomainIdentifier()
        self.router = HybridRouter(RuleBasedRouter(), MLRouter())
    
    async def retrieve(self, query: str) -> dict:
        """
        Full adaptive retrieval pipeline:
        1. Analyze query (modality, intent, complexity, domain)
        2. Route to appropriate indexes
        3. Execute retrieval
        4. Fuse and rank results
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
            'complexity': complexity.overall_complexity,
            'domain': domain.primary_domain,
        }
        
        # Step 3: Route
        decision = self.router.route(context)
        
        # Step 4: Execute retrieval (mock for now)
        results = await self._execute_retrieval(query, decision)
        
        return {
            'query': query,
            'routing': decision,
            'results': results,
        }
    
    async def _execute_retrieval(self, query: str, decision: RoutingDecision):
        # Mock retrieval - implement with actual vector DB
        return []
```

---

## 📝 Deliverables

1. Complete `solution.py` with all routing components
2. Test cases demonstrating routing decisions
3. Evaluation showing routing accuracy
4. Analysis of rule conflicts and resolutions

## ✅ Success Criteria

- Intent classifier achieves >80% accuracy
- Complexity estimator provides meaningful scores
- Rule-based router handles all defined scenarios
- Hybrid router correctly chooses between rules and ML
- End-to-end pipeline produces valid routing decisions

## 🔍 Hints

- Use lambda functions for rule conditions
- Sort rules by priority before evaluation
- Implement fallback for edge cases
- Log routing decisions for debugging

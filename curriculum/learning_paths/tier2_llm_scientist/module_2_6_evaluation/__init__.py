"""
Module 2.6: Evaluation

LLM evaluation components:
- Benchmark implementations (MMLU, TruthfulQA, GSM8K, HumanEval)
- Human evaluation
- Model-based evaluation
- Feedback analysis
"""

from .benchmarks import (
    MMLUEvaluator,
    TruthfulQAEvaluator,
    GSM8KEvaluator,
    HumanEvalEvaluator,
    BenchmarkRunner,
)
from .human_eval import (
    HumanEvaluator,
    EvaluationTask,
    AnnotationGuidelines,
    QualityControl,
)
from .model_based_eval import (
    LLMJudge,
    PairwiseEvaluator,
    ScoringRubric,
    ModelBasedEvaluator,
)
from .feedback_analysis import (
    ErrorCategorizer,
    PatternDetector,
    ImprovementSuggester,
    FeedbackAnalyzer,
)

__all__ = [
    # Benchmarks
    "MMLUEvaluator",
    "TruthfulQAEvaluator",
    "GSM8KEvaluator",
    "HumanEvalEvaluator",
    "BenchmarkRunner",
    # Human Evaluation
    "HumanEvaluator",
    "EvaluationTask",
    "AnnotationGuidelines",
    "QualityControl",
    # Model-Based Evaluation
    "LLMJudge",
    "PairwiseEvaluator",
    "ScoringRubric",
    "ModelBasedEvaluator",
    # Feedback Analysis
    "ErrorCategorizer",
    "PatternDetector",
    "ImprovementSuggester",
    "FeedbackAnalyzer",
]

"""
LLM Engineering Module
======================

Large Language Model components implemented from scratch and with PyTorch.
Includes transformer architectures, attention mechanisms, fine-tuning methods,
evaluation frameworks, and safety utilities.
"""

from .transformer import (
    MultiHeadAttention,
    LayerNorm,
    TransformerEncoderLayer,
    BERT,
    GPT2,
    LearnedPositionalEncoding,
    get_sinusoidal_encoding,
)

from .attention import (
    scaled_dot_product_attention,
    DotProductAttention,
    MultiHeadAttention as TorchMultiHeadAttention,
    SelfAttention,
    CausalSelfAttention,
    CrossAttention,
    RelativePositionAttention,
    FlashAttention,
    GroupedQueryAttention,
    AttentionWithRoPE,
    create_attention_mechanism,
)

from .fine_tuning import (
    FineTuningMethod,
    FineTuningConfig,
    LoRALayer,
    apply_lora_to_model,
    AdapterLayer,
    apply_adapters_to_model,
    FineTuner,
    QLoRA,
    create_fine_tuner,
    evaluate_model,
)

from .evaluation import (
    RAGEvaluator,
    MLEvaluator,
    LLMEvaluator,
    BenchmarkRunner,
    EvaluationReport,
    EvaluationConfig,
    Metric,
)

from .benchmarks.performance_evaluation import (
    BenchmarkResult,
    PerformanceMetrics,
    BenchmarkSuite,
    ModelBenchmark,
    AlgorithmBenchmark,
    benchmark_decorator,
    ScalabilityBenchmark,
    CostPerformanceAnalyzer,
)

from .benchmarks.component_benchmarks import (
    ComponentBenchmarkResult,
    MathOperationsBenchmark,
    ClassicalMLBenchmark,
    DeepLearningBenchmark,
    LLMComponentsBenchmark,
    ProductionComponentsBenchmark,
    run_comprehensive_benchmarks,
    run_specific_component_benchmarks,
)

__all__ = [
    # Transformer
    "MultiHeadAttention",
    "LayerNorm",
    "TransformerEncoderLayer",
    "BERT",
    "GPT2",
    "LearnedPositionalEncoding",
    "get_sinusoidal_encoding",
    # Attention
    "scaled_dot_product_attention",
    "DotProductAttention",
    "TorchMultiHeadAttention",
    "SelfAttention",
    "CausalSelfAttention",
    "CrossAttention",
    "RelativePositionAttention",
    "FlashAttention",
    "GroupedQueryAttention",
    "AttentionWithRoPE",
    "create_attention_mechanism",
    # Fine-tuning
    "FineTuningMethod",
    "FineTuningConfig",
    "LoRALayer",
    "apply_lora_to_model",
    "AdapterLayer",
    "apply_adapters_to_model",
    "FineTuner",
    "QLoRA",
    "create_fine_tuner",
    "evaluate_model",
    # Evaluation
    "RAGEvaluator",
    "MLEvaluator",
    "LLMEvaluator",
    "BenchmarkRunner",
    "EvaluationReport",
    "EvaluationConfig",
    "Metric",
    # Benchmarks
    "BenchmarkResult",
    "PerformanceMetrics",
    "BenchmarkSuite",
    "ModelBenchmark",
    "AlgorithmBenchmark",
    "benchmark_decorator",
    "ScalabilityBenchmark",
    "CostPerformanceAnalyzer",
    "ComponentBenchmarkResult",
    "MathOperationsBenchmark",
    "ClassicalMLBenchmark",
    "DeepLearningBenchmark",
    "LLMComponentsBenchmark",
    "ProductionComponentsBenchmark",
    "run_comprehensive_benchmarks",
    "run_specific_component_benchmarks",
]

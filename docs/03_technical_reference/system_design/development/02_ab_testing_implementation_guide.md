# A/B Testing Implementation Guide

## Overview

This document provides a comprehensive guide to the A/B testing functionality implementation in the RAG Engine Mini. The A/B testing system enables experimentation with different models, prompts, and algorithms, which was marked as pending in the project completion checklist.

## Architecture

### Component Structure

The A/B testing functionality follows the same architectural patterns as the rest of the RAG Engine:

```
┌─────────────────┐    ┌──────────────────┐    └─────────────────┐
│   API Layer     │────│  Application     │────│   Domain/       │
│   (routes)      │    │  Services/       │    │   Ports/        │
│                 │    │  Use Cases       │    │   Adapters      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │ HTTP Requests          │ Business Logic        │ Interfaces &
         │                        │                       │ Implementations
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  A/B Testing    │    │ ABTesting       │
│   Endpoints     │    │  Service        │    │ Service Port    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **API Routes** (`src/api/v1/routes_ab_testing.py`): FastAPI endpoints for A/B testing
2. **A/B Testing Service** (`src/application/services/ab_testing_service.py`): Core A/B testing logic
3. **Dependency Injection** (`src/core/bootstrap.py`): Service registration and wiring

## Implementation Details

### 1. A/B Testing Service

The `ABTestingService` implements the `ABTestingServicePort` interface and provides:

- **Experiment Management**: Creation, activation, and termination of experiments
- **Variant Assignment**: Consistent assignment of users to variants
- **Event Tracking**: Recording of experiment-related events
- **Statistical Analysis**: Built-in significance testing and result interpretation
- **Sample Size Calculation**: Power analysis for experiment planning

Key methods:
```python
async def create_experiment(experiment: ABExperiment) -> ABExperiment
async def get_experiment(experiment_id: str) -> Optional[ABExperiment]
async def list_experiments(status: Optional[ExperimentStatus] = None) -> List[ABExperiment]
async def assign_variant(experiment_id: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> ExperimentAssignment
async def track_event(experiment_id: str, user_id: str, variant_name: str, event_type: str, value: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)
async def get_experiment_results(experiment_id: str) -> Optional[ExperimentResult]
async def update_experiment_status(experiment_id: str, status: ExperimentStatus) -> ABExperiment
async def calculate_sample_size(baseline_conversion_rate: float, minimum_detectable_effect: float, significance_level: float = 0.05, power: float = 0.8) -> int
```

### 2. Core Concepts

#### Experiment
An A/B test with defined variants, metrics, and configuration.

#### Variants
Different versions of the system being tested (e.g., different LLM models).

#### Assignment
The process of determining which variant a user sees.

#### Events
Metrics collected during the experiment (e.g., response time, user satisfaction).

#### Results
Statistical analysis of the collected data showing which variant performed better.

### 3. API Endpoints

The API provides endpoints for:
- Creating experiments (`POST /ab-testing/experiments`)
- Listing experiments (`GET /ab-testing/experiments`)
- Getting experiment details (`GET /ab-testing/experiments/{experiment_id}`)
- Assigning users to variants (`POST /ab-testing/experiments/{experiment_id}/assign`)
- Tracking events (`POST /ab-testing/experiments/{experiment_id}/track`)
- Getting results (`GET /ab-testing/experiments/{experiment_id}/results`)
- Updating experiment status (`PATCH /ab-testing/experiments/{experiment_id}/status`)
- Calculating sample size (`POST /ab-testing/calculate-sample-size`)

## API Usage

### Creating an Experiment

```bash
POST /ab-testing/experiments
Content-Type: application/json

{
  "name": "GPT-3.5 vs GPT-4 Response Quality",
  "description": "Comparing response quality between GPT-3.5 and GPT-4 for RAG queries",
  "status": "draft",
  "variants": [
    {
      "name": "gpt-3.5-control",
      "description": "Using GPT-3.5 as the base model",
      "traffic_split": 0.5,
      "config": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
      },
      "variant_type": "control"
    },
    {
      "name": "gpt-4-treatment",
      "description": "Using GPT-4 as the improved model",
      "traffic_split": 0.5,
      "config": {
        "model": "gpt-4",
        "temperature": 0.7
      },
      "variant_type": "treatment"
    }
  ],
  "metrics": ["response_time", "user_satisfaction_score", "answer_relevance"],
  "created_by": "rag-engine-admin",
  "hypothesis": "GPT-4 will produce higher quality answers with similar response times",
  "owner": "ai-team"
}
```

### Assigning Users to Variants

```bash
POST /ab-testing/experiments/{experiment_id}/assign
Content-Type: application/json

{
  "user_id": "user-123",
  "context": {
    "user_type": "premium"
  }
}
```

### Tracking Events

```bash
POST /ab-testing/experiments/{experiment_id}/track
Content-Type: application/json

{
  "user_id": "user-123",
  "variant_name": "gpt-4-treatment",
  "event_type": "user_satisfaction_score",
  "value": 4.5,
  "metadata": {
    "session_id": "session-456",
    "query_complexity": "high"
  }
}
```

### Getting Experiment Results

```bash
GET /ab-testing/experiments/{experiment_id}/results
```

Response:
```json
{
  "experiment_id": "exp-123",
  "variant_results": {
    "gpt-3.5-control": {
      "total_events": 150,
      "conversion_rate": 0.15,
      "average_value": 3.8
    },
    "gpt-4-treatment": {
      "total_events": 148,
      "conversion_rate": 0.18,
      "average_value": 4.2
    }
  },
  "statistical_significance": {
    "conversion_rate_vs_gpt-3.5-control": {
      "gpt-4-treatment": {
        "p_value": 0.023,
        "significant": true,
        "effect_size": 0.45
      }
    }
  },
  "winner": "gpt-4-treatment",
  "is_significant": true,
  "conclusion": "Variant 'gpt-4-treatment' is the winner with medium effect size and statistical significance."
}
```

## Integration Points

### Dependency Injection

The A/B testing service is registered in `src/core/bootstrap.py`:

```python
ab_test_service = ABTestingService()

return {
    # ... other services
    "ab_test_service": ab_test_service,
}
```

### API Integration

The routes are included in the main application through the routes_ab_testing import.

## Statistical Analysis

The A/B testing functionality includes built-in statistical analysis:

- **T-Tests**: Comparing means between variants
- **Effect Size**: Magnitude of the difference between variants
- **Confidence Intervals**: Range of values likely to contain the true effect
- **Power Analysis**: Sample size calculations for adequate statistical power

## Use Cases in RAG Systems

A/B testing is particularly valuable for RAG systems:

1. **Model Comparison**: Testing different LLMs for response quality
2. **Prompt Engineering**: Comparing different prompt strategies
3. **Retrieval Methods**: Evaluating vector vs keyword search effectiveness
4. **Chunking Strategies**: Assessing different document segmentation approaches
5. **Reranking Algorithms**: Testing various re-ranking methods
6. **System Parameters**: Optimizing temperature, top_p, and other settings

## Performance Considerations

1. **Event Storage**: Efficient storage of experiment events
2. **Assignment Speed**: Fast variant assignment with consistent hashing
3. **Statistical Calculations**: Optimized statistical computations
4. **Memory Management**: Efficient handling of large experiment datasets

## Security Considerations

1. **Data Isolation**: Tenant-specific experiment data
2. **Configuration Security**: Safe handling of model configurations
3. **Result Access**: Proper authorization for experiment results

## Educational Value

This implementation demonstrates:

1. **Clean Architecture**: Clear separation of concerns
2. **Port/Adapter Pattern**: Interface-based design
3. **Statistical Rigor**: Proper experimental design and analysis
4. **API Design**: RESTful endpoint design
5. **Experimentation Methodology**: Scientific approach to optimization
6. **Data-Driven Decisions**: Evidence-based system improvements

## Testing

The A/B testing functionality includes comprehensive tests in `tests/unit/test_ab_testing_service.py`:

- Experiment creation and management
- User assignment logic
- Event tracking functionality
- Statistical analysis accuracy
- Error condition handling

## Conclusion

The A/B testing functionality completes a critical feature that was marked as pending in the project completion checklist. It follows the same architectural patterns as the rest of the RAG Engine Mini, ensuring consistency and maintainability. The implementation provides comprehensive tools for conducting rigorous experiments with proper statistical analysis, enabling data-driven optimization of RAG systems.

This addition brings the RAG Engine Mini significantly closer to full completion, providing users with powerful tools to optimize their systems based on empirical evidence.
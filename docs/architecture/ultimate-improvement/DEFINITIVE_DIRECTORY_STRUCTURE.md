# 🗂️ DEFINITIVE DIRECTORY STRUCTURE

**AI-Mastery-2026: Complete Repository Organization**

| Document Info | Details |
|---------------|---------|
| **Version** | 1.0 (Definitive Edition) |
| **Date** | March 31, 2026 |
| **Status** | Target Architecture - Approved |
| **Author** | AI Engineering Tech Lead |
| **Migration Complexity** | Medium-High |
| **Estimated Migration Time** | 8-10 weeks |

---

## 📋 EXECUTIVE SUMMARY

### Purpose

This document defines the **complete, final directory structure** for AI-Mastery-2026, designed to:

- ✅ Enable content discovery in **<15 seconds**
- ✅ Support **1,000+ modules** without reorganization
- ✅ Provide **crystal-clear separation** of concerns
- ✅ Enable **10x scalable growth**
- ✅ Support **multi-modal content** (text, video, interactive, code)
- ✅ Be **i18n-ready** for global audiences
- ✅ Last **10 years** with minimal changes

### Key Design Decisions

1. **Audience-First Top Level**: Separate `curriculum/` (students), `src/` (developers), `docs/` (reference)
2. **Domain-Driven Subdivision**: Organize by topic within each audience section
3. **Progressive Disclosure**: Beginner content isolated from advanced
4. **Single Source of Truth**: No duplicate content across directories
5. **Scalable Naming**: Conventions that work at 10x current size

### Current vs Target State

| Aspect | Current State | Target State | Improvement |
|--------|---------------|--------------|-------------|
| **Root Files** | 30+ markdown files | 5 core files only | 85% reduction |
| **Curriculum Location** | Scattered (curriculum/, docs/curriculum/, docs/01_student_guide/) | Single `curriculum/` directory | Unified |
| **Code Organization** | 25K files in flat src/ | Domain-driven src/ structure | Clear boundaries |
| **Documentation** | Mixed audience in docs/ | Diátaxis framework | Audience-specific |
| **Assessments** | Scattered across repo | Centralized `assessments/` | Easy tracking |

---

## 🌳 COMPLETE DIRECTORY TREE

### Root Level Structure

```
AI-Mastery-2026/
│
├── 📖 README.md                              # Main hub with audience gateways
├── 📖 CONTRIBUTING.md                        # Contribution guidelines
├── 📖 CODE_OF_CONDUCT.md                     # Community standards
├── 📖 LICENSE                                # MIT License
├── 📖 SECURITY.md                            # Security policy
│
├── 🎓 curriculum/                            # STRUCTURED LEARNING CONTENT
│   ├── README.md                             # Curriculum overview
│   ├── learning-paths/                       # Student pathways by tier
│   ├── tracks/                               # Specialized cross-cutting tracks
│   ├── assessments/                          # Centralized assessments
│   └── certifications/                       # Certification pathways
│
├── 💻 src/                                   # PRODUCTION CODE
│   ├── README.md                             # src/ overview
│   ├── core/                                 # Core utilities (from scratch)
│   ├── ml/                                   # Machine Learning
│   ├── llm/                                  # LLM Fundamentals
│   ├── rag/                                  # RAG Systems
│   ├── agents/                               # AI Agents
│   ├── production/                           # Production Infrastructure
│   ├── utils/                                # Shared utilities
│   └── data/                                 # Data pipelines
│
├── 📓 notebooks/                             # INTERACTIVE CONTENT
│   ├── README.md
│   ├── 01_mathematical_foundations/
│   ├── 02_classical_ml/
│   ├── 03_deep_learning/
│   ├── 04_llm_fundamentals/
│   ├── 05_rag_systems/
│   ├── 06_agents/
│   └── 07_production_deployment/
│
├── 📚 docs/                                  # DOCUMENTATION (Diátaxis)
│   ├── README.md                             # Documentation hub
│   ├── tutorials/                            # Learning-oriented
│   ├── how-to/                               # Goal-oriented
│   ├── reference/                            # Information-oriented
│   ├── explanation/                          # Understanding-oriented
│   └── architecture/                         # Architecture decisions
│
├── 👥 community/                             # COMMUNITY & GOVERNANCE
│   ├── README.md
│   ├── governance/                           # Decision-making processes
│   ├── recognition/                          # Contributor recognition
│   ├── events/                               # Community events
│   └── code-of-conduct/                      # Enforcement guidelines
│
├── 🏢 industry/                              # INDUSTRY CONNECTIONS
│   ├── README.md
│   ├── hiring-partners/                      # Partner program
│   ├── career-services/                      # Job preparation
│   ├── skill-verification/                   # Skill validation
│   └── advisory-board/                       # Industry advisors
│
├── 🧪 tests/                                 # TEST SUITES
│   ├── README.md
│   ├── unit/                                 # Unit tests
│   ├── integration/                          # Integration tests
│   ├── e2e/                                  # End-to-end tests
│   └── fixtures/                             # Test fixtures
│
├── 🔧 scripts/                               # AUTOMATION SCRIPTS
│   ├── README.md
│   ├── setup/                                # Environment setup
│   ├── build/                                # Build scripts
│   ├── deploy/                               # Deployment scripts
│   └── maintenance/                          # Maintenance tasks
│
├── 📦 config/                                # CONFIGURATION
│   ├── README.md
│   ├── ci-cd/                                # CI/CD configuration
│   ├── docker/                               # Docker configurations
│   ├── environments/                         # Environment configs
│   └── linting/                              # Linting rules
│
├── 📊 datasets/                              # DATASETS
│   ├── README.md
│   ├── raw/                                  # Raw datasets (gitignored)
│   ├── processed/                            # Processed datasets
│   └── external/                             # External dataset references
│
├── 🎨 templates/                             # TEMPLATES
│   ├── README.md
│   ├── modules/                              # Module templates
│   ├── assessments/                          # Assessment templates
│   ├── projects/                             # Project templates
│   └── documentation/                        # Documentation templates
│
├── 📝 assessments/                           # ASSESSMENT CONTENT
│   ├── README.md
│   ├── quizzes/                              # Quiz questions
│   ├── coding-challenges/                    # Coding problems
│   ├── projects/                             # Project specifications
│   └── rubrics/                              # Evaluation criteria
│
├── 🏆 projects/                              # PROJECT SHOWCASE
│   ├── README.md
│   ├── beginner/                             # Beginner projects
│   ├── intermediate/                         # Intermediate projects
│   ├── advanced/                             # Advanced projects
│   └── capstone/                             # Capstone projects
│
├── 📄 .gitignore                             # Git ignore rules
├── 📄 .pre-commit-config.yaml                # Pre-commit hooks
├── 📄 pyproject.toml                         # Python project config
├── 📄 setup.py                               # Setup script
├── 📄 requirements.txt                       # Base requirements
├── 📄 requirements-dev.txt                   # Development requirements
├── 📄 docker-compose.yml                     # Docker Compose
├── 📄 Dockerfile                             # Base Dockerfile
├── 📄 Makefile                               # Makefile commands
└── 📄 environment.yml                        # Conda environment
```

---

## 📖 ROOT LEVEL FILES

### Required Root Files (5 only)

| File | Purpose | Size Limit | Content |
|------|---------|------------|---------|
| **README.md** | Main hub, audience gateways | <500 lines | Overview, quick start, audience paths |
| **CONTRIBUTING.md** | Contribution guidelines | <300 lines | How to contribute, workflows |
| **CODE_OF_CONDUCT.md** | Community standards | <200 lines | Behavior expectations |
| **LICENSE** | Legal license | Standard | MIT License text |
| **SECURITY.md** | Security policy | <100 lines | Reporting vulnerabilities |

### What NOT to Put at Root

❌ **Never add these to root**:
- Tutorial content (goes in `curriculum/` or `docs/tutorials/`)
- Technical documentation (goes in `docs/`)
- Configuration files (goes in `config/`)
- Scripts (goes in `scripts/`)
- Additional markdown files (consolidate into README.md or docs/)

**Exception**: Files required by platforms (GitHub, PyPI, etc.):
- `.gitignore`, `.github/`, `setup.py`, `pyproject.toml`, `requirements*.txt`

---

## 🎓 CURRICULUM DIRECTORY (`curriculum/`)

### Purpose

**Primary home for all structured learning content** - organized by learning progression.

### Structure

```
curriculum/
│
├── README.md                                 # Curriculum overview
│
├── learning-paths/                           # Tier-based progression
│   ├── README.md                             # Learning path guide
│   ├── tier-01-beginner/                     # Foundations (0-6 months)
│   │   ├── README.md
│   │   ├── course-01-mathematics/
│   │   │   ├── module-01-linear-algebra/
│   │   │   │   ├── README.md                 # Module overview
│   │   │   │   ├── lesson-01-vectors.md
│   │   │   │   ├── lesson-02-matrices.md
│   │   │   │   ├── lesson-03-decompositions.md
│   │   │   │   ├── exercises/
│   │   │   │   │   ├── practice-problems.md
│   │   │   │   │   └── solutions.md
│   │   │   │   ├── quiz/
│   │   │   │   │   └── quiz-01.json
│   │   │   │   └── project/
│   │   │   │       └── matrix-operations-from-scratch/
│   │   │   ├── module-02-calculus/
│   │   │   └── module-03-probability/
│   │   ├── course-02-python-for-ml/
│   │   ├── course-03-neural-networks/
│   │   └── course-04-nlp-fundamentals/
│   ├── tier-02-intermediate/                 # LLM Scientist (6-12 months)
│   │   ├── README.md
│   │   ├── course-01-transformer-architecture/
│   │   ├── course-02-llm-pretraining/
│   │   ├── course-03-fine-tuning/
│   │   └── course-04-evaluation-methods/
│   ├── tier-03-advanced/                     # LLM Engineer (12-18 months)
│   │   ├── README.md
│   │   ├── course-01-running-llms/
│   │   ├── course-02-vector-storage/
│   │   ├── course-03-rag-systems/
│   │   ├── course-04-advanced-rag/
│   │   ├── course-05-ai-agents/
│   │   └── course-06-llm-security/
│   └── tier-04-production/                   # Production & DevOps (18-24 months)
│       ├── README.md
│       ├── course-01-deployment-strategies/
│       ├── course-02-monitoring-observability/
│       ├── course-03-scaling-optimization/
│       └── course-04-mlops-pipelines/
│
├── tracks/                                   # Cross-cutting specializations
│   ├── README.md
│   ├── track-01-mathematics/
│   ├── track-02-python-programming/
│   ├── track-03-machine-learning/
│   ├── track-04-deep-learning/
│   ├── track-05-nlp/
│   ├── track-06-llm-architecture/
│   ├── track-07-rag-systems/
│   ├── track-08-ai-agents/
│   ├── track-09-security-safety/
│   ├── track-10-production-devops/
│   ├── track-11-multimodal-systems/
│   ├── track-12-reinforcement-learning/
│   ├── track-13-causal-inference/
│   ├── track-14-time-series/
│   └── track-15-graph-neural-networks/
│
├── assessments/                              # Centralized assessments
│   ├── README.md
│   ├── quizzes/                              # All quiz questions
│   │   ├── tier-01/
│   │   │   ├── mathematics-quiz-01.json
│   │   │   ├── mathematics-quiz-02.json
│   │   │   └── ...
│   │   ├── tier-02/
│   │   ├── tier-03/
│   │   └── tier-04/
│   ├── coding-challenges/                    # Coding problems
│   │   ├── README.md
│   │   ├── easy/
│   │   │   ├── challenge-01.md
│   │   │   └── ...
│   │   ├── medium/
│   │   └── hard/
│   ├── projects/                             # Project specifications
│   │   ├── README.md
│   │   ├── beginner/
│   │   │   ├── project-01-spec.md
│   │   │   └── ...
│   │   ├── intermediate/
│   │   ├── advanced/
│   │   └── capstone/
│   └── rubrics/                              # Evaluation criteria
│       ├── README.md
│       ├── project-rubrics/
│       │   ├── beginner-rubric.md
│       │   ├── intermediate-rubric.md
│       │   └── advanced-rubric.md
│       ├── coding-rubrics/
│       └── quiz-rubrics/
│
└── certifications/                           # Certification pathways
    ├── README.md
    ├── foundations-certificate/
    │   ├── requirements.md
    │   ├── learning-outcomes.md
    │   └── verification.md
    ├── llm-engineer-certificate/
    ├── advanced-specialist-certificate/
    └── expert-mastery-certificate/
```

### Module Template Structure

Every module in `curriculum/learning-paths/` follows this structure:

```
module-XX-module-name/
│
├── README.md                                 # Module overview (required)
│   - Title, description, prerequisites
│   - Learning objectives (Bloom's taxonomy)
│   - Time estimate
│   - Module map (visual)
│
├── theory/                                   # Theoretical content
│   ├── 01-introduction.md
│   ├── 02-core-concepts.md
│   └── 03-advanced-topics.md
│
├── practice/                                 # Hands-on exercises
│   ├── 01-guided-exercise.md
│   ├── 02-independent-practice.md
│   └── 03-challenge.md
│
├── labs/                                     # Lab exercises
│   ├── lab-01/
│   │   ├── README.md
│   │   ├── starter-code/
│   │   └── solution/
│   └── lab-02/
│
├── projects/                                 # Module project
│   ├── specification.md
│   ├── starter-code/
│   ├── solution/
│   └── rubric.md
│
├── assessments/                              # Module assessments
│   ├── knowledge-check.md
│   ├── quiz-questions.json
│   └── coding-challenge.md
│
├── resources/                                # Additional resources
│   ├── further-reading.md
│   ├── video-links.md
│   └── tools-frameworks.md
│
└── instructor/                               # Instructor resources (optional)
    ├── teaching-guide.md
    ├── common-misconceptions.md
    └── discussion-prompts.md
```

### Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| **Tier directories** | `tier-XX-level/` | `tier-01-beginner/` |
| **Course directories** | `course-XX-name/` | `course-01-mathematics/` |
| **Module directories** | `module-XX-name/` | `module-01-linear-algebra/` |
| **Lesson files** | `lesson-XX-name.md` | `lesson-01-vectors.md` |
| **Lab directories** | `lab-XX-name/` | `lab-01-vector-ops/` |
| **Quiz files** | `quiz-XX-topic.json` | `quiz-01-matrices.json` |

---

## 💻 SOURCE CODE DIRECTORY (`src/`)

### Purpose

**Production-grade Python code** - organized by domain and functionality.

### Structure

```
src/
│
├── __init__.py                               # Package initialization
├── README.md                                 # src/ overview
│
├── core/                                     # Core utilities (from scratch)
│   ├── __init__.py
│   ├── math/
│   │   ├── __init__.py
│   │   ├── vectors.py                        # Vector operations
│   │   ├── matrices.py                       # Matrix operations
│   │   ├── calculus.py                       # Numerical calculus
│   │   ├── decompositions.py                 # Matrix decompositions (SVD, QR, Cholesky)
│   │   └── __init__.py
│   ├── probability/
│   │   ├── __init__.py
│   │   ├── distributions.py                  # Probability distributions
│   │   ├── bayes.py                          # Bayesian inference
│   │   └── hypothesis_testing.py             # Statistical tests
│   └── optimization/
│       ├── __init__.py
│       ├── optimizers.py                     # Gradient descent variants
│       └── loss_functions.py                 # Loss functions
│
├── ml/                                       # Machine Learning
│   ├── __init__.py
│   ├── classical/
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── decision_trees.py
│   │   ├── random_forests.py
│   │   ├── svm.py
│   │   ├── kmeans.py
│   │   └── __init__.py
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   ├── layers.py                         # Neural network layers
│   │   ├── activations.py                    # Activation functions
│   │   ├── losses.py                         # Loss functions
│   │   ├── mlp.py                            # Multi-layer perceptron
│   │   ├── cnn.py                            # Convolutional neural networks
│   │   └── rnn.py                            # Recurrent neural networks
│   └── vision/
│       ├── __init__.py
│       ├── resnet.py
│       ├── vit.py
│       └── __init__.py
│
├── llm/                                      # LLM Fundamentals
│   ├── __init__.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── attention.py                      # Attention mechanisms
│   │   ├── transformer.py                    # Transformer architecture
│   │   ├── tokenization.py                   # Tokenization algorithms
│   │   └── positional_encodings.py           # Positional encodings
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretraining.py                    # Pretraining procedures
│   │   ├── fine_tuning.py                    # Fine-tuning (LoRA, QLoRA)
│   │   └── __init__.py
│   └── alignment/
│       ├── __init__.py
│       ├── rlhf.py                           # RLHF implementation
│       └── dpo.py                            # Direct Preference Optimization
│
├── rag/                                      # RAG Systems
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py                           # Base chunker class
│   │   ├── fixed_size.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   └── hierarchical.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sentence_transformers.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   └── hybrid.py
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── cross_encoder.py
│   ├── vector_stores/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── faiss_store.py
│   │   └── qdrant_store.py
│   └── pipeline/
│       ├── __init__.py
│       ├── base.py
│       ├── standard.py
│       └── advanced.py
│
├── agents/                                   # AI Agents
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── react.py
│   │   └── planning.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py
│   │   └── long_term.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── search.py
│   │   └── code_interpreter.py
│   └── multi_agent/
│       ├── __init__.py
│       ├── coordinator.py
│       └── protocols.py
│
├── production/                               # Production Infrastructure
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── middleware/
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── tracing.py
│   │   └── alerting.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── docker.py
│   │   ├── kubernetes.py
│   │   └── vllm.py
│   └── security/
│       ├── __init__.py
│       ├── auth.py
│       ├── rate_limit.py
│       └── guardrails.py
│
├── utils/                                    # Shared utilities
│   ├── __init__.py
│   ├── errors.py
│   ├── logging.py
│   ├── config.py
│   └── types.py
│
└── data/                                     # Data pipelines
    ├── __init__.py
    ├── loading.py
    ├── preprocessing.py
    └── versioning.py
```

### Import Hierarchy Rules

**Rule 1: Intra-domain imports are free**
```python
# ✅ Allowed: Within same domain
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.hybrid import HybridRetriever
```

**Rule 2: Cross-domain imports require dependency injection**
```python
# ❌ Avoid: Direct cross-domain import
from src.llm.architecture.transformer import Transformer

# ✅ Prefer: Dependency injection
class RAGPipeline:
    def __init__(self, embedding_model: BaseEmbedding):
        self.embedding_model = embedding_model
```

**Rule 3: Utils are last resort**
```python
# ❌ Avoid: Dumping everything in utils
from src.utils.helpers import everything

# ✅ Prefer: Domain-specific utilities
from src.rag.utils.chunking_helpers import chunk_by_semantics
```

---

## 📚 DOCUMENTATION DIRECTORY (`docs/`)

### Purpose

**Reference documentation** following Diátaxis framework - organized by audience need.

### Structure (Diátaxis Framework)

```
docs/
│
├── README.md                                 # Documentation hub
│   - Audience gateways (Student | Developer | Instructor | Hiring Manager)
│   - Quick links
│   - Search bar
│
├── tutorials/                                # LEARNING-ORIENTED
│   ├── README.md
│   ├── getting-started/
│   │   ├── quickstart.md
│   │   ├── installation.md
│   │   └── first-project.md
│   ├── beginner/
│   │   ├── your-first-neural-network.md
│   │   └── training-lstm-text-generator.md
│   ├── intermediate/
│   │   ├── building-rag-system.md
│   │   └── fine-tuning-llm.md
│   └── advanced/
│       ├── multi-agent-systems.md
│       └── production-deployment.md
│
├── how-to/                                   # GOAL-ORIENTED
│   ├── README.md
│   ├── data-preparation/
│   │   ├── how-to-prepare-dataset.md
│   │   └── how-to-handle-imbalanced-data.md
│   ├── model-training/
│   │   ├── how-to-train-transformer.md
│   │   └── how-to-finetune-with-lora.md
│   ├── deployment/
│   │   ├── how-to-deploy-with-docker.md
│   │   └── how-to-setup-monitoring.md
│   └── troubleshooting/
│       ├── how-to-debug-slow-inference.md
│       └── how-to-fix-vanishing-gradients.md
│
├── reference/                                # INFORMATION-ORIENTED
│   ├── README.md
│   ├── api-reference/
│   │   ├── src.core.math.md
│   │   ├── src.ml.classical.md
│   │   ├── src.llm.architecture.md
│   │   └── src.rag.pipeline.md
│   ├── configuration/
│   │   ├── environment-variables.md
│   │   └── config-files.md
│   └── cli-reference/
│       ├── commands.md
│       └── examples.md
│
├── explanation/                              # UNDERSTANDING-ORIENTED
│   ├── README.md
│   ├── concepts/
│   │   ├── attention-mechanism.md
│   │   ├── transformer-architecture.md
│   │   └── rag-systems.md
│   ├── comparisons/
│   │   ├── pytorch-vs-tensorflow.md
│   │   └── rag-vs-finetuning.md
│   └── deep-dives/
│       ├── backpropagation-math.md
│       └── transformer-attention-derivation.md
│
├── architecture/                             # ARCHITECTURE DECISIONS
│   ├── README.md
│   ├── decisions/                            # Architecture Decision Records (ADRs)
│   │   ├── ADR-001-monorepo-structure.md
│   │   ├── ADR-002-python-first.md
│   │   └── ADR-003-diátaxis-framework.md
│   ├── diagrams/                             # Architecture diagrams
│   │   ├── system-overview.png
│   │   └── data-flow.png
│   └── ultimate-improvement/                 # Current improvement initiative
│       ├── ULTIMATE_REPOSITORY_VISION.md
│       ├── DEFINITIVE_DIRECTORY_STRUCTURE.md
│       ├── CURRICULUM_ARCHITECTURE.md
│       ├── CODE_ARCHITECTURE.md
│       ├── DOCUMENTATION_ARCHITECTURE.md
│       ├── STUDENT_JOURNEY_DESIGN.md
│       ├── CONTRIBUTOR_ECOSYSTEM.md
│       ├── INDUSTRY_INTEGRATION_HUB.md
│       ├── SCALABILITY_AND_PERFORMANCE.md
│       ├── MIGRATION_MASTERPLAN.md
│       ├── QUICK_REFERENCE_COMPENDIUM.md
│       └── IMPLEMENTATION_ROADMAP_2026.md
│
└── curriculum/                               # CURRICULUM REFERENCE
    ├── README.md
    ├── learning-paths/
    ├── tracks/
    └── assessments/
```

### Diátaxis Framework Application

| Quadrant | Purpose | Key Question | Content Type |
|----------|---------|--------------|--------------|
| **Tutorials** | Learning | "How do I learn X?" | Step-by-step guides |
| **How-to** | Doing | "How do I accomplish X?" | Task-oriented guides |
| **Reference** | Knowing | "What does X do?" | API docs, specs |
| **Explanation** | Understanding | "Why does X work?" | Concepts, theory |

---

## 📝 ASSESSMENTS DIRECTORY (`assessments/`)

### Purpose

**Centralized assessment content** - all quizzes, challenges, and project specifications.

### Structure

```
assessments/
│
├── README.md
│
├── quizzes/                                  # Quiz questions (JSON format)
│   ├── tier-01/
│   │   ├── mathematics/
│   │   │   ├── quiz-01-linear-algebra.json
│   │   │   ├── quiz-02-calculus.json
│   │   │   └── quiz-03-probability.json
│   │   └── python/
│   ├── tier-02/
│   ├── tier-03/
│   └── tier-04/
│
├── coding-challenges/                        # Coding problems
│   ├── README.md
│   ├── easy/
│   │   ├── 01-vector-operations.md
│   │   ├── 02-matrix-multiplication.md
│   │   └── ...
│   ├── medium/
│   │   ├── 01-implement-attention.md
│   │   ├── 02-build-rag-pipeline.md
│   │   └── ...
│   └── hard/
│       ├── 01-distributed-training.md
│       ├── 02-production-rag-system.md
│       └── ...
│
├── projects/                                 # Project specifications
│   ├── README.md
│   ├── beginner/
│   │   ├── 01-linear-regression-from-scratch.md
│   │   ├── 02-neural-network-mnist.md
│   │   └── ...
│   ├── intermediate/
│   │   ├── 01-transformer-implementation.md
│   │   ├── 02-rag-chatbot.md
│   │   └── ...
│   ├── advanced/
│   │   ├── 01-multi-agent-system.md
│   │   ├── 02-llm-finetuning-pipeline.md
│   │   └── ...
│   └── capstone/
│       ├── README.md
│       ├── specifications/
│       │   ├── 01-github-issue-classifier.md
│       │   ├── 02-medical-diagnosis-assistant.md
│       │   └── ...
│       └── rubrics/
│           ├── technical-rubric.md
│           └── presentation-rubric.md
│
└── rubrics/                                  # Evaluation criteria
    ├── README.md
    ├── project-rubrics/
    │   ├── beginner-rubric.md
    │   ├── intermediate-rubric.md
    │   ├── advanced-rubric.md
    │   └── capstone-rubric.md
    ├── coding-rubrics/
    │   ├── code-quality.md
    │   ├── testing.md
    │   └── documentation.md
    └── quiz-rubrics/
        └── quiz-design-guidelines.md
```

---

## 🏆 PROJECTS DIRECTORY (`projects/`)

### Purpose

**Project showcase and portfolio** - completed student projects and exemplars.

### Structure

```
projects/
│
├── README.md
│
├── beginner/                                 # Beginner projects
│   ├── 01-linear-regression-from-scratch/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── report.md
│   │   └── demo.mp4
│   └── ...
│
├── intermediate/                             # Intermediate projects
│   ├── 01-rag-chatbot/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── report.md
│   │   └── demo.mp4
│   └── ...
│
├── advanced/                                 # Advanced projects
│   ├── 01-multi-agent-system/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── report.md
│   │   └── demo.mp4
│   └── ...
│
└── capstone/                                 # Capstone projects
    ├── 01-github-issue-classifier/
    │   ├── README.md
    │   ├── code/
    │   ├── report.md
    │   ├── demo.mp4
    │   └── deployment/
    └── ...
```

---

## 🧪 TESTS DIRECTORY (`tests/`)

### Purpose

**Comprehensive test suites** - organized by test type and domain.

### Structure

```
tests/
│
├── README.md
├── conftest.py                               # Pytest configuration
│
├── unit/                                     # Unit tests
│   ├── core/
│   │   ├── test_vectors.py
│   │   ├── test_matrices.py
│   │   └── test_calculus.py
│   ├── ml/
│   │   ├── test_classical.py
│   │   └── test_deep_learning.py
│   └── llm/
│       ├── test_attention.py
│       └── test_transformer.py
│
├── integration/                              # Integration tests
│   ├── test_rag_pipeline.py
│   ├── test_agent_system.py
│   └── test_api_endpoints.py
│
├── e2e/                                      # End-to-end tests
│   ├── test_learning_path.py
│   ├── test_assessment_flow.py
│   └── test_deployment.py
│
└── fixtures/                                 # Test fixtures
    ├── sample_data/
    ├── mock_models/
    └── test_datasets/
```

---

## 🔧 SCRIPTS DIRECTORY (`scripts/`)

### Purpose

**Automation scripts** - setup, build, deploy, maintain.

### Structure

```
scripts/
│
├── README.md
│
├── setup/                                    # Environment setup
│   ├── install-dependencies.sh
│   ├── setup-environment.py
│   └── validate-installation.py
│
├── build/                                    # Build scripts
│   ├── build-docs.sh
│   ├── build-docker.sh
│   └── package-release.sh
│
├── deploy/                                   # Deployment scripts
│   ├── deploy-staging.sh
│   ├── deploy-production.sh
│   └── rollback.sh
│
├── maintenance/                              # Maintenance tasks
│   ├── cleanup-cache.py
│   ├── update-dependencies.py
│   └── validate-links.py
│
└── curriculum/                               # Curriculum scripts
    ├── generate-module.py
    ├── validate-assessments.py
    └── export-progress.py
```

---

## 📊 NAMING CONVENTIONS

### File Naming

| Type | Convention | Example |
|------|------------|---------|
| **Python modules** | `snake_case.py` | `linear_algebra.py` |
| **Markdown files** | `kebab-case.md` | `getting-started.md` |
| **Jupyter notebooks** | `NN_topic.ipynb` | `01_vectors_python.ipynb` |
| **Test files** | `test_topic.py` | `test_matrices.py` |
| **Configuration** | `kebab-case.ext` | `docker-compose.yml` |
| **Scripts** | `verb-noun.sh/py` | `setup-environment.py` |

### Directory Naming

| Type | Convention | Example |
|------|------------|---------|
| **Domain directories** | `snake_case` | `deep_learning` |
| **Tier/level directories** | `tier-XX-name` | `tier-01-beginner` |
| **Module directories** | `module-XX-name` | `module-01-linear-algebra` |
| **Lesson directories** | `lesson-XX-name` | `lesson-01-vectors` |
| **Track directories** | `track-XX-name` | `track-01-mathematics` |

### Branch Naming

| Type | Convention | Example |
|------|------------|---------|
| **Feature branches** | `feature/description` | `feature/add-rag-reranking` |
| **Bug fixes** | `fix/description` | `fix/attention-mask-bug` |
| **Documentation** | `docs/description` | `docs/add-api-reference` |
| **Curriculum** | `curriculum/description` | `curriculum/add-transformer-module` |
| **Release branches** | `release/X.Y.Z` | `release/1.0.0` |

---

## 🔗 CROSS-DIRECTORY LINKING STRATEGY

### When to Cross-Reference

**✅ Do cross-reference when**:
- Connecting theory (curriculum) to implementation (src)
- Linking tutorials to reference docs
- Referencing prerequisites
- Showing real-world examples

**❌ Don't cross-reference when**:
- Content belongs in the target directory (move it instead)
- Creating circular dependencies
- Linking to unstable/experimental content

### Link Format

```markdown
### Recommended Format

**Theory**: See [Attention Mechanisms](../../curriculum/learning-paths/tier-02/course-01-transformer/lesson-02-attention.md)

**Implementation**: See [`src/llm/architecture/attention.py`](../../src/llm/architecture/attention.py)

**Tutorial**: See [Building Your First Transformer](../../docs/tutorials/intermediate/building-transformer.md)

**Reference**: See [API Reference: MultiHeadAttention](../../docs/reference/api-reference/src.llm.architecture.md#multiheadattention)
```

---

## 🚫 ANTI-PATTERNS TO AVOID

### Anti-Pattern 1: Flat Directory Explosion

```
❌ WRONG: All modules in one directory
curriculum/
├── module-01/
├── module-02/
├── ...
├── module-500/   # 😱 500 items in one directory
```

```
✅ RIGHT: Hierarchical organization
curriculum/
├── tier-01-beginner/
│   ├── course-01-mathematics/
│   │   ├── module-01-linear-algebra/
│   │   └── module-02-calculus/
│   └── course-02-python/
```

---

### Anti-Pattern 2: Duplicate Content

```
❌ WRONG: Same content in multiple places
curriculum/learning-paths/rag/module-01.md
docs/tutorials/rag/module-01.md    # Duplicate!
src/rag/README.md                  # Also duplicate!
```

```
✅ RIGHT: Single source of truth
curriculum/learning-paths/rag/module-01.md    # Canonical location
docs/tutorials/rag/README.md                  # Links to curriculum
src/rag/README.md                             # Code overview only
```

---

### Anti-Pattern 3: Mixed Audiences

```
❌ WRONG: Student and developer content mixed
docs/
├── student-tutorial.md
├── api-reference.md
├── career-advice.md        # Doesn't belong here
└── hiring-guide.md         # Wrong audience
```

```
✅ RIGHT: Audience separation
docs/
├── tutorials/              # Students
├── reference/              # Developers
industry/
├── hiring-partners/        # Hiring managers
└── career-services/        # Students (career-focused)
```

---

### Anti-Pattern 4: Unclear Purpose

```
❌ WRONG: Generic directory names
misc/
stuff/
things/
other/
```

```
✅ RIGHT: Explicit directory names
scripts/
├── maintenance/
├── setup/
└── deployment/
```

---

### Anti-Pattern 5: Deep Nesting

```
❌ WRONG: Too many levels deep
curriculum/
└── tier-01/
    └── beginner/
        └── fundamentals/
            └── mathematics/
                └── linear-algebra/
                    └── module-01/
                        └── lesson-01/
                            └── content/
                                └── file.md    # 😱 10 levels deep!
```

```
✅ RIGHT: Reasonable depth (3-5 levels)
curriculum/
└── tier-01-beginner/
    └── course-01-mathematics/
        └── module-01-linear-algebra/
            └── lesson-01-vectors.md    # 4 levels
```

---

## 📐 SCALABILITY CONSIDERATIONS

### Handling 1,000+ Modules

**Strategy**: Hierarchical grouping with clear boundaries

```
At 100 modules:
curriculum/
└── modules/
    └── module-001/ ... module-100/

At 1,000 modules:
curriculum/
├── tier-01/
│   ├── course-01/
│   │   ├── module-001/ ... module-025/
│   │   └── module-026/ ... module-050/
│   └── course-02/

At 10,000 modules:
curriculum/
├── tier-01-beginner/
│   ├── track-01-mathematics/
│   │   ├── course-01-linear-algebra/
│   │   │   ├── module-001/ ... module-010/
│   │   │   └── module-011/ ... module-020/
│   │   └── course-02-calculus/
│   └── track-02-python/
```

### Handling 100+ Contributors

**Strategy**: Clear ownership and contribution pathways

```
Each directory has:
├── README.md              # Purpose and scope
├── OWNERS.md              # Maintainers for this directory
└── CONTRIBUTING.md        # How to contribute to this section
```

### Handling 10,000+ Students

**Strategy**: Performance-optimized structure

- CDN for static assets
- Search-optimized organization
- Lazy loading for large directories
- Caching strategies for frequently accessed content

---

## ✅ VALIDATION CHECKLIST

Before finalizing directory structure:

- [ ] **Student Test**: Can a beginner find "linear algebra" in <15 seconds?
- [ ] **Developer Test**: Can a developer find `attention.py` in <15 seconds?
- [ ] **Contributor Test**: Does a contributor know where to add a new module?
- [ ] **Scale Test**: Can this handle 10x more content without reorganization?
- [ ] **Duplication Test**: Is there any content in multiple places?
- [ ] **Audience Test**: Are different audiences served separately?
- [ ] **Naming Test**: Are all names consistent with conventions?
- [ ] **Link Test**: Do cross-directory links work correctly?
- [ ] **Git Test**: Does git handle the structure efficiently?
- [ ] **CI/CD Test**: Can CI/CD target specific directories?

---

## 📝 DOCUMENT HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | March 31, 2026 | AI Engineering Tech Lead | Initial definitive structure |

---

## 🔗 RELATED DOCUMENTS

This document is part of the **Ultimate Repository Improvement** series:

1. ✅ [ULTIMATE_REPOSITORY_VISION.md](./ULTIMATE_REPOSITORY_VISION.md) - Strategic vision
2. ✅ **DEFINITIVE_DIRECTORY_STRUCTURE.md** (this document) - Complete directory tree
3. 📋 [CURRICULUM_ARCHITECTURE.md](./CURRICULUM_ARCHITECTURE.md) - Curriculum structure
4. 💻 [CODE_ARCHITECTURE.md](./CODE_ARCHITECTURE.md) - Code organization
5. 📖 [DOCUMENTATION_ARCHITECTURE.md](./DOCUMENTATION_ARCHITECTURE.md) - Documentation system
6. 🎓 [STUDENT_JOURNEY_DESIGN.md](./STUDENT_JOURNEY_DESIGN.md) - Student experience
7. 👥 [CONTRIBUTOR_ECOSYSTEM.md](./CONTRIBUTOR_ECOSYSTEM.md) - Contribution system
8. 🏢 [INDUSTRY_INTEGRATION_HUB.md](./INDUSTRY_INTEGRATION_HUB.md) - Industry connections
9. ⚡ [SCALABILITY_AND_PERFORMANCE.md](./SCALABILITY_AND_PERFORMANCE.md) - Scaling strategy
10. 🔄 [MIGRATION_MASTERPLAN.md](./MIGRATION_MASTERPLAN.md) - Migration guide
11. 📖 [QUICK_REFERENCE_COMPENDIUM.md](./QUICK_REFERENCE_COMPENDIUM.md) - Quick reference
12. 📅 [IMPLEMENTATION_ROADMAP_2026.md](./IMPLEMENTATION_ROADMAP_2026.md) - 16-week plan

---

<div align="center">

**📁 Structure defined. Next: Curriculum architecture.**

[Next: CURRICULUM_ARCHITECTURE.md](./CURRICULUM_ARCHITECTURE.md)

</div>

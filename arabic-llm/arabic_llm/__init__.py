"""
Arabic LLM - Main Package

A comprehensive system for fine-tuning Arabic language models
to become expert linguists, poets, and language investigators.

Usage:
    import arabic_llm
    
    # Create training example
    example = arabic_llm.TrainingExample(...)
    
    # Process books
    processor = arabic_llm.BookProcessor(...)
    segments = processor.process_all_books()
    
    # Generate dataset
    generator = arabic_llm.DatasetGenerator(...)
    dataset = generator.generate(segments)
    
    # Run autonomous research
    agent = arabic_llm.ResearchAgent(...)
    agent.run(experiments=100)
"""

# Import version
from .version import (
    __version__,
    __version_info__,
    __author__,
    __email__,
    __license__,
)

# Import core components (flat namespace for common usage)
from .core import (
    TrainingExample,
    Role,
    Skill,
    Level,
    Domain,
    Style,
    TaskType,
    DatasetConfig,
    DatasetStatistics,
    validate_example,
    Book,
    TextSegment,
    BookProcessor,
    ExampleGenerator,
    DatasetGenerator,
)

# Import pipeline components
from .pipeline import (
    ArabicTextCleaner,
    CleanedBook,
    PipelineStats,
)

# Import integration components
from .integration import (
    SystemBookIntegration,
    HadithRecord,
    TafseerRecord,
)

# Import agents (optional - requires torch)
try:
    from .agents import (
        ResearchAgent,
        ExperimentProposal,
    )
except (ImportError, OSError) as e:
    # torch or other dependencies not available
    import warnings
    warnings.warn(f"Agents disabled: {e}")
    ResearchAgent = None
    ExperimentProposal = None

# Import utilities
from .utils import (
    setup_logging,
    read_jsonl,
    write_jsonl,
    count_arabic_chars,
    get_arabic_ratio,
)

# Define public API
__all__ = [
    # Version
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    "__license__",
    # Core (most commonly used)
    "TrainingExample",
    "Role",
    "Skill",
    "Level",
    "DatasetConfig",
    "DatasetStatistics",
    "validate_example",
    "Book",
    "TextSegment",
    "BookProcessor",
    "ExampleGenerator",
    "DatasetGenerator",
    # Pipeline
    "ArabicTextCleaner",
    "CleanedBook",
    "PipelineStats",
    # Integration
    "SystemBookIntegration",
    "HadithRecord",
    "TafseerRecord",
    # Agents
    "ResearchAgent",
    "ExperimentProposal",
    # Utils
    "setup_logging",
    "read_jsonl",
    "write_jsonl",
    "count_arabic_chars",
    "get_arabic_ratio",
]

# Package metadata
__package_name__ = "arabic_llm"
__description__ = "Arabic LLM Fine-Tuning System"
__url__ = "https://github.com/your-org/arabic-llm"
__status__ = "Production"

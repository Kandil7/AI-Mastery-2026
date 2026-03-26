"""
Arabic LLM - Core Business Logic

This subpackage contains the core business logic for Arabic LLM training:
- Data schemas (TrainingExample, Role, Skill, etc.)
- Instruction templates
- Book processing
- Dataset generation
"""

from .schema import (
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
    write_jsonl,
    read_jsonl,
    compute_statistics,
)

from .schema_enhanced import (
    # Enhanced roles (15 total)
    # Enhanced skills (48+ total)
    pass
)

from .templates import (
    Template,
    get_templates,
    get_random_template,
    get_template_by_id,
    ALL_TEMPLATES,
    POETRY_METERS,
    POETRY_TOPICS,
)

from .book_processor import (
    Book,
    TextSegment,
    BookProcessor,
    process_all_books,
)

from .dataset_generator import (
    ExampleGenerator,
    DatasetGenerator,
)

__all__ = [
    # Schema
    "TrainingExample",
    "Role",
    "Skill",
    "Level",
    "Domain",
    "Style",
    "TaskType",
    "DatasetConfig",
    "DatasetStatistics",
    "validate_example",
    "write_jsonl",
    "read_jsonl",
    "compute_statistics",
    # Templates
    "Template",
    "get_templates",
    "get_random_template",
    "get_template_by_id",
    "ALL_TEMPLATES",
    "POETRY_METERS",
    "POETRY_TOPICS",
    # Processing
    "Book",
    "TextSegment",
    "BookProcessor",
    "process_all_books",
    # Generation
    "ExampleGenerator",
    "DatasetGenerator",
]

"""
Arabic LLM Fine-Tuning Package

A comprehensive system for fine-tuning Arabic language models
to become expert linguists, poets, and language investigators.

Usage:
    from arabic_llm import schema, book_processor, dataset_generator, fine_tuning
"""

__version__ = "1.0.0"
__author__ = "Arabic LLM Project"

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

from .instruction_templates import (
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

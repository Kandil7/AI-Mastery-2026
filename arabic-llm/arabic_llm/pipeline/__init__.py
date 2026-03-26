"""
Arabic LLM - Data Processing Pipelines

This subpackage contains data processing pipelines:
- Text cleaning and normalization
- Text segmentation
- Quality validation
"""

from .cleaning import (
    TextCleaner,
    DataCleaningPipeline,
    BookMetadata,
    Page,
    Chapter,
    CleanedBook,
)

from .segmentation import (
    Segmenter,
    segment_by_page,
    segment_by_chapter,
)

from .validation import (
    QualityValidator,
    validate_arabic_ratio,
    validate_completeness,
)

__all__ = [
    # Cleaning
    "TextCleaner",
    "DataCleaningPipeline",
    "BookMetadata",
    "Page",
    "Chapter",
    "CleanedBook",
    # Segmentation
    "Segmenter",
    "segment_by_page",
    "segment_by_chapter",
    # Validation
    "QualityValidator",
    "validate_arabic_ratio",
    "validate_completeness",
]

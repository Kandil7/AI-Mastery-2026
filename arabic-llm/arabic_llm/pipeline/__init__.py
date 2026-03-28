"""
Arabic LLM - Data Processing Pipelines

This subpackage contains data processing pipelines:
- Text cleaning and normalization (7-stage pipeline)
- Text segmentation
- Quality validation
"""

from .cleaning import (
    ArabicTextCleaner,
    BookMetadata,
    Page,
    Chapter,
    CleanedBook,
    PipelineStats,
    setup_logging,
)

from .deduplication import (
    Document,
    DeduplicationStats,
    ExactDeduplicator,
    NearDuplicateDeduplicator,
    SentenceDeduplicator,
    ArabicDeduplicationPipeline,
)

# Segmentation and validation (TODO: implement)
# from .segmentation import (
#     Segmenter,
#     segment_by_page,
#     segment_by_chapter,
# )
# from .validation import (
#     QualityValidator,
#     validate_arabic_ratio,
#     validate_completeness,
# )

__all__ = [
    # Cleaning
    "ArabicTextCleaner",
    "BookMetadata",
    "Page",
    "Chapter",
    "CleanedBook",
    "PipelineStats",
    "setup_logging",
    # Deduplication
    "Document",
    "DeduplicationStats",
    "ExactDeduplicator",
    "NearDuplicateDeduplicator",
    "SentenceDeduplicator",
    "ArabicDeduplicationPipeline",
    # Segmentation (TODO)
    # "Segmenter",
    # "segment_by_page",
    # "segment_by_chapter",
    # Validation (TODO)
    # "QualityValidator",
    # "validate_arabic_ratio",
    # "validate_completeness",
]

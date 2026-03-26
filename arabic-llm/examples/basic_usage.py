"""
Arabic LLM - Basic Usage Example

This example demonstrates basic usage of the Arabic LLM package:
1. Load and process books
2. Generate training dataset
3. Validate examples
"""

from arabic_llm.core import (
    TrainingExample,
    Role,
    Skill,
    Level,
    BookProcessor,
    DatasetGenerator,
    DatasetConfig,
)
from arabic_llm.pipeline import DataCleaningPipeline
from arabic_llm.utils import setup_logging


def main():
    """Main example function"""
    
    # Setup logging
    logger = setup_logging("arabic_llm_example")
    logger.info("Starting Arabic LLM example")
    
    # Example 1: Create a training example manually
    logger.info("Creating training example...")
    example = TrainingExample(
        instruction="أعرب الجملة التالية إعراباً مفصلاً",
        input="العلمُ نورٌ والجهلُ ظلامٌ.",
        output="العلمُ: مبتدأ مرفوع وعلامة رفعه الضمة الظاهرة...",
        role=Role.TUTOR,
        skills=[Skill.NAHW, Skill.BALAGHA],
        level=Level.INTERMEDIATE,
        domain="education",
        difficulty=2,
    )
    
    logger.info(f"Created example with role={example.role.value}")
    logger.info(f"Skills: {[s.value for s in example.skills]}")
    
    # Example 2: Process books (commented out - requires actual data)
    """
    logger.info("Processing books...")
    processor = BookProcessor(
        books_dir="../datasets/extracted_books",
        metadata_dir="../datasets/metadata",
        output_dir="data/processed",
    )
    
    # Process up to 100 books
    segments = list(processor.process_books(max_books=100))
    logger.info(f"Processed {len(segments)} text segments")
    """
    
    # Example 3: Clean text
    logger.info("Testing text cleaning...")
    cleaner = DataCleaningPipeline(
        books_dir="../datasets/extracted_books",
        metadata_dir="../datasets/metadata",
        output_dir="data/cleaned",
        workers=4,
    )
    
    # Test cleaning on sample text
    sample_text = "  نص   تجريبي  مع  فراغات  متعددة  "
    cleaned_text = cleaner.clean_text(sample_text)
    logger.info(f"Original: '{sample_text}'")
    logger.info(f"Cleaned: '{cleaned_text}'")
    
    # Example 4: Generate dataset (commented out - requires processed data)
    """
    logger.info("Generating dataset...")
    config = DatasetConfig(
        target_examples=1000,
        role_distribution={
            "tutor": 0.35,
            "proofreader": 0.25,
            "poet": 0.20,
            "muhhaqiq": 0.15,
            "assistant_general": 0.05,
        }
    )
    
    generator = DatasetGenerator(
        books_dir="../datasets/extracted_books",
        metadata_dir="../datasets/metadata",
        output_dir="data/jsonl",
        config=config,
    )
    
    stats = generator.generate(target_examples=1000)
    logger.info(f"Generated {stats.total_examples} examples")
    logger.info(f"Role distribution: {stats.by_role}")
    """
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()

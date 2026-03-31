#!/usr/bin/env python3
"""
Complete End-to-End Arabic LLM Pipeline

This script runs the complete pipeline from raw books to trained model:
1. Data cleaning (7-stage pipeline)
2. Dataset generation (JSONL with roles/skills)
3. Quality verification
4. Training preparation

Usage:
    python scripts/complete_pipeline.py \
        --books-dir datasets/extracted_books \
        --metadata-dir datasets/metadata \
        --output-dir data/jsonl \
        --target-examples 61500

Author: Arabic LLM Project
Version: 2.4.0
Date: March 26, 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "arabic_llm"))

from arabic_llm.pipeline import DataCleaningPipeline
from arabic_llm.core import DatasetGenerator, DatasetConfig
from arabic_llm.utils import setup_logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Complete Arabic LLM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python complete_pipeline.py --target-examples 61500
  
  # Run with custom directories
  python complete_pipeline.py \\
      --books-dir datasets/extracted_books \\
      --metadata-dir datasets/metadata \\
      --output-dir data/jsonl
  
  # Run with specific number of workers
  python complete_pipeline.py --workers 16
        """
    )
    
    parser.add_argument(
        "--books-dir",
        type=str,
        default="../datasets/extracted_books",
        help="Path to extracted books directory"
    )
    
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="../datasets/metadata",
        help="Path to metadata directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/jsonl",
        help="Output directory for JSONL dataset"
    )
    
    parser.add_argument(
        "--target-examples",
        type=int,
        default=61500,
        help="Target number of training examples"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for cleaning"
    )
    
    parser.add_argument(
        "--max-books",
        type=int,
        default=None,
        help="Maximum number of books to process (None = all)"
    )
    
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip cleaning step (use already processed data)"
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation step"
    )
    
    return parser.parse_args()


def run_cleaning(args, logger):
    """Run data cleaning pipeline"""
    logger.info("="*70)
    logger.info("STEP 1: Data Cleaning")
    logger.info("="*70)
    
    cleaning_pipeline = DataCleaningPipeline(
        books_dir=args.books_dir,
        metadata_dir=args.metadata_dir,
        output_dir="data/processed",
        workers=args.workers,
    )
    
    logger.info(f"Processing books from: {args.books_dir}")
    logger.info(f"Using {args.workers} workers")
    
    if args.max_books:
        logger.info(f"Limiting to {args.max_books} books")
    
    # Run pipeline
    stats = cleaning_pipeline.run_pipeline(max_books=args.max_books)
    
    logger.info(f"✓ Cleaning complete!")
    logger.info(f"  Total books: {stats.get('total_books', 0)}")
    logger.info(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
    logger.info(f"  Average Arabic ratio: {stats.get('avg_arabic_ratio', 0):.1%}")
    
    return stats


def run_generation(args, logger):
    """Run dataset generation"""
    logger.info("="*70)
    logger.info("STEP 2: Dataset Generation")
    logger.info("="*70)
    
    config = DatasetConfig(
        target_examples=args.target_examples,
        role_distribution={
            "tutor": 0.35,
            "proofreader": 0.25,
            "poet": 0.20,
            "muhhaqiq": 0.15,
            "assistant_general": 0.05,
        }
    )
    
    generator = DatasetGenerator(
        books_dir=args.books_dir,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        config=config,
    )
    
    logger.info(f"Generating {args.target_examples:,} examples...")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Generate dataset
    stats = generator.generate()
    
    logger.info(f"✓ Generation complete!")
    logger.info(f"  Total examples: {stats.total_examples:,}")
    logger.info(f"  Role distribution:")
    for role, count in stats.by_role.items():
        logger.info(f"    {role}: {count:,} ({count/stats.total_examples*100:.1f}%)")
    logger.info(f"  Skill distribution:")
    for skill, count in list(stats.by_skill.items())[:5]:
        logger.info(f"    {skill}: {count:,}")
    if len(stats.by_skill) > 5:
        logger.info(f"    ... and {len(stats.by_skill) - 5} more skills")
    
    return stats


def generate_summary(cleaning_stats, generation_stats, args):
    """Generate pipeline summary report"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "2.4.0",
        "parameters": {
            "books_dir": args.books_dir,
            "metadata_dir": args.metadata_dir,
            "output_dir": args.output_dir,
            "target_examples": args.target_examples,
            "workers": args.workers,
            "max_books": args.max_books,
        },
        "cleaning": cleaning_stats,
        "generation": {
            "total_examples": generation_stats.total_examples,
            "by_role": generation_stats.by_role,
            "by_skill": generation_stats.by_skill,
            "by_level": generation_stats.by_level,
        },
    }
    
    # Save summary
    summary_file = Path(args.output_dir) / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


def main():
    """Main pipeline function"""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("arabic_llm_pipeline")
    
    print("="*70)
    print("Arabic LLM - Complete Pipeline")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target examples: {args.target_examples:,}")
    print(f"Workers: {args.workers}")
    print("="*70)
    
    # Initialize stats
    cleaning_stats = {}
    generation_stats = None
    
    # Step 1: Cleaning
    if not args.skip_cleaning:
        cleaning_stats = run_cleaning(args, logger)
    else:
        logger.info("Skipping cleaning step")
    
    # Step 2: Generation
    if not args.skip_generation:
        generation_stats = run_generation(args, logger)
    else:
        logger.info("Skipping generation step")
    
    # Generate summary
    if generation_stats:
        summary = generate_summary(cleaning_stats, generation_stats, args)
        
        # Print final summary
        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)
        print(f"Output directory: {args.output_dir}")
        print(f"Total examples: {generation_stats.total_examples:,}")
        print(f"Summary saved to: {summary['timestamp']}")
        print("="*70)
        
        logger.info("Pipeline completed successfully!")
    else:
        logger.info("Pipeline completed with skipped steps")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

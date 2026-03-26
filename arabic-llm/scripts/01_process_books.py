#!/usr/bin/env python3
"""
Step 1: Process Books

Process extracted books from Shamela dataset and prepare them for
training data generation.

Usage:
    python scripts/01_process_books.py \
        --books-dir ../datasets/extracted_books \
        --metadata-dir ../datasets/metadata \
        --output-dir data/raw
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add arabic_llm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arabic_llm.core import BookProcessor, process_all_books, DatasetConfig


def main():
    parser = argparse.ArgumentParser(description="Process Shamela books for Arabic LLM dataset")
    
    parser.add_argument(
        "--books-dir",
        type=str,
        required=True,
        help="Path to extracted books directory"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Path to metadata directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=1000,
        help="Maximum number of books to process"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to process (default: all linguistic categories)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    books_dir = Path(args.books_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    
    if not books_dir.exists():
        print(f"Error: Books directory not found: {books_dir}")
        sys.exit(1)
    
    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 1: Process Books")
    print("=" * 60)
    print(f"\nBooks directory: {books_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max books: {args.max_books}")
    
    # Create dataset config
    config = DatasetConfig()
    if args.categories:
        config.source_categories = args.categories
    
    # Initialize processor
    print("\nInitializing book processor...")
    processor = BookProcessor(
        books_dir=str(books_dir),
        metadata_dir=str(metadata_dir),
        output_dir=str(output_dir),
    )
    
    # Load metadata
    print("\nLoading book metadata...")
    num_books = processor.load_metadata()
    print(f"Loaded metadata for {num_books} books")
    
    # Show category distribution
    print("\nCategory distribution:")
    category_counts = {}
    for book in processor._books_cache.values():
        category_counts[book.category] = category_counts.get(book.category, 0) + 1
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat}: {count}")
    
    # Process books
    print(f"\nProcessing up to {args.max_books} books...")
    all_segments = []
    stats = {
        "by_category": {},
        "by_type": {},
        "by_author": {},
    }
    
    for segment in processor.process_books(
        categories=config.source_categories,
        max_books=args.max_books,
    ):
        all_segments.append(segment)
        
        # Update stats
        cat = segment.category
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        seg_type = segment.segment_type
        stats["by_type"][seg_type] = stats["by_type"].get(seg_type, 0) + 1
    
    # Save processed data
    print(f"\nSaving processed segments...")
    output_file = output_dir / "processed_segments.json"
    num_saved = processor.save_processed_data(all_segments, str(output_file))
    print(f"Saved {num_saved} segments to {output_file}")
    
    # Save processing report
    report = {
        "processed_at": datetime.now().isoformat(),
        "total_books_available": num_books,
        "books_processed": len(set(s.book_id for s in all_segments)),
        "total_segments": len(all_segments),
        "by_category": stats["by_category"],
        "by_type": stats["by_type"],
        "config": {
            "max_books": args.max_books,
            "categories": args.categories or "all_linguistic",
        }
    }
    
    report_file = output_dir / "processing_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved processing report to {report_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nTotal Segments: {len(all_segments)}")
    print(f"Books Processed: {report['books_processed']}")
    print(f"\nSegments by Category:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    print(f"\nSegments by Type:")
    for seg_type, count in stats["by_type"].items():
        print(f"  {seg_type}: {count}")
    
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

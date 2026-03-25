#!/usr/bin/env python3
"""
Step 2: Generate Dataset

Generate JSONL training dataset from processed book segments using
instruction templates.

Usage:
    python scripts/02_generate_dataset.py \
        --input-dir data/raw \
        --output-dir data/jsonl \
        --config configs/data_config.yaml
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_generator import DatasetGenerator
from schema import DatasetConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate Arabic LLM training dataset")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with processed segments"
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
        default="data/jsonl",
        help="Output directory for JSONL dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=50000,
        help="Target number of training examples"
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=1000,
        help="Maximum books to use"
    )
    parser.add_argument(
        "--books-dir",
        type=str,
        required=True,
        help="Path to extracted books directory"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Run 01_process_books.py first")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        config_data = {}
    else:
        config_data = load_config(str(config_path))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 2: Generate Dataset")
    print("=" * 60)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target examples: {args.target_examples}")
    
    # Create dataset config
    dataset_config = DatasetConfig(
        target_examples=args.target_examples,
    )
    
    # Override with config file values
    if "dataset" in config_data:
        if "target_examples" in config_data["dataset"]:
            dataset_config.target_examples = config_data["dataset"]["target_examples"]
    
    if "role_distribution" in config_data:
        dataset_config.role_distribution = config_data["role_distribution"]
    
    if "source_categories" in config_data:
        dataset_config.source_categories = config_data["source_categories"]
    
    # Initialize generator
    print("\nInitializing dataset generator...")
    generator = DatasetGenerator(
        books_dir=args.books_dir,
        metadata_dir=args.metadata_dir,
        output_dir=str(output_dir),
        config=dataset_config,
    )
    
    # Generate dataset
    stats = generator.generate(
        target_examples=args.target_examples,
        max_books=args.max_books,
    )
    
    # Generate train/val/test splits
    print("\nGenerating train/val/test splits...")
    split_config = config_data.get("data_split", {
        "train_ratio": 0.90,
        "val_ratio": 0.05,
        "test_ratio": 0.05,
    })
    
    counts = generator.generate_split_datasets(
        train_ratio=split_config.get("train_ratio", 0.90),
        val_ratio=split_config.get("val_ratio", 0.05),
        test_ratio=split_config.get("test_ratio", 0.05),
    )
    
    # Save generation report
    report = {
        "generated_at": datetime.now().isoformat(),
        "target_examples": args.target_examples,
        "actual_examples": stats.total_examples,
        "role_distribution": stats.role_counts,
        "skill_distribution": stats.skill_counts,
        "level_distribution": stats.level_counts,
        "source_books": stats.source_books,
        "split_counts": counts,
        "config": {
            "role_distribution": dict(dataset_config.role_distribution),
            "source_categories": dataset_config.source_categories,
        }
    }
    
    report_file = output_dir / "generation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved generation report to {report_file}")
    
    # Save sample examples
    sample_file = output_dir / "sample_examples.jsonl"
    import random
    from schema import read_jsonl
    
    all_examples = read_jsonl(str(output_dir / "training_data.jsonl"))
    sample_size = min(100, len(all_examples))
    sample_examples = random.sample(all_examples, sample_size)
    
    from schema import write_jsonl
    write_jsonl(sample_examples, str(sample_file))
    print(f"Saved {sample_size} sample examples to {sample_file}")
    
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'training_data.jsonl'} (full dataset)")
    print(f"  - {output_dir / 'train.jsonl'} (training set)")
    print(f"  - {output_dir / 'val.jsonl'} (validation set)")
    print(f"  - {output_dir / 'test.jsonl'} (test set)")
    print(f"  - {report_file}")
    print(f"  - {sample_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

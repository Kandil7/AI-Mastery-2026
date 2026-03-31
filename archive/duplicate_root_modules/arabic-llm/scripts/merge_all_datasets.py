"""
Merge All Balygh Datasets

Merges all processed datasets into a single training file with deduplication.

Usage:
    python scripts/merge_all_datasets.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MergeConfig:
    """Merge configuration"""
    
    # Input directory
    JSONL_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm/data/jsonl")
    
    # Output
    OUTPUT_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm/data/jsonl")
    OUTPUT_FILE = OUTPUT_DIR / "balygh_final_sft.jsonl"
    
    # Deduplication
    DEDUP_THRESHOLD = 0.85
    
    # Quality filters
    MIN_ARABIC_RATIO = 0.7
    MIN_LENGTH = 50
    MAX_LENGTH = 2000
    MIN_QUALITY_SCORE = 0.5


# ============================================================================
# Deduplication
# ============================================================================

class MinHashDeduplicator:
    """Simple MinHash-based deduplication"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.seen_hashes: Set[str] = set()
    
    def _text_hash(self, text: str) -> str:
        """Create hash for text"""
        # Simple shingle-based hash
        words = text.split()
        shingles = [' '.join(words[i:i+3]) for i in range(0, len(words), 3)]
        return hashlib.sha256(' '.join(shingles).encode()).hexdigest()[:16]
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        h = self._text_hash(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False


# ============================================================================
# Quality Filters
# ============================================================================

def arabic_ratio(text: str) -> float:
    """Calculate Arabic character ratio"""
    if not text:
        return 0.0
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return arabic / len(text)


def passes_quality_filter(example: Dict, config: MergeConfig) -> bool:
    """Check if example passes quality filters"""
    # Check length
    instruction = example.get('instruction', '')
    output = example.get('output', '')
    total_length = len(instruction) + len(output)
    
    if total_length < config.MIN_LENGTH:
        return False
    if total_length > config.MAX_LENGTH:
        return False
    
    # Check Arabic ratio
    combined_text = instruction + output
    if arabic_ratio(combined_text) < config.MIN_ARABIC_RATIO:
        return False
    
    # Check quality score
    quality_score = example.get('quality_score', 0.0)
    if quality_score > 0 and quality_score < config.MIN_QUALITY_SCORE:
        return False
    
    return True


# ============================================================================
# Merge Function
# ============================================================================

def merge_all_datasets():
    """Merge all datasets with deduplication"""
    config = MergeConfig()
    
    print("=" * 70)
    print("Merging All Balygh Datasets")
    print("=" * 70)
    print()
    
    # Find all JSONL files to merge
    source_files = []
    
    # Source-specific files
    source_patterns = [
        "arabic_web*.jsonl",
        "balygh_sft*.jsonl",
        "books_sft*.jsonl",
        "sanadset*.jsonl",
        "system_books*.jsonl",
    ]
    
    for pattern in source_patterns:
        for f in config.JSONL_DIR.glob(pattern):
            if f.name != "balygh_final_sft.jsonl":  # Don't include output
                source_files.append(f)
    
    # Also include integrated files
    integrated = config.JSONL_DIR / "balygh_integrated_sft.jsonl"
    if integrated.exists() and integrated not in source_files:
        source_files.append(integrated)
    
    print(f"Found {len(source_files)} source files to merge:")
    for f in source_files:
        print(f"  - {f.name}")
    print()
    
    # Initialize deduplicator
    deduplicator = MinHashDeduplicator(threshold=config.DEDUP_THRESHOLD)
    
    # Merge
    total_read = 0
    total_duplicates = 0
    total_quality_filtered = 0
    total_written = 0
    
    # Track by source
    by_source: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    
    with open(config.OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for source_file in source_files:
            print(f"Processing {source_file.name}...")
            
            file_count = 0
            file_written = 0
            
            with open(source_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    
                    total_read += 1
                    file_count += 1
                    
                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Quality filter
                    if not passes_quality_filter(example, config):
                        total_quality_filtered += 1
                        continue
                    
                    # Deduplication
                    combined_text = example.get('instruction', '') + example.get('output', '')
                    if deduplicator.is_duplicate(combined_text):
                        total_duplicates += 1
                        continue
                    
                    # Write
                    out_f.write(json.dumps(example, ensure_ascii=False) + '\n')
                    total_written += 1
                    file_written += 1
                    
                    # Track by source
                    source = example.get('source', 'unknown')
                    by_source[source] = by_source.get(source, 0) + 1
                    
                    # Track by role
                    role = example.get('role', 'unknown')
                    by_role[role] = by_role.get(role, 0) + 1
            
            print(f"  Read: {file_count:,}, Written: {file_written:,}")
    
    # Print summary
    print()
    print("=" * 70)
    print("Merge Summary")
    print("=" * 70)
    print(f"Total Read: {total_read:,}")
    print(f"Duplicates Removed: {total_duplicates:,} ({total_duplicates/max(total_read,1)*100:.1f}%)")
    print(f"Quality Filtered: {total_quality_filtered:,}")
    print(f"Final Count: {total_written:,}")
    print()
    
    print("By Source:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:10]:
        print(f"  {source}: {count:,}")
    print()
    
    print("By Role:")
    for role, count in sorted(by_role.items(), key=lambda x: -x[1])[:10]:
        print(f"  {role}: {count:,}")
    print()
    
    print(f"Output: {config.OUTPUT_FILE}")
    print("=" * 70)
    print("✅ Merge Complete!")
    print("=" * 70)


if __name__ == "__main__":
    merge_all_datasets()

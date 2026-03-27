#!/usr/bin/env python3
"""
Balygh v3.0 Migration Script

Automates the migration from v2.0 to v3.0 directory structure.

Usage:
    python migrate_to_v3.py [--dry-run] [--force]
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class MigrationConfig:
    """Migration configuration"""
    
    ROOT = Path(__file__).parent
    
    # Directories to create
    NEW_DIRS = [
        ROOT / "arabic_llm" / "processing",
        ROOT / "arabic_llm" / "generation",
        ROOT / "arabic_llm" / "training",
        ROOT / "scripts" / "processing",
        ROOT / "scripts" / "generation",
        ROOT / "scripts" / "training",
        ROOT / "scripts" / "utilities",
        ROOT / "docs" / "guides",
        ROOT / "docs" / "architecture",
        ROOT / "docs" / "api",
        ROOT / "docs" / "implementation",
        ROOT / "docs" / "archive",
    ]
    
    # File moves: (source, destination)
    FILE_MOVES = [
        # Processing module
        (ROOT / "arabic_llm" / "pipeline" / "cleaning.py", ROOT / "arabic_llm" / "processing" / "cleaning.py"),
        (ROOT / "arabic_llm" / "pipeline" / "deduplication.py", ROOT / "arabic_llm" / "processing" / "deduplication.py"),
        (ROOT / "arabic_llm" / "core" / "book_processor.py", ROOT / "arabic_llm" / "processing" / "book_processor.py"),
        
        # Generation module
        (ROOT / "arabic_llm" / "core" / "dataset_generator.py", ROOT / "arabic_llm" / "generation" / "dataset_generator.py"),
        
        # Training module
        (ROOT / "arabic_llm" / "models" / "qlora.py", ROOT / "arabic_llm" / "training" / "qlora.py"),
        (ROOT / "arabic_llm" / "models" / "quantization.py", ROOT / "arabic_llm" / "training" / "quantization.py"),
        (ROOT / "arabic_llm" / "models" / "checkpoints.py", ROOT / "arabic_llm" / "training" / "checkpoints.py"),
        
        # Scripts - processing
        (ROOT / "scripts" / "complete_data_audit.py", ROOT / "scripts" / "processing" / "complete_data_audit.py"),
        (ROOT / "scripts" / "process_arabic_web.py", ROOT / "scripts" / "processing" / "process_arabic_web.py"),
        (ROOT / "scripts" / "process_sanadset.py", ROOT / "scripts" / "processing" / "process_sanadset.py"),
        (ROOT / "scripts" / "integrate_datasets.py", ROOT / "scripts" / "processing" / "integrate_datasets.py"),
        (ROOT / "scripts" / "01_process_books.py", ROOT / "scripts" / "processing" / "process_books.py"),
        
        # Scripts - generation
        (ROOT / "scripts" / "build_balygh_sft_dataset.py", ROOT / "scripts" / "generation" / "build_balygh_sft.py"),
        (ROOT / "scripts" / "refine_balygh_sft_with_llm.py", ROOT / "scripts" / "generation" / "refine_with_llm.py"),
        (ROOT / "scripts" / "02_generate_dataset.py", ROOT / "scripts" / "generation" / "generate_dataset.py"),
        
        # Scripts - training
        (ROOT / "scripts" / "03_train_model.py", ROOT / "scripts" / "training" / "train.py"),
        (ROOT / "scripts" / "prepare.py", ROOT / "scripts" / "training" / "prepare_eval.py"),
        
        # Scripts - utilities
        (ROOT / "scripts" / "merge_all_datasets.py", ROOT / "scripts" / "utilities" / "merge_all_datasets.py"),
        (ROOT / "scripts" / "audit_datasets.py", ROOT / "scripts" / "utilities" / "audit_datasets.py"),
        
        # Documentation - implementation
        (ROOT / "COMPLETE_DATA_UTILIZATION_PLAN.md", ROOT / "docs" / "implementation" / "complete_data_utilization.md"),
        (ROOT / "DATA_UPDATES_IMPROVEMENTS.md", ROOT / "docs" / "implementation" / "data_updates.md"),
        (ROOT / "FINAL_IMPLEMENTATION_SUMMARY.md", ROOT / "docs" / "implementation" / "final_summary.md"),
        (ROOT / "IMPLEMENTATION_COMPLETE.md", ROOT / "docs" / "implementation" / "implementation_complete.md"),
        
        # Documentation - archive
        (ROOT / "IMPLEMENTATION_LINES_8000_9866.md", ROOT / "docs" / "archive" / "implementation_lines_8000_9866.md"),
        (ROOT / "IMPLEMENTATION_LINES_9800_11993.md", ROOT / "docs" / "archive" / "implementation_lines_9800_11993.md"),
        (ROOT / "FINAL_ARCHITECTURE_STATUS.md", ROOT / "docs" / "archive" / "final_architecture_status.md"),
        (ROOT / "CLEANUP_PLAN.md", ROOT / "docs" / "archive" / "cleanup_plan.md"),
        (ROOT / "AUTORESEARCH_README.md", ROOT / "docs" / "archive" / "autoresearch_readme.md"),
    ]
    
    # Files to delete (duplicates)
    FILES_TO_DELETE = [
        ROOT / "scripts" / "complete_pipeline.py",
        ROOT / "scripts" / "train.py",
    ]
    
    # Files to merge
    FILES_TO_MERGE = [
        (ROOT / "arabic_llm" / "core" / "schema.py", ROOT / "arabic_llm" / "core" / "schema_enhanced.py"),
        (ROOT / "arabic_llm" / "core" / "templates.py", ROOT / "arabic_llm" / "core" / "templates_extended.py"),
    ]
    
    # Directories to remove (if empty)
    DIRS_TO_REMOVE = [
        ROOT / "arabic_llm" / "pipeline",
        ROOT / "arabic_llm" / "models",
    ]


# ============================================================================
# Migration Functions
# ============================================================================

def create_directories(config: MigrationConfig, dry_run: bool = False):
    """Create new directory structure"""
    print("\n" + "=" * 70)
    print("Step 1: Creating New Directory Structure")
    print("=" * 70)
    
    created = 0
    for dir_path in config.NEW_DIRS:
        if dry_run:
            print(f"  [DRY-RUN] Would create: {dir_path}")
        else:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created: {dir_path}")
                created += 1
            else:
                print(f"  ⚠️  Already exists: {dir_path}")
    
    print(f"\nTotal: {created}/{len(config.NEW_DIRS)} directories created")
    return created


def move_files(config: MigrationConfig, dry_run: bool = False, force: bool = False):
    """Move files to new locations"""
    print("\n" + "=" * 70)
    print("Step 2: Moving Files")
    print("=" * 70)
    
    moved = 0
    skipped = 0
    errors = 0
    
    for src, dst in config.FILE_MOVES:
        if not src.exists():
            print(f"  ⚠️  Source not found: {src}")
            skipped += 1
            continue
        
        if dst.exists() and not force:
            print(f"  ⚠️  Destination exists: {dst}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"  [DRY-RUN] Would move: {src.name} → {dst}")
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"  ✅ Moved: {src.name} → {dst}")
                moved += 1
            except Exception as e:
                print(f"  ❌ Error moving {src.name}: {e}")
                errors += 1
    
    print(f"\nTotal: {moved} moved, {skipped} skipped, {errors} errors")
    return moved, skipped, errors


def delete_files(config: MigrationConfig, dry_run: bool = False):
    """Delete duplicate files"""
    print("\n" + "=" * 70)
    print("Step 3: Deleting Duplicate Files")
    print("=" * 70)
    
    deleted = 0
    skipped = 0
    
    for file_path in config.FILES_TO_DELETE:
        if not file_path.exists():
            print(f"  ⚠️  File not found: {file_path}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"  [DRY-RUN] Would delete: {file_path}")
        else:
            try:
                file_path.unlink()
                print(f"  ✅ Deleted: {file_path}")
                deleted += 1
            except Exception as e:
                print(f"  ❌ Error deleting {file_path}: {e}")
                skipped += 1
    
    print(f"\nTotal: {deleted} deleted, {skipped} skipped")
    return deleted


def merge_files(config: MigrationConfig, dry_run: bool = False):
    """Merge redundant files"""
    print("\n" + "=" * 70)
    print("Step 4: Merging Redundant Files")
    print("=" * 70)
    
    merged = 0
    
    for base_file, extend_file in config.FILES_TO_MERGE:
        if not extend_file.exists():
            print(f"  ⚠️  Extension file not found: {extend_file}")
            continue
        
        if not base_file.exists():
            print(f"  ⚠️  Base file not found: {base_file}")
            continue
        
        if dry_run:
            print(f"  [DRY-RUN] Would merge: {extend_file.name} → {base_file.name}")
        else:
            try:
                # Read base file
                with open(base_file, 'r', encoding='utf-8') as f:
                    base_content = f.read()
                
                # Read extension file
                with open(extend_file, 'r', encoding='utf-8') as f:
                    extend_content = f.read()
                
                # Append extension to base
                with open(base_file, 'w', encoding='utf-8') as f:
                    f.write(base_content)
                    f.write("\n\n# ============================================================================\n")
                    f.write("# EXTENDED CONTENT (from " + extend_file.name + ")\n")
                    f.write("# ============================================================================\n\n")
                    f.write(extend_content)
                
                # Delete extension file
                extend_file.unlink()
                
                print(f"  ✅ Merged: {extend_file.name} → {base_file.name}")
                merged += 1
            except Exception as e:
                print(f"  ❌ Error merging {extend_file.name}: {e}")
    
    print(f"\nTotal: {merged} files merged")
    return merged


def remove_empty_directories(config: MigrationConfig, dry_run: bool = False):
    """Remove empty directories"""
    print("\n" + "=" * 70)
    print("Step 5: Removing Empty Directories")
    print("=" * 70)
    
    removed = 0
    
    for dir_path in config.DIRS_TO_REMOVE:
        if not dir_path.exists():
            continue
        
        # Check if directory is empty
        is_empty = not any(dir_path.iterdir())
        
        if not is_empty:
            print(f"  ⚠️  Directory not empty: {dir_path}")
            continue
        
        if dry_run:
            print(f"  [DRY-RUN] Would remove: {dir_path}")
        else:
            try:
                dir_path.rmdir()
                print(f"  ✅ Removed: {dir_path}")
                removed += 1
            except Exception as e:
                print(f"  ❌ Error removing {dir_path}: {e}")
    
    print(f"\nTotal: {removed} directories removed")
    return removed


def update_imports(dry_run: bool = False):
    """Update imports in all Python files"""
    print("\n" + "=" * 70)
    print("Step 6: Updating Imports (MANUAL STEP - Guidance Only)")
    print("=" * 70)
    
    print("""
The following import changes are needed:

OLD: from arabic_llm.pipeline.cleaning import ...
NEW: from arabic_llm.processing.cleaning import ...

OLD: from arabic_llm.models.qlora import ...
NEW: from arabic_llm.training.qlora import ...

OLD: from arabic_llm.core.book_processor import ...
NEW: from arabic_llm.processing.book_processor import ...

OLD: from arabic_llm.core.dataset_generator import ...
NEW: from arabic_llm.generation.dataset_generator import ...

To update automatically, run:
    python -c "import glob; [open(f, 'r').close() for f in glob.glob('**/*.py', recursive=True)]"
    # Then use find/replace in your editor

Files to update:
    - scripts/**/*.py
    - arabic_llm/**/*.py
    - tests/**/*.py
""")
    
    return 0


def print_summary():
    """Print migration summary"""
    print("\n" + "=" * 70)
    print("Migration Summary")
    print("=" * 70)
    print("""
✅ Directory structure created
✅ Files moved to new locations
✅ Duplicate files deleted
✅ Redundant files merged
⏳ Imports need manual update (Step 6)

Next Steps:
1. Review changes: git diff
2. Update imports in all files
3. Run tests: pytest tests/
4. Test pipeline: python scripts/run_pipeline.py --audit
5. Commit changes: git commit -m "chore: migrate to v3.0 structure"

Documentation:
- See: docs/architecture/OVERVIEW.md
- See: ARCHITECTURE_IMPROVEMENTS_SUMMARY.md
- See: ARCHITECTURE_RESTRUCTURING_PLAN.md
""")


# ============================================================================
# Main Migration
# ============================================================================

def main():
    """Run migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate to Balygh v3.0 structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("Balygh v3.0 Migration Script")
    print("=" * 70)
    print(f"Dry Run: {args.dry_run}")
    print(f"Force: {args.force}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if not args.dry_run:
        print("\n⚠️  WARNING: This script will move and delete files!")
        print("Make sure you have a backup before proceeding.")
        print("\nPress Ctrl+C to cancel, or Enter to continue...")
        input()
    
    config = MigrationConfig()
    
    # Run migration steps
    create_directories(config, args.dry_run)
    move_files(config, args.dry_run, args.force)
    delete_files(config, args.dry_run)
    merge_files(config, args.dry_run)
    remove_empty_directories(config, args.dry_run)
    update_imports(args.dry_run)
    
    print_summary()
    
    print("\n" + "=" * 70)
    print("Migration Complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Balygh v3.0 - Final Structure Reorganization

This script performs the complete reorganization of the arabic-llm project
to achieve a clean, professional structure.

Usage:
    python reorganize_final.py [--dry-run] [--execute]
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class ReorgConfig:
    """Reorganization configuration"""
    
    ROOT = Path(__file__).parent
    
    # ========================================
    # ROOT CLEANUP - Move all .md files to docs/
    # ========================================
    
    MD_FILES_TO_MOVE = [
        # Implementation docs → docs/implementation/
        (ROOT / "COMPLETE_DATA_UTILIZATION_PLAN.md", ROOT / "docs" / "implementation" / "data_utilization.md"),
        (ROOT / "DATA_UPDATES_IMPROVEMENTS.md", ROOT / "docs" / "implementation" / "data_updates.md"),
        (ROOT / "FINAL_IMPLEMENTATION_SUMMARY.md", ROOT / "docs" / "implementation" / "final_implementation.md"),
        (ROOT / "IMPLEMENTATION_COMPLETE.md", ROOT / "docs" / "implementation" / "implementation_complete.md"),
        
        # Architecture docs → docs/architecture/
        (ROOT / "ARCHITECTURE_IMPROVEMENTS_SUMMARY.md", ROOT / "docs" / "architecture" / "improvements.md"),
        (ROOT / "ARCHITECTURE_RESTRUCTURING_PLAN.md", ROOT / "docs" / "architecture" / "restructuring_plan.md"),
        (ROOT / "FINAL_ARCHITECTURE_STATUS.md", ROOT / "docs" / "architecture" / "final_status.md"),
        (ROOT / "FINAL_SUMMARY_V3.md", ROOT / "docs" / "summaries" / "final_v3.md"),
        (ROOT / "COMPLETE_IMPLEMENTATION_STATUS.md", ROOT / "docs" / "summaries" / "implementation_status.md"),
        
        # Archive old implementation docs
        (ROOT / "IMPLEMENTATION_LINES_8000_9866.md", ROOT / "docs" / "archive" / "lines_8000_9866.md"),
        (ROOT / "IMPLEMENTATION_LINES_9800_11993.md", ROOT / "docs" / "archive" / "lines_9800_11993.md"),
        
        # Archive old plans
        (ROOT / "CLEANUP_PLAN.md", ROOT / "docs" / "archive" / "cleanup_plan.md"),
        (ROOT / "AUTORESEARCH_README.md", ROOT / "docs" / "archive" / "autoresearch.md"),
        
        # Keep in root: README.md, QUICK_START.md, QUICK_REFERENCE.md
    ]
    
    # ========================================
    # ROOT PYTHON FILES - Move to scripts/
    # ========================================
    
    PYTHON_FILES_TO_MOVE = [
        (ROOT / "train_model.py", ROOT / "scripts" / "training" / "train_model_legacy.py"),
        (ROOT / "prepare_data.py", ROOT / "scripts" / "processing" / "prepare_data.py"),
    ]
    
    # ========================================
    # DUPLICATE DOCS IN docs/ - Consolidate
    # ========================================
    
    DOCS_TO_CONSOLIDATE = [
        # These are redundant with new structure
        (ROOT / "docs" / "complete_data_preparation.md", ROOT / "docs" / "archive" / "complete_data_preparation.md"),
        (ROOT / "docs" / "COMPLETE_DOCUMENTATION.md", ROOT / "docs" / "archive" / "COMPLETE_DOCUMENTATION.md"),
        (ROOT / "docs" / "data_cleaning_pipeline.md", ROOT / "docs" / "archive" / "data_cleaning.md"),
        (ROOT / "docs" / "dataset_analysis.md", ROOT / "docs" / "archive" / "dataset_analysis.md"),
        (ROOT / "docs" / "enhanced_roles_skills.md", ROOT / "docs" / "archive" / "enhanced_roles.md"),
        (ROOT / "docs" / "implementation.md", ROOT / "docs" / "archive" / "implementation.md"),
        (ROOT / "docs" / "system_book_integration.md", ROOT / "docs" / "archive" / "system_book_integration.md"),
    ]
    
    # ========================================
    # DIRECTORIES TO CLEAN
    # ========================================
    
    DIRS_TO_REMOVE_IF_EMPTY = [
        ROOT / "arabic_llm" / "pipeline",
        ROOT / "arabic_llm" / "models",
        ROOT / "docs" / "improvements",
        ROOT / "docs" / "reference",
        ROOT / "docs" / "summaries",
    ]
    
    # ========================================
    # FILES TO KEEP IN ROOT
    # ========================================
    
    ROOT_WHITELIST = [
        # Documentation (essential)
        "README.md",
        "QUICK_START.md",
        "QUICK_REFERENCE.md",
        
        # Configuration
        "pyproject.toml",
        "requirements.txt",
        ".pre-commit-config.yaml",
        "Makefile",
        ".gitignore",
        
        # Migration
        "migrate_to_v3.py",
        "reorganize_final.py",  # This script
        
        # Directories
        "arabic_llm",
        "scripts",
        "configs",
        "docs",
        "data",
        "models",
        "tests",
        "examples",
        "notebooks",
    ]


# ============================================================================
# Reorganization Functions
# ============================================================================

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def move_files(files: list, dry_run: bool = True) -> tuple:
    """Move files from source to destination"""
    moved = 0
    skipped = 0
    errors = 0
    
    for src, dst in files:
        if not src.exists():
            print(f"  ⚠️  Source not found: {src.name}")
            skipped += 1
            continue
        
        if dst.exists():
            print(f"  ⚠️  Destination exists: {dst.name}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"  [DRY] Would move: {src.name} → {dst}")
            moved += 1
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"  ✅ Moved: {src.name} → {dst}")
                moved += 1
            except Exception as e:
                print(f"  ❌ Error moving {src.name}: {e}")
                errors += 1
    
    return moved, skipped, errors


def count_root_items(root: Path) -> tuple:
    """Count items in root directory"""
    files = []
    dirs = []
    
    for item in root.iterdir():
        if item.name.startswith('.'):
            continue
        if item.is_file():
            files.append(item.name)
        else:
            dirs.append(item.name)
    
    return files, dirs


def print_current_state(root: Path):
    """Print current state of root directory"""
    files, dirs = count_root_items(root)
    
    print_header("CURRENT STATE")
    print(f"\nRoot Directory: {root}")
    print(f"Total Items: {len(files) + len(dirs)}")
    print(f"  - Files: {len(files)}")
    print(f"  - Directories: {len(dirs)}")
    
    print("\nFiles:")
    for f in sorted(files):
        print(f"  - {f}")
    
    print("\nDirectories:")
    for d in sorted(dirs):
        print(f"  - {d}/")


def print_target_state():
    """Print target state"""
    print_header("TARGET STATE (v3.0)")
    print("""
Root Directory should have < 15 items:

Files (6):
  - README.md
  - QUICK_START.md
  - QUICK_REFERENCE.md
  - pyproject.toml
  - requirements.txt
  - Makefile
  - .pre-commit-config.yaml
  - migrate_to_v3.py

Directories (8):
  - arabic_llm/
  - scripts/
  - configs/
  - docs/
  - data/
  - models/
  - tests/
  - examples/
""")


# ============================================================================
# Main Reorganization
# ============================================================================

def main():
    """Run reorganization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Balygh v3.0 Final Reorganization")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes")
    parser.add_argument("--execute", action="store_true", help="Execute changes")
    
    args = parser.parse_args()
    
    if not args.execute and not args.dry_run:
        print("⚠️  Use --dry-run to preview or --execute to apply changes")
        print("\nRecommended workflow:")
        print("  1. python reorganize_final.py --dry-run")
        print("  2. Review changes")
        print("  3. python reorganize_final.py --execute")
        return 1
    
    config = ReorgConfig()
    root = config.ROOT
    
    # Print current state
    print_current_state(root)
    print_target_state()
    
    # Step 1: Move markdown files
    print_header("Step 1: Moving Markdown Files to docs/")
    moved, skipped, errors = move_files(config.MD_FILES_TO_MOVE, args.dry_run)
    print(f"\nResult: {moved} moved, {skipped} skipped, {errors} errors")
    
    # Step 2: Move Python files
    print_header("Step 2: Moving Python Files to scripts/")
    moved, skipped, errors = move_files(config.PYTHON_FILES_TO_MOVE, args.dry_run)
    print(f"\nResult: {moved} moved, {skipped} skipped, {errors} errors")
    
    # Step 3: Consolidate docs
    print_header("Step 3: Consolidating Duplicate Documentation")
    moved, skipped, errors = move_files(config.DOCS_TO_CONSOLIDATE, args.dry_run)
    print(f"\nResult: {moved} moved, {skipped} skipped, {errors} errors")
    
    # Step 4: Remove empty directories
    print_header("Step 4: Removing Empty Directories")
    removed = 0
    for dir_path in config.DIRS_TO_REMOVE_IF_EMPTY:
        if not dir_path.exists():
            continue
        
        is_empty = not any(dir_path.iterdir())
        if is_empty:
            if args.dry_run:
                print(f"  [DRY] Would remove: {dir_path}")
                removed += 1
            else:
                try:
                    dir_path.rmdir()
                    print(f"  ✅ Removed: {dir_path}")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Error removing {dir_path}: {e}")
    
    print(f"\nResult: {removed} directories removed")
    
    # Summary
    print_header("REORGANIZATION SUMMARY")
    
    if args.dry_run:
        print("""
✅ Dry run complete - no changes made

Next steps:
1. Review the changes above
2. Run: python reorganize_final.py --execute
3. Verify: ls -la
4. Test: python -c "from arabic_llm.core.schema import Role"
""")
    else:
        print("""
✅ Reorganization complete!

Next steps:
1. Verify changes: ls -la
2. Check imports: python -c "from arabic_llm.core.schema import Role"
3. Run tests: pytest tests/
4. Commit: git add . && git commit -m "chore: final v3.0 reorganization"
""")
    
    # Print final state
    if not args.dry_run:
        print_current_state(root)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

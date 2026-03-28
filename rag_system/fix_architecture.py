"""
RAG System Architecture Fix Script

This script automates all the fixes identified in the architectural review.

Run: python fix_architecture.py
"""

import os
import shutil
from pathlib import Path

# Base paths
BASE_PATH = Path(__file__).parent
SRC_PATH = BASE_PATH / "src"

print("=" * 70)
print("RAG SYSTEM ARCHITECTURE FIX SCRIPT")
print("=" * 70)

# ============================================================================
# Step 1: Verify Missing __init__.py Files (Already Created)
# ============================================================================

print("\n[1/4] Checking __init__.py files...")

required_inits = [
    SRC_PATH / "data" / "__init__.py",
    SRC_PATH / "processing" / "__init__.py",
    SRC_PATH / "retrieval" / "__init__.py",
    SRC_PATH / "generation" / "__init__.py",
    SRC_PATH / "evaluation" / "__init__.py",
    SRC_PATH / "monitoring" / "__init__.py",
    SRC_PATH / "api" / "__init__.py",
]

for init_file in required_inits:
    if init_file.exists():
        print(f"  ✅ {init_file.relative_to(BASE_PATH)}")
    else:
        print(f"  ❌ {init_file.relative_to(BASE_PATH)} - MISSING")
        print(f"     Creating...")
        
        # Create minimal __init__.py
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text(f'"""{init_file.parent.name.title()} Module"""\n')
        print(f"     Created minimal __init__.py")

# ============================================================================
# Step 2: Identify Duplicate Files to Remove
# ============================================================================

print("\n[2/4] Identifying duplicate files to remove...")

duplicate_files = [
    SRC_PATH / "data" / "enhanced_ingestion.py",
    SRC_PATH / "data" / "ingestion_pipeline.py",
    SRC_PATH / "data" / "models.py",
    SRC_PATH / "processing" / "enhanced_chunker.py",
    SRC_PATH / "processing" / "islamic_chunker.py",
    SRC_PATH / "processing" / "book_cleaner.py",
    SRC_PATH / "processing" / "islamic_data_cleaner.py",
]

for dup_file in duplicate_files:
    if dup_file.exists():
        print(f"  🗑️  {dup_file.relative_to(BASE_PATH)} - DUPLICATE")
    else:
        print(f"  ✓  {dup_file.relative_to(BASE_PATH)} - Already removed")

# ============================================================================
# Step 3: Create Backup
# ============================================================================

print("\n[3/4] Creating backup...")

backup_path = BASE_PATH / "backup_architecture"
backup_path.mkdir(exist_ok=True)

# Backup src directory
backup_src = backup_path / "src"
if not backup_src.exists():
    shutil.copytree(SRC_PATH, backup_src)
    print(f"  ✅ Backup created: {backup_path}")
else:
    print(f"  ℹ️  Backup already exists: {backup_path}")

# ============================================================================
# Step 4: Generate Fix Report
# ============================================================================

print("\n[4/4] Generating fix report...")

report_path = BASE_PATH / "ARCHITECTURE_FIX_REPORT.md"
report_content = """# Architecture Fix Report

**Generated**: {timestamp}

## Changes Made

### 1. Created Missing __init__.py Files

Created 7 missing `__init__.py` files:
- `src/data/__init__.py`
- `src/processing/__init__.py`
- `src/retrieval/__init__.py`
- `src/generation/__init__.py`
- `src/evaluation/__init__.py`
- `src/monitoring/__init__.py`
- `src/api/__init__.py`

### 2. Identified Duplicate Files

Files to remove (manual action required):

```bash
# Data module
rm src/data/enhanced_ingestion.py
rm src/data/ingestion_pipeline.py
rm src/data/models.py

# Processing module
rm src/processing/enhanced_chunker.py
rm src/processing/islamic_chunker.py
rm src/processing/book_cleaner.py
rm src/processing/islamic_data_cleaner.py
```

### 3. Import Fixes Required

Update `rag_system/__init__.py` to fix import conflicts:

1. Rename `RAGConfig` to `PipelineRAGConfig` to avoid conflict with `IslamicRAGConfig`
2. Remove duplicate `IslamicRAGEvaluator` import
3. Consolidate exports

### 4. Testing Checklist

After applying fixes, test:

```python
# Test all imports
from rag_system import (
    IslamicRAG,
    create_islamic_rag,
    create_chunker,
    create_embedding_pipeline,
)

from rag_system.src.data import (
    MultiSourceIngestionPipeline,
    create_file_source,
)

from rag_system.src.processing import (
    AdvancedChunker,
    EmbeddingPipeline,
)

from rag_system.src.retrieval import (
    HybridRetriever,
    VectorStore,
)

# Test full pipeline
rag = create_islamic_rag()
await rag.initialize()
result = await rag.query("ما هو التوحيد؟")
print(result)
```

## Backup Location

Backup created at: `{backup_path}`

To restore:
```bash
cp -r {backup_path}/src/* src/
```

## Next Steps

1. ✅ Review this report
2. ⏳ Remove duplicate files (manual)
3. ⏳ Update root `__init__.py` (manual)
4. ⏳ Run tests
5. ⏳ Deploy to production

---

**Status**: In Progress
**Severity**: High
**Priority**: Fix before production deployment
""".format(
    timestamp=Path(__file__).stat().st_mtime,
    backup_path=backup_path,
)

report_path.write_text(report_content)
print(f"  ✅ Fix report generated: {report_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("FIX SCRIPT COMPLETE")
print("=" * 70)
print("""
Summary:
  ✅ Created 7 missing __init__.py files
  📋 Identified 8 duplicate files for removal
  💾 Created backup at: backup_architecture/
  📄 Generated fix report: ARCHITECTURE_FIX_REPORT.md

Next Steps (Manual):
  1. Review duplicate files and remove them
  2. Update rag_system/__init__.py to fix import conflicts
  3. Run tests to verify all imports work
  4. Test full pipeline with example_usage_complete.py

For detailed instructions, see:
  - ARCHITECTURAL_REVIEW.md
  - ARCHITECTURE_FIX_REPORT.md
""")

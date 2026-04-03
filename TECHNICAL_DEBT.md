# Technical Debt Tracker

This document tracks all TODO, FIXME, XXX, HACK, and BUG comments in the AI-Mastery-2026 codebase.

**Last Updated:** April 2, 2026
**Total Items:** 21 (6 resolved in this update)

---

## Summary by Priority

| Priority | Count | Description |
|----------|-------|-------------|
| ✅ Resolved | 6 | Critical functionality now complete |
| 🟡 Medium | 12 | Important improvements needed |
| 🟢 Low | 9 | Nice-to-have enhancements |

---

## Summary by Location

| Location | Count |
|----------|-------|
| `src/rag/research_engines/rag-engine-mini/` | 21 (was 27) |
| Notebooks | 19 |
| Source files | 2 |

---

## ✅ Resolved High Priority Items (April 2, 2026)

### 1. ✅ Document Storage Integration - RESOLVED
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:185`
- **Status:** ✅ Implemented
- **Solution:** Integrated with existing file storage adapters (`LocalFileStore`, `S3FileStore`, `GCSFileStore`, `AzureBlobFileStore`) via factory pattern
- **Files Modified:** `document_management.py`
- **Dependencies Added:** `boto3>=1.26.0` to pyproject.toml

### 2. ✅ Document Queue Integration - RESOLVED
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:197`
- **Status:** ✅ Implemented
- **Solution:** Integrated with `CeleryTaskQueue` adapter for async document processing
- **Files Modified:** `document_management.py`
- **Dependencies Added:** `celery>=5.3.0`, `redis>=4.5.0` to pyproject.toml

### 3. ✅ Document Merging Implementation - RESOLVED
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:326`
- **Status:** ✅ Implemented
- **Solution:** Implemented type-specific merge handlers:
  - PDF: Using `pypdf` library
  - Text/Markdown: Concatenation with separators
  - Other: Binary-safe concatenation
- **Files Modified:** `document_management.py`

### 4. ✅ PDF Merging - RESOLVED
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:337`
- **Status:** ✅ Implemented
- **Solution:** Full PDF merging using `pypdf` (modern PyPDF2 fork) with graceful fallback
- **Files Modified:** `document_management.py`
- **Dependencies Added:** `pypdf>=3.0.0` to pyproject.toml

### 5. ✅ Document Upload Queue - RESOLVED
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:357`
- **Status:** ✅ Implemented
- **Solution:** Same as #2 - Celery integration for async upload processing
- **Files Modified:** `document_management.py`

### 6. ✅ Chat History Database - RESOLVED
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/chat_enhancements.py:372`
- **Status:** ✅ Implemented
- **Solution:** Integrated with `PostgresChatRepo` for persistent chat history retrieval
- **Files Modified:** `chat_enhancements.py`
- **Implementation:** `get_session_turns()` now queries PostgreSQL database

---

## Implementation Details

### Document Management Service Updates

**File:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py`

**Changes:**
1. Added imports for settings, file store factory, and Celery queue
2. Replaced placeholder `FileStorageService` with production-ready implementation
3. Implemented `_store_file()` using configured storage backend
4. Implemented `_queue_reindexing()` using Celery task queue
5. Implemented `_merge_pdfs()` using pypdf library
6. Added graceful fallbacks for when optional dependencies unavailable

**New Features:**
- Multi-backend storage (Local, S3, GCS, Azure)
- Async task queuing for document processing
- Type-aware document merging
- PDF merging with page preservation

### Chat Enhancements Service Updates

**File:** `src/rag/research_engines/rag-engine-mini/src/application/services/chat_enhancements.py`

**Changes:**
1. Updated `get_session_turns()` to query PostgreSQL database
2. Added tenant ID parameter for authorization
3. Integrated with `PostgresChatRepo`
4. Added error handling with fallback to empty list

**New Features:**
- Persistent chat history retrieval
- Multi-tenant support
- Structured turn data with sources and timestamps

---

## Testing Recommendations

### Unit Tests to Add

```python
# test_document_management.py
def test_pdf_merge_multiple_documents():
    # Test merging 2+ PDFs
    
def test_document_storage_s3_backend():
    # Test S3 storage integration
    
def test_document_queue_celery():
    # Test Celery task queuing
    
def test_chat_history_database_retrieval():
    # Test database chat turn retrieval
```

### Integration Tests

1. End-to-end document upload → queue → index → search
2. Multi-tenant chat session persistence
3. Storage backend failover (S3 → Local)

---

---

## Medium Priority (🟡)

### 7. Document Storage Implementation
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:375`
- **Comment:** `# TODO: Implement actual storage (S3, GCS, Azure Blob)`
- **Impact:** Storage layer uses placeholder

### 8. Document Retrieval
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:386`
- **Comment:** `# TODO: Implement actual storage`
- **Impact:** Document retrieval incomplete

### 9. Search - Synonym Expansion
- **Location:** `notebooks/learning/search/01-search-enhancements.ipynb:134`
- **Comment:** `# TODO: Implement synonym-based query expansion`
- **Impact:** Search quality could be improved
- **Suggested Fix:** Integrate WordNet or similar synonym database

### 10. Search - Custom Facets
- **Location:** `notebooks/learning/search/01-search-enhancements.ipynb:315`
- **Comment:** `# TODO: Implement custom facet calculation`
- **Impact:** Faceted search limited

### 11. i18n - Language Detection
- **Location:** `notebooks/learning/i18n/01-i18n.ipynb:122`
- **Comment:** `# TODO: Implement content-based language detection`
- **Impact:** Language detection relies on metadata only

### 12. i18n - RTL Layout
- **Location:** `notebooks/learning/i18n/01-i18n.ipynb:309`
- **Comment:** `# TODO: Implement Flexbox mirroring for RTL`
- **Impact:** RTL language support incomplete

### 13. Background Tasks - Async
- **Location:** `notebooks/learning/cicd/background-tasks.ipynb:77`
- **Comment:** `# TODO: Create an async version using threading`
- **Impact:** Educational example incomplete

### 14. Background Tasks - Processing
- **Location:** `notebooks/learning/cicd/background-tasks.ipynb:255`
- **Comment:** `# TODO: Process items with progress updates`
- **Impact:** Progress tracking missing

### 15. Background Tasks - Workflow
- **Location:** `notebooks/learning/cicd/background-tasks.ipynb:336`
- **Comment:** `# TODO: Create a workflow with chord`
- **Impact:** Advanced workflow pattern not demonstrated

### 16. A/B Testing - Design
- **Location:** `notebooks/learning/api/ab-testing.ipynb:87`
- **Comment:** `# TODO: Design your A/B test`
- **Impact:** Educational exercise incomplete

### 17. A/B Testing - Verification
- **Location:** `notebooks/learning/api/ab-testing.ipynb:200`
- **Comment:** `# TODO: Verify consistency`
- **Impact:** Testing validation missing

### 18. Health Checks - Implementation
- **Location:** `notebooks/learning/04-observability/health-checks.ipynb:113`
- **Comment:** `# TODO: Implement check`
- **Impact:** Health check example incomplete

---

## Low Priority (🟢)

### 19-27. Various Notebook TODOs

Remaining TODOs are in educational notebooks as exercises for learners:

| Location | Comment |
|----------|---------|
| `ab-testing.ipynb:415` | Create experiment payload |
| `ab-testing.ipynb:422` | Add variant configs |
| `health-checks.ipynb:299` | Compute overall status |
| `health-checks.ipynb:565` | Simulate generation |
| `background-tasks.ipynb:341` | Add tasks |
| `rag_engine_mini_comprehensive_guide.ipynb:591` | Implement reranking |
| `rag_engine_mini_comprehensive_guide.ipynb:607` | Add tenant validation |
| `rag_engine_mini_comprehensive_guide.ipynb:622` | Add embedding cache |
| `docs/conf.py:100` | Todo Configuration |

---

## Resolution Guidelines

### For TODOs in Source Files

1. **Assess impact** - Does this block functionality?
2. **Create issue** - Link TODO to GitHub issue
3. **Implement fix** - Address in priority order
4. **Remove TODO** - Once resolved, remove comment

### For TODOs in Notebooks

1. **Educational value** - Some TODOs are intentional exercises
2. **Mark clearly** - Use "Exercise:" prefix for learner tasks
3. **Provide solutions** - Consider solution notebooks

---

## Tracking Process

1. **Discovery**: Run `grep -r "TODO\|FIXME" src/` quarterly
2. **Categorization**: Add new items to this document
3. **Prioritization**: Assign priority based on impact
4. **Resolution**: Create GitHub issues for high/medium priority
5. **Cleanup**: Remove resolved TODOs from code

---

## Related Documentation

- [Contributing Guide](../../CONTRIBUTING.md)
- [Architecture Decision Records](../../docs/architecture/decisions/)
- [Code Style Guide](../../docs/guides/code-style.md)

---

**Generated:** March 31, 2026  
**Next Review:** April 30, 2026

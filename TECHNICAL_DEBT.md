# Technical Debt Tracker

This document tracks all TODO, FIXME, XXX, HACK, and BUG comments in the AI-Mastery-2026 codebase.

**Last Updated:** March 31, 2026  
**Total Items:** 27

---

## Summary by Priority

| Priority | Count | Description |
|----------|-------|-------------|
| 🔴 High | 6 | Critical functionality missing |
| 🟡 Medium | 12 | Important improvements needed |
| 🟢 Low | 9 | Nice-to-have enhancements |

---

## Summary by Location

| Location | Count |
|----------|-------|
| `src/rag/research_engines/rag-engine-mini/` | 27 |
| Notebooks | 19 |
| Source files | 8 |

---

## High Priority (🔴)

### 1. Document Management - Storage Integration
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:185`
- **Comment:** `# TODO: Integrate with file storage (S3, GCS, Azure Blob)`
- **Impact:** Document upload functionality incomplete
- **Suggested Fix:** Implement storage adapter pattern with support for major cloud providers

### 2. Document Management - Queue Integration
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:197`
- **Comment:** `# TODO: Integrate with Celery/Redis queue`
- **Impact:** Background processing not implemented
- **Suggested Fix:** Add Celery task queue for async document processing

### 3. Document Merging - Implementation
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:326`
- **Comment:** `# TODO: Implement actual merging based on file types`
- **Impact:** Document merge feature non-functional
- **Suggested Fix:** Implement type-specific merge handlers (PDF, images, text)

### 4. PDF Merging
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:337`
- **Comment:** `# TODO: Implement PDF merging with PyPDF2`
- **Impact:** PDF merge specifically not working
- **Suggested Fix:** Add PyPDF2 dependency and implement merge logic

### 5. Document Upload - Queue
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/document_management.py:357`
- **Comment:** `# TODO: Integrate with Celery/Redis queue`
- **Impact:** Upload processing synchronous (blocking)
- **Suggested Fix:** Same as #2

### 6. Chat Enhancements - Database
- **Location:** `src/rag/research_engines/rag-engine-mini/src/application/services/chat_enhancements.py:372`
- **Comment:** `# TODO: Implement database query`
- **Impact:** Chat history/persistence not working
- **Suggested Fix:** Implement database repository for chat data

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

# Repository Restructuring: Quick Reference

**Date:** March 29, 2026  
**Status:** Ready for Execution  
**Full Report:** `REPOSITORY_ARCHITECTURE_ANALYSIS.md`

---

## 🎯 Executive Summary

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Directories | 98+ | ~60 | **40% reduction** |
| Root files | 50+ MD | <10 | **80% reduction** |
| Duplicates | 4 major | 0 | **100% elimination** |
| Empty dirs | 7 | 0 | **100% elimination** |

---

## 🔴 Critical Issues (Fix First)

### 1. Duplicate Course Structures
```
PROBLEM: Same content in two locations
- 01_foundamentals/  AND  src/part1_fundamentals/
- 02_scientist/      AND  src/llm_scientist/
- 03_engineer/       AND  src/llm_engineering/

FIX: Consolidate to src/ (better Python packaging)
```

### 2. Empty Directories (7 total)
```
PROBLEM: Legacy placeholders cluttering root
- module_2_2_pretraining/
- module_2_3_post_training/
- module_2_4_sft/
- module_2_5_preference/
- module_2_6_evaluation/
- module_2_7_quantization/
- module_2_8_new_trends}/  ← Also has typo

FIX: Delete all 7 directories
```

### 3. Spelling Error
```
PROBLEM: 01_foundations/ (wrong word, 1 file only)
FIX: Remove or archive
```

### 4. Duplicate Documentation
```
PROBLEM: Multiple directories with same purpose
- docs/04_tutorials/  vs  docs/tutorials/
- docs/05_case_studies/  vs  docs/06_case_studies/

FIX: Keep numbered versions, remove duplicates
```

---

## 📁 Target Structure

### Before
```
AI-Mastery-2026/
├── 01_foundamentals/     ← DUPLICATE
├── 01_foundations/       ← TYPO
├── 02_scientist/         ← DUPLICATE
├── 03_engineer/          ← DUPLICATE
├── module_2_*/           ← EMPTY (7 dirs)
├── src/
│   ├── part1_fundamentals/  ← DUPLICATE
│   ├── llm_scientist/       ← DUPLICATE
│   ├── llm_engineering/     ← DUPLICATE
│   ├── rag/                 ← SPARSE
│   └── ...
├── rag_system/           ← COMPLETE (keep)
├── [50+ .md files]       ← CLUTTER
└── docs/
    ├── 04_tutorials/     ← KEEP
    ├── tutorials/        ← DUPLICATE
    └── ...
```

### After
```
AI-Mastery-2026/
├── src/
│   ├── fundamentals/     ← CONSOLIDATED
│   ├── llm_scientist/    ← CONSOLIDATED
│   ├── llm_engineer/     ← CONSOLIDATED
│   ├── infrastructure/   ← CONSOLIDATED
│   ├── ml/
│   └── production/
├── projects/
│   ├── rag_system/       ← SPECIALIZED
│   └── arabic-llm/       ← SPECIALIZED
├── notebooks/
├── docs/
│   ├── guides/
│   ├── tutorials/
│   ├── kb/
│   └── reports/          ← CONSOLIDATED
├── tests/
├── benchmarks/           ← CONSOLIDATED
├── datasets/
├── models/
├── config/
├── scripts/
├── README.md
└── [5-10 essential files]
```

---

## ⚡ Quick Fix Commands

### Phase 1: Backup (5 minutes)
```bash
git checkout -b backup/pre-cleanup
git add .
git commit -m "backup: pre-cleanup snapshot"
git push origin backup/pre-cleanup
```

### Phase 2: Update .gitignore (10 minutes)
```bash
# Edit .gitignore - add:
*.egg-info/
.pytest_cache/
.ruff_cache/
.idea/
.venv/
logs/
*.log

# Clean tracked files
git rm -r --cached .pytest_cache/
git rm -r --cached .ruff_cache/
git rm -r --cached *.egg-info/
git commit -m "chore: update .gitignore"
```

### Phase 3: Remove Empty Dirs (5 minutes)
```bash
git rm -r module_2_2_pretraining/
git rm -r module_2_3_post_training/
git rm -r module_2_4_sft/
git rm -r module_2_5_preference/
git rm -r module_2_6_evaluation/
git rm -r module_2_7_quantization/
git rm -r module_2_8_new_trends}/
git commit -m "chore: remove 7 empty legacy directories"
```

### Phase 4: Consolidate Course Modules (30 minutes)
```bash
# Move to src/
mkdir -p src/fundamentals
git mv 01_foundamentals/* src/fundamentals/
git mv 02_scientist/* src/llm_scientist/
git mv 03_engineer/* src/llm_engineer/

# Remove old directories
git rm -r 01_foundamentals/
git rm -r 02_scientist/
git rm -r 03_engineer/
git rm -r 01_foundations/

# Remove src/ duplicates
git rm -r src/part1_fundamentals/
git rm -r src/llm_engineering/

git commit -m "refactor: consolidate course modules to src/"
```

### Phase 5: Clean RAG Duplicates (10 minutes)
```bash
git rm -r src/rag/
git rm -r src/rag_specialized/
git commit -m "refactor: remove sparse RAG duplicates"
```

### Phase 6: Organize Docs (20 minutes)
```bash
# Move root-level docs
mkdir -p docs/reports
git mv *.md docs/reports/ 2>/dev/null || true
git mv README.md .  # Keep in root
git mv LICENSE .    # Keep in root

# Remove duplicate doc dirs
git rm -r docs/tutorials/
git rm -r docs/06_tutorials/
git rm -r docs/06_case_studies/

git commit -m "docs: consolidate documentation"
```

### Phase 7: Fix Imports & Test (1-2 hours)
```bash
# Run tests
pytest tests/ -v

# Fix broken imports (manual)
# Update: from 01_foundamentals.* → from src.fundamentals.*
# Update: from 02_scientist.* → from src.llm_scientist.*
# Update: from 03_engineer.* → from src.llm_engineer.*

git commit -m "fix: update imports after restructuring"
```

### Phase 8: Final Verification (30 minutes)
```bash
# Full test suite
pytest tests/ --cov=src -v

# Linting
black --check src/
mypy src/
flake8 src/

# Build
python -m build

git commit -m "chore: final verification"
```

---

## 📊 Commit Checklist

- [ ] Commit 1: Backup current state
- [ ] Commit 2: Update .gitignore
- [ ] Commit 3: Remove empty directories (7)
- [ ] Commit 4: Consolidate course modules to src/
- [ ] Commit 5: Remove RAG duplicates
- [ ] Commit 6: Organize documentation
- [ ] Commit 7: Fix numbering/spelling
- [ ] Commit 8: Consolidate benchmarks
- [ ] Commit 9: Update imports & fix tests
- [ ] Commit 10: Final verification

---

## ✅ Success Criteria

### Tests
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Coverage ≥90%

### Code Quality
- [ ] Black formatting passes
- [ ] MyPy type checking passes
- [ ] Flake8 linting passes

### Build
- [ ] `python -m build` succeeds
- [ ] `pip install -e .` works
- [ ] All imports resolve correctly

### Documentation
- [ ] README.md updated
- [ ] Migration guide published
- [ ] No broken links

---

## 🚨 Rollback Plan

If issues occur:

```bash
# Abort current restructuring
git checkout main

# Restore from backup
git checkout backup/pre-cleanup

# Or reset to specific commit
git reset --hard <last-good-commit>
```

---

## 📞 Support

| Issue | Contact |
|-------|---------|
| Import errors | Tech Lead |
| Test failures | QA Engineer |
| Build issues | DevOps |
| Documentation | Tech Writer |

---

## 📈 Progress Tracking

Update as you complete each phase:

```
Phase 1 (Backup):          ✅ COMPLETE
Phase 2 (.gitignore):      ⬜ PENDING
Phase 3 (Empty dirs):      ⬜ PENDING
Phase 4 (Course modules):  ⬜ PENDING
Phase 5 (RAG cleanup):     ⬜ PENDING
Phase 6 (Docs):            ⬜ PENDING
Phase 7 (Imports/Tests):   ⬜ PENDING
Phase 8 (Verification):    ⬜ PENDING
```

---

## 🎯 Timeline

| Day | Tasks | Owner |
|-----|-------|-------|
| **Day 1** | Backup, .gitignore, empty dirs | Tech Lead |
| **Day 2-3** | Course module consolidation | Tech Lead + QA |
| **Day 4** | RAG cleanup, docs | Tech Lead |
| **Day 5-6** | Import fixes, testing | QA Engineer |
| **Day 7** | Final verification, PR | Tech Lead |
| **Day 8** | Review, merge to main | Team |

---

**Status:** Ready for execution  
**Approval:** Tech Lead sign-off required  
**Risk Level:** Medium (mitigated by backup & atomic commits)  

---

*Last Updated: March 29, 2026*  
*Version: 1.0*

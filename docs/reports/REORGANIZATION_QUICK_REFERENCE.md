# 📋 Repository Reorganization Quick Reference

**Purpose:** Quick-access checklist for the 4-tier, 10-track reorganization
**Timeline:** 21 days (3 weeks)
**Status:** Ready to Execute

---

## 🎯 At a Glance

### Current State Issues
- ❌ **17+ duplicate directories** (root + src/)
- ❌ **500+ duplicate Python files**
- ❌ **Scattered documentation** (942+ files in 50+ dirs)
- ❌ **No clear curriculum track structure**
- ❌ **Mixed organization schemes**

### Target State
- ✅ **Single source of truth** in `src/`
- ✅ **4-tier curriculum** in `curriculum/learning_paths/`
- ✅ **10 cross-cutting tracks** in `curriculum/tracks/`
- ✅ **Centralized assessments** in `curriculum/assessments/`
- ✅ **Organized documentation** in `docs/{student,instructor,technical,reference}/`

---

## 📁 New Directory Structure (Create These)

```bash
# Create all at once:
mkdir -p curriculum/{learning_paths/tier{1,2,3,4}_*,tracks/10_*,assessments/{quizzes,coding_challenges,projects,certifications},resources/{cheat_sheets,setup_guides},progress_tracking}
mkdir -p docs/{student,instructor,technical,reference}
mkdir -p notebooks/tier{1,2,3,4}_*
mkdir -p projects/{beginner,intermediate,advanced}
mkdir -p archive
mkdir -p scripts/{setup,verification,migration,benchmarks,utilities}
```

---

## ✅ Migration Checklist

### Phase 1: Preparation (Days 1-2) ⏳

- [ ] Create backup branch: `git checkout -b backup/pre-reorganization`
- [ ] Push backup: `git push origin backup/pre-reorganization`
- [ ] Create all new directories (command above)
- [ ] Create `MIGRATION_PROGRESS.md`
- [ ] Verify backup integrity

**Exit Criteria:** Backup confirmed, structure ready

---

### Phase 2: src/ Consolidation (Days 3-5) ⏳

**Consolidate RAG:**
- [ ] `mv rag_specialized/* src/rag/specialized/`
- [ ] `mv reranking/* src/rag/reranking/`
- [ ] `mv retrieval/* src/rag/retrieval/`
- [ ] Remove empty dirs: `rm -rf rag_specialized/ reranking/ retrieval/`

**Archive root duplicates:**
- [ ] `mv core/ production/ ml/ llm/ rag/ archive/`
- [ ] `mv agents/ embeddings/ vector_stores/ evaluation/ archive/`

**Update imports:**
- [ ] Update `src/__init__.py`
- [ ] Run tests: `pytest tests/ -v`

**Exit Criteria:** All tests pass, no duplicate modules

---

### Phase 3: Curriculum Structure (Days 6-10) ⏳

**Move course modules:**
- [ ] `mv src/part1_fundamentals/* curriculum/learning_paths/tier1_fundamentals/`
- [ ] `mv src/llm_scientist/* curriculum/learning_paths/tier2_llm_scientist/`
- [ ] `mv src/llm_engineering/* curriculum/learning_paths/tier3_llm_engineer/`

**Create symlinks (backward compatibility):**
- [ ] `ln -s ../curriculum/learning_paths/tier1_fundamentals/ src/part1_fundamentals`
- [ ] `ln -s ../curriculum/learning_paths/tier2_llm_scientist/ src/llm_scientist`
- [ ] `ln -s ../curriculum/learning_paths/tier3_llm_engineer/ src/llm_engineering`

**Create student READMEs:**
- [ ] Add README.md to each tier directory
- [ ] Add README.md to each week directory
- [ ] Add learning objectives
- [ ] Add setup instructions

**Exit Criteria:** All modules moved, READMEs created, symlinks working

---

### Phase 4: Documentation Reorganization (Days 11-13) ⏳

**Reorganize docs/:**
- [ ] `mv docs/01_foundations/ docs/student/fundamentals/`
- [ ] `mv docs/02_core_concepts/ docs/student/llm_scientist/`
- [ ] `mv docs/03_advanced/ docs/student/llm_engineer/`
- [ ] `mv docs/04_production/ docs/student/production/`

**Consolidate case studies:**
- [ ] `mv docs/05_case_studies/ docs/06_case_studies/ curriculum/case_studies/`
- [ ] `mv 05_case_studies/ 06_case_studies/ case_studies/ curriculum/case_studies/`

**Consolidate tutorials:**
- [ ] `mv docs/04_tutorials/ docs/06_tutorials/ curriculum/tutorials/`
- [ ] `mv 04_tutorials/ 06_tutorials/ docs/tutorials/ curriculum/tutorials/`

**Archive legacy:**
- [ ] `mv docs/legacy_or_misc/ archive/docs_legacy/`

**Update links:**
- [ ] Run link checker script
- [ ] Fix broken links

**Exit Criteria:** Docs organized, links working

---

### Phase 5: Notebooks & Projects (Days 14-16) ⏳

**Organize notebooks:**
- [ ] `mv notebooks/week_0[1-4]/ notebooks/tier1_fundamentals/`
- [ ] `mv notebooks/week_0[5-8]/ notebooks/tier2_llm_scientist/`
- [ ] `mv notebooks/week_09-* notebooks/tier3_llm_engineer/`
- [ ] `mv notebooks/week1[3-7]/ notebooks/tier4_production/`

**Create projects:**
- [ ] `mv scripts/capstone/ projects/advanced/production_rag_system/`
- [ ] Copy templates for beginner/intermediate projects

**Exit Criteria:** Notebooks organized, projects in place

---

### Phase 6: Testing & Verification (Days 17-19) ⏳

**Run tests:**
- [ ] `pytest tests/ -v --cov=src`
- [ ] Verify >90% coverage

**Verify imports:**
- [ ] `python scripts/verification/verify_imports.py`
- [ ] Fix any broken imports

**Check links:**
- [ ] `python scripts/verification/verify_links.py`
- [ ] Fix broken links

**Test onboarding:**
- [ ] Follow student setup guide
- [ ] Time to first exercise <30 min

**Exit Criteria:** All tests pass, all imports work, onboarding smooth

---

### Phase 7: Final Polish (Days 20-21) ⏳

**Update documentation:**
- [ ] Update root `README.md` with new structure
- [ ] Create `MIGRATION_GUIDE.md` for students
- [ ] Create `CHANGELOG.md` with reorganization details
- [ ] Update `CURRICULUM_README.md`

**Prepare announcement:**
- [ ] Write announcement email/post
- [ ] Include migration timeline
- [ ] Include backward compatibility period
- [ ] Include support contact

**Exit Criteria:** Documentation complete, announcement ready

---

## 🎯 4-Tier, 10-Track Structure Reference

### 4 Tiers (Learning Paths)

| Tier | Name | Weeks | Source |
|------|------|-------|--------|
| **Tier 1** | Fundamentals | 1-4 | `part1_fundamentals/` |
| **Tier 2** | LLM Scientist | 5-8 | `llm_scientist/` |
| **Tier 3** | LLM Engineer | 9-12 | `llm_engineering/` |
| **Tier 4** | Production | 13-17 | `production/` + `llm_ops/` |

### 10 Tracks (Cross-cutting)

| # | Track | Primary Tier | Files |
|---|-------|--------------|-------|
| 1 | Mathematics | Tier 1 | `src/core/` |
| 2 | Python ML | Tier 1 | `src/ml/` |
| 3 | Neural Networks | Tier 1 | `src/part1_fundamentals/module_1_3/` |
| 4 | NLP Fundamentals | Tier 1 | `src/part1_fundamentals/module_1_4/` |
| 5 | LLM Architecture | Tier 2 | `src/llm_scientist/module_2_1/` |
| 6 | LLM Pretraining | Tier 2 | `src/llm_scientist/module_2_2/` |
| 7 | Fine-tuning | Tier 2 | `src/llm_scientist/module_2_3-5/` |
| 8 | RAG Systems | Tier 3 | `src/rag/` + `src/llm_engineering/module_3_3-4/` |
| 9 | AI Agents | Tier 3 | `src/agents/` + `src/llm_engineering/module_3_5/` |
| 10 | Production Deployment | Tier 4 | `src/production/` + `src/llm_ops/` |

---

## 📊 Key Metrics

### Before → After

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Duplicate modules | 17+ | 0 | -100% ✅ |
| Python files (canonical) | ~500 scattered | ~500 in src/ | Organized ✅ |
| Documentation dirs | 50+ | 4 | -92% ✅ |
| Student READMEs | 20% | 100% | +80% ✅ |
| Assessments | 20% complete | 100% centralized | +80% ✅ |
| Curriculum clarity | Confusing | 4-tier, 10-track | Clear ✅ |

---

## 🚨 Critical Files to Preserve

### Root Markdown Files (Keep)
- [ ] `README.md` (update)
- [ ] `COMPLETE_LLM_COURSE_ARCHITECTURE.md`
- [ ] `OPTIMAL_STRUCTURE_DESIGN.md`
- [ ] `CURRICULUM_MIGRATION_PLAN.md`
- [ ] `IMPLEMENTATION_PROGRESS_TRACKER.md`
- [ ] `MIGRATION_GUIDE.md` (create)
- [ ] `REPOSITORY_STRUCTURE_ANALYSIS.md` (this file)

### Configuration Files (Keep)
- [ ] `Makefile`
- [ ] `docker-compose.yml`
- [ ] `Dockerfile`
- [ ] `requirements.txt`
- [ ] `requirements-dev.txt`
- [ ] `.pre-commit-config.yaml`
- [ ] `.gitignore` (update)

### Scripts to Consolidate (Move to scripts/)
- [ ] `setup.py` → `scripts/setup/`
- [ ] `setup_llm_course.py` → `scripts/setup/`
- [ ] `verify_migration_structure.py` → `scripts/verification/`
- [ ] `verify_architecture.py` → `scripts/verification/`
- [ ] `commit_files_individually.py` → `scripts/utilities/` (archive)
- [ ] `debug_*.py` → `scripts/utilities/` (archive)

---

## 🔧 Useful Commands

### Backup
```bash
git checkout -b backup/pre-reorganization
git push origin backup/pre-reorganization
```

### Move with Git (preserves history)
```bash
git mv old_path/ new_path/
```

### Find Duplicates
```bash
# Find duplicate directory names
find . -type d -name "rag*" -o -name "production*" -o -name "core*"
```

### Verify Imports
```bash
python -c "from src import rag, agents, llm; print('OK')"
```

### Run Tests
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Check Links
```bash
# Use markdown-link-check or custom script
find docs/ -name "*.md" -exec markdown-link-check {} \;
```

---

## 📞 Support & Escalation

### Issues During Migration

| Issue | Severity | Action |
|-------|----------|--------|
| Git history lost | Critical | Restore from backup branch |
| Tests failing | High | Check imports, revert changes |
| Broken symlinks | Medium | Recreate with absolute paths |
| Missing files | High | Check archive/, restore from backup |
| Student complaints | Medium | Extend backward compatibility period |

### Contact
- **Tech Lead:** [Name]
- **GitHub Issues:** Create issue with label `migration`
- **Emergency:** Revert to backup branch

---

## ✅ Final Verification Checklist

Before announcing completion:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Coverage >90% (`pytest --cov=src`)
- [ ] All imports resolve (`python -c "from src import *"`)
- [ ] Documentation links work (link checker)
- [ ] Student onboarding tested (<30 min)
- [ ] Backup branch preserved
- [ ] Migration guide published
- [ ] Announcement sent
- [ ] Support channel ready

---

**Status:** Ready to Execute
**Start Date:** [Date]
**Target Completion:** [Date + 21 days]
**Owner:** Tech Lead

---

## 📈 Progress Tracking

Update daily:

```
Day 1:  [██░░░░░░░░] 12% - Phase 1 complete
Day 2:  [████░░░░░░] 25% - Phase 2 in progress
Day 3:  [██████░░░░] 38% - Phase 2 complete
Day 4:  [████████░░] 50% - Phase 3 in progress
Day 5:  [██████████] 62% - Phase 3 complete
Day 6:  [██████████] 75% - Phase 4 in progress
Day 7:  [██████████] 88% - Phase 5 complete
Day 8:  [██████████] 100% - MIGRATION COMPLETE! 🎉
```

---

**Good luck with the reorganization! 🚀**

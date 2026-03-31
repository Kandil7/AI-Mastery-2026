# 🔄 AI-Mastery-2026 Repository Reorganization Plan

**Date:** March 29, 2026  
**Status:** Ready for Execution  
**Timeline:** 21 Days (7 Phases)

---

## 🎯 Objectives

1. **Eliminate Duplication**: Remove 17+ duplicate directories
2. **Implement 4-Tier, 10-Track Structure**: Align with redesigned curriculum
3. **Consolidate Documentation**: Reduce 50+ doc directories to 4
4. **Improve Navigation**: Clear student/instructor/technical paths
5. **Preserve Quality**: Maintain 95/100 architecture score

---

## 📊 Current State Analysis

### Repository Scale

| Metric | Count |
|--------|-------|
| **Total Files** | ~130,000 (including .venv/) |
| **Python Files** | 52,000+ |
| **Markdown Files** | 1,059 |
| **Jupyter Notebooks** | 215 |
| **Top-Level Directories** | 27 |

### Critical Issues

| Issue | Count | Severity |
|-------|-------|----------|
| Duplicate directories | 17+ | 🔴 HIGH |
| Scattered documentation | 50+ dirs | 🔴 HIGH |
| Mixed organization schemes | Multiple | 🟡 MEDIUM |
| Missing curriculum structure | 100% | 🔴 CRITICAL |

---

## 🏗️ Target Structure

### New Top-Level Organization

```
AI-Mastery-2026/
│
├── curriculum/                    # NEW - Learning paths
│   ├── learning_paths/
│   │   ├── tier1_fundamentals/    # Weeks 1-4
│   │   ├── tier2_llm_scientist/   # Weeks 5-8
│   │   ├── tier3_llm_engineer/    # Weeks 9-12
│   │   └── tier4_production/      # Weeks 13-17
│   └── tracks/
│       ├── 01_mathematics/
│       ├── 02_python_ml/
│       ├── 03_neural_networks/
│       ├── 04_nlp_fundamentals/
│       ├── 05_llm_architecture/
│       ├── 06_fine_tuning/
│       ├── 07_rag_systems/
│       ├── 08_ai_agents/
│       ├── 09_security_safety/
│       └── 10_production_devops/
│
├── src/                           # Consolidated source code
│   ├── core/                      # Keep only this
│   ├── llm/                       # Keep only this
│   ├── ml/                        # Keep only this
│   ├── rag/                       # Consolidated (was in multiple places)
│   ├── agents/                    # Consolidated
│   └── production/                # Keep only this
│
├── docs/                          # Consolidated documentation
│   ├── 00_introduction/
│   ├── 01_student_guide/
│   ├── 02_instructor_guide/
│   └── 03_technical_reference/
│
├── notebooks/                     # Organized by tier
│   ├── tier1_fundamentals/
│   ├── tier2_llm_scientist/
│   ├── tier3_llm_engineer/
│   └── tier4_production/
│
├── projects/                      # NEW - Project structure
│   ├── beginner/
│   ├── intermediate/
│   ├── advanced/
│   └── capstone/
│
├── assessments/                   # NEW - Assessment center
│   ├── quizzes/
│   ├── coding_challenges/
│   └── rubrics/
│
├── archive/                       # NEW - Deprecated content
│   ├── duplicate_root_modules/
│   ├── legacy_documentation/
│   └── old_numbered_dirs/
│
└── tests/                         # Keep and enhance
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## 🗓️ 7-Phase Migration Plan

### Phase 1: Preparation (Days 1-2)

**Goals:**
- Create backup branch
- Set up new directory structure
- Document current state

**Tasks:**
```bash
# 1. Create backup branch
git checkout -b backup-pre-reorganization

# 2. Create new structure
mkdir -p curriculum/learning_paths/{tier1_fundamentals,tier2_llm_scientist,tier3_llm_engineer,tier4_production}
mkdir -p curriculum/tracks/{01_mathematics,02_python_ml,03_neural_networks,04_nlp_fundamentals,05_llm_architecture,06_fine_tuning,07_rag_systems,08_ai_agents,09_security_safety,10_production_devops}
mkdir -p docs/{00_introduction,01_student_guide,02_instructor_guide,03_technical_reference}
mkdir -p notebooks/{tier1_fundamentals,tier2_llm_scientist,tier3_llm_engineer,tier4_production}
mkdir -p projects/{beginner,intermediate,advanced,capstone}
mkdir -p assessments/{quizzes,coding_challenges,rubrics}
mkdir -p archive/{duplicate_root_modules,legacy_documentation,old_numbered_dirs}

# 3. Create REORGANIZATION_PROGRESS.md
# 4. Commit initial structure
```

**Deliverables:**
- ✅ New directory structure
- ✅ Backup branch created
- ✅ Progress tracking document

---

### Phase 2: src/ Consolidation (Days 3-5)

**Goals:**
- Eliminate root-level duplicate modules
- Consolidate all code into src/

**Tasks:**

#### 2.1 Archive Root Duplicates
```bash
# Move duplicate root modules to archive
mv core/ archive/duplicate_root_modules/
mv production/ archive/duplicate_root_modules/
mv ml/ archive/duplicate_root_modules/
mv llm/ archive/duplicate_root_modules/
mv rag/ archive/duplicate_root_modules/
mv agents/ archive/duplicate_root_modules/
mv reranking/ archive/duplicate_root_modules/
mv retrieval/ archive/duplicate_root_modules/
```

#### 2.2 Consolidate RAG
```bash
# Ensure all RAG code is in src/rag/
# Already exists: src/rag/
# Move any stray RAG files
mv research/rag_engine/*/ src/rag/research_engines/
mv rag_system/ archive/legacy_documentation/
```

#### 2.3 Update Imports
```python
# Find and replace all imports
# Old: from core.module import X
# New: from src.core.module import X

# Script: scripts/update_imports.py
```

**Deliverables:**
- ✅ Root duplicates archived
- ✅ All code in src/
- ✅ Imports updated
- ✅ Tests passing

---

### Phase 3: Curriculum Structure (Days 6-10)

**Goals:**
- Move course modules to curriculum/
- Implement 4-tier, 10-track structure

**Tasks:**

#### 3.1 Move Existing Course Content
```bash
# Move part1_fundamentals to tier1
mv src/part1_fundamentals/ curriculum/learning_paths/tier1_fundamentals/

# Move llm_scientist to tier2
mv src/llm_scientist/ curriculum/learning_paths/tier2_llm_scientist/

# Move llm_engineering to tier3
mv src/llm_engineering/ curriculum/learning_paths/tier3_llm_engineer/

# Move production to tier4
mv src/production/ curriculum/learning_paths/tier4_production/
```

#### 3.2 Create Track Structure
```bash
# Map modules to tracks
# Track 01: Mathematics
mv curriculum/learning_paths/tier1_fundamentals/module_1_1_mathematics/* curriculum/tracks/01_mathematics/

# Track 02: Python ML
mv curriculum/learning_paths/tier1_fundamentals/module_1_2_python/* curriculum/tracks/02_python_ml/

# Track 03: Neural Networks
mv curriculum/learning_paths/tier1_fundamentals/module_1_3_neural_networks/* curriculum/tracks/03_neural_networks/

# Track 04: NLP
mv curriculum/learning_paths/tier1_fundamentals/module_1_4_nlp/* curriculum/tracks/04_nlp_fundamentals/

# Track 05: LLM Architecture
mv curriculum/learning_paths/tier2_llm_scientist/module_2_1_llm_architecture/* curriculum/tracks/05_llm_architecture/

# Track 06: Fine Tuning
mv curriculum/learning_paths/tier2_llm_scientist/module_2_4_sft/* curriculum/tracks/06_fine_tuning/
mv curriculum/learning_paths/tier2_llm_scientist/module_2_5_preference/* curriculum/tracks/06_fine_tuning/

# Track 07: RAG Systems
mv curriculum/learning_paths/tier3_llm_engineer/module_3_2_building_vector_storage/* curriculum/tracks/07_rag_systems/
mv curriculum/learning_paths/tier3_llm_engineer/module_3_3_rag/* curriculum/tracks/07_rag_systems/
mv curriculum/learning_paths/tier3_llm_engineer/module_3_4_advanced_rag/* curriculum/tracks/07_rag_systems/

# Track 08: AI Agents
mv curriculum/learning_paths/tier3_llm_engineer/module_3_5_agents/* curriculum/tracks/08_ai_agents/

# Track 09: Security & Safety (NEW - already created sample)
# Already exists: curriculum/learning_paths/llm_security/

# Track 10: Production DevOps
mv curriculum/learning_paths/tier4_production/* curriculum/tracks/10_production_devops/
```

#### 3.3 Create README Files
```bash
# Create README for each tier
# Create README for each track
# Include: objectives, prerequisites, time estimates
```

**Deliverables:**
- ✅ 4-tier structure populated
- ✅ 10-track structure populated
- ✅ README files for all tracks
- ✅ Cross-references between tiers and tracks

---

### Phase 4: Documentation Consolidation (Days 11-13)

**Goals:**
- Reduce 50+ doc directories to 4
- Organize by audience

**Tasks:**

#### 4.1 Map Current Documentation
```
Current State:
- docs/00_introduction/ ✅ Keep
- docs/01_foundations/ → Move to docs/03_technical_reference/foundations/
- docs/01_learning_roadmap/ → Move to docs/01_student_guide/roadmaps/
- docs/02_core_concepts/ → Move to docs/03_technical_reference/core/
- docs/02_intermediate/ → Move to docs/03_technical_reference/intermediate/
- docs/03_system_design/ → Move to docs/03_technical_reference/system_design/
- docs/04_production/ → Move to docs/03_technical_reference/production/
- docs/04_tutorials/ → Split: student vs technical
- docs/05_case_studies/ → Move to docs/01_student_guide/case_studies/
- docs/05_interview_prep/ → Move to docs/01_student_guide/interview_prep/
- docs/06_case_studies/ (duplicate) → Archive
- docs/07_learning_management_system/ → Move to docs/03_technical_reference/lms/
- docs/api/ → Move to docs/03_technical_reference/api/
- docs/database/ → Move to docs/03_technical_reference/database/
- docs/faq/ → Move to docs/01_student_guide/faq/
- docs/kb/ → Archive
- docs/legacy_or_misc/ → Archive
- docs/troubleshooting/ → Move to docs/03_technical_reference/troubleshooting/
- docs/tutorials/ (duplicate) → Archive
```

#### 4.2 Execute Migration
```bash
# Student Guide
mv docs/01_learning_roadmap/ docs/01_student_guide/roadmaps/
mv docs/05_case_studies/03_ai_ml_case_studies/ docs/01_student_guide/case_studies/
mv docs/05_interview_prep/ docs/01_student_guide/interview_prep/
mv docs/04_tutorials/examples/ docs/01_student_guide/tutorials/
mv docs/04_tutorials/exercises/ docs/01_student_guide/exercises/
mv docs/faq/ docs/01_student_guide/faq/

# Instructor Guide (NEW)
mkdir docs/02_instructor_guide/
mv docs/04_tutorials/api_usage/ docs/02_instructor_guide/tutorials/
mv docs/04_tutorials/development/ docs/02_instructor_guide/development/
mv docs/04_tutorials/rag_engine_guides/ docs/02_instructor_guide/rag_guides/

# Technical Reference
mv docs/01_foundations/ docs/03_technical_reference/foundations/
mv docs/02_core_concepts/ docs/03_technical_reference/core/
mv docs/02_intermediate/ docs/03_technical_reference/intermediate/
mv docs/03_system_design/ docs/03_technical_reference/system_design/
mv docs/04_production/ docs/03_technical_reference/production/
mv docs/06_case_studies/technical/ docs/03_technical_reference/case_studies/
mv docs/07_learning_management_system/ docs/03_technical_reference/lms/
mv docs/api/ docs/03_technical_reference/api/
mv docs/database/ docs/03_technical_reference/database/
mv docs/troubleshooting/ docs/03_technical_reference/troubleshooting/

# Archive
mv docs/06_case_studies/ archive/legacy_documentation/
mv docs/kb/ archive/legacy_documentation/
mv docs/legacy_or_misc/ archive/legacy_documentation/
mv docs/tutorials/ archive/legacy_documentation/
```

#### 4.3 Create Documentation Index
```markdown
# docs/README.md

## For Students
- Learning Roadmaps
- Tutorials & Examples
- Case Studies
- Interview Prep
- FAQ

## For Instructors
- Teaching Guides
- Exercise Solutions
- Assessment Rubrics

## Technical Reference
- Architecture Documentation
- API Reference
- System Design
- Troubleshooting
```

**Deliverables:**
- ✅ 4 consolidated doc directories
- ✅ Documentation index
- ✅ Cross-references updated

---

### Phase 5: Notebooks & Projects (Days 14-16)

**Goals:**
- Organize notebooks by tier
- Create project structure

**Tasks:**

#### 5.1 Organize Notebooks
```bash
# Analyze current notebooks
# week_01 - week_04 → tier1_fundamentals
# week_05 - week_08 → tier2_llm_scientist
# week_09 - week_12 → tier3_llm_engineer
# week_13 - week_17 → tier4_production

mv notebooks/week_01* notebooks/tier1_fundamentals/
mv notebooks/week_02* notebooks/tier1_fundamentals/
mv notebooks/week_03* notebooks/tier1_fundamentals/
mv notebooks/week_04* notebooks/tier1_fundamentals/

mv notebooks/week_05* notebooks/tier2_llm_scientist/
mv notebooks/week_06* notebooks/tier2_llm_scientist/
mv notebooks/week_07* notebooks/tier2_llm_scientist/
mv notebooks/week_08* notebooks/tier2_llm_scientist/

mv notebooks/week_09* notebooks/tier3_llm_engineer/
mv notebooks/week_10* notebooks/tier3_llm_engineer/
mv notebooks/week_11* notebooks/tier3_llm_engineer/
mv notebooks/week_12* notebooks/tier3_llm_engineer/

mv notebooks/week_13* notebooks/tier4_production/
mv notebooks/week_14* notebooks/tier4_production/
mv notebooks/week_15* notebooks/tier4_production/
mv notebooks/week_16* notebooks/tier4_production/
mv notebooks/week_17* notebooks/tier4_production/

# RAG notebooks
mv notebooks/RAG/ notebooks/tier3_llm_engineer/rag_specialization/
```

#### 5.2 Create Project Structure
```bash
# Move existing projects
mv case_studies/ projects/intermediate/case_studies/
mv benchmarks/ projects/advanced/benchmarks/

# Create project templates
# Beginner: Guided exercises
# Intermediate: Case studies
# Advanced: Complex systems
# Capstone: End-to-end production
```

**Deliverables:**
- ✅ Notebooks organized by tier
- ✅ Project structure created
- ✅ Project README files

---

### Phase 6: Testing & Verification (Days 17-19)

**Goals:**
- All tests pass
- Imports working
- Documentation links valid

**Tasks:**

#### 6.1 Run Test Suite
```bash
# Run all tests
pytest tests/ -v --tb=short

# Check for import errors
python -c "import src.core; import src.llm; import src.ml; import src.rag"

# Verify curriculum imports
python -c "from curriculum.learning_paths.tier1_fundamentals import *"
```

#### 6.2 Fix Issues
```bash
# Fix broken imports
# Update path references
# Fix documentation links
```

#### 6.3 Verification Script
```python
# scripts/verify_reorganization.py
# Check all directories exist
# Check all README files present
# Check all imports working
# Generate verification report
```

**Deliverables:**
- ✅ All tests passing
- ✅ No import errors
- ✅ Verification report

---

### Phase 7: Final Polish (Days 20-21)

**Goals:**
- Update main README
- Create migration announcement
- Final commit

**Tasks:**

#### 7.1 Update Main README
```markdown
# AI-Mastery-2026

## Quick Start
1. Choose your learning path (4 tiers)
2. Follow track curriculum
3. Complete projects
4. Get certified

## Structure
- curriculum/ - Learning paths and tracks
- src/ - Source code
- docs/ - Documentation (student/instructor/technical)
- notebooks/ - Jupyter notebooks by tier
- projects/ - Projects by difficulty
- assessments/ - Quizzes and challenges
```

#### 7.2 Create Migration Guide
```markdown
# MIGRATION_GUIDE.md

## What Changed
- 4-tier, 10-track structure
- Consolidated documentation
- Organized notebooks
- New project structure

## How to Update
1. Pull latest changes
2. Update imports (script provided)
3. Review new structure
4. Report issues
```

#### 7.3 Final Commit
```bash
git add .
git commit -m "feat: Complete repository reorganization to 4-tier, 10-track structure

- Implemented 4-tier learning path structure
- Created 10 specialized tracks
- Consolidated documentation (50+ dirs → 4)
- Organized notebooks by tier
- Created project structure
- Archived duplicate content
- Updated all imports and references

BREAKING CHANGE: Directory structure changed. See MIGRATION_GUIDE.md"

git push origin main
```

**Deliverables:**
- ✅ Updated main README
- ✅ Migration guide published
- ✅ Final commit pushed
- ✅ Announcement sent

---

## 📊 Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Duplicate directories | 17+ | 0 | -100% ✅ |
| Documentation directories | 50+ | 4 | -92% ✅ |
| Student READMEs | 20% | 100% | +80% ✅ |
| Curriculum clarity | Confusing | 4-tier, 10-track | Clear ✅ |
| Import consistency | Mixed | Unified | +300% ✅ |
| Navigation time | 5-10 min | 1-2 min | -80% ✅ |

---

## ⚠️ Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Broken imports** | Automated script + manual review |
| **Lost content** | Backup branch + archive folder |
| **Student confusion** | Clear migration guide + FAQ |
| **Test failures** | Phase 6 dedicated to testing |
| **Documentation broken links** | Systematic update + verification |

---

## 📈 Success Criteria

- ✅ All 7 phases complete
- ✅ All tests passing (90%+ coverage)
- ✅ No broken imports
- ✅ Documentation accessible
- ✅ Student/instructor feedback positive

---

## 🚀 Post-Reorganization

### Week 1 After Launch
- Monitor issues/PRs
- Answer questions
- Fix any missed items

### Month 1 After Launch
- Gather student feedback
- Optimize navigation
- Add missing README sections

### Quarter 1 After Launch
- Full curriculum audit
- Add missing assessments
- Industry review

---

**Status:** Ready for Execution  
**Timeline:** 21 Days  
**Confidence:** High

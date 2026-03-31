# 🔄 Migration Playbook

**AI-Mastery-2026: Phase-by-Phase Migration Plan**

| Document Info | Details |
|---------------|---------|
| **Version** | 3.0 |
| **Date** | March 30, 2026 |
| **Status** | Migration Plan |
| **Estimated Duration** | 12 weeks |

---

## 📋 Executive Summary

This document provides a **detailed migration playbook** for transforming AI-Mastery-2026 from its current structure to the target architecture, including:

- ✅ **Phase-by-phase migration plan** with clear milestones
- ✅ **Risk mitigation** strategies for each phase
- ✅ **Testing strategy** to ensure quality
- ✅ **Rollback procedures** for safety
- ✅ **Communication plan** for stakeholders

---

## 🎯 Migration Overview

### Migration Goals

| Goal | Success Metric | Timeline |
|------|----------------|----------|
| **Zero data loss** | All content preserved | Throughout |
| **Zero broken links** | 100% link validity | End of each phase |
| **Minimal disruption** | <4 hours downtime per phase | Per phase |
| **Full testing** | All tests pass | End of each phase |
| **User communication** | 100% stakeholder notified | Before each phase |

### Migration Principles

1. **Incremental:** Small, reversible changes
2. **Tested:** Comprehensive testing before each move
3. **Documented:** All changes recorded
4. **Communicated:** Stakeholders informed
5. **Safe:** Rollback plan for each phase

---

## 📅 Phase 1: Foundation Setup (Week 1-2)

### Objectives

- [ ] Create new directory structure
- [ ] Set up redirect infrastructure
- [ ] Establish backup procedures
- [ ] Create migration scripts

### Tasks

#### Week 1: Infrastructure Setup

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| 1 | Create target directories | Tech Lead | ⚪ |
| 2 | Set up backup system | DevOps | ⚪ |
| 3 | Create migration scripts | Dev Lead | ⚪ |
| 4 | Test backup/restore | QA | ⚪ |
| 5 | Document procedures | Tech Writer | ⚪ |

#### Week 2: Validation

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| 1 | Run full backup | DevOps | ⚪ |
| 2 | Verify backup integrity | QA | ⚪ |
| 3 | Test migration scripts on copy | Dev Lead | ⚪ |
| 4 | Fix script issues | Dev Lead | ⚪ |
| 5 | Phase 1 review | All | ⚪ |

### Scripts to Create

```bash
# scripts/migration/phase1-setup.sh
#!/bin/bash
# Phase 1: Foundation Setup

set -e

echo "=== Phase 1: Foundation Setup ==="

# Create new directory structure
echo "Creating target directories..."
mkdir -p docs/tutorials
mkdir -p docs/howto
mkdir -p docs/reference
mkdir -p docs/explanation
mkdir -p curriculum/learning-paths
mkdir -p curriculum/tracks
mkdir -p curriculum/assessments
mkdir -p careers
mkdir -p community
mkdir -p i18n

# Create README files for new directories
echo "Creating README files..."
# ... (create basic README for each)

# Create backup
echo "Creating backup..."
BACKUP_DIR="backups/phase1-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
cp -r docs "$BACKUP_DIR/"
cp -r curriculum "$BACKUP_DIR/"
cp -r src "$BACKUP_DIR/"

echo "Phase 1 complete!"
```

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Directory creation fails | Low | Low | Manual creation fallback |
| Backup incomplete | Medium | High | Multiple backup locations |
| Script errors | Medium | Medium | Test on copy first |

### Success Criteria

- [ ] All target directories exist
- [ ] Backup completed successfully
- [ ] Migration scripts tested
- [ ] Team trained on procedures

---

## 📅 Phase 2: Root Level Cleanup (Week 3-4)

### Objectives

- [ ] Consolidate root-level markdown files
- [ ] Create audience gateways in README.md
- [ ] Set up redirect files

### Tasks

#### Week 3: Content Consolidation

| Task | Source | Destination | Status |
|------|--------|-------------|--------|
| Move historical reports | Root `*.md` | `docs/internal/reports/` | ⚪ |
| Consolidate curriculum docs | Root `CURRICULUM_*.md` | `curriculum/README.md` | ⚪ |
| Archive implementation docs | Root `COMPLETE_*.md` | `docs/internal/reports/` | ⚪ |
| Update root README.md | `README.md` | Enhanced version | ⚪ |

#### Week 4: Gateway Creation

| Task | Description | Status |
|------|-------------|--------|
| Create student gateway | `/for-students/` section in README | ⚪ |
| Create instructor gateway | `/for-instructors/` section | ⚪ |
| Create contributor gateway | `/for-contributors/` section | ⚪ |
| Create hiring manager gateway | `/careers/` link | ⚪ |

### Redirect Strategy

```markdown
<!-- docs/01_foundations/README.md (old location) -->
# This page has moved

> ⚠️ **Redirect Notice**
> 
> This content has been moved to a new location.
> 
> **New location:** [Foundations Tutorials](../tutorials/beginner/README.md)
> 
> You will be redirected automatically in 5 seconds...

<meta http-equiv="refresh" content="5; url=../tutorials/beginner/README.md">
```

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Lost content during move | Low | High | Verify before delete |
| Broken internal links | High | Medium | Update all links |
| User confusion | Medium | Medium | Clear redirects |

### Success Criteria

- [ ] Root level has <10 markdown files
- [ ] All redirects working
- [ ] README.md has all gateways
- [ ] No broken links

---

## 📅 Phase 3: Curriculum Consolidation (Week 5-6)

### Objectives

- [ ] Consolidate curriculum structure
- [ ] Move assessments into curriculum
- [ ] Standardize module templates

### Tasks

#### Week 5: Structure Reorganization

| Task | Source | Destination | Status |
|------|--------|-------------|--------|
| Reorganize learning paths | `curriculum/learning_paths/` | `curriculum/learning-paths/` | ⚪ |
| Move assessments | `assessments/` (root) | `curriculum/assessments/` | ⚪ |
| Consolidate tracks | `curriculum/tracks/` | Keep, enhance | ⚪ |
| Update all internal links | Throughout | Updated paths | ⚪ |

#### Week 6: Template Standardization

| Task | Description | Status |
|------|-------------|--------|
| Apply module template | All modules | ⚪ |
| Standardize quizzes | All quiz files | ⚪ |
| Update project specs | All projects | ⚪ |
| Create missing rubrics | Projects without rubrics | ⚪ |

### Migration Script

```python
# scripts/migration/phase3_curriculum.py
"""Phase 3: Curriculum Consolidation"""

import os
import shutil
from pathlib import Path

def migrate_curriculum():
    """Migrate curriculum structure."""
    
    base = Path("curriculum")
    
    # Rename learning_paths to learning-paths
    old_paths = base / "learning_paths"
    new_paths = base / "learning-paths"
    if old_paths.exists():
        print(f"Moving {old_paths} → {new_paths}")
        shutil.move(str(old_paths), str(new_paths))
    
    # Move assessments from root
    root_assessments = Path("assessments")
    curriculum_assessments = base / "assessments"
    if root_assessments.exists() and not curriculum_assessments.exists():
        print(f"Moving {root_assessments} → {curriculum_assessments}")
        shutil.move(str(root_assessments), str(curriculum_assessments))
    
    # Update all internal links
    update_links()

def update_links():
    """Update all internal links to new paths."""
    # Implementation for link updating
    pass

if __name__ == "__main__":
    migrate_curriculum()
```

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Broken curriculum links | High | High | Comprehensive link check |
| Lost assessment data | Low | High | Backup before move |
| Inconsistent templates | Medium | Medium | Template validation |

### Success Criteria

- [ ] All curriculum content in new structure
- [ ] Assessments integrated
- [ ] All modules use standard template
- [ ] Link checker passes 100%

---

## 📅 Phase 4: Documentation Reorganization (Week 7-8)

### Objectives

- [ ] Apply Diátaxis framework
- [ ] Migrate all docs to new structure
- [ ] Create documentation hub

### Tasks

#### Week 7: Diátaxis Migration

| Task | Source | Destination | Status |
|------|--------|-------------|--------|
| Migrate tutorials | `docs/01_foundations/`, `docs/04_tutorials/` | `docs/tutorials/` | ⚪ |
| Migrate how-to | `docs/guides/`, `docs/04_production/` | `docs/howto/` | ⚪ |
| Migrate explanation | `docs/02_core_concepts/` | `docs/explanation/` | ⚪ |
| Migrate reference | `docs/reference/`, `docs/api/` | `docs/reference/` | ⚪ |

#### Week 8: Hub Creation

| Task | Description | Status |
|------|-------------|--------|
| Create docs/README.md | Documentation hub | ⚪ |
| Set up cross-references | Between doc types | ⚪ |
| Create search index | For all docs | ⚪ |
| Test navigation | User testing | ⚪ |

### Diátaxis Mapping

| Current Location | Content Type | New Location |
|------------------|--------------|--------------|
| `docs/00_introduction/` | Tutorials | `docs/tutorials/getting-started/` |
| `docs/01_foundations/` | Tutorials | `docs/tutorials/beginner/` |
| `docs/02_intermediate/` | Tutorials | `docs/tutorials/intermediate/` |
| `docs/03_advanced/` | Tutorials | `docs/tutorials/advanced/` |
| `docs/02_core_concepts/` | Explanation | `docs/explanation/concepts/` |
| `docs/03_system_design/` | Explanation | `docs/explanation/architecture/` |
| `docs/04_production/` | How-to | `docs/howto/deployment/` |
| `docs/05_case_studies/` | Explanation | `docs/explanation/case-studies/` |
| `docs/06_tutorials/` | Tutorials | Merge into `docs/tutorials/` |
| `docs/reference/` | Reference | Keep, enhance |
| `docs/api/` | Reference | `docs/reference/api/` |

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Lost documentation | Low | High | Full backup |
| Broken cross-references | High | Medium | Automated link check |
| User disorientation | Medium | Medium | Clear navigation |

### Success Criteria

- [ ] All docs in Diátaxis structure
- [ ] Documentation hub complete
- [ ] Cross-references working
- [ ] Search functional

---

## 📅 Phase 5: Code Organization (Week 9-10)

### Objectives

- [ ] Reorganize src/ subdirectories
- [ ] Update all imports
- [ ] Verify all tests pass

### Tasks

#### Week 9: src/ Reorganization

| Task | Source | Destination | Status |
|------|--------|-------------|--------|
| Create subdirectories | `src/core/*.py` | `src/core/math/`, etc. | ⚪ |
| Organize ml/ | `src/ml/*.py` | `src/ml/classical/`, etc. | ⚪ |
| Organize llm/ | `src/llm/*.py` | `src/llm/architecture/`, etc. | ⚪ |
| Update __init__.py | All modules | New exports | ⚪ |

#### Week 10: Import Updates

| Task | Description | Status |
|------|-------------|--------|
| Update all imports | Throughout codebase | ⚪ |
| Fix circular dependencies | If any | ⚪ |
| Run all tests | Verify functionality | ⚪ |
| Update documentation | Code examples | ⚪ |

### Import Migration Guide

```python
# Old imports (to be replaced)
from src.core.math_operations import Vector
from src.ml.classical import LinearRegression
from src.llm.attention import MultiHeadAttention

# New imports (target)
from src.core.math.vectors import Vector
from src.ml.classical.linear_regression import LinearRegression
from src.llm.architecture.attention import MultiHeadAttention

# Migration script
# scripts/migration/update-imports.py
```

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Broken imports | High | High | Comprehensive testing |
| Circular dependencies | Medium | High | Import analysis |
| Test failures | Medium | Medium | Fix before merge |

### Success Criteria

- [ ] All imports updated
- [ ] All tests pass
- [ ] No circular dependencies
- [ ] Documentation updated

---

## 📅 Phase 6: New Features (Week 11-12)

### Objectives

- [ ] Create careers/ hub
- [ ] Set up community/ structure
- [ ] Create i18n/ framework

### Tasks

#### Week 11: Career & Community

| Task | Description | Status |
|------|-------------|--------|
| Create careers/README.md | Career hub | ⚪ |
| Populate job pathways | ML Engineer, LLM Engineer, etc. | ⚪ |
| Create interview prep | Question bank | ⚪ |
| Set up community/ | Contribution, mentorship | ⚪ |

#### Week 12: i18n & Finalization

| Task | Description | Status |
|------|-------------|--------|
| Create i18n/ structure | Language directories | ⚪ |
| Set up translation framework | i18n tools | ⚪ |
| Final link check | All links | ⚪ |
| Migration complete review | All phases | ⚪ |

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Incomplete features | Medium | Low | Prioritize core features |
| Translation gaps | High | Low | Mark as work in progress |
| Timeline slippage | Medium | Medium | Buffer time included |

### Success Criteria

- [ ] Careers hub functional
- [ ] Community structure ready
- [ ] i18n framework in place
- [ ] All phases complete

---

## 🧪 Testing Strategy

### Test Types

| Test Type | Scope | Tools | Frequency |
|-----------|-------|-------|-----------|
| **Unit Tests** | Individual functions | pytest | Every change |
| **Integration Tests** | Module interactions | pytest | Every change |
| **Link Tests** | All internal/external links | markdown-link-check | Every phase |
| **Import Tests** | All imports work | pytest | Every change |
| **E2E Tests** | Full user journeys | pytest | Every phase |

### Test Checklist

#### Phase Completion Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Link checker: 100% valid
- [ ] Import checker: No errors
- [ ] Documentation builds
- [ ] Docker builds succeed

### Validation Scripts

```bash
# scripts/migration/validate-phase.sh
#!/bin/bash
# Validate phase completion

set -e

PHASE=$1

echo "=== Validating Phase $PHASE ==="

# Run tests
echo "Running tests..."
make test

# Check links
echo "Checking links..."
find docs/ -name "*.md" -exec markdown-link-check {} \;

# Check imports
echo "Checking imports..."
python -c "import src; print('Imports OK')"

# Check documentation
echo "Building docs..."
mkdocs build

echo "=== Phase $PHASE Validation Complete ==="
```

---

## ↩️ Rollback Procedures

### Rollback Triggers

| Trigger | Action |
|---------|--------|
| Critical bug found | Immediate rollback |
| >10% test failures | Rollback and fix |
| Data loss detected | Restore from backup |
| User reports critical issues | Investigate, rollback if needed |

### Rollback Steps

```bash
# scripts/migration/rollback.sh
#!/bin/bash
# Rollback to previous state

set -e

PHASE=$1
BACKUP_DIR="backups/phase${PHASE}-*"

echo "=== Rollback Phase $PHASE ==="

# Find latest backup
LATEST_BACKUP=$(ls -td $BACKUP_DIR | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No backup found for phase $PHASE"
    exit 1
fi

echo "Restoring from $LATEST_BACKUP..."

# Restore based on phase
case $PHASE in
    1)
        cp -r "$LATEST_BACKUP/docs" .
        cp -r "$LATEST_BACKUP/curriculum" .
        ;;
    2)
        # Restore root level
        ;;
    # ... other phases
esac

echo "Rollback complete!"
```

### Rollback Decision Tree

```
Issue Detected
      │
      ▼
┌─────────────────┐
│ Can it be fixed │
│ in <1 hour?     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   Yes       No
    │         │
    ▼         ▼
  Fix    ┌─────────────┐
         │ Is data     │
         │ at risk?    │
         └──────┬──────┘
                │
           ┌────┴────┐
           │         │
          Yes       No
           │         │
           ▼         ▼
      Rollback   Fix in
      immediately next release
```

---

## 📢 Communication Plan

### Stakeholder Communication

| Stakeholder | Channel | Frequency | Content |
|-------------|---------|-----------|---------|
| **Core Team** | Slack/Discord | Daily | Progress, blockers |
| **Contributors** | GitHub Discussions | Weekly | Phase updates |
| **Students** | README banner | Per phase | Downtime notices |
| **Instructors** | Email | Per phase | Impact on teaching |

### Communication Templates

#### Phase Start Announcement

```markdown
## 🔄 Migration Phase X Starting

**Date:** [Start Date]
**Duration:** [X] weeks
**Impact:** [Minimal/Moderate/Significant]

### What's Changing
[Brief description]

### What to Expect
- [Expected impacts]

### How to Help
- [Ways contributors can help]

### Questions?
[Contact information]
```

#### Phase Complete Announcement

```markdown
## ✅ Migration Phase X Complete!

**Date:** [Completion Date]
**Status:** All tests passing

### What Was Done
- [List of completed tasks]

### What's New
- [New features/structure]

### Known Issues
- [Any remaining issues]

### Next Phase
- [Preview of next phase]

### Thank You
[Thanks to contributors]
```

---

## 📊 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Zero data loss** | 100% content preserved | Backup comparison |
| **Zero broken links** | 100% valid | Link checker |
| **Test pass rate** | 100% | CI/CD |
| **User satisfaction** | >4/5 | Survey |
| **Timeline adherence** | ±1 week | Project tracking |
| **Rollback incidents** | 0 | Incident log |

---

## 📋 Phase Gate Checklist

### Before Starting Each Phase

- [ ] Previous phase complete and validated
- [ ] Backup completed
- [ ] Team briefed
- [ ] Rollback plan ready
- [ ] Communication sent

### Before Ending Each Phase

- [ ] All tasks complete
- [ ] Tests passing
- [ ] Links validated
- [ ] Documentation updated
- [ ] Stakeholders notified

---

**Document Status:** ✅ **COMPLETE - Migration Playbook**

**Next Document:** [QUICK_REFERENCE_GUIDE.md](./QUICK_REFERENCE_GUIDE.md)

---

*Document Version: 3.0 | Last Updated: March 30, 2026 | AI-Mastery-2026*

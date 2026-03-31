# Curriculum Integration Map

## FineTuningLLMs → AI-Mastery-2026

This document maps the FineTuningLLMs repository content to the AI-Mastery-2026 curriculum structure, identifying gaps, integration approaches, and attribution requirements.

---

## 1. Curriculum Overview

### AI-Mastery-2026 Tier Structure

```
Tier 1: AI Fundamentals
├── Python for AI
├── Math for AI
└── ML Basics

Tier 2: LLM Scientist
├── Track 01: LLM Fundamentals
├── Track 02: Transformer Architecture
├── Track 03: Prompt Engineering
├── Track 04: Embeddings & Vector Search
├── Track 05: RAG Systems
└── Track 06: Fine-Tuning ← PRIMARY INTEGRATION

Tier 3: LLM Engineer
├── Track 01: LLM Application Development
├── Track 02: Production RAG
├── Track 03: Running LLMs ← SECONDARY INTEGRATION
└── Track 04: LLM Ops

Tier 4: Production
├── Track 01: Deployment Patterns
├── Track 02: Scaling & Optimization
├── Track 03: Monitoring & Observability
└── Track 04: Security & Compliance ← TERTIARY INTEGRATION
```

---

## 2. Detailed Mapping

### 2.1 Tier 2, Track 06: Fine-Tuning Track

#### Current Module Structure (Before Integration)

| Module | Topic | Content Status | Gap |
|--------|-------|----------------|-----|
| 6.1 | Introduction to Fine-Tuning | Outline only | Needs content |
| 6.2 | When to Fine-Tune vs Prompt | Brief notes | Needs examples |
| 6.3 | Fine-Tuning Methods Overview | High-level | Needs depth |
| 6.4 | Hands-On Lab | Not defined | Needs complete lab |
| 6.5 | Project | Not defined | Needs specification |

#### FineTuningLLMs Content Mapping

| FineTuningLLMs | AI-Mastery-2026 Module | Integration Approach |
|----------------|----------------------|---------------------|
| Chapter 0 (LLM Fundamentals) | Module 6.1 | Embed & Adapt |
| Chapter 0 (Transformer Review) | Module 6.2 | Embed & Adapt |
| Chapter 2 (Data Preparation) | Module 6.4 | Embed & Adapt |
| Chapter 3 (PEFT & LoRA) | Module 6.3 | Embed & Adapt |
| Chapter 3 (LoRA Lab) | Module 6.4 | Embed as Lab |
| Chapter 4 (SFTTrainer) | Module 6.4 | Embed as Lab |
| Chapter 5 (Flash Attention) | Module 6.3 | Reference |

#### Gap Analysis & Solutions

| Current Gap | How FineTuningLLMs Fills It | Integration Approach |
|-------------|----------------------------|---------------------|
| No hands-on notebooks | 7 complete chapter notebooks | Embed Chapters 2-5 |
| Limited theory depth | Comprehensive theory in chapters | Adapt theory sections |
| No working examples | Complete working code | Use as-is with attribution |
| No troubleshooting | FAQ.md with solutions | Embed troubleshooting section |
| No assessment | Can add quizzes to notebooks | Enhance with knowledge checks |

#### Attribution Requirements

```markdown
## Attribution for Module 6.x

Content in this module is adapted from:
- Repository: FineTuningLLMs by dvgodoy
- URL: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License
- Book: "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

Specific adaptations:
- Added learning objectives
- Integrated knowledge check questions
- Aligned with AI-Mastery-2026 assessment structure
```

---

### 2.2 Tier 3, Track 03: Running LLMs Track

#### Current Module Structure (Before Integration)

| Module | Topic | Content Status | Gap |
|--------|-------|----------------|-----|
| 3.1 | LLM Inference Basics | Outline only | Needs content |
| 3.2 | Quantization for Inference | Brief notes | Needs depth |
| 3.3 | Model Formats (GGUF, etc.) | Not defined | Needs content |
| 3.4 | Local Deployment | Not defined | Needs lab |
| 3.5 | Cloud Deployment | Not defined | Needs examples |

#### FineTuningLLMs Content Mapping

| FineTuningLLMs | AI-Mastery-2026 Module | Integration Approach |
|----------------|----------------------|---------------------|
| Chapter 1 (Quantization) | Module 3.2 | Embed & Adapt |
| Chapter 0 (LLM Basics) | Module 3.1 | Reference |
| Chapter 6 (GGUF Conversion) | Module 3.3 | Embed & Adapt |
| Chapter 6 (Ollama Deployment) | Module 3.4 | Embed as Lab |
| Appendix A (GPU Setup) | Module 3.1 | Embed as Setup Guide |

#### Gap Analysis & Solutions

| Current Gap | How FineTuningLLMs Fills It | Integration Approach |
|-------------|----------------------------|---------------------|
| No quantization depth | Complete Chapter 1 on quantization | Embed Chapter 1 |
| No GGUF coverage | Chapter 6 covers GGUF conversion | Embed Chapter 6 Part 1 |
| No local deployment lab | Ollama integration in Chapter 6 | Embed as hands-on lab |
| No environment setup guide | Appendix A provides setup | Embed as prerequisite |

#### Attribution Requirements

```markdown
## Attribution for Module 3.x

Quantization and GGUF conversion content adapted from:
- Repository: FineTuningLLMs by dvgodoy
- URL: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License

Specific content:
- Chapter 1: Quantization techniques (8-bit, 4-bit, BitsAndBytes)
- Chapter 6: GGUF conversion and llama.cpp integration
```

---

### 2.3 Tier 4, Production Deployment Track

#### Current Module Structure (Before Integration)

| Module | Topic | Content Status | Gap |
|--------|-------|----------------|-----|
| 4.1 | Deployment Patterns | Outline only | Needs examples |
| 4.2 | Model Optimization | Brief notes | Needs techniques |
| 4.3 | Monitoring & Observability | Not defined | Needs content |
| 4.4 | Production Troubleshooting | Not defined | Needs guide |

#### FineTuningLLMs Content Mapping

| FineTuningLLMs | AI-Mastery-2026 Module | Integration Approach |
|----------------|----------------------|---------------------|
| Chapter 6 (Deployment) | Module 4.1 | Embed & Adapt |
| Chapter 5 (Flash Attention) | Module 4.2 | Embed & Adapt |
| Chapter 6 (Ollama API) | Module 4.1 | Embed as Example |
| FAQ.md (Troubleshooting) | Module 4.4 | Adapt & Expand |

#### Gap Analysis & Solutions

| Current Gap | How FineTuningLLMs Fills It | Integration Approach |
|-------------|----------------------------|---------------------|
| No deployment examples | Ollama deployment in Chapter 6 | Embed as pattern example |
| Limited optimization | Flash Attention in Chapter 5 | Embed optimization section |
| No troubleshooting guide | FAQ.md with solutions | Adapt for production context |

#### Attribution Requirements

```markdown
## Attribution for Module 4.x

Production deployment and optimization content adapted from:
- Repository: FineTuningLLMs by dvgodoy
- URL: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License

Specific content:
- Chapter 5: Flash Attention optimization
- Chapter 6: Local deployment patterns with Ollama
- FAQ.md: Troubleshooting common issues
```

---

## 3. Notebook Integration Mapping

### 3.1 Notebook Directory Structure

```
AI-Mastery-2026/
└── notebooks/
    └── 04_llm_fundamentals/
        └── fine-tuning/
            ├── 01_introduction_to_fine_tuning.ipynb    (from Chapter 0)
            ├── 02_quantization_fundamentals.ipynb      (from Chapter 1)
            ├── 03_data_preparation.ipynb               (from Chapter 2)
            ├── 04_tokenization_chat_templates.ipynb    (from Chapter 2)
            ├── 05_peft_lora_theory.ipynb               (from Chapter 3)
            ├── 06_lora_implementation.ipynb            (from Chapter 3)
            ├── 07_sfttrainer_deep_dive.ipynb           (from Chapter 4)
            ├── 08_flash_attention_optimization.ipynb   (from Chapter 5)
            ├── 09_gguf_conversion.ipynb                (from Chapter 6)
            ├── 10_local_deployment_ollama.ipynb        (from Chapter 6)
            ├── 11_environment_setup.ipynb              (from Appendix A)
            └── 12_troubleshooting_guide.ipynb          (from FAQ.md)
```

### 3.2 Notebook Enhancement Requirements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Learning Objectives | Add header with module objectives | High |
| Knowledge Checks | Insert quiz questions throughout | High |
| Try It Yourself | Add exercises after key concepts | High |
| Progress Markers | Add completion checkpoints | Medium |
| Solution Cells | Provide hidden solution cells | Medium |
| Troubleshooting Tips | Add tips from FAQ.md | Medium |
| Assessment Integration | Link to quiz system | High |

### 3.3 Notebook Modification Checklist

For each notebook:

```
[ ] Add learning objectives header
[ ] Insert knowledge check questions (3-5 per notebook)
[ ] Add "Try It Yourself" exercises
[ ] Include troubleshooting tips
[ ] Add progress tracking markers
[ ] Update imports for src/ structure
[ ] Add attribution footer
[ ] Test execution on Colab
[ ] Add estimated completion time
[ ] Include prerequisite links
```

---

## 4. Assessment Integration

### 4.1 Quiz Question Mapping

| Notebook | Quiz Topics | Question Count |
|----------|-------------|----------------|
| Chapter 0 | LLM basics, transformer architecture | 10 |
| Chapter 1 | Quantization types, BitsAndBytes | 8 |
| Chapter 2 | Data formatting, tokenization | 10 |
| Chapter 3 | PEFT, LoRA theory | 10 |
| Chapter 4 | SFTTrainer, training loops | 8 |
| Chapter 5 | Flash Attention, optimization | 6 |
| Chapter 6 | GGUF, deployment | 8 |

### 4.2 Project Rubric Mapping

| Project Component | FineTuningLLMs Source | Rubric Weight |
|-------------------|----------------------|---------------|
| Data Preparation | Chapter 2 | 20% |
| Model Selection | Chapter 0, 3 | 15% |
| Fine-Tuning Implementation | Chapter 3, 4 | 30% |
| Optimization | Chapter 5 | 15% |
| Deployment | Chapter 6 | 20% |

---

## 5. Integration Summary Table

### 5.1 Complete Mapping Overview

| FineTuningLLMs Content | AI-Mastery-2026 Location | Approach | Attribution |
|------------------------|-------------------------|----------|-------------|
| Chapter 0 | Tier 2, Module 6.1-6.2 | Embed & Adapt | Full credit |
| Chapter 1 | Tier 3, Module 3.2 | Embed & Adapt | Full credit |
| Chapter 2 | Tier 2, Module 6.4 | Embed & Adapt | Full credit |
| Chapter 3 | Tier 2, Module 6.3-6.4 | Embed & Adapt | Full credit |
| Chapter 4 | Tier 2, Module 6.4 | Embed & Adapt | Full credit |
| Chapter 5 | Tier 2, Module 6.3 / Tier 4, Module 4.2 | Embed & Adapt | Full credit |
| Chapter 6 | Tier 3, Module 3.3-3.4 / Tier 4, Module 4.1 | Embed & Adapt | Full credit |
| Appendix A | Tier 3, Module 3.1 | Embed | Full credit |
| FAQ.md | All tracks | Adapt | Full credit |

### 5.2 Integration Effort Estimate

| Phase | Tasks | Estimated Hours |
|-------|-------|-----------------|
| Setup | Directory structure, attribution docs | 8 |
| Adaptation | Notebook modifications, enhancements | 24 |
| Assessment | Quiz questions, rubrics | 16 |
| Testing | Execute all notebooks, fix issues | 12 |
| Documentation | Module READMEs, guides | 8 |
| **Total** | | **68 hours** |

---

## 6. Prerequisite Mapping

### 6.1 Student Prerequisites

Before starting Fine-Tuning Track, students should complete:

| Prerequisite | AI-Mastery-2026 Module | FineTuningLLMs Assumption |
|--------------|----------------------|---------------------------|
| Python Programming | Tier 1, Module 1 | Assumed |
| PyTorch Basics | Tier 1, Module 3 | Assumed |
| Transformer Basics | Tier 2, Track 01 | Covered in Chapter 0 |
| Hugging Face Basics | Tier 2, Track 01 | Covered in Chapter 0 |
| GPU Fundamentals | Tier 3, Track 03 | Covered in Appendix A |

### 6.2 Hardware Prerequisites

| Option | Requirements | Notes |
|--------|--------------|-------|
| Local | GPU with 8GB+ VRAM | RTX 3060 or better |
| Colab Free | T4 GPU (16GB) | Sufficient for most examples |
| Colab Pro | A100 GPU (40GB) | Recommended for larger models |
| Cloud | Runpod/Lambda Labs | Pay-as-you-go option |

---

## 7. Version Compatibility

### 7.1 Library Version Alignment

| Library | FineTuningLLMs | AI-Mastery-2026 | Action |
|---------|----------------|-----------------|--------|
| Python | 3.9+ | 3.10+ | ✅ Compatible |
| PyTorch | 2.0+ | 2.1+ | ✅ Compatible |
| Transformers | 4.35+ | 4.37+ | ✅ Compatible |
| PEFT | 0.6+ | 0.7+ | ✅ Compatible |
| BitsAndBytes | 0.41+ | 0.43+ | ✅ Compatible |
| Accelerate | 0.24+ | 0.25+ | ✅ Compatible |

### 7.2 Required Updates

| File | Update Needed | Reason |
|------|---------------|--------|
| requirements.txt | Version alignment | Match AI-Mastery-2026 versions |
| Notebook imports | Update for src/ | Integration with codebase |
| Model references | Add alternatives | Provide multiple model options |

---

## 8. Content Enhancement Plan

### 8.1 Theory Enhancements

| Topic | Enhancement | Source |
|-------|-------------|--------|
| LoRA Mathematics | Add detailed derivations | Original + supplements |
| Quantization Theory | Add visual diagrams | Original + supplements |
| Flash Attention | Add performance benchmarks | Original + supplements |

### 8.2 Practical Enhancements

| Topic | Enhancement | Source |
|-------|-------------|--------|
| Data Preparation | Add more dataset examples | AI-Mastery-2026 |
| Evaluation | Add comprehensive metrics | AI-Mastery-2026 |
| Deployment | Add Docker examples | AI-Mastery-2026 |
| Monitoring | Add observability integration | AI-Mastery-2026 |

### 8.3 Assessment Enhancements

| Topic | Enhancement | Source |
|-------|-------------|--------|
| Knowledge Checks | Add 5-10 questions per module | AI-Mastery-2026 |
| Projects | Define 3 capstone projects | AI-Mastery-2026 |
| Rubrics | Create detailed grading rubrics | AI-Mastery-2026 |

---

## 9. Attribution Implementation

### 9.1 File-Level Attribution

Add to each adapted notebook:

```markdown
---
title: [Module Title]
original_author: dvgodoy
original_source: https://github.com/dvgodoy/FineTuningLLMs
original_license: MIT
adaptations_by: AI-Mastery-2026 Team
adaptation_date: 2026
---

## Attribution

This notebook is adapted from the FineTuningLLMs repository by dvgodoy:
https://github.com/dvgodoy/FineTuningLLMs

Original License: MIT License
Associated Book: "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

Modifications made for AI-Mastery-2026 curriculum:
- Added learning objectives
- Integrated knowledge check questions
- Added "Try It Yourself" exercises
- Updated imports for curriculum structure
```

### 9.2 Module-Level Attribution

Add to each module README:

```markdown
## Content Attribution

This module incorporates content from the FineTuningLLMs repository:

- **Original Author:** dvgodoy (Daniel Godoy)
- **Repository:** https://github.com/dvgodoy/FineTuningLLMs
- **License:** MIT License
- **Associated Book:** "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

We gratefully acknowledge this excellent resource and encourage students to 
explore the original repository for additional examples and insights.
```

### 9.3 Course-Level Attribution

Add to main curriculum documentation:

```markdown
## Fine-Tuning Track Attribution

The Fine-Tuning Track (Tier 2, Track 06) is built upon the excellent 
FineTuningLLMs repository by dvgodoy. This integration enables us to provide 
students with production-quality, hands-on fine-tuning education.

We maintain full attribution and comply with the MIT License requirements.
Students are encouraged to explore the original repository for additional 
learning resources.
```

---

## 10. Integration Timeline

### Phase 1: Setup (Week 1)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Download and review all notebooks | Content inventory |
| 3 | Create directory structure | Folder structure |
| 4 | Create attribution documentation | ATTRIBUTION_AND_LEGAL.md |
| 5 | Test notebooks on Colab | Compatibility report |

### Phase 2: Adaptation (Week 2-3)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-3 | Add learning objectives to notebooks | Enhanced notebooks |
| 4-7 | Insert knowledge checks | Quiz-ready notebooks |
| 8-10 | Add exercises and solutions | Complete labs |
| 11-12 | Update imports and structure | Integrated codebase |
| 13-14 | Test all adapted notebooks | Verified notebooks |

### Phase 3: Integration (Week 4)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Map to curriculum modules | Module READMEs |
| 3-5 | Write quiz questions | Question bank |
| 6-7 | Define project specifications | Project docs |
| 8-9 | Create rubrics | Assessment rubrics |
| 10 | Final review | Integration complete |

### Phase 4: Testing & Launch (Week 5-6)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-5 | Beta test with students | Feedback report |
| 6-8 | Fix issues | Updated content |
| 9-10 | Final review and launch | Published track |

---

## 11. Success Criteria

### 11.1 Integration Success

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Notebook execution | 100% pass | All notebooks run on Colab |
| Attribution compliance | 100% | Legal review passed |
| Content alignment | >90% | Maps to curriculum objectives |
| Assessment coverage | 100% | All modules have quizzes |

### 11.2 Student Success

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| First fine-tune time | <2 hours | Time to complete first lab |
| Track completion rate | >80% | Students finishing track |
| Satisfaction rating | >4.0/5.0 | Student feedback |
| Project completion | >30% | Students completing capstone |

---

## Appendix: Quick Reference

### Integration Contact Points

| Role | Responsibility |
|------|----------------|
| Curriculum Lead | Overall integration oversight |
| Content Developer | Notebook adaptation |
| Assessment Designer | Quiz and rubric creation |
| Legal Review | Attribution compliance |
| QA Tester | Notebook execution testing |

### Key Documents

| Document | Purpose |
|----------|---------|
| REPO_ANALYSIS_FINE_TUNING.md | Repository assessment |
| CURRICULUM_INTEGRATION_MAP.md | This document |
| ENHANCED_FINE_TUNING_TRACK.md | Track structure |
| NOTEBOOK_INTEGRATION_PLAN.md | Notebook details |
| ATTRIBUTION_AND_LEGAL.md | Legal compliance |

---

*Document Version: 1.0*  
*Created: March 30, 2026*  
*For: AI-Mastery-2026 Curriculum Integration*

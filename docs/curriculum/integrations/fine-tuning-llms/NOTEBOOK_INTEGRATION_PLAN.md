# Notebook Integration Plan

## FineTuningLLMs Notebooks → AI-Mastery-2026

This document details the strategy for integrating the 7 chapter notebooks from the FineTuningLLMs repository into the AI-Mastery-2026 curriculum.

---

## 1. Notebook Inventory

### 1.1 Source Notebooks

| Notebook | Source | Topics | Estimated Duration |
|----------|--------|--------|-------------------|
| `Chapter0.ipynb` | FineTuningLLMs | LLM Fundamentals, Transformer Architecture | 3-4 hours |
| `Chapter1.ipynb` | FineTuningLLMs | Quantization (8-bit, 4-bit, BitsAndBytes) | 2-3 hours |
| `Chapter2.ipynb` | FineTuningLLMs | Data Formatting, Tokenization, Chat Templates | 4-5 hours |
| `Chapter3.ipynb` | FineTuningLLMs | PEFT & LoRA Theory + Implementation | 4-5 hours |
| `Chapter4.ipynb` | FineTuningLLMs | SFTTrainer Deep Dive | 5-6 hours |
| `Chapter5.ipynb` | FineTuningLLMs | Flash Attention & Optimization | 3-4 hours |
| `Chapter6.ipynb` | FineTuningLLMs | GGUF Conversion & Local Deployment | 4-5 hours |

### 1.2 Supplementary Notebooks

| Notebook | Source | Topics | Priority |
|----------|--------|--------|----------|
| `AppendixA.ipynb` | FineTuningLLMs | GPU Environment Setup | High |
| `AppendixB.ipynb` | FineTuningLLMs | Additional Examples | Medium |
| `Troubleshooting.ipynb` | FineTuningLLMs | Common Issues & Solutions | High |

---

## 2. Integration Options Analysis

### 2.1 Option 1: Direct Reference

**Description:** Link to original notebooks in the FineTuningLLMs repository.

**Pros:**
- ✅ No maintenance burden
- ✅ Always up-to-date with original
- ✅ Proper attribution is clear
- ✅ Minimal legal complexity

**Cons:**
- ❌ Students leave curriculum platform
- ❌ No customization for curriculum flow
- ❌ Dependencies on external repo availability
- ❌ Cannot add curriculum-specific assessments
- ❌ Broken links if repo changes

**Verdict:** ❌ Not Recommended for core content

---

### 2.2 Option 2: Fork & Adapt

**Description:** Copy notebooks to curriculum repository with modifications.

**Pros:**
- ✅ Full control over content
- ✅ Can add assessments and exercises
- ✅ Integrated with curriculum flow
- ✅ Can fix issues and update independently
- ✅ Works offline

**Cons:**
- ⚠️ Maintenance burden (sync with updates)
- ⚠️ Must maintain attribution carefully
- ⚠️ Larger repository size
- ⚠️ Potential for divergence from original

**Verdict:** ✅ Recommended for core notebooks

---

### 2.3 Option 3: Embed as Git Submodule

**Description:** Add FineTuningLLMs as a git submodule.

**Pros:**
- ✅ Clear separation of original content
- ✅ Easy to update from upstream
- ✅ Maintains git history
- ✅ Clear attribution

**Cons:**
- ⚠️ More complex git workflow
- ⚠️ Students need to initialize submodules
- ⚠️ Cannot easily modify content
- ⚠️ CI/CD complexity

**Verdict:** ⚠️ Consider for supplementary content

---

### 2.4 Option 4: Hybrid Approach

**Description:** Embed core notebooks, link to advanced/supplementary.

**Pros:**
- ✅ Best of both worlds
- ✅ Core content integrated and enhanced
- ✅ Advanced content stays linked (less maintenance)
- ✅ Balanced maintenance burden

**Cons:**
- ⚠️ Slightly complex structure
- ⚠️ Need clear navigation between embedded and linked

**Verdict:** ✅✅ STRONGLY RECOMMENDED

---

## 3. Recommended Integration Approach

### 3.1 Hybrid Strategy

| Notebook Type | Approach | Rationale |
|---------------|----------|-----------|
| Core Chapters (0-6) | Fork & Adapt | Essential curriculum content |
| Appendix A (Setup) | Fork & Adapt | Required for all students |
| Appendix B (Examples) | Direct Reference | Supplementary, less critical |
| Troubleshooting | Fork & Adapt | Critical for student success |

### 3.2 Directory Structure

```
AI-Mastery-2026/
└── notebooks/
    └── 04_llm_fundamentals/
        └── fine-tuning/
            ├── README.md                          # Track overview
            ├── setup/
            │   ├── 00_environment_setup.ipynb     # From Appendix A
            │   └── requirements.txt               # Pinned dependencies
            ├── fundamentals/
            │   ├── 01_introduction_to_fine_tuning.ipynb  # From Chapter 0 (Part 1)
            │   └── 02_transformer_architecture.ipynb     # From Chapter 0 (Part 2)
            ├── quantization/
            │   └── 03_quantization_fundamentals.ipynb    # From Chapter 1
            ├── data/
            │   ├── 04_data_preparation.ipynb             # From Chapter 2 (Part 1)
            │   └── 05_tokenization_chat_templates.ipynb  # From Chapter 2 (Part 2)
            ├── lora/
            │   ├── 06_peft_lora_theory.ipynb             # From Chapter 3 (Part 1)
            │   └── 07_lora_implementation.ipynb          # From Chapter 3 (Part 2)
            ├── training/
            │   └── 08_sfttrainer_deep_dive.ipynb         # From Chapter 4
            ├── optimization/
            │   └── 09_flash_attention.ipynb              # From Chapter 5
            ├── deployment/
            │   ├── 10_gguf_conversion.ipynb              # From Chapter 6 (Part 1)
            │   └── 11_local_deployment_ollama.ipynb      # From Chapter 6 (Part 2)
            └── resources/
                ├── troubleshooting_guide.ipynb            # From FAQ.md
                ├── solution_notebooks/                    # Completed examples
                │   └── ...
                └── datasets/                              # Sample datasets
                    └── ...
```

### 3.3 External References

```
AI-Mastery-2026/
└── docs/
    └── curriculum/
        └── integrations/
            └── fine-tuning-llms/
                └── external_resources.md  # Links to supplementary content
```

---

## 4. Notebook Enhancement Checklist

### 4.1 Standard Enhancements (All Notebooks)

```
Enhancement Checklist for Each Notebook:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] HEADER SECTION
    [ ] Learning objectives (3-5 bullet points)
    [ ] Estimated completion time
    [ ] Prerequisites list
    [ ] Attribution notice (dvgodoy, MIT License)
    [ ] Module number and title

[ ] CONTENT ENHANCEMENTS
    [ ] Knowledge check questions (every 2-3 sections)
    [ ] "Try It Yourself" exercises
    [ ] Troubleshooting tips (from FAQ.md)
    [ ] Visual diagrams where helpful
    [ ] Code comments for complex sections

[ ] ASSESSMENT INTEGRATION
    [ ] Quiz question markers (for extraction)
    [ ] Exercise solution cells (collapsed)
    [ ] Progress checkpoint markers
    [ ] Completion criteria

[ ] TECHNICAL UPDATES
    [ ] Update imports for src/ structure
    [ ] Pin library versions
    [ ] Add error handling examples
    [ ] Test on Colab (free tier)
    [ ] Add GPU memory requirements note

[ ] FOOTER SECTION
    [ ] Summary of key takeaways
    [ ] Links to next module
    [ ] Links to resources
    [ ] Full attribution notice
    [ ] Feedback link
```

### 4.2 Enhancement Templates

#### Learning Objectives Header Template

```markdown
# Module X: [Title]

## Learning Objectives

By the end of this module, you will be able to:
- [ ] Objective 1 (Bloom's taxonomy: Understand/Apply/Analyze)
- [ ] Objective 2
- [ ] Objective 3
- [ ] Objective 4
- [ ] Objective 5

## Module Info

| Property | Value |
|----------|-------|
| Estimated Time | X hours |
| Difficulty | Beginner/Intermediate/Advanced |
| Prerequisites | [List of prerequisites] |
| GPU Required | Yes (minimum X GB VRAM) |
| Colab Compatible | Yes |

## Attribution

This module is adapted from the FineTuningLLMs repository by dvgodoy:
- Repository: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License
- Book: "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

Modifications made for AI-Mastery-2026 curriculum.
```

#### Knowledge Check Template

```markdown
## 🧠 Knowledge Check

<details>
<summary>Question 1: [Question text]</summary>

**Answer:** [Answer]

**Explanation:** [Detailed explanation]

</details>

<details>
<summary>Question 2: [Question text]</summary>

**Answer:** [Answer]

**Explanation:** [Detailed explanation]

</details>
```

#### Try It Yourself Template

```markdown
## 💪 Try It Yourself

**Exercise X.X:** [Exercise title]

**Task:** [Clear description of what to do]

**Hints:**
- Hint 1
- Hint 2

<details>
<summary>View Solution</summary>

```python
# Solution code here
```

</details>
```

#### Troubleshooting Tip Template

```markdown
> **💡 Troubleshooting Tip**
>
> **Issue:** [Common issue description]
>
> **Solution:** [How to fix it]
>
> **Prevention:** [How to avoid it in the future]
```

---

## 5. Notebook-by-Notebook Enhancement Plan

### 5.1 Chapter 0 → Modules 1 & 2

**Source:** `Chapter0.ipynb`  
**Target:** `01_introduction_to_fine_tuning.ipynb`, `02_transformer_architecture.ipynb`

| Enhancement | Details |
|-------------|---------|
| Split | Divide into 2 notebooks (intro + architecture) |
| Add | Business use cases for fine-tuning |
| Add | Decision tree: Fine-tune vs Prompt |
| Add | Interactive architecture diagrams |
| Add | 10 knowledge check questions |
| Add | Lab: Load and inspect different models |

---

### 5.2 Chapter 1 → Module 3

**Source:** `Chapter1.ipynb`  
**Target:** `03_quantization_fundamentals.ipynb`

| Enhancement | Details |
|-------------|---------|
| Add | Memory savings calculator |
| Add | Visual comparison of precision levels |
| Add | Cost comparison (cloud GPU hours) |
| Add | 8 knowledge check questions |
| Add | Lab: Compare 16-bit, 8-bit, 4-bit inference |
| Update | Latest BitsAndBytes version |

---

### 5.3 Chapter 2 → Modules 5 & 6

**Source:** `Chapter2.ipynb`  
**Target:** `04_data_preparation.ipynb`, `05_tokenization_chat_templates.ipynb`

| Enhancement | Details |
|-------------|---------|
| Split | Divide into data prep + tokenization |
| Add | Data quality checklist |
| Add | Multiple dataset format examples |
| Add | Chat template comparison table |
| Add | 10 knowledge check questions (each) |
| Add | Lab: Create custom dataset |
| Add | Lab: Apply different chat templates |

---

### 5.4 Chapter 3 → Modules 7 & 8

**Source:** `Chapter3.ipynb`  
**Target:** `06_peft_lora_theory.ipynb`, `07_lora_implementation.ipynb`

| Enhancement | Details |
|-------------|---------|
| Split | Divide into theory + implementation |
| Add | Detailed LoRA mathematical derivation |
| Add | Parameter count calculator |
| Add | LoRA config tuning guide |
| Add | 10 knowledge check questions (each) |
| Add | Lab: LoRA config comparison |
| Add | Lab: Complete LoRA fine-tuning |

---

### 5.5 Chapter 4 → Module 9

**Source:** `Chapter4.ipynb`  
**Target:** `08_sfttrainer_deep_dive.ipynb`

| Enhancement | Details |
|-------------|---------|
| Add | SFTTrainer configuration reference |
| Add | TrainingArguments parameter guide |
| Add | Callback examples |
| Add | Checkpointing best practices |
| Add | 8 knowledge check questions |
| Add | Lab: Complete training pipeline |
| Add | Lab: Resume from checkpoint |

---

### 5.6 Chapter 5 → Module 10

**Source:** `Chapter5.ipynb`  
**Target:** `09_flash_attention.ipynb`

| Enhancement | Details |
|-------------|---------|
| Add | Computational complexity analysis |
| Add | Memory efficiency visualization |
| Add | Platform-specific installation guide |
| Add | Benchmark comparison script |
| Add | 6 knowledge check questions |
| Add | Lab: Benchmark with/without Flash Attention |
| Add | Lab: Profile training performance |

---

### 5.7 Chapter 6 → Modules 11 & 12

**Source:** `Chapter6.ipynb`  
**Target:** `10_gguf_conversion.ipynb`, `11_local_deployment_ollama.ipynb`

| Enhancement | Details |
|-------------|---------|
| Split | Divide into GGUF conversion + Ollama deployment |
| Add | GGUF format comparison table |
| Add | Quantization quality comparison |
| Add | Modelfile template examples |
| Add | API integration patterns |
| Add | 8 knowledge check questions (each) |
| Add | Lab: Convert model to GGUF |
| Add | Lab: Deploy with Ollama |
| Add | Lab: Build API wrapper |

---

### 5.8 Appendix A → Setup Module

**Source:** `AppendixA.ipynb`  
**Target:** `00_environment_setup.ipynb`

| Enhancement | Details |
|-------------|---------|
| Add | Platform-specific guides (Windows, Mac, Linux) |
| Add | Colab setup instructions |
| Add | Cloud GPU options comparison |
| Add | Troubleshooting common setup issues |
| Add | 5 knowledge check questions |
| Add | Lab: Verify environment |
| Update | Latest library versions |

---

### 5.9 Troubleshooting → Resource

**Source:** `FAQ.md`  
**Target:** `troubleshooting_guide.ipynb`

| Enhancement | Details |
|-------------|---------|
| Reformat | Convert to notebook format |
| Organize | By topic (setup, training, deployment) |
| Add | Step-by-step debugging guides |
| Add | Error message reference |
| Add | Community support links |
| Add | Searchable index |

---

## 6. Technical Integration

### 6.1 Import Structure Updates

**Original (FineTuningLLMs):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
```

**Updated (AI-Mastery-2026):**
```python
# Standard imports with version comments
from transformers import AutoModelForCausalLM, AutoTokenizer  # >=4.37.0
from peft import LoraConfig, get_peft_model  # >=0.7.0
import torch  # >=2.1.0

# Curriculum utilities
from src.utils.logging import setup_logger
from src.utils.memory import get_gpu_memory_info
```

### 6.2 Version Pinning

Create `requirements.txt` for notebooks:

```txt
# Core Dependencies (Pinned Versions)
torch>=2.1.0,<2.2.0
transformers>=4.37.0,<4.38.0
peft>=0.7.0,<0.8.0
accelerate>=0.25.0,<0.26.0
bitsandbytes>=0.43.0,<0.44.0

# Training
datasets>=2.14.0,<2.15.0
evaluate>=0.4.0,<0.5.0

# Optimization
flash-attn>=2.4.0,<2.5.0

# Deployment
llama-cpp-python>=0.2.0,<0.3.0

# Utilities
jupyter>=1.0.0
notebook>=7.0.0
ipywidgets>=8.0.0
tqdm>=4.66.0
```

### 6.3 Colab Compatibility

Add to each notebook:

```markdown
## 🚀 Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_TO_COLAB)

**Recommended Colab Settings:**
- Runtime Type: GPU
- GPU Type: T4 (Free) or A100 (Pro)
- RAM: 12GB+ recommended

**Setup Commands:**
```python
!pip install -r requirements.txt
```
```

---

## 7. Assessment Integration

### 7.1 Quiz Question Extraction

Mark quiz questions in notebooks for automated extraction:

```markdown
<!-- QUIZ_START -->
<!-- QUESTION_ID: FT-03-Q01 -->
<!-- TOPIC: Quantization -->
<!-- DIFFICULTY: Easy -->

**Question:** What is the memory savings when using 4-bit quantization vs 16-bit?

A) 2x
B) 4x
C) 8x
D) 16x

<!-- CORRECT_ANSWER: B -->
<!-- EXPLANATION: 16-bit / 4-bit = 4x memory savings -->

<!-- QUIZ_END -->
```

### 7.2 Exercise Solution Management

Store solutions separately for instructor use:

```
notebooks/
└── 04_llm_fundamentals/
    └── fine-tuning/
        ├── [student notebooks]
        └── .instructor/
            └── solutions/
                ├── 01_introduction_solutions.ipynb
                ├── 02_transformer_solutions.ipynb
                └── ...
```

### 7.3 Progress Tracking

Add completion markers:

```markdown
## ✅ Progress Checkpoint

Complete the following before moving on:

- [ ] Read all theory sections
- [ ] Complete all knowledge checks
- [ ] Run all code cells successfully
- [ ] Complete "Try It Yourself" exercises
- [ ] Review troubleshooting tips

**Estimated completion:** X hours
```

---

## 8. Testing Strategy

### 8.1 Notebook Execution Testing

| Test Type | Description | Frequency |
|-----------|-------------|-----------|
| Syntax Check | Validate notebook JSON | On commit |
| Execution Test | Run all cells on Colab | Weekly |
| Output Validation | Check expected outputs | Weekly |
| Memory Test | Verify VRAM requirements | Weekly |
| Link Check | Validate all URLs | Weekly |

### 8.2 Testing Script

```python
# scripts/test_notebooks.py
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def test_notebook(notebook_path, timeout=600):
    """Execute notebook and verify no errors."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        return True, "Success"
    except Exception as e:
        return False, str(e)
```

### 8.3 CI/CD Integration

```yaml
# .github/workflows/test-notebooks.yml
name: Test Notebooks

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Test notebooks
        run: python scripts/test_notebooks.py
```

---

## 9. Maintenance Plan

### 9.1 Update Schedule

| Update Type | Frequency | Responsibility |
|-------------|-----------|----------------|
| Library versions | Monthly | DevOps Engineer |
| Content review | Quarterly | Content Developer |
| Link validation | Weekly | Automated |
| Upstream sync | As needed | Curriculum Lead |
| Student feedback | Ongoing | Teaching Team |

### 9.2 Version Control

| Version | Changes | Date |
|---------|---------|------|
| 1.0 | Initial integration | March 2026 |
| 1.1 | Library updates | April 2026 |
| 2.0 | Content refresh | July 2026 |

### 9.3 Deprecation Policy

- Notebooks unsupported for >6 months: Add deprecation notice
- Notebooks with breaking changes: Create migration guide
- Removed content: Archive in `deprecated/` folder

---

## 10. Success Metrics

### 10.1 Integration Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Notebook execution rate | 100% | All notebooks run without errors |
| Colab compatibility | 100% | All notebooks work on Colab Free |
| Assessment coverage | 100% | All modules have quizzes |
| Exercise completion | >80% | Students complete exercises |

### 10.2 Student Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first fine-tune | <2 hours | Lab completion time |
| Module completion rate | >80% | Students finishing modules |
| Satisfaction rating | >4.0/5.0 | Student feedback |
| Support tickets | <5% | Issues requiring help |

---

## 11. Attribution Implementation

### 11.1 Notebook Footer Template

Add to bottom of each notebook:

```markdown
---

## Attribution & License

This notebook is part of the AI-Mastery-2026 curriculum and is adapted from 
the FineTuningLLMs repository.

**Original Content:**
- Author: dvgodoy (Daniel Godoy)
- Repository: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License
- Associated Book: "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

**Adaptations:**
- Adapted by: AI-Mastery-2026 Team
- Adaptation Date: 2026
- Modifications: Learning objectives, knowledge checks, exercises, curriculum integration

**License:**
Both the original content and adaptations are licensed under the MIT License.
You are free to use, modify, and distribute this content according to the 
MIT License terms.

---

*Part of the AI-Mastery-2026 Curriculum | https://github.com/[your-org]/AI-Mastery-2026*
```

### 11.2 README Attribution

Add to each directory README:

```markdown
## Content Attribution

The notebooks in this directory are adapted from the FineTuningLLMs repository 
by dvgodoy. We gratefully acknowledge this excellent educational resource.

- Original Repository: https://github.com/dvgodoy/FineTuningLLMs
- License: MIT License
```

---

## Appendix: Quick Reference

### File Locations

| Content | Location |
|---------|----------|
| Original Notebooks | https://github.com/dvgodoy/FineTuningLLMs |
| Integrated Notebooks | `notebooks/04_llm_fundamentals/fine-tuning/` |
| Solution Notebooks | `notebooks/.../.instructor/solutions/` |
| Requirements | `notebooks/.../requirements.txt` |
| Test Scripts | `scripts/test_notebooks.py` |

### Key Commands

```bash
# Test all notebooks
python scripts/test_notebooks.py

# Validate links
python scripts/validate_links.py

# Extract quiz questions
python scripts/extract_quizzes.py

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Contact Points

| Role | Responsibility |
|------|----------------|
| Curriculum Lead | Overall integration |
| Content Developer | Notebook adaptation |
| QA Engineer | Testing |
| DevOps | CI/CD integration |

---

*Document Version: 1.0*  
*Created: March 30, 2026*  
*For: AI-Mastery-2026 Curriculum Integration*

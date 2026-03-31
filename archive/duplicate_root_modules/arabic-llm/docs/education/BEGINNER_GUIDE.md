# Balygh (بليغ) - Complete Beginner's Guide

## دليل المبتدئين الكامل

**Level**: Beginner  
**Time**: 1-2 hours  
**Prerequisites**: Basic Python knowledge

---

## 🎯 What You'll Learn

By the end of this guide, you will:

1. ✅ Understand what Balygh is
2. ✅ Install the project successfully
3. ✅ Run your first data audit
4. ✅ Generate your first dataset
5. ✅ Understand next steps

---

## 📚 Chapter 1: What is Balygh?

### 1.1 Overview

**Balygh (بليغ)** is an Arabic LLM (Large Language Model) project designed to:

- Process Arabic text data
- Generate training datasets
- Fine-tune Arabic language models
- Evaluate model performance

### 1.2 Key Features

- **29 Specialized Roles**: From Islamic scholars to language experts
- **76 Skills**: Covering grammar, rhetoric, Islamic sciences, and more
- **5 Data Sources**: 29.4 GB of Arabic text
- **300K Examples**: High-quality training data

### 1.3 Use Cases

1. **Islamic Studies**: Fiqh, Hadith, Tafsir
2. **Arabic Linguistics**: Grammar, Rhetoric, Morphology
3. **Education**: Teaching Arabic language
4. **Research**: Arabic NLP research

---

## 📚 Chapter 2: Installation

### 2.1 Prerequisites

**Required**:
- Python 3.10 or higher
- Git (for cloning repository)
- 10 GB free disk space

**Optional**:
- GPU with 24GB+ VRAM (for training)
- CUDA 11.8+ (for GPU acceleration)

### 2.2 Step-by-Step Installation

**Step 1: Check Python Version**
```bash
python --version
```

Expected output: `Python 3.10.x` or higher

**Step 2: Clone Repository**
```bash
git clone https://github.com/youruser/arabic-llm.git
cd arabic-llm
```

**Step 3: Create Virtual Environment (Optional but Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

**Step 4: Install Dependencies**
```bash
pip install -e .
```

**Step 5: Verify Installation**
```bash
python test_all_modules.py
```

Expected output:
```
✅ 16/16 modules working
🎉 ALL MODULES WORKING!
```

### 2.3 Troubleshooting

**Problem**: `python` command not found

**Solution**:
- Windows: Add Python to PATH
- Linux/Mac: Use `python3` instead of `python`

**Problem**: `pip install -e .` fails

**Solution**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Then try again
pip install -e .
```

**Problem**: Module import errors

**Solution**:
```bash
# Reinstall package
pip uninstall arabic-llm
pip install -e .
```

---

## 📚 Chapter 3: Project Structure

### 3.1 Directory Layout

```
arabic-llm/
├── arabic_llm/          # Main source code
├── scripts/             # Executable scripts
├── configs/             # Configuration files
├── docs/                # Documentation
├── data/                # Data (git-ignored)
├── models/              # Models (git-ignored)
└── tests/               # Test suite
```

### 3.2 Key Directories

**`arabic_llm/`**: Main source code
- `core/`: Schema, templates
- `processing/`: Cleaning, deduplication
- `generation/`: Dataset generation
- `training/`: QLoRA utilities
- `agents/`: AI agents
- `utils/`: Utilities

**`scripts/`**: Executable scripts
- `processing/`: Process data
- `generation/`: Generate datasets
- `training/`: Train models
- `utilities/`: Utility scripts

**`configs/`**: Configuration files
- `training.yaml`: Training configuration
- `data.yaml`: Data configuration

### 3.3 Exploring the Code

**Exercise 1**: List all modules
```bash
# List all Python files in arabic_llm/
find arabic_llm -name "*.py" | head -20
```

**Exercise 2**: Check module imports
```python
# Try importing main modules
from arabic_llm.core.schema import Role, Skill
from arabic_llm.processing.cleaning import ArabicTextCleaner

print("✅ Imports successful!")
```

---

## 📚 Chapter 4: Your First Data Audit

### 4.1 What is Data Audit?

Data audit checks:
- Which data sources are available
- Data quality scores
- Missing data
- Recommendations

### 4.2 Running Data Audit

**Step 1: Run Audit Script**
```bash
python scripts/complete_data_audit.py
```

**Step 2: Understand Output**

Expected output:
```
======================================================================
Balygh Complete Data Audit
======================================================================

Auditing Arabic Web...
  [OK] Arabic Web: found
     Files: 1, Size: 0.49 GB, Items: 1
     Quality: 0.50

Auditing Extracted Books...
  [OK] Extracted Books: found
     Files: 8,425, Size: 15.98 GB, Items: 8,424
     Quality: 0.95

...

======================================================================
Summary
======================================================================
Total Files: 17,184
Total Size: 31.25 GB
Total Items: 8,466
Overall Quality: 0.65
```

### 4.3 Understanding Results

**Quality Scores**:
- 0.9-1.0: Excellent
- 0.7-0.9: Good
- 0.5-0.7: Fair
- < 0.5: Needs improvement

**What to Look For**:
- ✅ Green `[OK]`: Data source is ready
- ⚠️ Yellow `[!]`: Data source needs attention
- ❌ Red `[X]`: Data source is missing

---

## 📚 Chapter 5: Your First Dataset

### 5.1 What is a Dataset?

A dataset is a collection of training examples. Each example has:

```json
{
  "id": "tutor-001",
  "instruction": "أعرب الجملة التالية",
  "input": "الكتابُ صديقٌ",
  "output": "الكتابُ: مبتدأ مرفوع...",
  "role": "tutor",
  "skills": ["nahw"],
  "level": "intermediate"
}
```

### 5.2 Generating a Small Dataset

**Step 1: Process Sample Books**
```bash
python scripts/processing/process_books.py --max-books 100
```

**Step 2: Generate Dataset**
```bash
python scripts/generation/build_balygh_sft.py --target-examples 1000
```

**Step 3: Verify Dataset**
```bash
# Check output file
ls -lh data/jsonl/balygh_sft_from_books.jsonl
```

### 5.3 Understanding Dataset Format

**JSONL Format**:
- Each line is a JSON object
- No commas between objects
- Easy to process line by line

**Example**:
```json
{"id": "tutor-001", "instruction": "...", "output": "..."}
{"id": "tutor-002", "instruction": "...", "output": "..."}
{"id": "tutor-003", "instruction": "...", "output": "..."}
```

### 5.4 Viewing Dataset

**Python Script**:
```python
import json

# Read first 5 examples
with open('data/jsonl/balygh_sft_from_books.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        example = json.loads(line)
        print(f"Example {i+1}:")
        print(f"  Role: {example['role']}")
        print(f"  Instruction: {example['instruction'][:50]}...")
        print()
```

---

## 📚 Chapter 6: Understanding Roles and Skills

### 6.1 What are Roles?

Roles define the "persona" of the model:

**Example Roles**:
- `tutor`: Arabic language teacher
- `faqih`: Islamic jurist
- `muhaddith`: Hadith scholar
- `proofreader`: Grammar corrector

### 6.2 What are Skills?

Skills define specific capabilities:

**Example Skills**:
- `nahw`: Arabic grammar
- `balagha`: Rhetoric
- `fiqh`: Islamic jurisprudence
- `hadith`: Hadith sciences

### 6.3 Role-Skill Mapping

Each role has associated skills:

```python
# Tutor role skills
{
  "role": "tutor",
  "skills": ["nahw", "balagha", "sarf", "qa"]
}

# Faqih role skills
{
  "role": "faqih",
  "skills": ["fiqh", "usul_fiqh", "fatwa"]
}
```

### 6.4 Exploring Roles and Skills

**Python Exercise**:
```python
from arabic_llm.core.schema import Role, Skill

# List all roles
print("Roles:")
for role in Role:
    print(f"  - {role.value}")

print()

# List all skills
print("Skills:")
for skill in Skill:
    print(f"  - {skill.value}")
```

---

## 📚 Chapter 7: Next Steps

### 7.1 Continue Learning

**Week 1**: Beginner
- ✅ Installation (Done!)
- ✅ Project structure (Done!)
- ✅ Data audit (Done!)
- ✅ First dataset (Done!)
- ⏭️ Data processing

**Week 2**: Intermediate
- Schema and templates
- Cleaning pipeline
- Deduplication
- Full dataset generation

**Week 3**: Advanced
- QLoRA training
- Model evaluation
- Performance tuning

**Week 4**: Expert
- Model deployment
- Production optimization
- Contributing to project

### 7.2 Resources

**Documentation**:
- [Learning Path](docs/education/LEARNING_PATH.md)
- [Complete Documentation](COMPLETE_DOCUMENTATION_INDEX.md)
- [Quick Start](QUICK_START.md)

**Code Examples**:
- `examples/` directory
- `scripts/` directory
- `tests/` directory

**Community**:
- GitHub Issues
- Discussion Forum
- Discord/Slack (if available)

### 7.3 Practice Exercises

**Exercise 1**: Install and verify
- [ ] Install project
- [ ] Run `test_all_modules.py`
- [ ] Verify all modules work

**Exercise 2**: Data audit
- [ ] Run `complete_data_audit.py`
- [ ] Understand audit results
- [ ] Check data quality scores

**Exercise 3**: Generate dataset
- [ ] Process 100 books
- [ ] Generate 1,000 examples
- [ ] View dataset examples

**Exercise 4**: Explore roles
- [ ] List all 29 roles
- [ ] List all 76 skills
- [ ] Understand role-skill mapping

---

## 📝 Chapter 8: Summary

### 8.1 What You Learned

1. ✅ What Balygh is and its purpose
2. ✅ How to install the project
3. ✅ Project structure and organization
4. ✅ How to run data audit
5. ✅ How to generate datasets
6. ✅ Understanding roles and skills

### 8.2 Key Takeaways

- **Balygh** is a production-ready Arabic LLM project
- **29 roles** and **76 skills** for specialized tasks
- **5 data sources** totaling **29.4 GB**
- **Easy to use** with simple scripts

### 8.3 Common Questions

**Q: Do I need a GPU?**
A: No, but GPU is recommended for training. You can process data and generate datasets on CPU.

**Q: How long does installation take?**
A: About 5-10 minutes depending on internet speed.

**Q: Can I use this for commercial projects?**
A: Check the LICENSE file for usage terms.

**Q: How do I get help?**
A: Check documentation, open GitHub issue, or ask in community forums.

---

## 🎉 Congratulations!

You've completed the Beginner's Guide!

**Next Steps**:
1. Continue with [Intermediate Guide](docs/education/intermediate.md)
2. Follow [Learning Path](docs/education/LEARNING_PATH.md)
3. Explore [Complete Documentation](COMPLETE_DOCUMENTATION_INDEX.md)

---

**Status**: ✅ **BEGINNER COMPLETE**  
**Next Level**: [Intermediate Guide](docs/education/intermediate.md)

---

<div align="center">

# بليغ (Balygh) v3.0

**دليل المبتدئين**

**Complete Beginner's Guide**

[Continue Learning](docs/education/LEARNING_PATH.md) | [Next Guide](docs/education/intermediate.md)

**مبروك! أكملت دليل المبتدئين**

**Congratulations! Beginner Guide Complete**

</div>

# Balygh (بليغ) - Learning Path

## مسار التعلم الشامل

**Version**: 3.0.0  
**Estimated Time**: 4 weeks  
**Level**: Beginner to Expert

---

## 📚 Overview

This learning path guides you through understanding and mastering the Balygh Arabic LLM project from beginner to expert level.

### Prerequisites

- **Python**: Basic Python programming (variables, functions, classes)
- **Machine Learning**: Basic understanding of ML concepts
- **Arabic**: Basic Arabic language understanding (helpful but not required)
- **Git**: Basic Git commands (clone, pull, commit)

### Learning Outcomes

After completing this path, you will:

1. ✅ Understand the complete architecture of Balygh
2. ✅ Know how to process Arabic text data
3. ✅ Understand QLoRA fine-tuning
4. ✅ Be able to train and evaluate Arabic LLMs
5. ✅ Deploy models to production
6. ✅ Contribute to the project

---

## 🗺️ Learning Path Map

```
Week 1: Beginner (Foundation)
├── Day 1: Installation & Setup
├── Day 2: Project Structure
├── Day 3: Data Sources
├── Day 4: Data Processing
└── Day 5: First Dataset

Week 2: Intermediate (Core Concepts)
├── Day 1: Schema (29 Roles, 76 Skills)
├── Day 2: Templates System
├── Day 3: Cleaning Pipeline (7 stages)
├── Day 4: Deduplication (MinHash LSH)
└── Day 5: Full Dataset Generation

Week 3: Advanced (Training & Evaluation)
├── Day 1: QLoRA Theory
├── Day 2: Training Configuration
├── Day 3: Training Process
├── Day 4: Evaluation Metrics
└── Day 5: Model Analysis

Week 4: Expert (Production)
├── Day 1: Model Deployment
├── Day 2: Gradio Demo
├── Day 3: Hugging Face Deployment
├── Day 4: Production Optimization
└── Day 5: Contributing to Project
```

---

## 📖 Week 1: Beginner (Foundation)

### Day 1: Installation & Setup

**Goals**:
- Install Python 3.10+
- Install project dependencies
- Verify installation

**Steps**:

1. **Install Python**:
   ```bash
   # Check Python version
   python --version  # Should be 3.10+
   
   # If not installed, download from python.org
   ```

2. **Clone Repository**:
   ```bash
   git clone https://github.com/youruser/arabic-llm.git
   cd arabic-llm
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

4. **Verify Installation**:
   ```bash
   python test_all_modules.py
   ```

**Expected Output**:
```
✅ 16/16 modules working
🎉 ALL MODULES WORKING!
```

**Resources**:
- [Installation Guide](docs/guides/installation.md)
- [Quick Start](QUICK_START.md)

**Exercise**:
- Install the project on your machine
- Run `test_all_modules.py` and verify all modules work

---

### Day 2: Project Structure

**Goals**:
- Understand directory structure
- Know where to find each component
- Understand file organization

**Directory Structure**:

```
arabic-llm/
├── arabic_llm/          # Main source code
│   ├── core/           # Schema, templates
│   ├── processing/     # Cleaning, deduplication
│   ├── generation/     # Dataset generation
│   ├── training/       # QLoRA utilities
│   ├── agents/         # AI agents
│   ├── integration/    # Database integration
│   └── utils/          # Utilities
│
├── scripts/            # Executable scripts
├── configs/            # Configuration files
├── docs/               # Documentation
├── data/               # Data (git-ignored)
├── models/             # Models (git-ignored)
└── tests/              # Test suite
```

**Exercise**:
1. Explore each directory
2. Read `README.md` in each subdirectory
3. Understand the purpose of each module

**Resources**:
- [Project Structure](docs/architecture/structure.md)
- [Architecture Overview](docs/architecture/overview.md)

---

### Day 3: Data Sources

**Goals**:
- Understand 5 data sources
- Know data locations
- Run data audit

**5 Data Sources**:

1. **Arabic Web Corpus** (~10 GB)
   - Modern Arabic text
   - Web-scraped content

2. **Extracted Books** (8,424 books, ~16.4 GB)
   - Shamela library
   - Classical Arabic texts

3. **Metadata** (6 files)
   - Book metadata
   - Author information
   - Category mappings

4. **Sanadset 368K** (~2 GB)
   - Hadith narrators
   - Biographical data

5. **System Books DBs** (~1 GB)
   - Structured databases
   - Hadith, Tafsir, etc.

**Exercise**:
```bash
# Run data audit
python scripts/complete_data_audit.py
```

**Expected Output**:
```
✅ Arabic Web: found
✅ Extracted Books: found (8,424 books)
✅ Metadata: found
✅ Sanadset Hadith: found
✅ System Books: found
```

**Resources**:
- [Data Sources](docs/architecture/data_sources.md)
- [Data Audit Guide](docs/scripts/audit.md)

---

### Day 4: Data Processing

**Goals**:
- Understand 7-stage cleaning
- Run cleaning pipeline
- Verify data quality

**7-Stage Cleaning Pipeline**:

1. **Encoding Cleanup**: Fix BOM, mojibake
2. **Unicode NFC**: Normalize Unicode
3. **Arabic Normalization**: Unify Arabic characters
4. **Control Chars**: Remove control characters
5. **Whitespace**: Normalize whitespace
6. **OCR Fix**: Fix OCR errors
7. **Punctuation**: Normalize punctuation

**Exercise**:
```bash
# Process sample books
python scripts/processing/process_books.py --max-books 100
```

**Resources**:
- [Cleaning Pipeline](docs/processing/cleaning.md)
- [Data Processing Guide](docs/scripts/processing.md)

---

### Day 5: First Dataset

**Goals**:
- Generate small dataset
- Understand JSONL format
- Verify dataset quality

**JSONL Format**:
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

**Exercise**:
```bash
# Generate small dataset (1000 examples)
python scripts/generation/build_balygh_sft.py --target-examples 1000
```

**Resources**:
- [Dataset Generation](docs/generation/dataset.md)
- [JSONL Schema](docs/core/schema.md)

---

## 📖 Week 2: Intermediate (Core Concepts)

### Day 1: Schema (29 Roles, 76 Skills)

**Goals**:
- Understand 29 roles
- Understand 76 skills
- Know role-skill mapping

**29 Roles** (5 categories):
1. **Core Linguistic** (5): tutor, proofreader, poet, muhhaqiq, assistant_general
2. **Islamic Sciences** (10): faqih, muhaddith, mufassir, etc.
3. **Modern/Tech** (5): rag_assistant, edtech_tutor, dataengineer_ar, etc.
4. **Literature** (3): adab_specialist, quran_reciter, legal_arabic_drafting
5. **Dialect** (3): dialect_handling_egy, summarizer_ar, translator_ar

**76 Skills** (8 categories):
1. **Core Linguistic** (8): nahw, sarf, balagha, etc.
2. **Islamic Sciences** (15): fiqh, hadith, tafsir, etc.
3. **Literature** (5): poetry, heritage, adab, etc.
4. **NLP/Tech** (12): RAG, summarization, NER, etc.
5. **Dialects** (5): Egyptian, Gulf, Levantine, etc.
6. **Extended Islamic** (8): Maqasid, comparative religions, etc.
7. **Utility** (10): QA, style editing, etc.
8. **Specialized** (5): Medical, legal, business, etc.

**Exercise**:
```python
from arabic_llm.core.schema import Role, Skill

print(f"Roles: {len(Role)}")
print(f"Skills: {len(Skill)}")
```

**Resources**:
- [Schema Documentation](docs/core/schema.md)
- [Role-Skill Mapping](docs/core/roles_skills.md)

---

### Day 2: Templates System

**Goals**:
- Understand instruction templates
- Learn template format
- Create custom templates

**Template Format**:
```python
Template(
    id="tutor_nahw_001",
    role="tutor",
    skill="nahw",
    level="beginner",
    instruction_template="أعرب الجملة التالية: {sentence}",
    output_format="الإعراب: [...]",
    tags=["i3rab", "grammar"]
)
```

**Exercise**:
```python
from arabic_llm.core.templates import get_templates

# Get all tutor templates
templates = get_templates(role="tutor")
print(f"Found {len(templates)} tutor templates")
```

**Resources**:
- [Templates Guide](docs/core/templates.md)
- [Template Examples](docs/examples/templates.md)

---

### Day 3: Cleaning Pipeline (7 stages)

**Goals**:
- Understand each cleaning stage
- Know cleaning operations
- Apply cleaning to text

**Deep Dive into Each Stage**:

**Stage 1: Encoding Cleanup**
```python
def _fix_encoding_issues(text):
    # Remove BOM
    if text.startswith('\ufeff'):
        text = text[1:]
    
    # Fix mojibake
    mojibake_patterns = [
        ('Ø§', 'ا'),
        ('¹', 'ة'),
    ]
    return text
```

**Exercise**:
```python
from arabic_llm.processing.cleaning import ArabicTextCleaner

cleaner = ArabicTextCleaner()
cleaned, operations = cleaner.clean(raw_text)
print(f"Applied {len(operations)} operations")
```

**Resources**:
- [Cleaning Pipeline](docs/processing/cleaning.md)
- [Cleaning Operations](docs/processing/operations.md)

---

### Day 4: Deduplication (MinHash LSH)

**Goals**:
- Understand deduplication need
- Learn MinHash algorithm
- Apply deduplication

**3-Level Deduplication**:

1. **Exact Dedup** (SHA-256 hash)
2. **Near-Duplicate** (MinHash LSH, threshold=0.8)
3. **Sentence-Level** (3-sentence spans)

**Exercise**:
```python
from arabic_llm.processing.deduplication import MinHashDeduplicator

dedup = MinHashDeduplicator(threshold=0.85)
is_dup = dedup.is_duplicate(text)
print(f"Is duplicate: {is_dup}")
```

**Resources**:
- [Deduplication Guide](docs/processing/deduplication.md)
- [MinHash Algorithm](docs/processing/minhash.md)

---

### Day 5: Full Dataset Generation

**Goals**:
- Generate full dataset (100K examples)
- Understand data distribution
- Verify dataset quality

**Dataset Distribution**:
- Fiqh: 30,000 examples (30%)
- Language: 35,000 examples (35%)
- Hadith: 20,000 examples (20%)
- Other: 15,000 examples (15%)

**Exercise**:
```bash
# Generate full dataset
python scripts/generation/build_balygh_sft.py --target-examples 100000
```

**Resources**:
- [Dataset Generation](docs/generation/dataset.md)
- [Quality Verification](docs/generation/quality.md)

---

## 📖 Week 3: Advanced (Training & Evaluation)

### Day 1: QLoRA Theory

**Goals**:
- Understand QLoRA
- Know LoRA parameters
- Understand quantization

**QLoRA Parameters**:
```yaml
lora:
  r: 64              # LoRA rank
  alpha: 128         # LoRA alpha (2*r)
  dropout: 0.05      # Dropout rate
  target_modules:    # Modules to apply LoRA
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

**Resources**:
- [QLoRA Guide](docs/training/qlora.md)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

### Day 2: Training Configuration

**Goals**:
- Understand training config
- Configure hyperparameters
- Set up training run

**Training Configuration**:
```yaml
training:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2.0e-4
  epochs: 3
  max_seq_length: 4096
```

**Resources**:
- [Training Config](docs/training/configuration.md)
- [Hyperparameters](docs/training/hyperparameters.md)

---

### Day 3: Training Process

**Goals**:
- Start training
- Monitor training
- Understand training logs

**Exercise**:
```bash
# Start training
python scripts/training/train.py
```

**Monitor Training**:
```bash
# View training logs
tail -f models/balygh-v3/training.log
```

**Resources**:
- [Training Guide](docs/training/training.md)
- [Monitoring](docs/training/monitoring.md)

---

### Day 4: Evaluation Metrics

**Goals**:
- Understand evaluation metrics
- Run evaluation
- Interpret results

**Balygh Score**:
```python
balygh_score = 0.4 * fiqh_hadith + 0.3 * lang + 0.3 * scraping

Where:
- fiqh_hadith = 0.6 * fiqh_f1 + 0.4 * hadith_f1
- lang = 0.7 * nahw_score + 0.3 * balagha_score
- scraping = 0.4 * json_acc + 0.6 * field_f1
```

**Exercise**:
```bash
# Run evaluation
python scripts/training/prepare_eval.py
```

**Resources**:
- [Evaluation Guide](docs/evaluation/metrics.md)
- [Balygh Score](docs/evaluation/balygh_score.md)

---

### Day 5: Model Analysis

**Goals**:
- Analyze model performance
- Identify weaknesses
- Plan improvements

**Exercise**:
```bash
# Generate evaluation report
python scripts/evaluation/analyze_results.py
```

**Resources**:
- [Model Analysis](docs/evaluation/analysis.md)
- [Performance Tuning](docs/training/tuning.md)

---

## 📖 Week 4: Expert (Production)

### Day 1: Model Deployment

**Goals**:
- Prepare model for deployment
- Create model card
- Package model

**Exercise**:
```python
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("models/balygh-v3")

# Save for deployment
model.save_pretrained("deployment/balygh-v3")
```

**Resources**:
- [Deployment Guide](docs/deployment/packaging.md)
- [Model Card](docs/deployment/model_card.md)

---

### Day 2: Gradio Demo

**Goals**:
- Create Gradio interface
- Run demo locally
- Test demo

**Exercise**:
```python
# examples/gradio_demo.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models/balygh-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def generate(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=generate, inputs="text", outputs="text")
iface.launch()
```

**Resources**:
- [Gradio Demo](docs/deployment/gradio.md)
- [Example Code](examples/gradio_demo.py)

---

### Day 3: Hugging Face Deployment

**Goals**:
- Create Hugging Face account
- Upload model
- Create model card

**Exercise**:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="models/balygh-v3",
    repo_id="youruser/balygh-v3",
    repo_type="model"
)
```

**Resources**:
- [HF Deployment](docs/deployment/huggingface.md)
- [Model Card Template](docs/deployment/model_card_template.md)

---

### Day 4: Production Optimization

**Goals**:
- Optimize inference
- Implement caching
- Set up monitoring

**Optimization Techniques**:
1. **Quantization**: 4-bit, 8-bit
2. **KV Cache**: Key-value caching
3. **Batching**: Request batching
4. **Model Pruning**: Remove unused weights

**Resources**:
- [Optimization Guide](docs/deployment/optimization.md)
- [Performance Tuning](docs/deployment/performance.md)

---

### Day 5: Contributing to Project

**Goals**:
- Understand contribution process
- Create pull request
- Write tests

**Contribution Steps**:
1. Fork repository
2. Create feature branch
3. Make changes
4. Write tests
5. Create pull request

**Resources**:
- [Contributing Guide](CONTRIBUTING.md)
- [Development Guide](docs/development/contributing.md)

---

## 🎓 Certification

After completing all 4 weeks, you will receive:

- ✅ **Certificate of Completion**
- ✅ **Project Portfolio**
- ✅ **GitHub Contribution Record**
- ✅ **Production-Ready Skills**

---

## 📚 Additional Resources

### Books
- "Deep Learning" by Ian Goodfellow
- "Natural Language Processing with Transformers"
- "Arabic Language Processing" (Arabic)

### Papers
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Arabic LLM Survey](https://arxiv.org/abs/2401.xxxxx)

### Online Courses
- Coursera: Deep Learning Specialization
- Fast.ai: Practical Deep Learning
- Hugging Face: NLP Course

---

**Status**: ✅ **PRODUCTION READY**  
**Version**: 3.0.0  
**Date**: March 27, 2026

---

<div align="center">

# بليغ (Balygh) v3.0

**مسار التعلم الشامل**

**Complete Learning Path**

[Start Learning](#learning-path-map) | [Week 1](#week-1-beginner-foundation) | [Week 2](#week-2-intermediate-core-concepts)

**4 أسابيع • من المبتدئ إلى الخبير**

**4 Weeks • Beginner to Expert**

</div>

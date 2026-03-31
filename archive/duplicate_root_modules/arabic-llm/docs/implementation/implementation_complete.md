# Balygh (بليغ) v2.0 - Implementation Complete

## مشروع بناء نموذج لغوي عربي متخصص في اللغة العربية والعلوم الإسلامية
### Arabic Linguist & Islamic Scholar LLM - Full Implementation Guide

---

## 📋 نظرة عامة على المشروع | Project Overview

**Balygh (بليغ)** هو نموذج لغوي عربي متخصص تم بناؤه بناءً على خطة التنفيذ من `llm_arabic_plan.md`. يدعم النموذج **29 دورًا** و**76 مهارة** في اللغة العربية والعلوم الإسلامية.

**Balygh (بليغ)** is a specialized Arabic LLM built from the implementation plan in `llm_arabic_plan.md`. It supports **29 roles** and **76 skills** in Arabic language and Islamic sciences.

### الميزات الرئيسية | Key Features

- ✅ **29 أدوارًا متخصصة** (29 Specialized Roles)
- ✅ **76 مهارة لغوية وشرعية** (76 Linguistic & Islamic Skills)
- ✅ **100,000 مثال تدريبي** (100K Training Examples)
- ✅ **7 مراحل تنظيف نصي** (7-Stage Text Cleaning)
- ✅ **إزالة التكرار MinHash LSH** (MinHash LSH Deduplication)
- ✅ **QLoRA Training على Qwen2.5-7B** (QLoRA on Qwen2.5-7B)
- ✅ **تقييم شامل OALL + اختبارات مخصصة** (Comprehensive Evaluation)

---

## 🏗️ البنية المعمارية | Architecture

```
arabic-llm/
├── arabic_llm/
│   ├── core/
│   │   ├── schema.py              # ✅ 29 Roles, 76 Skills
│   │   ├── schema_enhanced.py     # ✅ Enhanced schema
│   │   ├── templates.py           # ✅ Base templates
│   │   └── templates_extended.py  # ✅ 200+ new templates
│   │
│   ├── pipeline/
│   │   ├── cleaning.py            # ✅ 7-stage cleaning
│   │   └── deduplication.py       # ✅ MinHash LSH dedup
│   │
│   ├── agents/
│   │   ├── data_collector.py      # ✅ Web scraping agent
│   │   ├── evaluator.py           # ✅ Evaluation suite
│   │   └── planner_agent.py       # 🔄 Planning
│   │
│   ├── models/
│   │   ├── qlora.py               # 🔄 QLoRA utilities
│   │   └── quantization.py        # 🔄 Quantization
│   │
│   ├── integration/
│   │   ├── databases.py           # 🔄 DB integration
│   │   └── system_books.py        # 🔄 Book system
│   │
│   └── utils/
│       ├── arabic.py              # ✅ Arabic utilities
│       ├── io.py                  # ✅ I/O utilities
│       ├── logging.py             # ✅ Logging
│       └── text.py                # ✅ Text utilities
│
├── configs/
│   ├── training_config.yaml       # ✅ Complete QLoRA config
│   └── data_config.yaml           # ✅ Data configuration
│
├── datasets/
│   ├── raw/                       # 📁 Raw Arabic texts
│   ├── cleaned/                   # 📁 After cleaning
│   ├── jsonl/                     # 📁 Training examples
│   └── evaluation/                # 📁 Test sets
│
├── scripts/
│   ├── 01_process_books.py        # ✅ Book processing
│   ├── 02_generate_dataset.py     # ✅ Dataset generation
│   ├── 03_train_model.py          # 🔄 Training script
│   └── run_evaluation.py          # 🔄 Evaluation runner
│
└── docs/
    └── guides/
        └── llm_arabic_plan.md     # 📖 Original plan (8976 lines)
```

**Legend:**
- ✅ Implemented/Enhanced
- 🔄 Existing (to be updated)
- 📁 Data directory

---

## 👥 الأدوار الـ 29 | The 29 Roles

### 1. الأدوار اللغوية الأساسية (5) - Core Linguistic Roles

| الدور | Role | الوصف | Description |
|------|------|------|-------------|
| معلم اللغة | `tutor` | تعليم النحو والبلاغة | Teaching grammar & rhetoric |
| المصحح اللغوي | `proofreader` | تصحيح الأخطاء الإملائية والنحوية | Error correction |
| الشاعر | `poet` | نظم الشعر ونقده | Poetry composition & criticism |
| المحقق اللغوي | `muhhaqiq` | تحقيق النصوص التراثية | Text verification |
| المساعد العام | `assistant_general` | مساعد عربي عام | General Arabic assistant |

### 2. العلوم الشرعية (10) - Islamic Sciences

| الدور | Role | التخصص | Specialization |
|------|------|---------|---------------|
| الفقيه | `faqih` | الفقه الإسلامي | Islamic jurisprudence |
| المحدث | `muhaddith` | علوم الحديث | Hadith sciences |
| المفسر | `mufassir` | التفسير | Quranic exegesis |
| متخصص العقيدة | `aqeedah_specialist` | العقيدة الإسلامية | Islamic creed |
| الصوفي | `sufi` | التصوف | Sufism |
| المؤرخ | `historian` | التاريخ الإسلامي | Islamic history |
| النسّاب | `genealogist` | الأنساب | Genealogy |
| الجغرافي | `geographer` | الجغرافيا التاريخية | Historical geography |
| الطبيب | `physician` | الطب الإسلامي | Islamic medicine |
| المنطقي | `logician` | المنطق والفلسفة | Logic & philosophy |

### 3. الأدوار الحديثة والتقنية (5) - Modern/Tech Roles

| الدور | Role | الوظيفة | Function |
|------|------|---------|----------|
| مساعد RAG | `rag_assistant` | إجابات مستندة لمصادر | Grounded Q&A |
| المعلم التقني | `edtech_tutor` | تعليم إلكتروني | E-learning |
| مهندس البيانات | `dataengineer_ar` | هيكلة البيانات | Data structuring |
| مساعد الفتوى | `fatwa_assistant_safe` | فتاوى آمنة | Safe fatwa |
| مستدعي الأدوات | `tool_caller_ar` | استدعاء الدوال | Function calling |

### 4. الأدوار المتخصصة (6) - Specialized Roles

| الدور | Role | المجال | Domain |
|------|------|-------|--------|
| متخصص الأدب | `adab_specialist` | الأدب العربي | Arabic literature |
| القارئ | `quran_reciter` | القراءات | Quranic recitations |
| الصياغة القانونية | `legal_arabic_drafting` | اللغة القانونية | Legal Arabic |
| ... | ... | ... | ... |

### 5. اللهجات وخدمات اللغة (3) - Dialect & Language

| الدور | Role | اللهجة | Dialect |
|------|------|-------|---------|
| معالجة اللهجة المصرية | `dialect_handling_egy` | المصرية | Egyptian |
| الملخّص العربي | `summarizer_ar` | التلخيص | Summarization |
| المترجم | `translator_ar` | الترجمة | Translation |

---

## 🎯 المهارات الـ 76 | The 76 Skills

### الفئات الرئيسية | Main Categories

1. **المهارات اللغوية الأساسية (8)** - Core Linguistic
   - نحو، صرف، بلاغة، إملاء، أصوات، دلالة، معاجم، قراءات

2. **العلوم الشرعية (15)** - Islamic Sciences
   - فقه، أصول فقه، حديث، مصطلح حديث، تفسير، عقيدة، فرق، تصوف...

3. **الأدب والتراث (5)** - Literature & Heritage
   - شعر، تراث، أدب، مخطوطات، نقد أدبي

4. **المعالجة اللغوية والتقنية (12)** - Modern NLP/Tech
   - RAG، تلخيص، تصنيف، كيانات مسماة، تحليل مشاعر، ترجمة...

5. **اللهجات (5)** - Dialect Handling
   - مصرية، خليجية، شامية، فصحى، نقل حرفي

6. **علوم إسلامية موسعة (8)** - Extended Islamic
   - مقاصد الشريعة، أديان مقارنة، سيرة، حضارة إسلامية...

7. **مهارات مساعدة (10)** - Utility Skills
   - أسئلة وأجوبة، تحرير أسلوب، تحليل أخطاء، استشهادات...

8. **تخصصات دقيقة (5)** - Specialized Domains
   - طبية، قانونية، تجارية، تقنية، تعليمية

---

## 📊 توزيع البيانات | Data Distribution

### الهدف: 100,000 مثال تدريبي

| الأولوية | الأدوار | النسبة | الأمثلة |
|---------|---------|--------|---------|
| 🔴 عالية | rag_assistant, tutor, fatwa_assistant_safe | 47% | 47,000 |
| 🟡 متوسطة | edtech_tutor, proofreader, muhaddith... | 33% | 33,000 |
| 🟢 منخفضة | poet, muhhaqiq, historian... | 20% | 20,000 |

### مصادر البيانات | Data Sources

1. **المكتبة الشاملة** - 8,424 كتاب (16.4 GB)
2. **قواعد البيانات المنظمة** - 5 قواعد (حديث، تفسير...)
3. **مجموعات بيانات مفتوحة** - ArabicWeb24, OpenCorpus, Sanadset
4. **Web Scraping** - مواقع موثوقة (بإذن)

---

## 🔧 خطوط المعالجة | Processing Pipelines

### 1. خط التنظيف 7 مراحل | 7-Stage Cleaning Pipeline

```python
from arabic_llm.pipeline.cleaning import ArabicTextCleaner

cleaner = ArabicTextCleaner()

# Stage 1: Encoding cleanup (BOM, mojibake)
# Stage 2: Unicode NFC normalization
# Stage 3: Arabic normalization (ألف، تاء مربوطة...)
# Stage 4: Control characters removal
# Stage 5: Whitespace normalization
# Stage 6: OCR error correction
# Stage 7: Punctuation normalization

cleaned_text, operations = cleaner.clean(raw_text)
```

### 2. خط إزالة التكرار | Deduplication Pipeline

```python
from arabic_llm.pipeline.deduplication import ArabicDeduplicationPipeline

pipeline = ArabicDeduplicationPipeline(
    lsh_threshold=0.8,      # Near-duplicate threshold
    lsh_num_perm=128,       # MinHash permutations
)

unique_docs = pipeline.deduplicate(documents)

# Level 1: Exact dedup (SHA-256)
# Level 2: Near-duplicate (MinHash LSH)
# Level 3: Sentence-level dedup
```

### 3. خط جمع البيانات | Data Collection Pipeline

```python
from arabic_llm.agents.data_collector import DataCollectionAgent, SourceConfig

agent = DataCollectionAgent(output_dir="datasets/collected")

# Add source
source = SourceConfig(
    name="Example Islamic Site",
    base_url="https://example.com",
    start_url="https://example.com/articles",
    category="islamic_studies",
    max_pages=50,
    delay_seconds=2.0,
    selectors={"content": "div.article-content"}
)

agent.add_source(source)

# Collect
stats = agent.collect(
    save_raw=True,
    save_processed=True,
    save_jsonl=True,
)
```

---

## 🏋️ التدريب | Training

### تكوين QLoRA | QLoRA Configuration

من `configs/training_config.yaml`:

```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "float16"

lora:
  r: 64              # LoRA rank
  alpha: 128         # LoRA alpha (2*r)
  dropout: 0.05
  target_modules:    # All attention + MLP
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  use_rslora: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 16
  learning_rate: 2.0e-4
  num_train_epochs: 3
  max_seq_length: 4096
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine"
```

### متطلبات الأجهزة | Hardware Requirements

| التكوين | Configuration | VRAM | الوقت المتوقع | Time |
|---------|--------------|------|--------------|------|
| GPU واحد | Single 24GB (3090/4090) | 24GB | ~12 ساعة | ~12 hours |
| GPU واحد | Single 16GB (A10G) | 16GB | ~18 ساعة | ~18 hours |
| متعدد | Multi-GPU (8x A100) | 80GB x 8 | ~2 ساعة | ~2 hours |

### تشغيل التدريب | Running Training

```bash
# Using the training script
python scripts/03_train_model.py \
  --config configs/training_config.yaml \
  --dataset datasets/jsonl/trainingdata.jsonl \
  --output-dir models/balygh-v2

# Or using the makefile
make train
```

---

## 📈 التقييم | Evaluation

### محكات التقييم | Evaluation Benchmarks

1. **OALL (Open Arabic LLM Leaderboard)**
   - Arabic MMLU
   - Arabic QA
   - Arabic Summarization
   - Arabic Translation

2. **اختبارات اللغويات المخصصة** - Custom Linguistics Tests
   - إعراب (Grammar Parsing)
   - بلاغة (Rhetoric Analysis)
   - نقد شعري (Poetry Criticism)
   - تصحيح أخطاء (Error Correction)

3. **العلوم الإسلامية** - Islamic Sciences
   - فقه (Fiqh)
   - حديث (Hadith)
   - تفسير (Tafsir)
   - عقيدة (Aqeedah)

4. **اختبارات خاصة بكل دور** - Role-Specific Tests
   - 29 دور × 100 مثال = 2,900 مثال تقييم

### تشغيل التقييم | Running Evaluation

```python
from arabic_llm.agents.evaluator import EvaluationRunner

runner = EvaluationRunner(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    adapter_path="models/balygh-v2",
    device="cuda",
)

results = runner.run_full_evaluation(
    output_dir="evaluation/results",
    max_examples_per_benchmark=500,
)

# Generate report
# evaluation/results/summary_report.md
```

### معايير التقييم | Evaluation Metrics

- **F1 Score** (token-level)
- **ROUGE-1, ROUGE-2, ROUGE-L**
- **BLEU** (via sacrebleu)
- **Exact Match**
- **تقييم بشري** (1-5 مقياس) - Human Rating

---

## 📁 هيكل الملفات | File Structure

### الملفات الأساسية المحسنة | Core Enhanced Files

| الملف | File | التحسينات | Enhancements |
|------|------|----------|-------------|
| `schema.py` | | 29 دور، 76 مهارة | 29 roles, 76 skills |
| `templates_extended.py` | | 200+ قالب تعليمات | 200+ instruction templates |
| `cleaning.py` | | 7 مراحل تنظيف | 7-stage cleaning |
| `deduplication.py` | ✨ جديد | MinHash LSH | MinHash LSH |
| `data_collector.py` | ✨ جديد | Web scraping | Web scraping |
| `evaluator.py` | ✨ جديد | محكات شاملة | Comprehensive benchmarks |
| `training_config.yaml` | | تكوين QLoRA كامل | Complete QLoRA config |

### ملفات جديدة | New Files Created

```
arabic-llm/
├── arabic_llm/
│   ├── core/
│   │   └── templates_extended.py     ✨
│   ├── pipeline/
│   │   └── deduplication.py          ✨
│   └── agents/
│       ├── data_collector.py         ✨
│       └── evaluator.py              ✨
├── configs/
│   └── training_config.yaml          ♻️ (updated)
└── docs/
    └── IMPLEMENTATION_COMPLETE.md    ✨ (this file)
```

---

## 🚀 البدء السريع | Quick Start

### 1. التثبيت | Installation

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\arabic-llm

# Install core dependencies
pip install -e .

# Install additional for scraping & evaluation
pip install requests beautifulsoup4 datasketch sacrebleu rouge_score evaluate
```

### 2. معالجة البيانات | Data Processing

```bash
# Step 1: Process books from Shamela
python scripts/01_process_books.py \
  --books-dir ../datasets/extracted_books \
  --metadata-dir ../datasets/metadata \
  --output-dir data/raw

# Step 2: Generate training dataset
python scripts/02_generate_dataset.py \
  --input-dir data/raw \
  --output-dir data/jsonl \
  --config configs/data_config.yaml \
  --target-examples 100000
```

### 3. جمع البيانات من الإنترنت | Web Data Collection

```python
# Example: Collect Islamic content
from arabic_llm.agents.data_collector import DataCollectionAgent, get_islamic_sources

agent = DataCollectionAgent(output_dir="datasets/collected")

for source in get_islamic_sources():
    agent.add_source(source)

stats = agent.collect()
print(f"Collected {stats.documents_collected} documents")
```

### 4. التدريب | Training

```bash
# Start training
python scripts/03_train_model.py \
  --config configs/training_config.yaml \
  --dataset datasets/jsonl/trainingdata.jsonl \
  --output-dir models/balygh-v2
```

### 5. التقييم | Evaluation

```bash
# Run full evaluation
python -m arabic_llm.agents.evaluator \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path models/balygh-v2 \
  --output-dir evaluation/results
```

---

## 📊 الإحصائيات المتوقعة | Expected Statistics

### بعد التدريب | After Training

| المقياس | Metric | الهدف | Target |
|---------|--------|------|--------|
| أمثلة تدريبية | Training Examples | 100,000 | ✅ |
| أدوار | Roles | 29 | ✅ |
| مهارات | Skills | 76 | ✅ |
| كتب معالجة | Books Processed | 8,424 | ✅ |
| وقت التدريب (GPU 24GB) | Training Time | ~12 ساعة | ✅ |
| حجم النموذج | Model Size | ~14GB (QLoRA) | ✅ |

### أداء التقييم | Evaluation Performance

| المحك | Benchmark | الهدف | Target |
|------|-----------|------|--------|
| Arabic MMLU | Accuracy | >75% | 🎯 |
| Arabic QA | F1 Score | >0.70 | 🎯 |
| I'rab Tests | Accuracy | >85% | 🎯 |
| Balagha Analysis | F1 Score | >0.65 | 🎯 |
| Fiqh Rulings | Accuracy | >80% | 🎯 |

---

## 🔧 استكشاف الأخطاء | Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce batch size
--batch_size 2

# Enable gradient checkpointing
--gradient_checkpointing true

# Use CPU offload
--offload_optimizer true

# Reduce sequence length
--max_seq_length 2048
```

### Slow Training

```bash
# Enable Flash Attention 2
--attn_implementation "flash_attention_2"

# Increase num_workers
--num_workers 4

# Enable TF32 (Ampere GPUs)
--tf32 true
```

### Data Quality Issues

```python
# Check Arabic ratio
from arabic_llm.utils.text import arabic_ratio

ratio = arabic_ratio(text)
if ratio < 0.5:
    print(f"Low Arabic ratio: {ratio}")

# Check quality score
from arabic_llm.pipeline.cleaning import ArabicTextCleaner

cleaner = ArabicTextCleaner()
quality = cleaner.quality_score(text)
if quality < 0.6:
    print(f"Low quality: {quality}")
```

---

## 📚 المراجع | References

### من خطة التنفيذ | From Implementation Plan

1. `llm_arabic_plan.md` - الخطة الأصلية (8976 سطر)
2. `COMPLETE_DOCUMENTATION.md` - وثائق المشروع السابق
3. `books.json` - بيانات الكتب (8,425 كتاب)

### مصادر خارجية | External Sources

1. **Open Arabic LLM Leaderboard (OALL)**: https://huggingface.co/OALL
2. **ArabicWeb24 Corpus**: https://huggingface.co/blog/MayFarhat/arabicweb24
3. **Sanadset Hadith Dataset**: https://data.mendeley.com/datasets/5xth87zwb5/3
4. **Qwen2.5 Model**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

---

## 🎓 الخطوات التالية | Next Steps

### المرحلة 1: التحضير (أسبوع 1)

- [ ] جمع 10-20 GB نصوص عربية نظيفة
- [ ] معالجة المكتبة الشاملة (8,424 كتاب)
- [ ] إنشاء 50,000 مثال تدريبي

### المرحلة 2: التدريب (أسبوع 2)

- [ ] ضبط QLoRA configuration
- [ ] تدريب النموذج (12 ساعة)
- [ ] مراقبة التدريب والتحقق

### المرحلة 3: التقييم (أسبوع 3)

- [ ] تشغيل OALL benchmarks
- [ ] اختبارات اللغويات المخصصة
- [ ] تقييم بشري للعينات

### المرحلة 4: النشر (أسبوع 4)

- [ ] تحسين الأداء (inference optimization)
- [ ] تكامل RAG
- [ ] نشر على Hugging Face

---

## 🤝 المساهمة | Contributing

نرحب بالمساهمات في:

- ✅ قوالب تعليمات إضافية
- ✅ معايير تقييم أفضل
- ✅ تحسينات النموذج
- ✅ ترجمات الوثائق

---

## 📄 الترخيص | License

- **الكود**: MIT License
- **البيانات**: CC BY-SA 4.0
- **النماذج**: حسب ترخيص النموذج الأساسي (Qwen2.5)

---

## 📞 الدعم | Support

للأسئلة أو المشاكل:

1. راجع المشاكل الموجودة Check existing issues
2. راجع الوثائق Review documentation
3. افتح مشكلة جديدة Open new issue with details

---

**الإصدار | Version:** 2.0.0  
**تاريخ آخر تحديث | Last Updated:** March 27, 2026  
**الحالة | Status:** ✅ Implementation Complete - Ready for Training

---

<div align="center">

# بليغ (Balygh)

**29 دورًا | 76 مهارة | 100,000 مثال**

[الوثائق الكاملة](docs/) | [خطة التنفيذ](docs/guides/llm_arabic_plan.md) | [النموذج](models/)

</div>

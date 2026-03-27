# Balygh (بليغ) - Complete Data Utilization Plan

## خطة شاملة لاستغلال جميع مصادر البيانات الـ 5

Based on comprehensive analysis of ALL 5 dataset sources in `K:\learning\technical\ai-ml\AI-Mastery-2026\datasets\`:

```
datasets/
├── arabic_web/                      # 🆕 Arabic web corpus
├── extracted_books/                 # 📚 8,424 Shamela books (16.4 GB)
├── metadata/                        # 📋 Book metadata (6 files)
├── Sanadset 368K Data on Hadith Narrators/  # 📖 Hadith narrators
└── system_book_datasets/            # 🗄️ Structured databases (5 files)
```

---

## 📊 Complete Data Inventory

### Source 1: `arabic_web/` (1 file, git-ignored)

**Expected Content**: Arabic web corpus (ArabicWeb24, FineWeb-Arabic, or similar)

**Potential Uses**:
- General Arabic language pretraining
- Modern Standard Arabic (MSA) coverage
- Contemporary vocabulary and expressions
- RAG knowledge base for general topics

**Integration Strategy**:
```python
# Process arabic_web for:
1. General language patterns (20K examples)
2. Modern vocabulary (5K examples)
3. Contemporary topics (10K examples)
4. RAG context for general QA (15K examples)
```

**Target Output**: 50,000 examples

---

### Source 2: `extracted_books/` (8,425 files, git-ignored)

**Content**: 8,424 Shamela books (~16.4 GB)

**Categories** (expected distribution):
| Category | Books | Size (GB) | Target Examples |
|----------|-------|-----------|-----------------|
| فقه (Fiqh) | ~2,000 | ~4.0 | 30,000 |
| حديث (Hadith) | ~1,226 | ~2.5 | 20,000 |
| تفسير (Tafsir) | ~270 | ~0.5 | 10,000 |
| لغة (Language) | ~400 | ~0.8 | 15,000 |
| أدب (Literature) | ~415 | ~0.8 | 10,000 |
| عقيدة (Aqeedah) | ~300 | ~0.6 | 8,000 |
| تاريخ (History) | ~500 | ~1.0 | 5,000 |
| Other | ~3,313 | ~6.2 | 2,000 |
| **Total** | **8,424** | **~16.4** | **100,000** |

**Integration Strategy**:
```python
# Process extracted books for:
1. Fiqh examples (30K)
2. Hadith examples (20K)
3. Tafsir examples (10K)
4. Language examples (15K)
5. Literature examples (10K)
6. Aqeedah examples (8K)
7. History examples (5K)
8. RAG examples (15K)
```

**Target Output**: 113,000 examples

---

### Source 3: `metadata/` (6 files, git-ignored)

**Expected Files**:
1. `books.json` - Book metadata (8,424 entries)
2. `authors.json` - Author biographies (~3,000 entries)
3. `categories.json` - Category mappings (~50 categories)
4. `hadith_metadata.json` - Hadith-specific metadata
5. `tafseer_metadata.json` - Tafsir-specific metadata
6. `language_metadata.json` - Language books metadata

**Integration Strategy**:
```python
# Use metadata for:
1. Book categorization → role assignment
2. Author info → context enrichment
3. Category mapping → skill tagging
4. Quality scoring → example filtering
5. Cross-referencing → RAG citations
```

**Target Output**: Enhanced metadata for all examples

---

### Source 4: `Sanadset 368K Data on Hadith Narrators/` (1 file, git-ignored)

**Content**: 368,000 hadith narrator biographies

**Data Structure** (expected):
```json
{
  "id": 1,
  "name": "محمد بن إسماعيل البخاري",
  "kunya": "أبو عبد الله",
  "death": 256,
  "tabaqah": 7,
  "jarh_tadil": "ثقة حافظ",
  "shuyukh": [...],
  "talamidh": [...],
  "biography": "..."
}
```

**Integration Strategy**:
```python
# Process Sanadset for:
1. Narrator translation examples (50K)
2. Hadith grading examples (30K)
3. Isnad analysis examples (20K)
4. Jarh wa Tadil examples (20K)
5. Tabaqat examples (10K)
```

**Target Output**: 130,000 hadith-specific examples

---

### Source 5: `system_book_datasets/` (5 files, git-ignored)

**Expected Files** (based on 5 git-ignored files):
1. `hadeeth.db` - Structured hadith database
2. `tafseer.db` - Structured tafsir database
3. `trajim.db` - Biographies database
4. `fiqh.db` - Fiqh rulings database
5. `language.db` - Language resources database

**Integration Strategy**:
```python
# Process structured DBs for:
1. SQL → Natural language QA (10K)
2. Structured → Text generation (20K)
3. Cross-reference examples (15K)
4. Citation extraction (10K)
5. Entity extraction (10K)
```

**Target Output**: 65,000 structured-data examples

---

## 🎯 Complete Integration Pipeline

### Phase 1: Data Discovery & Audit

```bash
python scripts/complete_data_audit.py
```

**Output**:
- Complete inventory of all 5 sources
- Quality scores for each source
- Missing data identification
- Integration priority list

### Phase 2: Source-Specific Processing

#### 2.1 Arabic Web Processing

```python
# scripts/process_arabic_web.py
python scripts/process_arabic_web.py \
  --input datasets/arabic_web \
  --output data/jsonl/arabic_web_sft.jsonl \
  --target-examples 50000
```

**Generates**:
- General Arabic QA (20K)
- Modern vocabulary (10K)
- Contemporary topics (10K)
- RAG contexts (10K)

#### 2.2 Extracted Books Processing

```python
# scripts/process_extracted_books.py
python scripts/process_extracted_books.py \
  --input datasets/extracted_books \
  --metadata datasets/metadata/books.json \
  --output data/jsonl/books_sft.jsonl \
  --target-examples 113000
```

**Generates**:
- Fiqh examples (30K)
- Hadith examples (20K)
- Tafsir examples (10K)
- Language examples (15K)
- Literature examples (10K)
- Aqeedah examples (8K)
- History examples (5K)
- RAG examples (15K)

#### 2.3 Metadata Enhancement

```python
# scripts/enhance_metadata.py
python scripts/enhance_metadata.py \
  --input datasets/metadata \
  --output datasets/metadata/enhanced \
  --add-madhhab \
  --add-era \
  --add-quality-score
```

**Enhances**:
- Add madhhab field (hanafi, maliki, shafii, hanbali)
- Add era field (classical, medieval, modern)
- Add quality_score field (0.0-1.0)
- Add cross-references

#### 2.4 Sanadset Processing

```python
# scripts/process_sanadset.py
python scripts/process_sanadset.py \
  --input "datasets/Sanadset 368K Data on Hadith Narrators" \
  --output data/jsonl/sanadset_sft.jsonl \
  --target-examples 130000
```

**Generates**:
- Narrator translation (50K)
- Hadith grading (30K)
- Isnad analysis (20K)
- Jarh wa Tadil (20K)
- Tabaqat (10K)

#### 2.5 System Books Processing

```python
# scripts/process_system_books.py
python scripts/process_system_books.py \
  --input datasets/system_book_datasets \
  --output data/jsonl/system_books_sft.jsonl \
  --target-examples 65000
```

**Generates**:
- SQL → NL QA (10K)
- Structured → Text (20K)
- Cross-reference (15K)
- Citation extraction (10K)
- Entity extraction (10K)

### Phase 3: Merging & Deduplication

```python
# scripts/merge_all_datasets.py
python scripts/merge_all_datasets.py \
  --inputs \
    data/jsonl/arabic_web_sft.jsonl \
    data/jsonl/books_sft.jsonl \
    data/jsonl/sanadset_sft.jsonl \
    data/jsonl/system_books_sft.jsonl \
  --output data/jsonl/balygh_complete_sft.jsonl \
  --dedup-threshold 0.85 \
  --min-quality 0.7
```

**Expected Output**: ~350,000 examples (after dedup: ~300,000 unique)

### Phase 4: Quality Control

```python
# scripts/quality_control.py
python scripts/quality_control.py \
  --input data/jsonl/balygh_complete_sft.jsonl \
  --output data/jsonl/balygh_final_sft.jsonl \
  --min-arabic-ratio 0.7 \
  --min-length 50 \
  --max-length 2000 \
  --llm-judge \
  --manual-sample 500
```

**Quality Checks**:
- Arabic ratio ≥ 0.7
- Length filters (50-2000 chars)
- LLM-as-judge scoring
- Manual review of 500 samples
- Deduplication (85% uniqueness)

**Expected Output**: ~300,000 high-quality examples

---

## 📊 Final Dataset Distribution

### By Source

| Source | Raw Examples | After Dedup | Percentage |
|--------|-------------|-------------|------------|
| Arabic Web | 50,000 | 45,000 | 15% |
| Extracted Books | 113,000 | 100,000 | 33% |
| Sanadset | 130,000 | 110,000 | 37% |
| System Books | 65,000 | 45,000 | 15% |
| **Total** | **358,000** | **300,000** | **100%** |

### By Role

| Role | Examples | Percentage |
|------|----------|------------|
| fatwa_assistant_safe | 45,000 | 15% |
| muhaddith | 60,000 | 20% |
| tutor | 45,000 | 15% |
| rag_assistant | 40,000 | 13% |
| mufassir | 25,000 | 8% |
| faqih | 25,000 | 8% |
| edtech_tutor | 20,000 | 7% |
| proofreader | 15,000 | 5% |
| dialect_handling_egy | 10,000 | 3% |
| Other 20 roles | 15,000 | 5% |
| **Total** | **300,000** | **100%** |

### By Domain

| Domain | Examples | Percentage |
|--------|----------|------------|
| Islamic Studies | 150,000 | 50% |
| Linguistics | 60,000 | 20% |
| General Arabic | 45,000 | 15% |
| Literature | 30,000 | 10% |
| Other | 15,000 | 5% |
| **Total** | **300,000** | **100%** |

---

## 🚀 Quick Start Commands

### Complete Pipeline (One Command)

```bash
python scripts/run_complete_pipeline.py --all-sources
```

### Step-by-Step

```bash
# Step 1: Audit all sources
python scripts/complete_data_audit.py

# Step 2: Process each source
python scripts/process_arabic_web.py
python scripts/process_extracted_books.py
python scripts/process_sanadset.py
python scripts/process_system_books.py

# Step 3: Merge all
python scripts/merge_all_datasets.py

# Step 4: Quality control
python scripts/quality_control.py

# Step 5: Train
python scripts/03_train_model.py \
  --dataset data/jsonl/balygh_final_sft.jsonl \
  --output-dir models/balygh-complete-v1
```

---

## 📈 Expected Results

### Training Data Statistics

| Metric | Value |
|--------|-------|
| **Total Examples** | 300,000 |
| **Unique Examples** | 280,000 (93%) |
| **Avg Example Length** | 450 characters |
| **Arabic Ratio** | 0.88 |
| **Quality Score** | 0.82 |
| **Roles Covered** | 29/29 |
| **Skills Covered** | 76/76 |
| **Domains Covered** | 12/12 |

### Training Time Estimates

| GPU Configuration | Time for 300K examples |
|------------------|----------------------|
| Single RTX 3090 (24GB) | ~36 hours |
| Single RTX 4090 (24GB) | ~30 hours |
| Single A100 (80GB) | ~18 hours |
| 8x A100 (80GB each) | ~3 hours |

---

## 🎯 Quality Improvements

### Before Integration

| Metric | Value |
|--------|-------|
| Data Sources Used | 0/5 |
| Total Examples | 0 |
| Roles Covered | 5/29 |
| Quality Score | N/A |

### After Integration

| Metric | Value | Improvement |
|--------|-------|-------------|
| Data Sources Used | 5/5 | +5 sources |
| Total Examples | 300,000 | +300K |
| Roles Covered | 29/29 | +24 roles |
| Quality Score | 0.82 | New |
| Deduplication | 93% unique | New |

---

## 📋 Implementation Checklist

### Phase 1: Audit (Week 1)
- [ ] Create complete_data_audit.py ✅
- [ ] Run audit on all 5 sources
- [ ] Generate quality report
- [ ] Identify critical gaps

### Phase 2: Processing (Week 2-3)
- [ ] Create process_arabic_web.py ✅
- [ ] Create process_extracted_books.py ✅
- [ ] Create process_sanadset.py ✅
- [ ] Create process_system_books.py ✅
- [ ] Process all 5 sources
- [ ] Generate 358K raw examples

### Phase 3: Quality Control (Week 4)
- [ ] Create merge_all_datasets.py ✅
- [ ] Apply deduplication (85% threshold)
- [ ] Apply quality filters
- [ ] LLM-as-judge evaluation
- [ ] Manual review (500 samples)
- [ ] Generate 300K final examples

### Phase 4: Training (Week 5)
- [ ] Prepare training config
- [ ] Start training (36 hours)
- [ ] Monitor training
- [ ] Evaluate results

### Phase 5: Deployment (Week 6)
- [ ] Push to Hugging Face
- [ ] Create Gradio demo
- [ ] Write documentation
- [ ] Public release

---

## 🔍 Scripts to Create

### Priority 1: Core Processing

| Script | Status | Purpose |
|--------|--------|---------|
| `complete_data_audit.py` | 🟡 To Create | Audit all 5 sources |
| `process_arabic_web.py` | 🟡 To Create | Process arabic_web |
| `process_extracted_books.py` | ✅ Exists | Process books |
| `process_sanadset.py` | 🟡 To Create | Process Sanadset |
| `process_system_books.py` | 🟡 To Create | Process system DBs |
| `merge_all_datasets.py` | 🟡 To Create | Merge all sources |
| `quality_control.py` | 🟡 To Create | Quality filtering |
| `run_complete_pipeline.py` | 🟡 To Create | One-command pipeline |

### Priority 2: Enhancement

| Script | Status | Purpose |
|--------|--------|---------|
| `enhance_metadata.py` | 🟡 To Create | Enhance metadata |
| `create_eval_datasets.py` | 🟡 To Create | Create eval sets |
| `llm_judge.py` | 🟡 To Create | LLM quality scoring |

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    5 DATA SOURCES                           │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ arabic_web   │extracted_    │  metadata    │  Sanadset     │
│ (50K ex)     │books         │  (6 files)   │  (130K ex)    │
│              │(113K ex)     │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              SYSTEM BOOK DATASETS (5 DBs)                   │
│              (65K examples)                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              SOURCE-SPECIFIC PROCESSING                     │
│  • process_arabic_web.py                                    │
│  • process_extracted_books.py                               │
│  • process_sanadset.py                                      │
│  • process_system_books.py                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              MERGE & DEDUPLICATE                            │
│  • merge_all_datasets.py                                    │
│  • MinHash LSH (threshold=0.85)                            │
│  • Output: 358K → 300K unique examples                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              QUALITY CONTROL                                │
│  • Arabic ratio ≥ 0.7                                       │
│  • Length filters (50-2000 chars)                          │
│  • LLM-as-judge scoring                                     │
│  • Manual review (500 samples)                             │
│  • Output: 300K high-quality examples                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              TRAINING                                       │
│  • QLoRA (r=64, alpha=128)                                 │
│  • Qwen2.5-7B-Instruct                                      │
│  • 3 epochs, ~36 hours (RTX 3090)                          │
│  • Output: balygh-complete-v1                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              DEPLOYMENT                                     │
│  • Hugging Face publishing                                  │
│  • Gradio demo                                              │
│  • API launch                                               │
│  • Documentation                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Summary

### What You Have

✅ **5 Complete Data Sources**:
1. `arabic_web/` - Modern Arabic web corpus
2. `extracted_books/` - 8,424 Shamela books (16.4 GB)
3. `metadata/` - 6 metadata files
4. `Sanadset 368K/` - 368K hadith narrators
5. `system_book_datasets/` - 5 structured databases

✅ **Existing Scripts**:
- `build_balygh_sft_dataset.py`
- `refine_balygh_sft_with_llm.py`
- `audit_datasets.py`
- `integrate_datasets.py`

### What You Need

🟡 **8 New Scripts** (Priority 1):
1. `complete_data_audit.py`
2. `process_arabic_web.py`
3. `process_sanadset.py`
4. `process_system_books.py`
5. `merge_all_datasets.py`
6. `quality_control.py`
7. `llm_judge.py`
8. `run_complete_pipeline.py`

### Expected Outcome

🎯 **After Full Integration**:
- **300,000 high-quality examples**
- **All 29 roles covered**
- **All 76 skills covered**
- **Quality score: 0.82+**
- **Ready for production training**

---

**Status**: 🟡 Ready for Phase 2 Implementation  
**Next Step**: Create `complete_data_audit.py` to get exact counts from all 5 sources

---

<div align="center">

# بليغ (Balygh) - Complete Data Plan

**5 مصادر بيانات • 300,000 مثال • 29 دور • 76 مهارة**

[Audit](scripts/complete_data_audit.py) | [Process](scripts/process_all.py) | [Train](scripts/03_train_model.py)

</div>

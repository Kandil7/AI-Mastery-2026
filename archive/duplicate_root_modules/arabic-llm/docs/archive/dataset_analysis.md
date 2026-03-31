# Dataset Analysis and Recommendations

## تحليل مجموعة البيانات والتوصيات

Analysis of the Shamela dataset (8,423 books) for Arabic LLM fine-tuning.

---

## 1. Dataset Overview

### Total Statistics
- **Books**: 8,423 total (8,423 extracted - 100%)
- **Size**: ~16.4 GB
- **Authors**: 3,146
- **Categories**: 41
- **Time Period**: Pre-Islamic to modern (7th-20th century)

---

## 2. Key Categories for Linguistic Training

### Primary Categories (High Priority)

| ID | Category | Books | Priority | Use Case |
|----|----------|-------|----------|----------|
| 29 | كتب اللغة | ~400 | ⭐⭐⭐ | Core linguistics, vocabulary |
| 31 | النحو والصرف | ~150 | ⭐⭐⭐ | Grammar, morphology |
| 35 | البلاغة | ~100 | ⭐⭐⭐ | Rhetoric, figures of speech |
| 34 | الدواوين الشعرية | ~200 | ⭐⭐ | Poetry composition |
| 33 | العروض والقوافي | ~50 | ⭐⭐ | Poetry meters, prosody |
| 32 | الأدب | ~415 | ⭐⭐ | Literary Arabic, prose |

### Secondary Categories (Medium Priority)

| ID | Category | Books | Priority | Use Case |
|----|----------|-------|----------|----------|
| 3 | التفسير | 270 | ⭐ | Classical Arabic, Quranic exegesis |
| 5 | التجويد والقراءات | ~100 | ⭐ | Quranic recitation |
| 6 | كتب السنة | 1,226 | ⭐ | Prophetic traditions |
| 24 | السيرة النبوية | ~200 | ⭐ | Biographical Arabic |
| 26 | التراجم والطبقات | 555 | ⭐ | Classical biographies |

### Tertiary Categories (Low Priority)

| ID | Category | Books | Priority | Use Case |
|----|----------|-------|----------|----------|
| 1 | العقيدة | 794 | - | Theological content |
| 14-19 | الفقه | ~2,000 | - | Jurisprudence (too specialized) |
| 23 | الرقائق والآداب | 619 | - | Ethics and manners |

---

## 3. Recommended Book Selection

### For Grammar (نحو) Training
Books from category 31 (النحو والصرف):
- Classical grammar treatises
- Ibn Malik's works (Alfiyyah)
- Sibawayh's Al-Kitab
- Medieval grammar commentaries

**Estimated Examples**: 8,000-10,000

### For Rhetoric (بلاغة) Training
Books from category 35 (البلاغة):
- Al-Jurjani's works (Dala'il, Asrar)
- Al-Sakkaki's Miftah al-Ulum
- Al-Qazwini's Talkhis
- Commentaries on rhetoric

**Estimated Examples**: 5,000-7,000

### For Poetry (شعر) Training
Books from categories 34 + 33:
- Pre-Islamic poetry (Mu'allaqat)
- Abbasid poetry (Mutannabi, Abu Tammam)
- Andalusian poetry
- Poetry diwans with commentary

**Estimated Examples**: 10,000-15,000

### For Language (لغة) Training
Books from category 29:
- Ibn Faris's Maqayis
- Ibn Manzur's Lisan al-Arab
- Al-Fayruzabadi's Al-Qamus
- Classical dictionaries

**Estimated Examples**: 15,000-20,000

---

## 4. Updated Data Configuration

Based on dataset analysis, here's the recommended configuration:

```yaml
# Updated data_config.yaml recommendations

source_categories:
  # Primary (70% of data)
  - "كتب اللغة"           # Weight: 3.0
  - "النحو والصرف"        # Weight: 3.0
  - "البلاغة"            # Weight: 2.5
  - "الدواوين الشعرية"    # Weight: 2.0
  - "العروض والقوافي"     # Weight: 2.0
  
  # Secondary (25% of data)
  - "الأدب"              # Weight: 1.5
  - "التفسير"            # Weight: 1.0
  - "كتب السنة"          # Weight: 1.0
  
  # Tertiary (5% of data)
  - "التراجم والطبقات"    # Weight: 0.5
  - "السيرة النبوية"      # Weight: 0.5

# Updated role distribution based on available data
role_distribution:
  tutor: 0.35           # Reduced from 40% (more grammar books)
  proofreader: 0.25     # Same (good coverage)
  poet: 0.20            # Increased from 15% (rich poetry corpus)
  muhhaqiq: 0.15        # Same (classical texts available)
  assistant_general: 0.05  # Same

# Book weights for sampling
book_weights:
  "كتب اللغة": 3.0
  "النحو والصرف": 3.0
  "البلاغة": 2.5
  "الدواوين الشعرية": 2.0
  "العروض والقوافي": 2.0
  "الأدب": 1.5
  "التفسير": 1.0
  "كتب السنة": 1.0
  "التراجم والطبقات": 0.5
  "السيرة النبوية": 0.5
```

---

## 5. Content Type Distribution

Based on file analysis:

| Content Type | Estimated % | Best For |
|--------------|-------------|----------|
| Prose (نثر) | 60% | Grammar, rhetoric, general |
| Poetry (شعر) | 20% | Poetry composition, analysis |
| Hadith (حديث) | 10% | Classical Arabic, verification |
| Mixed (مختلط) | 10% | Various tasks |

---

## 6. Quality Considerations

### High-Quality Sources
1. **Verified classical texts** (categories 29, 31, 35)
2. **Well-preserved poetry diwans** (category 34)
3. **Standard grammar treatises** (category 31)

### Potential Issues
1. **OCR errors** in some extracted texts
2. **Formatting inconsistencies** across books
3. **Diacritics variation** (some books fully voweled, others not)
4. **Encoding issues** (check for corrupted characters)

### Recommendations
1. **Preprocessing pipeline**:
   - Normalize Arabic characters (أ/ا, ي/ى)
   - Standardize diacritics handling
   - Remove OCR artifacts
   - Validate UTF-8 encoding

2. **Quality filtering**:
   - Minimum Arabic character ratio: 0.85
   - Maximum repetition ratio: 0.20
   - Minimum unique words: 0.25

---

## 7. Estimated Dataset Size

Based on 8,423 books and target processing:

| Processing Level | Books | Segments/Book | Total Segments |
|-----------------|-------|---------------|----------------|
| Full corpus | 8,423 | 50 | 421,150 |
| Primary only | ~900 | 100 | 90,000 |
| Primary + Secondary | ~2,500 | 60 | 150,000 |
| Balanced sample | ~1,500 | 80 | 120,000 |

**Recommended**: Process ~1,500 books from primary + secondary categories to generate 50,000-100,000 high-quality examples.

---

## 8. Special Collections to Prioritize

### Grammar Core (النحو الأساسي)
- Kitab al-Ayn (Farahidi)
- Al-Kitab (Sibawayh)
- Al-Jami' (Mubarrad)
- Al-Insaf (Ibn al-Anbari)
- Sharh Ibn Aqil (on Alfiyyah)

### Rhetoric Core (البلاغة الأساسية)
- Dala'il al-I'jaz (Jurjani)
- Asrar al-Balagha (Jurjani)
- Miftah al-Ulum (Sakkaki)
- Talkhis al-Miftah (Qazwini)

### Poetry Core (الشعر الأساسي)
- Mu'allaqat al-Sab'
- Diwan Abu Tammam
- Diwan al-Mutannabi
- Diwan Ibn al-Farid
- Al-Hamasa (Buhturi)

---

## 9. Implementation Priority

### Phase 1 (Week 1-2): Core Linguistic Books
- Process categories: 29, 31, 35
- Target: 20,000 examples
- Focus: Grammar + Rhetoric templates

### Phase 2 (Week 3-4): Poetry and Literature
- Process categories: 32, 33, 34
- Target: 15,000 examples
- Focus: Poetry composition + Analysis

### Phase 3 (Week 5-6): Classical Texts
- Process categories: 3, 6, 24, 26
- Target: 15,000 examples
- Focus: Muhhaqiq + Advanced templates

### Total Target: 50,000 examples

---

## 10. Metadata Utilization

### Author-Based Filtering
```python
# Prioritize renowned linguists
priority_authors = [
    "سيبويه",     # Sibawayh
    "الزجاج",     # Al-Zajjaj
    "ابن جني",    # Ibn Jinni
    "الزمخشري",   # Zamakhshari
    "الجرجاني",   # Jurjani
    "السخاكي",    # Sakkaki
    "ابن مالك",   # Ibn Malik
]
```

### Time Period Filtering
```python
# Classical period (most authoritative)
classical_period = (0, 800)  # 1st-8th century AH
```

---

## 11. File Processing Notes

### File Naming Convention
```
{book_id}_{title}.txt
Example: 10018_النحو_الواضح_في_قواعد_اللغة_العربية.txt
```

### Encoding
- All files: UTF-8
- Arabic text: Proper Unicode
- Check for: mojibake, encoding errors

### Size Distribution
- Average book: 1-5 MB
- Largest books: 10-20 MB (dictionaries, collections)
- Smallest books: 0.1-0.5 MB (short treatises)

---

## 12. Next Steps

1. **Run category analysis**:
   ```bash
   python scripts/analyze_categories.py \
       --metadata-dir datasets/metadata \
       --output-dir arabic-llm/data/analysis
   ```

2. **Generate book priority list**:
   ```bash
   python scripts/generate_priority_list.py \
       --categories "كتب اللغة,النحو والصرف,البلاغة" \
       --output arabic-llm/configs/priority_books.json
   ```

3. **Update processing script** with weights:
   ```bash
   python scripts/01_process_books.py \
       --priority-list arabic-llm/configs/priority_books.json \
       --weighted-sampling true
   ```

---

**Analysis Date**: March 25, 2026  
**Dataset Version**: Shamela 4.0  
**Total Books**: 8,423  
**Status**: Ready for Processing

# Arabic LLM - Complete Implementation Plan

## خطة التنفيذ الكاملة

**Based on**: docs/guides/llm arabic_plan.md  
**Date**: March 26, 2026  
**Version**: 2.4.0  
**Status**: 📋 **IMPLEMENTATION GUIDE**  

---

## 🎯 Executive Summary

This document provides a **complete, practical implementation plan** for building an Arabic LLM based on the comprehensive plan in `llm arabic_plan.md`.

### Current Status (March 26, 2026)

✅ **COMPLETE**:
- Schema: 19 roles, 45+ skills
- Templates: 75+ instruction templates
- Data cleaning: 7-stage pipeline
- Dataset generator: 61,500 examples
- Training scripts: QLoRA, Qwen2.5-7B
- RAG system: Islamic/Arabic content
- Autonomous research: Autoresearch pattern

📋 **TODO**:
- Data collection from web
- Quality measurement tools
- Open corpora integration
- Complete end-to-end demo

---

## 📊 Part 1: Data Requirements

### What We Have ✅

| Data Type | Amount | Source |
|-----------|--------|--------|
| **Books** | 8,424 books | Shamela (extracted) |
| **Raw Text** | 16.4 GB | extracted_books/ |
| **Metadata** | Complete | books.json, authors.json |
| **System DBs** | 148 MB | hadeeth.db, tafseer.db |
| **Training Examples** | 61,500 | Generated JSONL |

### What You Need for Your Project

| Goal | Examples | Raw Data | Use Case |
|------|----------|----------|----------|
| **Prototype** | 10k-20k | 1-5 GB | Quick testing |
| **Medium (like docs)** | 50k-60k | 10-20 GB | arabic-linguist-v1 |
| **Production** | 100k+ | 50 GB+ | Full deployment |

---

## 🗂️ Part 2: Complete Data Collection Strategy

### Option 1: Use Existing Data (RECOMMENDED) ✅

**Already implemented in this project**:

```bash
# Your data is ready!
datasets/
├── extracted_books/     # 8,424 books (16.4 GB)
├── metadata/
│   ├── books.json       # Complete metadata
│   ├── authors.json     # 3,146 authors
│   └── categories.json  # 41 categories
└── system_book_datasets/
    ├── hadeeth.db       # Hadith collections
    └── tafseer.db       # Quranic exegesis
```

**Usage**:
```python
from arabic_llm.core import DatasetGenerator

generator = DatasetGenerator(
    books_dir="datasets/extracted_books",
    metadata_dir="datasets/metadata",
    output_dir="data/jsonl",
    config=DatasetConfig(target_examples=61500),
)

stats = generator.generate()
print(f"Generated {stats.total_examples} examples")
```

### Option 2: Collect from Web (If You Need More)

#### 2.1 Open Corpora (Free, Legal)

**Recommended Sources**:

| Corpus | Size | License | Use Case |
|--------|------|---------|----------|
| **ArabicWeb24** | 24GB | Open | General Arabic |
| **OpenITI** | 10GB | Open | Classical Islamic texts |
| **Arabic Wikipedia** | 1.5GB | CC-BY-SA | General knowledge |
| **Sanadset 368K** | 368K hadith | Open | Hadith NER |
| **COLD** | Legal docs | Open | Legal Arabic |

**Download Script**:
```python
# scripts/download_open_corpora.py
import requests
import json
from pathlib import Path

def download_arabic_web24():
    """Download ArabicWeb24 corpus"""
    url = "https://huggingface.co/datasets/MayFarhat/ArabicWeb24"
    # Use datasets library
    from datasets import load_dataset
    dataset = load_dataset("MayFarhat/ArabicWeb24")
    return dataset

def download_openiti():
    """Download OpenITI classical texts"""
    url = "https://github.com/OpenITI/Corpus"
    # Clone or download
    pass

if __name__ == "__main__":
    print("Downloading open corpora...")
    arabic_web = download_arabic_web24()
    openiti = download_openiti()
    print(f"Downloaded {len(arabic_web)} documents")
```

#### 2.2 Web Scraping (Careful with Legal)

**Safe Sources for Scraping**:

| Source | Content | Legal Status |
|--------|---------|--------------|
| **Dar Al-Ifta** | Fatwas | Check ToS |
| **IslamWeb** | Articles | Check ToS |
| **Arabic News** | News articles | CC or ToS |
| **Educational Sites** | Lessons | Often OK for education |

**Scraper Template**:
```python
# scripts/scrape_islamic_sites.py
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path

class IslamicSiteScraper:
    """Scraper for Islamic websites (fatwas, articles)"""
    
    def __init__(self, base_url, output_dir="data/scraped"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_fatwa(self, url):
        """Scrape single fatwa"""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        fatwa = {
            'question': self._extract_question(soup),
            'answer': self._extract_answer(soup),
            'scholar': self._extract_scholar(soup),
            'source': url,
        }
        
        return fatwa
    
    def _extract_question(self, soup):
        """Extract question text"""
        # Customize per site
        return soup.find('div', class_='question').text.strip()
    
    def _extract_answer(self, soup):
        """Extract answer text"""
        return soup.find('div', class_='answer').text.strip()
    
    def _extract_scholar(self, soup):
        """Extract scholar name"""
        return soup.find('span', class_='scholar').text.strip()
    
    def save(self, fatwas, filename="fatwas.jsonl"):
        """Save to JSONL"""
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            for fatwa in fatwas:
                f.write(json.dumps(fatwa, ensure_ascii=False) + '\n')

# Usage
if __name__ == "__main__":
    scraper = IslamicSiteScraper("https://example-ifta.com")
    fatwas = []
    
    # Scrape multiple fatwas
    for i in range(100):
        url = f"https://example-ifta.com/fatwa/{i}"
        try:
            fatwa = scraper.scrape_fatwa(url)
            fatwas.append(fatwa)
        except:
            pass
    
    scraper.save(fatwas)
    print(f"Saved {len(fatwas)} fatwas")
```

---

## 🧹 Part 3: Preprocessing Pipeline

### Complete Arabic Preprocessing

**Already implemented**: `arabic_llm/pipeline/cleaning.py`

**7-Stage Pipeline** (from COMPLETE_DOCUMENTATION.md):

```python
from arabic_llm.pipeline import DataCleaningPipeline

pipeline = DataCleaningPipeline(
    books_dir="datasets/extracted_books",
    metadata_dir="datasets/metadata",
    output_dir="data/processed",
    workers=8,  # Parallel processing
)

# Run complete pipeline
stats = pipeline.run_pipeline(max_books=8424)

print(f"Processed {stats['total_books']} books")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Average Arabic ratio: {stats['avg_arabic_ratio']:.1%}")
```

**Stage Details**:

| Stage | Operation | Example |
|-------|-----------|---------|
| **1. Encoding** | Fix BOM, mojibake | `\ufeff` → removed |
| **2. Unicode** | NFC normalization | `أ` (decomposed) → `أ` (composed) |
| **3. Arabic** | Normalize forms | `أ، إ، آ` → `ا` |
| **4. Control** | Remove control chars | Keep `\n\r\t` |
| **5. Whitespace** | Normalize spaces | `  ` → ` ` |
| **6. OCR** | Fix OCR errors | `٠١٢` → `012` |
| **7. Punctuation** | Normalize | `،` → `,` |

### Additional Preprocessing for Web Data

```python
# scripts/preprocess_web_data.py
import re
import unicodedata
from pathlib import Path

class ArabicWebPreprocessor:
    """Preprocessor for web-scraped Arabic text"""
    
    def __init__(self):
        self.arabic_range = ('\u0600', '\u06FF')
    
    def detect_language(self, text, threshold=0.7):
        """Detect if text is primarily Arabic"""
        arabic_chars = sum(1 for c in text if self.arabic_range[0] <= c <= self.arabic_range[1])
        return arabic_chars / len(text) >= threshold
    
    def remove_html(self, text):
        """Remove HTML tags"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_emoji(self, text):
        """Remove emoji characters"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_urls_emails(self, text):
        """Remove URLs and emails"""
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def normalize(self, text):
        """Complete normalization pipeline"""
        # Remove HTML
        text = self.remove_html(text)
        
        # Remove emoji
        text = self.remove_emoji(text)
        
        # Remove URLs/emails
        text = self.remove_urls_emails(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Arabic normalization
        text = re.sub(r'[أإآ]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        
        # Whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def filter_quality(self, text, min_length=50, max_length=10000):
        """Filter by quality metrics"""
        if len(text) < min_length or len(text) > max_length:
            return False
        
        if not self.detect_language(text):
            return False
        
        return True
    
    def process_file(self, input_file, output_file):
        """Process single file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Normalize
        text = self.normalize(text)
        
        # Filter
        if not self.filter_quality(text):
            print(f"Filtered out: {input_file}")
            return
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Processed: {input_file} → {output_file}")

# Usage
if __name__ == "__main__":
    preprocessor = ArabicWebPreprocessor()
    
    # Process single file
    preprocessor.process_file("data/scraped/fatwa.txt", "data/processed/fatwa_clean.txt")
    
    # Process directory
    from pathlib import Path
    input_dir = Path("data/scraped")
    output_dir = Path("data/processed")
    
    for file in input_dir.glob("*.txt"):
        preprocessor.process_file(file, output_dir / file.name)
```

---

## 📏 Part 4: Quality Measurement

### Quality Metrics

**Already implemented**: `arabic_llm/pipeline/cleaning.py`

```python
from arabic_llm.utils import count_arabic_chars, get_arabic_ratio

def measure_quality(text):
    """Measure text quality"""
    metrics = {
        'arabic_ratio': get_arabic_ratio(text),
        'total_chars': len(text),
        'total_words': len(text.split()),
        'arabic_chars': count_arabic_chars(text),
    }
    
    # Quality thresholds
    metrics['passed'] = (
        metrics['arabic_ratio'] >= 0.70 and
        metrics['total_chars'] >= 100 and
        metrics['total_chars'] <= 10000
    )
    
    return metrics

# Usage
text = "العلمُ نورٌ والجهلُ ظلامٌ."
metrics = measure_quality(text)

print(f"Arabic ratio: {metrics['arabic_ratio']:.1%}")
print(f"Total chars: {metrics['total_chars']}")
print(f"Passed quality: {metrics['passed']}")
```

### Comprehensive Quality Report

```python
# scripts/generate_quality_report.py
import json
from pathlib import Path
from arabic_llm.utils import get_arabic_ratio, count_arabic_chars

def generate_quality_report(data_dir="data/processed"):
    """Generate comprehensive quality report"""
    report = {
        'total_files': 0,
        'passed_files': 0,
        'avg_arabic_ratio': 0,
        'avg_chars': 0,
        'distribution': {
            'excellent': 0,  # > 90% Arabic
            'good': 0,       # 70-90%
            'fair': 0,       # 50-70%
            'poor': 0,       # < 50%
        }
    }
    
    arabic_ratios = []
    char_counts = []
    
    # Process all files
    data_path = Path(data_dir)
    for file in data_path.glob("*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        ratio = get_arabic_ratio(text)
        chars = len(text)
        
        arabic_ratios.append(ratio)
        char_counts.append(chars)
        
        # Distribution
        if ratio >= 0.90:
            report['distribution']['excellent'] += 1
        elif ratio >= 0.70:
            report['distribution']['good'] += 1
        elif ratio >= 0.50:
            report['distribution']['fair'] += 1
        else:
            report['distribution']['poor'] += 1
        
        report['total_files'] += 1
        if ratio >= 0.70:
            report['passed_files'] += 1
    
    # Averages
    report['avg_arabic_ratio'] = sum(arabic_ratios) / len(arabic_ratios) if arabic_ratios else 0
    report['avg_chars'] = sum(char_counts) / len(char_counts) if char_counts else 0
    
    # Save report
    with open(data_path / "quality_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Quality Report:")
    print(f"  Total files: {report['total_files']}")
    print(f"  Passed: {report['passed_files']} ({report['passed_files']/report['total_files']*100:.1f}%)")
    print(f"  Avg Arabic ratio: {report['avg_arabic_ratio']:.1%}")
    print(f"  Distribution: {report['distribution']}")
    
    return report

if __name__ == "__main__":
    generate_quality_report()
```

---

## 🎓 Part 5: Role/Skill Implementation

### Already Implemented ✅

**From schema_enhanced.py**:
- 19 roles (tutor, faqih, muhaddith, dataengineer_ar, rag_assistant, etc.)
- 45+ skills (nahw, balagha, fiqh, rag_grounded_answering, etc.)

**From templates.py**:
- 75+ instruction templates
- Organized by role/skill/level

### Usage Example

```python
from arabic_llm.core import get_random_template, Role, Skill

# Get template for specific role/skill
template = get_random_template(role="rag_assistant", skill="rag_grounded_answering")

print(f"Template ID: {template.id}")
print(f"Instruction: {template.instruction_template}")
print(f"Output Format: {template.output_format}")

# Fill template
filled = template.format_instruction(
    question="ما حكم الصلاة؟",
    context="قال الله تعالى: {أقيموا الصلاة}..."
)

print(f"\nFilled Instruction:\n{filled}")
```

---

## 🚀 Part 6: Complete End-to-End Pipeline

### Full Pipeline Script

```python
# scripts/complete_pipeline.py
"""
Complete End-to-End Arabic LLM Pipeline

Usage:
    python scripts/complete_pipeline.py \
        --books-dir datasets/extracted_books \
        --metadata-dir datasets/metadata \
        --output-dir data/jsonl \
        --target-examples 61500
"""

import argparse
from pathlib import Path
from arabic_llm.pipeline import DataCleaningPipeline
from arabic_llm.core import DatasetGenerator, DatasetConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-dir", default="datasets/extracted_books")
    parser.add_argument("--metadata-dir", default="datasets/metadata")
    parser.add_argument("--output-dir", default="data/jsonl")
    parser.add_argument("--target-examples", type=int, default=61500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    print("="*70)
    print("Arabic LLM - Complete Pipeline")
    print("="*70)
    
    # Step 1: Clean data
    print("\n[1/2] Cleaning data...")
    cleaning_pipeline = DataCleaningPipeline(
        books_dir=args.books_dir,
        metadata_dir=args.metadata_dir,
        output_dir="data/processed",
        workers=args.workers,
    )
    
    cleaning_stats = cleaning_pipeline.run_pipeline()
    print(f"✓ Cleaned {cleaning_stats['total_books']} books")
    
    # Step 2: Generate dataset
    print("\n[2/2] Generating dataset...")
    config = DatasetConfig(target_examples=args.target_examples)
    generator = DatasetGenerator(
        books_dir=args.books_dir,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        config=config,
    )
    
    dataset_stats = generator.generate()
    print(f"✓ Generated {dataset_stats.total_examples} examples")
    
    # Summary
    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    print(f"Input: {cleaning_stats['total_books']} books ({cleaning_stats['total_size_mb']:.2f} MB)")
    print(f"Output: {dataset_stats.total_examples} examples")
    print(f"Role distribution: {dataset_stats.by_role}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
```

---

## 📊 Part 7: Training

### QLoRA Training Script

**Already implemented**: `arabic-llm/scripts/03_train_model.py`

```bash
# Run training
python arabic-llm/scripts/03_train_model.py \
    --dataset data/jsonl/train.jsonl \
    --output-dir models/arabic-linguist-v1 \
    --config arabic-llm/configs/training_config.yaml
```

**Training Configuration** (from `training_config.yaml`):
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "float16"

lora:
  r: 64
  alpha: 128
  dropout: 0.05

training:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2.0e-4
  epochs: 3
  max_seq_length: 2048
```

---

## ✅ Implementation Checklist

### Phase 1: Data Collection ✅
- [x] Use existing Shamela data (8,424 books)
- [ ] Download open corpora (optional)
- [ ] Web scraping (optional, careful with legal)

### Phase 2: Preprocessing ✅
- [x] 7-stage cleaning pipeline
- [ ] Web data preprocessor (if scraping)
- [x] Quality measurement

### Phase 3: Schema & Templates ✅
- [x] 19 roles, 45+ skills
- [x] 75+ instruction templates
- [x] Template filling examples

### Phase 4: Dataset Generation ✅
- [x] DatasetGenerator
- [x] Role balancing
- [x] JSONL output

### Phase 5: Training ✅
- [x] QLoRA training script
- [x] Configuration files
- [x] Checkpoint saving

### Phase 6: Evaluation ⏳
- [ ] OALL benchmarks
- [ ] Custom test sets
- [ ] Human evaluation

---

## 🎯 Next Steps

1. **Run Complete Pipeline**:
   ```bash
   python arabic-llm/scripts/complete_pipeline.py
   ```

2. **Train Model**:
   ```bash
   python arabic-llm/scripts/03_train_model.py
   ```

3. **Evaluate**:
   ```bash
   python arabic-llm/scripts/04_evaluate.py
   ```

4. **Deploy**:
   ```bash
   python arabic-llm/scripts/05_deploy.py
   ```

---

**Version**: 2.4.0  
**Date**: March 26, 2026  
**Status**: 📋 **COMPLETE IMPLEMENTATION GUIDE**  
**Next**: Run pipeline and train model

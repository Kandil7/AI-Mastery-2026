"""
Process Sanadset Hadith Narrators Dataset for Balygh

Converts Sanadset 368K hadith narrators data into SFT training examples
for hadith sciences, narrator biographies, and isnad analysis.

Usage:
    python scripts/process_sanadset.py
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SanadsetConfig:
    """Configuration for Sanadset processing"""
    
    # Input/Output paths
    INPUT_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/datasets/Sanadset 368K Data on Hadith Narrators")
    OUTPUT_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm/data/jsonl")
    
    # Target counts
    TARGET_NARRATOR_TRANSLATION = 50000
    TARGET_HADITH_GRADING = 30000
    TARGET_ISNAD_ANALYSIS = 20000
    TARGET_JARH_TADIL = 20000
    TARGET_TABAQAT = 10000
    
    # Quality filters
    MIN_ARABIC_RATIO = 0.6
    MIN_LENGTH = 50
    MAX_LENGTH = 2000


# ============================================================================
# Example Generators
# ============================================================================

class SanadsetExampleGenerator:
    """Generate SFT examples from Sanadset data"""
    
    def __init__(self, config: SanadsetConfig):
        self.config = config
        self.seen_hashes = set()
    
    def _text_hash(self, text: str) -> str:
        """Create hash for deduplication"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        h = self._text_hash(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
    
    def generate_narrator_translation(self, narrator: Dict) -> Optional[Dict]:
        """Generate narrator biography (tarjama) example"""
        name = narrator.get('name', '')
        biography = narrator.get('biography', '')
        kunya = narrator.get('kunya', '')
        death = narrator.get('death', '')
        
        if not name or not biography:
            return None
        
        # Build input text
        input_text = f"الاسم: {name}\n"
        if kunya:
            input_text += f"الكنية: {kunya}\n"
        if death:
            input_text += f"سنة الوفاة: {death}\n"
        
        if self._is_duplicate(input_text + biography):
            return None
        
        example = {
            "id": f"sanad-narrator-{hashlib.md5(name.encode()).hexdigest()[:12]}",
            "instruction": f"ترجم للراوي التالي بالتفصيل:\n\n{name}",
            "input": input_text,
            "output": biography[:1500],
            "role": "muhaddith",
            "skills": ["hadith", "hadith_mustalah"],
            "level": "advanced",
            "domain": "islamic_studies",
            "style": "fusha_classical",
            "task_type": "explanation",
            "difficulty": 4,
            "source": "sanadset",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_hadith_grading(self, narrator: Dict) -> Optional[Dict]:
        """Generate hadith grading example"""
        name = narrator.get('name', '')
        jarh_tadil = narrator.get('jarh_tadil', '')
        grade = narrator.get('grade', '')
        
        if not name:
            return None
        
        # Determine grade from jarh_tadil
        if not grade:
            if 'ثقة' in jarh_tadil or 'صدوق' in jarh_tadil:
                grade = 'ثقة'
            elif 'ضعيف' in jarh_tadil:
                grade = 'ضعيف'
            elif 'كذاب' in jarh_tadil or 'متروك' in jarh_tadil:
                grade = 'متروك'
            else:
                grade = 'مجهول الحال'
        
        input_text = f"الراوي: {name}\nالجرح والتعديل: {jarh_tadil[:200] if jarh_tadil else 'غير معروف'}"
        
        if self._is_duplicate(input_text):
            return None
        
        example = {
            "id": f"sanad-grade-{hashlib.md5(name.encode()).hexdigest()[:12]}",
            "instruction": f"بيّن درجة هذا الراوي للحديث مع التعليل:\n\n{input_text}",
            "input": "",
            "output": f"الدرجة: {grade}\n\nالتعليل: {jarh_tadil[:500] if jarh_tadil else 'يحتاج إلى مزيد من البحث'}",
            "role": "muhaddith",
            "skills": ["hadith", "hadith_mustalah"],
            "level": "advanced",
            "domain": "islamic_studies",
            "style": "fusha_classical",
            "task_type": "evaluation",
            "difficulty": 4,
            "source": "sanadset",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_isnad_analysis(self, narrator: Dict) -> Optional[Dict]:
        """Generate isnad (chain of narration) analysis example"""
        name = narrator.get('name', '')
        shuyukh = narrator.get('shuyukh', [])  # Teachers
        talamidh = narrator.get('talamidh', [])  # Students
        tabaqah = narrator.get('tabaqah', '')
        
        if not name:
            return None
        
        input_text = f"الراوي: {name}\n"
        if tabaqah:
            input_text += f"الطبقة: {tabaqah}\n"
        if shuyukh and isinstance(shuyukh, list) and len(shuyukh) > 0:
            input_text += f"عدد شيوخه: {len(shuyukh)}\n"
        if talamidh and isinstance(talamidh, list) and len(talamidh) > 0:
            input_text += f"عدد تلاميذه: {len(talamidh)}\n"
        
        if self._is_duplicate(input_text):
            return None
        
        # Build isnad analysis output
        output = f"تحليل الإسناد:\n\n"
        output += f"الراوي: {name}\n"
        output += f"الطبقة: {tabaqah if tabaqah else 'غير محددة'}\n\n"
        
        if shuyukh and isinstance(shuyukh, list):
            output += f"شيوخه (من روى عنهم):\n"
            for i, shaykh in enumerate(shuyukh[:10], 1):
                if isinstance(shaykh, str):
                    output += f"{i}. {shaykh}\n"
            if len(shuyukh) > 10:
                output += f"... و{len(shuyukh) - 10} آخرين\n"
        
        output += "\n"
        
        if talamidh and isinstance(talamidh, list):
            output += f"تلاميذه (من روى عنه):\n"
            for i, talmidh in enumerate(talamidh[:10], 1):
                if isinstance(talmidh, str):
                    output += f"{i}. {talmidh}\n"
            if len(talamidh) > 10:
                output += f"... و{len(talamidh) - 10} آخرين\n"
        
        example = {
            "id": f"sanad-isnad-{hashlib.md5(name.encode()).hexdigest()[:12]}",
            "instruction": f"حلّل سند هذا الراوي واذكر شيوخه وتلاميذه:\n\n{input_text}",
            "input": "",
            "output": output[:1500],
            "role": "muhaddith",
            "skills": ["hadith", "hadith_mustalah"],
            "level": "advanced",
            "domain": "islamic_studies",
            "style": "fusha_classical",
            "task_type": "analysis",
            "difficulty": 4,
            "source": "sanadset",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_jarh_tadil(self, narrator: Dict) -> Optional[Dict]:
        """Generate jarh wa tadil (criticism and praise) example"""
        name = narrator.get('name', '')
        jarh_tadil = narrator.get('jarh_tadil', '')
        biography = narrator.get('biography', '')
        
        if not name or not jarh_tadil:
            return None
        
        input_text = f"الراوي: {name}\nالتعديل: {jarh_tadil[:300]}"
        
        if self._is_duplicate(input_text):
            return None
        
        example = {
            "id": f"sanad-jarh-{hashlib.md5(name.encode()).hexdigest()[:12]}",
            "instruction": f"بيّن معنى الجرح والتعديل التالي للراوي:\n\n{input_text}",
            "input": "",
            "output": f"شرح الجرح والتعديل:\n{jarh_tadil[:800]}\n\n{biography[:500] if biography else ''}",
            "role": "muhaddith",
            "skills": ["hadith", "hadith_mustalah"],
            "level": "specialist",
            "domain": "islamic_studies",
            "style": "fusha_classical",
            "task_type": "explanation",
            "difficulty": 5,
            "source": "sanadset",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_tabaqat(self, narrator: Dict) -> Optional[Dict]:
        """Generate tabaqat (generation/class) example"""
        name = narrator.get('name', '')
        tabaqah = narrator.get('tabaqah', '')
        death = narrator.get('death', '')
        biography = narrator.get('biography', '')
        
        if not name:
            return None
        
        input_text = f"الراوي: {name}\n"
        if tabaqah:
            input_text += f"الطبقة: {tabaqah}\n"
        if death:
            input_text += f"الوفاة: {death}\n"
        
        if self._is_duplicate(input_text):
            return None
        
        example = {
            "id": f"sanad-tabaqat-{hashlib.md5(name.encode()).hexdigest()[:12]}",
            "instruction": f"بيّن طبقة هذا الراوي ومن عاصره من المحدثين:\n\n{input_text}",
            "input": "",
            "output": f"الطبقة: {tabaqah if tabaqah else 'غير محددة'}\n\nعاصر من المحدثين: [...]\n\n{biography[:500] if biography else ''}",
            "role": "muhaddith",
            "skills": ["hadith", "hadith_mustalah"],
            "level": "advanced",
            "domain": "islamic_studies",
            "style": "fusha_classical",
            "task_type": "explanation",
            "difficulty": 4,
            "source": "sanadset",
            "quality_score": 0.0
        }
        
        return example


# ============================================================================
# Main Processing Function
# ============================================================================

def process_sanadset():
    """Process Sanadset dataset"""
    config = SanadsetConfig()
    
    print("=" * 70)
    print("Processing Sanadset Hadith Narrators Dataset")
    print("=" * 70)
    print()
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = SanadsetExampleGenerator(config)
    
    # Collect all examples
    all_examples = {
        "narrator_translation": [],
        "hadith_grading": [],
        "isnad_analysis": [],
        "jarh_tadil": [],
        "tabaqat": [],
    }
    
    # Find all JSON files
    json_files = list(config.INPUT_DIR.glob("*.json"))
    jsonl_files = list(config.INPUT_DIR.glob("*.jsonl"))
    csv_files = list(config.INPUT_DIR.glob("*.csv"))
    
    all_files = json_files + jsonl_files + csv_files
    
    print(f"Found {len(all_files)} data files in Sanadset directory")
    print()
    
    total_narrators = 0
    
    # Process each file
    for i, file_path in enumerate(all_files, 1):
        print(f"Processing file {i}/{len(all_files)}: {file_path.name}")
        
        try:
            # Load data
            narrators = []
            
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        narrators = data
                    elif isinstance(data, dict) and 'narrators' in data:
                        narrators = data['narrators']
            
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            narrators.append(json.loads(line))
            
            elif file_path.suffix == '.csv':
                # Skip CSV for now (would need pandas or csv module)
                print(f"  Skipping CSV file: {file_path.name}")
                continue
            
            total_narrators += len(narrators)
            print(f"  Loaded {len(narrators):,} narrators")
            
            # Generate examples for each narrator
            for j, narrator in enumerate(narrators):
                if not isinstance(narrator, dict):
                    continue
                
                # Generate different types
                if len(all_examples["narrator_translation"]) < config.TARGET_NARRATOR_TRANSLATION:
                    ex = generator.generate_narrator_translation(narrator)
                    if ex:
                        all_examples["narrator_translation"].append(ex)
                
                if len(all_examples["hadith_grading"]) < config.TARGET_HADITH_GRADING:
                    ex = generator.generate_hadith_grading(narrator)
                    if ex:
                        all_examples["hadith_grading"].append(ex)
                
                if len(all_examples["isnad_analysis"]) < config.TARGET_ISNAD_ANALYSIS:
                    ex = generator.generate_isnad_analysis(narrator)
                    if ex:
                        all_examples["isnad_analysis"].append(ex)
                
                if len(all_examples["jarh_tadil"]) < config.TARGET_JARH_TADIL:
                    ex = generator.generate_jarh_tadil(narrator)
                    if ex:
                        all_examples["jarh_tadil"].append(ex)
                
                if len(all_examples["tabaqat"]) < config.TARGET_TABAQAT:
                    ex = generator.generate_tabaqat(narrator)
                    if ex:
                        all_examples["tabaqat"].append(ex)
                
                # Check progress
                if j % 10000 == 0:
                    total = sum(len(exs) for exs in all_examples.values())
                    print(f"    Processed {j:,}/{len(narrators):,} narrators, {total:,} examples generated")
            
            # Check if we have enough
            total_examples = sum(len(exs) for exs in all_examples.values())
            target_total = (config.TARGET_NARRATOR_TRANSLATION + config.TARGET_HADITH_GRADING +
                          config.TARGET_ISNAD_ANALYSIS + config.TARGET_JARH_TADIL +
                          config.TARGET_TABAQAT)
            
            if total_examples >= target_total:
                print(f"\nReached target example count ({target_total:,})")
                break
        
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    print()
    print(f"Total narrators processed: {total_narrators:,}")
    print()
    print("Saving examples...")
    
    # Save examples
    total_saved = 0
    for example_type, examples in all_examples.items():
        if examples:
            output_file = config.OUTPUT_DIR / f"sanadset_{example_type}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            
            print(f"  ✅ {example_type}: {len(examples):,} examples → {output_file}")
            total_saved += len(examples)
    
    # Merge all
    all_examples_merged = []
    for examples in all_examples.values():
        all_examples_merged.extend(examples)
    
    if all_examples_merged:
        merged_file = config.OUTPUT_DIR / "sanadset_all.jsonl"
        with open(merged_file, 'w', encoding='utf-8') as f:
            for ex in all_examples_merged:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"  ✅ Merged: {len(all_examples_merged):,} examples → {merged_file}")
    
    print()
    print("=" * 70)
    print(f"Processing Complete! Total: {total_saved:,} examples from {total_narrators:,} narrators")
    print("=" * 70)


if __name__ == "__main__":
    process_sanadset()

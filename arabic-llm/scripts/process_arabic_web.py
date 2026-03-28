"""
Process Arabic Web Corpus for Balygh

Converts Arabic web corpus (ArabicWeb24, FineWeb-Arabic, etc.) into
SFT training examples for general Arabic language, modern vocabulary,
and contemporary topics.

Usage:
    python scripts/process_arabic_web.py
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
class ArabicWebConfig:
    """Configuration for Arabic web processing"""
    
    # Input/Output paths
    INPUT_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/datasets/arabic_web")
    OUTPUT_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm/data/jsonl")
    
    # Target counts
    TARGET_GENERAL_QA = 20000
    TARGET_VOCABULARY = 10000
    TARGET_TOPICS = 10000
    TARGET_RAG = 10000
    
    # Quality filters
    MIN_ARABIC_RATIO = 0.7
    MIN_LENGTH = 100
    MAX_LENGTH = 2000
    
    # Deduplication
    DEDUP_THRESHOLD = 0.85


# ============================================================================
# Text Processing
# ============================================================================

class ArabicTextProcessor:
    """Process Arabic web text"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Clean Arabic text"""
        # Remove HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        return text.strip()
    
    @staticmethod
    def arabic_ratio(text: str) -> float:
        """Calculate Arabic character ratio"""
        if not text:
            return 0.0
        arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return arabic / len(text)
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Arabic sentence delimiters
        sentences = re.split(r'[.!?؟]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if len(p.strip()) > 100]


# ============================================================================
# Example Generators
# ============================================================================

class ArabicWebExampleGenerator:
    """Generate SFT examples from Arabic web corpus"""
    
    def __init__(self, config: ArabicWebConfig):
        self.config = config
        self.processor = ArabicTextProcessor()
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
    
    def generate_general_qa(self, text: str) -> Optional[Dict]:
        """Generate general QA example"""
        text = self.processor.clean(text)
        
        # Quality checks
        if len(text) < self.config.MIN_LENGTH:
            return None
        if self.processor.arabic_ratio(text) < self.config.MIN_ARABIC_RATIO:
            return None
        if self._is_duplicate(text):
            return None
        
        # Split into paragraphs
        paragraphs = self.processor.split_into_paragraphs(text)
        
        if not paragraphs:
            return None
        
        # Use first paragraph as context
        context = paragraphs[0][:1000]
        
        # Generate question
        example = {
            "id": f"ar-web-qa-{hashlib.md5(context.encode()).hexdigest()[:12]}",
            "instruction": f"ما الموضوع الرئيسي في النص التالي؟\n\n{context[:500]}...",
            "input": context,
            "output": f"الموضوع الرئيسي هو: {context[:200]}...",
            "role": "assistant_general",
            "skills": ["reading_comprehension", "summarization"],
            "level": "intermediate",
            "domain": "general",
            "style": "fusha_modern",
            "task_type": "qa",
            "difficulty": 2,
            "source": "arabic_web",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_vocabulary_example(self, text: str) -> Optional[Dict]:
        """Generate vocabulary learning example"""
        text = self.processor.clean(text)
        
        # Quality checks
        if len(text) < self.config.MIN_LENGTH:
            return None
        if self.processor.arabic_ratio(text) < self.config.MIN_ARABIC_RATIO:
            return None
        if self._is_duplicate(text):
            return None
        
        # Extract sentences with interesting words
        sentences = self.processor.split_into_sentences(text)
        
        if len(sentences) < 2:
            return None
        
        # Find sentence with potentially interesting vocabulary
        target_sentence = max(sentences, key=len)[:300]
        
        example = {
            "id": f"ar-web-vocab-{hashlib.md5(target_sentence.encode()).hexdigest()[:12]}",
            "instruction": f"استخرج الكلمات المهمة من الجملة التالية واشرح معانيها:\n\n{target_sentence}",
            "input": target_sentence,
            "output": f"الكلمات المهمة:\n1. [...]: [...]\n2. [...]: [...]\n3. [...]: [...]",
            "role": "edtech_tutor",
            "skills": ["lexicography", "semantics"],
            "level": "intermediate",
            "domain": "linguistics",
            "style": "fusha_modern",
            "task_type": "explanation",
            "difficulty": 2,
            "source": "arabic_web",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_topic_example(self, text: str) -> Optional[Dict]:
        """Generate contemporary topic example"""
        text = self.processor.clean(text)
        
        # Quality checks
        if len(text) < self.config.MIN_LENGTH:
            return None
        if self.processor.arabic_ratio(text) < self.config.MIN_ARABIC_RATIO:
            return None
        if self._is_duplicate(text):
            return None
        
        # Use first 1500 chars as context
        context = text[:1500]
        
        example = {
            "id": f"ar-web-topic-{hashlib.md5(context.encode()).hexdigest()[:12]}",
            "instruction": f"لخّص النص التالي في فقرة واحدة:\n\n{context}",
            "input": context,
            "output": f"ملخص النص: {context[:300]}...",
            "role": "summarizer_ar",
            "skills": ["summarization", "analysis"],
            "level": "intermediate",
            "domain": "general",
            "style": "fusha_modern",
            "task_type": "summarization",
            "difficulty": 2,
            "source": "arabic_web",
            "quality_score": 0.0
        }
        
        return example
    
    def generate_rag_example(self, text: str) -> Optional[Dict]:
        """Generate RAG context example"""
        text = self.processor.clean(text)
        
        # Quality checks
        if len(text) < self.config.MIN_LENGTH:
            return None
        if self.processor.arabic_ratio(text) < self.config.MIN_ARABIC_RATIO:
            return None
        if self._is_duplicate(text):
            return None
        
        # Use as RAG context
        context = text[:1000]
        
        # Generate question that can be answered from context
        example = {
            "id": f"ar-web-rag-{hashlib.md5(context.encode()).hexdigest()[:12]}",
            "instruction": f"أجب عن السؤال التالي بناءً على النص المعطى فقط:\n\nالسؤال: ما المعلومات الرئيسية في هذا النص؟",
            "input": f"النص:\n{context}",
            "output": f"المعلومات الرئيسية:\n{context[:300]}...",
            "role": "rag_assistant",
            "skills": ["rag_grounded_answering", "reading_comprehension"],
            "level": "intermediate",
            "domain": "general",
            "style": "fusha_modern",
            "task_type": "rag_qa",
            "difficulty": 2,
            "source": "arabic_web",
            "quality_score": 0.0
        }
        
        return example


# ============================================================================
# Main Processing Function
# ============================================================================

def process_arabic_web():
    """Process Arabic web corpus"""
    config = ArabicWebConfig()
    
    print("=" * 70)
    print("Processing Arabic Web Corpus")
    print("=" * 70)
    print()
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ArabicWebExampleGenerator(config)
    
    # Collect all examples
    all_examples = {
        "qa": [],
        "vocabulary": [],
        "topics": [],
        "rag": [],
    }
    
    # Find all text files
    text_files = list(config.INPUT_DIR.rglob("*.txt"))
    json_files = list(config.INPUT_DIR.rglob("*.json"))
    jsonl_files = list(config.INPUT_DIR.rglob("*.jsonl"))
    
    all_files = text_files + json_files + jsonl_files
    
    print(f"Found {len(all_files)} files in Arabic web corpus")
    print()
    
    # Process each file
    for i, file_path in enumerate(all_files, 1):
        print(f"Processing file {i}/{len(all_files)}: {file_path.name}")
        
        try:
            # Load text
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, ensure_ascii=False)
            elif file_path.suffix == '.jsonl':
                text = ""
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        text += line + "\n"
            else:
                continue
            
            # Generate examples
            if len(text) < config.MIN_LENGTH:
                continue
            
            # Generate different types
            qa_ex = generator.generate_general_qa(text)
            if qa_ex and len(all_examples["qa"]) < config.TARGET_GENERAL_QA:
                all_examples["qa"].append(qa_ex)
            
            vocab_ex = generator.generate_vocabulary_example(text)
            if vocab_ex and len(all_examples["vocabulary"]) < config.TARGET_VOCABULARY:
                all_examples["vocabulary"].append(vocab_ex)
            
            topic_ex = generator.generate_topic_example(text)
            if topic_ex and len(all_examples["topics"]) < config.TARGET_TOPICS:
                all_examples["topics"].append(topic_ex)
            
            rag_ex = generator.generate_rag_example(text)
            if rag_ex and len(all_examples["rag"]) < config.TARGET_RAG:
                all_examples["rag"].append(rag_ex)
            
            # Check if we have enough
            total = sum(len(exs) for exs in all_examples.values())
            if total >= (config.TARGET_GENERAL_QA + config.TARGET_VOCABULARY + 
                        config.TARGET_TOPICS + config.TARGET_RAG):
                print("Reached target example count")
                break
        
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    # Save examples
    print()
    print("Saving examples...")
    
    total_saved = 0
    for example_type, examples in all_examples.items():
        if examples:
            output_file = config.OUTPUT_DIR / f"arabic_web_{example_type}.jsonl"
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
        merged_file = config.OUTPUT_DIR / "arabic_web_all.jsonl"
        with open(merged_file, 'w', encoding='utf-8') as f:
            for ex in all_examples_merged:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"  ✅ Merged: {len(all_examples_merged):,} examples → {merged_file}")
    
    print()
    print("=" * 70)
    print(f"Processing Complete! Total: {total_saved:,} examples")
    print("=" * 70)


if __name__ == "__main__":
    process_arabic_web()

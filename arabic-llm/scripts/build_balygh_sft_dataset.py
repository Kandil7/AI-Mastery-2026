"""
Build Balygh SFT Dataset from Books

Converts extracted books (Shamela corpus) into Supervised Fine-Tuning (SFT) examples
for Arabic fiqh, linguistics (nahw/balagha), and Islamic sciences.

Based on llm_arabic_plan.md implementation plan (lines 8000-9866)

Usage:
    python build_balygh_sft_dataset.py --target-examples 5000
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Iterable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Default paths
BOOKS_DIR = Path("datasets/extracted_books")
META_PATH = Path("datasets/metadata/books.json")
OUT_PATH = Path("datasets/jsonl/balygh_sft_from_books.jsonl")
LOG_PATH = Path("logs/build_dataset.log")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingExample:
    """Single SFT training example"""
    id: str
    instruction: str
    input: str
    output: str
    role: str
    skills: List[str]
    level: str
    domain: str
    style: str
    task_type: str
    difficulty: int
    source: str
    book_title: str
    book_category: str
    book_id: Optional[int] = None
    quality_score: float = 0.0
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii)


# =============================================================================
# Utilities
# =============================================================================

def load_books_meta() -> Dict[int, Dict]:
    """Load book metadata from JSON file"""
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return {int(b["id"]): b for b in meta}


def read_book_text(book_id: int) -> str:
    """Read book text from file"""
    path = BOOKS_DIR / f"{book_id}.txt"
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_paragraphs(text: str, min_len: int = 150, max_len: int = 1200) -> List[str]:
    """
    Split text into paragraphs for SFT examples.
    
    Args:
        text: Input text
        min_len: Minimum paragraph length
        max_len: Maximum paragraph length (longer ones will be chunked)
    
    Returns:
        List of paragraph strings
    """
    paras = re.split(r"\n\s*\n", text)
    clean = []
    
    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        
        if len(p) < min_len:
            continue
        
        if len(p) > max_len:
            # Chunk long paragraphs
            for i in range(0, len(p), max_len):
                chunk = p[i:i+max_len].strip()
                if len(chunk) >= min_len:
                    clean.append(chunk)
        else:
            clean.append(p)
    
    return clean


def is_fiqh_category(cat: str) -> bool:
    """Check if category is fiqh-related"""
    if not cat:
        return False
    cat = cat.strip()
    keywords = ["فقه", "عبادات", "معاملات", "فتاوى", "فقهية", "أحكام", "حلال", "حرام"]
    return any(k in cat for k in keywords)


def is_hadith_category(cat: str) -> bool:
    """Check if category is hadith-related"""
    if not cat:
        return False
    cat = cat.strip()
    keywords = ["حديث", "سنة", "أحاديث", "صحيح", "سنن", "مسند"]
    return any(k in cat for k in keywords)


def is_tafsir_category(cat: str) -> bool:
    """Check if category is tafsir-related"""
    if not cat:
        return False
    cat = cat.strip()
    keywords = ["تفسير", "قرآن", "آيات", "تنزيل", "تأويل"]
    return any(k in cat for k in keywords)


def is_language_category(cat: str) -> bool:
    """Check if category is language-related"""
    if not cat:
        return False
    cat = cat.strip()
    keywords = ["نحو", "لغة", "إعراب", "بلاغة", "بيان", "بديع", "صرف", "معاجم"]
    return any(k in cat for k in keywords)


def is_literature_category(cat: str) -> bool:
    """Check if category is literature-related"""
    if not cat:
        return False
    cat = cat.strip()
    keywords = ["أدب", "شعر", "نقد", "تراجم", "طبقات", "تاريخ"]
    return any(k in cat for k in keywords)


# =============================================================================
# Template Generators
# =============================================================================

def gen_fiqh_examples(paragraph: str, book_meta: Dict, max_examples: int = 2) -> List[TrainingExample]:
    """
    Convert fiqh paragraph into 1-2 SFT examples.
    Uses paragraph as reference text for safe fatwa responses.
    """
    examples = []
    title = book_meta.get("title", "").strip()
    cat = book_meta.get("cat_name", "").strip()
    book_id = book_meta.get("id")
    
    # Example 1: Summarize the fiqh issue
    instruction = (
        "بناءً على النص الفقهي التالي، لخّص المسألة التي يتحدّث عنها المؤلف، "
        "واذكر حكمها بإيجاز مع بيان إن كان فيه خلاف بين المذاهب.\n\n"
        "لا تُضِفْ شيئًا من عندك خارج ما في النص قدر الإمكان."
    )
    
    output_stub = (
        "المسألة كما وردت في النص: [...]\n\n"
        "الحكم كما ذكره المؤلف: [...]\n\n"
        "الخلاف بين الفقهاء (إن ذُكر في النص): [...]\n\n"
        "ملاحظة: هذه الصياغة مبنية على النص المعطى فقط، "
        "وللفتوى في حالة معيّنة يُرجى مراجعة دار الإفتاء المختصة."
    )
    
    ex1 = TrainingExample(
        id=f"fiqh-summary-{book_id}-{len(examples)}",
        instruction=instruction,
        input=paragraph,
        output=output_stub,
        role="fatwa_assistant_safe",
        skills=["fiqh", "fatwa", "rag_grounded_answering"],
        level="intermediate",
        domain="islamic_studies",
        style="fusha_modern",
        task_type="qa",
        difficulty=3,
        source=f"book:{book_id}",
        book_title=title,
        book_category=cat,
        book_id=book_id,
        quality_score=0.0
    )
    examples.append(ex1)
    
    # Example 2: Compare madhhab opinions (if applicable)
    if len(paragraph) > 300:
        instruction2 = (
            "استخرج من النص الفقهي التالي أقوال المذاهب الفقهية المذكورة، "
            "وقارن بينها في جدول منظم مع ذكر الأدلة إن وُجدت."
        )
        
        output_stub2 = (
            "جدول أقوال المذاهب:\n"
            "| المذهب | القول | الدليل |\n"
            "|---------|-------|--------|\n"
            "| ... | ... | ... |\n\n"
            "الراجح من الأقوال (إن ذُكر): [...]"
        )
        
        ex2 = TrainingExample(
            id=f"fiqh-comparison-{book_id}-{len(examples)}",
            instruction=instruction2,
            input=paragraph,
            output=output_stub2,
            role="faqih",
            skills=["fiqh", "comparative_fiqh"],
            level="advanced",
            domain="islamic_studies",
            style="fusha_classical",
            task_type="comparison",
            difficulty=4,
            source=f"book:{book_id}",
            book_title=title,
            book_category=cat,
            book_id=book_id,
            quality_score=0.0
        )
        examples.append(ex2)
    
    return examples[:max_examples]


def gen_nahw_examples(paragraph: str, book_meta: Dict, max_examples: int = 2) -> List[TrainingExample]:
    """
    Extract sentences from paragraph and create i'rab (grammar parsing) examples.
    """
    examples = []
    title = book_meta.get("title", "").strip()
    cat = book_meta.get("cat_name", "").strip()
    book_id = book_meta.get("id")
    
    # Extract sentences
    sentences = re.split(r"[.!؟\n]", paragraph)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15 and len(s.strip()) < 200]
    
    if not sentences:
        return []
    
    # Select best sentence (medium length, Arabic)
    sentence = max(sentences, key=len)[:150]
    
    # Check Arabic ratio
    arabic_chars = sum(1 for c in sentence if '\u0600' <= c <= '\u06FF')
    if arabic_chars / len(sentence) < 0.7:
        return []
    
    # Example 1: I'rab
    instruction = "أعرب الجملة التالية إعرابًا تفصيليًا، ثم استخرج القاعدة النحوية الأساسية وشرحها بإيجاز."
    
    output_stub = (
        "الإعراب:\n"
        "- ...\n\n"
        "القاعدة النحوية المستفادة:\n"
        "- ...\n\n"
        "شواهد أخرى على القاعدة:\n"
        "- ..."
    )
    
    ex1 = TrainingExample(
        id=f"nahw-ierab-{book_id}-{len(examples)}",
        instruction=instruction,
        input=sentence,
        output=output_stub,
        role="tutor",
        skills=["nahw"],
        level="intermediate",
        domain="linguistics",
        style="fusha_classical",
        task_type="explanation",
        difficulty=3,
        source=f"book:{book_id}",
        book_title=title,
        book_category=cat,
        book_id=book_id,
        quality_score=0.0
    )
    examples.append(ex1)
    
    return examples[:max_examples]


def gen_balagha_examples(paragraph: str, book_meta: Dict) -> List[TrainingExample]:
    """
    Extract rhetorical devices from paragraph.
    """
    examples = []
    title = book_meta.get("title", "").strip()
    cat = book_meta.get("cat_name", "").strip()
    book_id = book_meta.get("id")
    
    instruction = (
        "من النص التالي، استخرج مثالًا واحدًا على صورة بلاغية "
        "(تشبيه أو استعارة أو كناية)، ثم بيّن نوعها ووجه الشبه أو العلاقة بإيجاز."
    )
    
    output_stub = (
        "الصورة البلاغية المستخرجة: [...]\n"
        "نوعها: تشبيه/استعارة/كناية.\n"
        "وجه الشبه أو العلاقة: [...]\n"
        "نوع التشبيه/الاستعارة: [...]\n"
        "الأثر البلاغي: [...]"
    )
    
    ex = TrainingExample(
        id=f"balagha-figure-{book_id}-{len(examples)}",
        instruction=instruction,
        input=paragraph,
        output=output_stub,
        role="tutor",
        skills=["balagha"],
        level="intermediate",
        domain="linguistics",
        style="fusha_classical",
        task_type="explanation",
        difficulty=3,
        source=f"book:{book_id}",
        book_title=title,
        book_category=cat,
        book_id=book_id,
        quality_score=0.0
    )
    examples.append(ex)
    
    return examples


def gen_hadith_examples(paragraph: str, book_meta: Dict) -> List[TrainingExample]:
    """
    Create hadith takhreej and analysis examples.
    """
    examples = []
    title = book_meta.get("title", "").strip()
    cat = book_meta.get("cat_name", "").strip()
    book_id = book_meta.get("id")
    
    # Check if paragraph contains hadith markers
    hadith_markers = ["حدثنا", "أخبرنا", "قال رسول الله", "عن النبي", "ﷺ"]
    
    if not any(marker in paragraph for marker in hadith_markers):
        return []
    
    instruction = (
        "خرّج الحديث التالي، وبيّن درجته، واشرح معاني الألفاظ الغريبة فيه."
    )
    
    output_stub = (
        "التخريج:\n"
        "- المخرج: [...]\n"
        "- الكتاب: [...]\n"
        "- الرقم: [...]\n"
        "- درجة الحديث: [صحيح/حسن/ضعيف]\n\n"
        "شرح الألفاظ:\n"
        "- [...]: [...]\n\n"
        "الفوائد المستنبطة من الحديث:\n"
        "- ..."
    )
    
    ex = TrainingExample(
        id=f"hadith-takhreej-{book_id}-{len(examples)}",
        instruction=instruction,
        input=paragraph,
        output=output_stub,
        role="muhaddith",
        skills=["hadith", "hadith_mustalah"],
        level="advanced",
        domain="islamic_studies",
        style="fusha_classical",
        task_type="explanation",
        difficulty=4,
        source=f"book:{book_id}",
        book_title=title,
        book_category=cat,
        book_id=book_id,
        quality_score=0.0
    )
    examples.append(ex)
    
    return examples


def gen_tafsir_examples(paragraph: str, book_meta: Dict) -> List[TrainingExample]:
    """
    Create tafsir examples from Quranic exegesis paragraphs.
    """
    examples = []
    title = book_meta.get("title", "").strip()
    cat = book_meta.get("cat_name", "").strip()
    book_id = book_meta.get("id")
    
    # Check for Quranic verse markers
    quran_markers = ["قوله تعالى", "قال الله", "آية", "سورة"]
    
    if not any(marker in paragraph for marker in quran_markers):
        return []
    
    instruction = (
        "فسّر الآية الكريمة المذكورة في النص، مبيّنًا المعنى الإجمالي، "
        "أسباب النزول إن وُجدت، والفوائد المستنبطة."
    )
    
    output_stub = (
        "الآية الكريمة: [...]\n\n"
        "التفسير الإجمالي: [...]\n\n"
        "أسباب النزول (إن ذُكرت): [...]\n\n"
        "الفوائد والعبر:\n"
        "- ...\n\n"
        "الأحكام الفقهية (إن وُجدت): [...]"
    )
    
    ex = TrainingExample(
        id=f"tafsir-verse-{book_id}-{len(examples)}",
        instruction=instruction,
        input=paragraph,
        output=output_stub,
        role="mufassir",
        skills=["tafsir", "quran_sciences"],
        level="advanced",
        domain="islamic_studies",
        style="fusha_classical",
        task_type="explanation",
        difficulty=4,
        source=f"book:{book_id}",
        book_title=title,
        book_category=cat,
        book_id=book_id,
        quality_score=0.0
    )
    examples.append(ex)
    
    return examples


def gen_literature_examples(paragraph: str, book_meta: Dict) -> List[TrainingExample]:
    """
    Create literature and poetry analysis examples.
    """
    examples = []
    title = book_meta.get("title", "").strip()
    cat = book_meta.get("cat_name", "").strip()
    book_id = book_meta.get("id")
    
    # Check for poetry markers
    poetry_markers = ["قال", "بيت", "شعر", "قصيدة"]
    
    if not any(marker in paragraph for marker in poetry_markers):
        return []
    
    instruction = (
        "حلّل النص الأدبي التالي تحليلًا أدبيًا، مبيّنًا الصور البلاغية، "
        "الأفكار الرئيسية، والخصائص الأسلوبية."
    )
    
    output_stub = (
        "الأفكار الرئيسية:\n"
        "1. [...]\n\n"
        "الصور البلاغية:\n"
        "- [...]: [...]\n\n"
        "الخصائص الأسلوبية:\n"
        "- [...]\n\n"
        "النقد الأدبي:\n"
        "- نقاط القوة: [...]\n"
        "- نقاط الضعف: [...]"
    )
    
    ex = TrainingExample(
        id=f"literature-analysis-{book_id}-{len(examples)}",
        instruction=instruction,
        input=paragraph,
        output=output_stub,
        role="adab_specialist",
        skills=["adab", "literary_criticism", "balagha"],
        level="advanced",
        domain="literature",
        style="fusha_classical",
        task_type="analysis",
        difficulty=4,
        source=f"book:{book_id}",
        book_title=title,
        book_category=cat,
        book_id=book_id,
        quality_score=0.0
    )
    examples.append(ex)
    
    return examples


# =============================================================================
# Main Pipeline
# =============================================================================

def iterate_core_books(max_books: int = 200) -> Iterable[TrainingExample]:
    """
    Iterate through core books (fiqh, language, hadith, tafsir) and generate SFT examples.
    """
    meta = load_books_meta()
    book_ids = sorted(meta.keys())
    
    count = 0
    for bid in book_ids[:max_books]:
        book_meta = meta[bid]
        cat = book_meta.get("cat_name", "")
        
        text = read_book_text(bid)
        if not text or len(text) < 1000:
            continue
        
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            continue
        
        # Generate examples based on category
        if is_fiqh_category(cat):
            for p in paragraphs[:50]:  # Max 50 paragraphs per book
                for ex in gen_fiqh_examples(p, book_meta):
                    yield ex
                    count += 1
        
        elif is_hadith_category(cat):
            for p in paragraphs[:50]:
                for ex in gen_hadith_examples(p, book_meta):
                    yield ex
                    count += 1
        
        elif is_tafsir_category(cat):
            for p in paragraphs[:50]:
                for ex in gen_tafsir_examples(p, book_meta):
                    yield ex
                    count += 1
        
        elif is_language_category(cat):
            for p in paragraphs[:50]:
                for ex in gen_nahw_examples(p, book_meta):
                    yield ex
                    count += 1
                for ex in gen_balagha_examples(p, book_meta):
                    yield ex
                    count += 1
        
        elif is_literature_category(cat):
            for p in paragraphs[:30]:
                for ex in gen_literature_examples(p, book_meta):
                    yield ex
                    count += 1
        
        # Progress logging
        if count % 500 == 0:
            print(f"Generated {count} examples from book {bid}...")


def main(target_examples: int = 5000, max_books: int = 200):
    """
    Main function to generate SFT dataset from books.
    
    Args:
        target_examples: Target number of examples to generate
        max_books: Maximum number of books to process
    """
    print(f"📚 Building Balygh SFT Dataset from Books")
    print(f"   Target: {target_examples} examples")
    print(f"   Max books: {max_books}")
    print(f"   Output: {OUT_PATH}")
    print()
    
    count = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for ex in iterate_core_books(max_books):
            out.write(ex.to_json() + "\n")
            count += 1
            
            if count >= target_examples:
                break
    
    print(f"✅ Generated {count} raw SFT examples at {OUT_PATH}")
    print()
    print("Next steps:")
    print("1. Review samples manually")
    print("2. Run refine_balygh_sft_with_llm.py to improve output quality")
    print("3. Merge with other datasets for training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Balygh SFT Dataset from Books")
    parser.add_argument("--target-examples", type=int, default=5000, help="Target number of examples")
    parser.add_argument("--max-books", type=int, default=200, help="Maximum number of books to process")
    
    args = parser.parse_args()
    
    main(target_examples=args.target_examples, max_books=args.max_books)

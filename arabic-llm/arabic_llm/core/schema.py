"""
Arabic LLM Fine-Tuning Data Schema - Balygh (بليغ)

This module defines the JSONL schema for Arabic language model training data,
following the specification from llm_arabic_plan.md

Schema supports 29 roles across 5 categories:
- Linguistic (5): tutor, proofreader, poet, muhhaqiq, assistant_general
- Islamic Sciences (10): faqih, muhaddith, mufassir, aqeedah_specialist, sufi, etc.
- Modern/Tech (5): rag_assistant, edtech_tutor, dataengineer_ar, etc.
- Specialized Knowledge (6): historian, genealogist, geographer, physician, etc.
- Dialect & Language (3): dialect_handling_egy, translator_ar, summarizer_ar

And 76 skills covering:
- Linguistic sciences (nahw, sarf, balagha, etc.)
- Islamic sciences (fiqh, hadith, tafsir, aqeedah, etc.)
- Technical skills (RAG, NER, summarization, etc.)
- Dialect handling (Egyptian, Gulf, Levantine)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum
import json
import hashlib
import uuid
from datetime import datetime


# ============================================================================
# ROLE ENUMERATION - 29 Roles Total
# ============================================================================

class Role(Enum):
    """
    Training example roles - 29 total roles across 5 categories.
    
    Categories:
    1. Core Linguistic (5): Basic Arabic language tasks
    2. Islamic Sciences (10): Religious and Islamic studies
    3. Modern/Tech (5): AI, RAG, education technology
    4. Specialized Knowledge (6): History, medicine, logic, etc.
    5. Dialect & Language (3): Dialect handling and translation
    """
    
    # ────────────────────────────────────────────────────────────────────────
    # Core Linguistic Roles (5) - الأدوار اللغوية الأساسية
    # ────────────────────────────────────────────────────────────────────────
    TUTOR = "tutor"  # معلم اللغة العربية - Arabic language teacher
    PROOFREADER = "proofreader"  # المصحح اللغوي - Grammar corrector
    POET = "poet"  # الشاعر والناقد الأدبي - Poet and literary critic
    MUHHAQIQ = "muhhaqiq"  # المحقق اللغوي - Text investigator
    ASSISTANT_GENERAL = "assistant_general"  # المساعد العام - General assistant
    
    # ────────────────────────────────────────────────────────────────────────
    # Islamic Sciences Roles (10) - علوم شرعية
    # ────────────────────────────────────────────────────────────────────────
    FAQIH = "faqih"  # الفقيه - Islamic jurist
    MUHADDITH = "muhaddith"  # المحدث - Hadith scholar
    MUFASSIR = "mufassir"  # المفسر - Quranic exegete
    AQEEDAH_SPECIALIST = "aqeedah_specialist"  # متخصص العقيدة - Creed specialist
    SUFI = "sufi"  # الصوفي - Sufism scholar
    HISTORIAN = "historian"  # المؤرخ - Islamic historian
    GENEALOGIST = "genealogist"  # النسّاب - Genealogist
    GEOGRAPHER = "geographer"  # الجغرافي - Historical geographer
    PHYSICIAN = "physician"  # الطبيب - Classical Islamic physician
    LOGICIAN = "logician"  # المنطقي - Logic scholar
    
    # ────────────────────────────────────────────────────────────────────────
    # Modern/Tech Roles (5) - أدوار حديثة وتقنية
    # ────────────────────────────────────────────────────────────────────────
    RAG_ASSISTANT = "rag_assistant"  # مساعد RAG - RAG-based assistant
    EDTECH_TUTOR = "edtech_tutor"  # المعلم التقني - Educational technology tutor
    DATAENGINEER_AR = "dataengineer_ar"  # مهندس البيانات العربي - Arabic data engineer
    FATWA_ASSISTANT_SAFE = "fatwa_assistant_safe"  # مساعد الفتوى الآمن - Safe fatwa assistant
    TOOL_CALLER_AR = "tool_caller_ar"  # مستدعي الأدوات - Tool/function caller
    
    # ────────────────────────────────────────────────────────────────────────
    # Literature & Specialized (3) - الأدب والتخصصات
    # ────────────────────────────────────────────────────────────────────────
    ADAB_SPECIALIST = "adab_specialist"  # متخصص الأدب - Arabic literature specialist
    QURAN_RECITER = "quran_reciter"  # القارئ - Quran reciter
    LEGAL_ARABIC_DRAFTING = "legal_arabic_drafting"  # الصياغة القانونية - Legal drafter
    
    # ────────────────────────────────────────────────────────────────────────
    # Dialect & Language Services (3) - اللهجات وخدمات اللغة
    # ────────────────────────────────────────────────────────────────────────
    DIALECT_HANDLING_EGY = "dialect_handling_egy"  # معالجة اللهجة المصرية - Egyptian dialect
    SUMMARIZER_AR = "summarizer_ar"  # الملخّص العربي - Arabic summarizer
    TRANSLATOR_AR = "translator_ar"  # المترجم - Arabic translator


# ============================================================================
# SKILL ENUMERATION - 76 Skills Total
# ============================================================================

class Skill(Enum):
    """
    Linguistic and domain skills - 76 total skills across 8 categories.
    
    Categories:
    1. Core Linguistic (8): Arabic grammar, morphology, rhetoric
    2. Islamic Sciences (15): Fiqh, hadith, tafsir, aqeedah, etc.
    3. Literature & Heritage (5): Poetry, adab, classical texts
    4. Modern NLP/Tech (12): RAG, NER, summarization, etc.
    5. Dialect Handling (5): Egyptian, Gulf, Levantine dialects
    6. Extended Islamic (8): Quran sciences, maqasid, seerah, etc.
    7. Utility Skills (10): Citation, parsing, QA generation, etc.
    8. Specialized Domains (5): Medical, legal, comparative religions
    """
    
    # ────────────────────────────────────────────────────────────────────────
    # Core Linguistic Skills (8) - المهارات اللغوية الأساسية
    # ────────────────────────────────────────────────────────────────────────
    NAHW = "nahw"  # النحو - Arabic grammar
    SARF = "sarf"  # الصرف - Morphology
    BALAGHA = "balagha"  # البلاغة - Rhetoric
    ORTHOGRAPHY = "orthography"  # الإملاء - Spelling
    PHONOLOGY = "phonology"  # الأصوات اللغوية - Phonology
    SEMANTICS = "semantics"  # الدلالة - Semantics
    LEXICOGRAPHY = "lexicography"  # المعاجم - Lexicography
    QIRAAT = "qiraat"  # القراءات - Quranic recitations
    
    # ────────────────────────────────────────────────────────────────────────
    # Islamic Sciences Skills (15) - العلوم الشرعية
    # ────────────────────────────────────────────────────────────────────────
    FIQH = "fiqh"  # الفقه - Islamic jurisprudence
    USUL_FIQH = "usul_fiqh"  # أصول الفقه - Principles of jurisprudence
    HADITH = "hadith"  # الحديث - Prophetic traditions
    HADITH_MUSTALAH = "hadith_mustalah"  # مصطلح الحديث - Hadith terminology
    TAFSIR = "tafsir"  # التفسير - Quranic exegesis
    AQEEDAH = "aqeedah"  # العقيدة - Islamic creed
    SECTS = "sects"  # الفرق الإسلامية - Islamic sects
    TASAWWUF = "tasawwuf"  # التصوف - Sufism
    ZAKAT = "zakat"  # الزكاة - Islamic alms
    INHERITANCE = "inheritance"  # المواريث - Islamic inheritance
    FATWA = "fatwa"  # الفتوى - Islamic legal rulings
    JUDICIAL = "judicial"  # القضاء - Islamic judiciary
    SEERAH = "seerah"  # السيرة - Prophet's biography
    QURAN_SCIENCES = "quran_sciences"  # علوم القرآن - Quranic sciences
    COMPARATIVE_FIQH = "comparative_fiqh"  # الفقه المقارن - Comparative fiqh
    
    # ────────────────────────────────────────────────────────────────────────
    # Literature & Heritage Skills (5) - الأدب والتراث
    # ────────────────────────────────────────────────────────────────────────
    POETRY = "poetry"  # الشعر - Poetry composition and criticism
    HERITAGE = "heritage"  # التراث - Classical Islamic heritage
    ADAB = "adab"  # الأدب - Arabic literature
    MANUSCRIPTS = "arabic_manuscripts"  # المخطوطات - Arabic manuscripts
    LITERARY_CRITICISM = "literary_criticism"  # النقد الأدبي - Literary criticism
    
    # ────────────────────────────────────────────────────────────────────────
    # Modern NLP & Tech Skills (12) - المعالجة اللغوية والتقنية
    # ────────────────────────────────────────────────────────────────────────
    RAG_RETRIEVAL = "rag_retrieval"  # استرجاع RAG - RAG retrieval
    RAG_GROUNDED_ANSWERING = "rag_grounded_answering"  # الإجابة المستندة - Grounded answering
    FUNCTION_CALLING_AR = "function_calling_ar"  # استدعاء الدوال - Function calling
    SUMMARIZATION = "summarization"  # التلخيص - Text summarization
    TEXT_CLASSIFICATION = "text_classification"  # تصنيف النصوص - Text classification
    NAMED_ENTITY_AR = "named_entity_ar"  # الكيانات المسماة - Arabic NER
    SENTIMENT_AR = "sentiment_ar"  # تحليل المشاعر - Sentiment analysis
    TRANSLATION_AR_EN = "translation_ar_en"  # الترجمة - Arabic-English translation
    ASSESSMENT_DESIGN = "assessment_design"  # تصميم التقييمات - Assessment design
    CURRICULUM_ALIGNED_AR = "curriculum_aligned_ar"  # مناهج تعليمية - Curriculum alignment
    STRUCTURED_OUTPUT_AR = "structured_output_ar"  # مخرجات منظمة - Structured output
    DATA_STRUCTURING = "data_structuring"  # هيكلة البيانات - Data structuring
    
    # ────────────────────────────────────────────────────────────────────────
    # Dialect Handling Skills (5) - معالجة اللهجات
    # ────────────────────────────────────────────────────────────────────────
    DIALECT_EGY = "dialect_egy"  # اللهجة المصرية - Egyptian Arabic
    DIALECT_GLF = "dialect_glf"  # اللهجة الخليجية - Gulf Arabic
    DIALECT_LEV = "dialect_lev"  # اللهجة الشامية - Levantine Arabic
    DIALECT_MSA = "dialect_msa"  # العربية الفصحى الحديثة - Modern Standard Arabic
    TRANSLITERATION = "transliteration"  # النقل الحرفي - Transliteration
    
    # ────────────────────────────────────────────────────────────────────────
    # Extended Islamic Skills (8) - علوم إسلامية موسعة
    # ────────────────────────────────────────────────────────────────────────
    MAQASID_SHARIAH = "maqasid_shariah"  # مقاصد الشريعة - Maqasid al-Shariah
    COMPARATIVE_RELIGIONS = "comparative_religions"  # الأديان المقارنة - Comparative religions
    ISLAMIC_HISTORY = "islamic_history"  # التاريخ الإسلامي - Islamic history
    ISLAMIC_CIVILIZATION = "islamic_civilization"  # الحضارة الإسلامية - Islamic civilization
    ARABIC_GEOGRAPHY = "arabic_geography"  # الجغرافيا العربية - Arabic geography
    ISLAMIC_MEDICINE = "islamic_medicine"  # الطب الإسلامي - Islamic medicine
    ISLAMIC_PHILOSOPHY = "islamic_philosophy"  # الفلسفة الإسلامية - Islamic philosophy
    ISLAMIC_ECONOMICS = "islamic_economics"  # الاقتصاد الإسلامي - Islamic economics
    
    # ────────────────────────────────────────────────────────────────────────
    # Utility Skills (10) - مهارات مساعدة
    # ────────────────────────────────────────────────────────────────────────
    QA = "qa"  # أسئلة وأجوبة - Q&A
    STYLE_EDITING = "style_editing"  # تحرير الأسلوب - Style editing
    ERROR_ANALYSIS_AR = "error_analysis_ar"  # تحليل الأخطاء - Error analysis
    CITATION_EXTRACTION = "citation_extraction"  # استخراج الاستشهادات - Citation extraction
    DOCUMENT_PARSING = "document_parsing"  # تحليل الوثائق - Document parsing
    QA_GENERATION = "qa_generation"  # توليد الأسئلة - QA generation
    CONSISTENCY_CHECK = "consistency_check"  # فحص الاتساق - Consistency checking
    SIMPLIFICATION_AR = "simplification_ar"  # التبسيط اللغوي - Simplification
    EXPLANATION = "explanation"  # الشرح - Explanation
    ANALYSIS = "analysis"  # التحليل - Analysis
    
    # ────────────────────────────────────────────────────────────────────────
    # Specialized Domain Skills (5) - تخصصات دقيقة
    # ────────────────────────────────────────────────────────────────────────
    MEDICAL_ARABIC = "medical_arabic"  # العربية الطبية - Medical Arabic
    LEGAL_ARABIC = "legal_arabic"  # العربية القانونية - Legal Arabic
    BUSINESS_ARABIC = "business_arabic"  # العربية التجارية - Business Arabic
    TECHNICAL_ARABIC = "technical_arabic"  # العربية التقنية - Technical Arabic
    EDUCATIONAL_ARABIC = "educational_arabic"  # العربية التعليمية - Educational Arabic


class Level(Enum):
    """
    Proficiency levels - 4 levels for curriculum learning.
    
    Used to structure training examples from simple to complex,
    enabling progressive learning during fine-tuning.
    """
    BEGINNER = "beginner"  # مبتدئ - Basic concepts, simple tasks
    INTERMEDIATE = "intermediate"  # متوسط - Standard tasks
    ADVANCED = "advanced"  # متقدم - Complex analysis, specialized topics
    SPECIALIST = "specialist"  # متخصص - Expert-level, research-grade tasks


class Domain(Enum):
    """
    Content domains - 12 domains covering all Arabic knowledge areas.
    
    Each domain maps to specific roles and skills for targeted training.
    """
    EDUCATION = "education"  # تعليم - Educational content
    BUSINESS = "business"  # أعمال - Business Arabic
    ACADEMIC = "academic"  # أكاديمي - Academic writing
    ISLAMIC_STUDIES = "islamic_studies"  # دراسات إسلامية - Islamic sciences
    GENERAL = "general"  # عام - General knowledge
    HERITAGE = "heritage"  # تراث - Classical Islamic heritage
    LITERATURE = "literature"  # أدب - Arabic literature
    LINGUISTICS = "linguistics"  # لغويات - Linguistic sciences
    LAW = "law"  # قانون - Legal Arabic
    MEDICINE = "medicine"  # طب - Medical Arabic
    TECHNOLOGY = "technology"  # تقنية - Technical Arabic
    MEDIA = "media"  # إعلام - Media and journalism


class Style(Enum):
    """
    Language styles - 8 styles covering Arabic language registers.
    
    From classical Quranic Arabic to modern dialects.
    """
    FUSHА_CLASSICAL = "fusha_classical"  # فصحى تراثية - Classical Arabic
    FUSHА_MODERN = "fusha_modern"  # فصحى حديثة - Modern Standard Arabic
    QURANIC = "quranic"  # قرآني - Quranic style
    HADITH = "hadith"  # حديثي - Hadith style
    LITERARY = "literary"  # أدبي - Literary style
    ACADEMIC_FORMAL = "academic_formal"  # أكاديمي رسمي - Academic formal
    DIALECT_EGYPTIAN = "dialect_egyptian"  # عامية مصرية - Egyptian Arabic
    DIALECT_LEVANTINE = "dialect_levantine"  # عامية شامية - Levantine Arabic
    DIALECT_GULF = "dialect_gulf"  # عامية خليجية - Gulf Arabic
    MIXED = "mixed"  # مختلط - Mixed registers


class TaskType(Enum):
    """
    Task types - 15 task types for diverse training objectives.
    
    Covers all major NLP and educational task formats.
    """
    EXPLANATION = "explanation"  # شرح - Explain a concept
    QA = "qa"  # سؤال وجواب - Question answering
    CORRECTION = "correction"  # تصحيح - Error correction
    REWRITE = "rewrite"  # إعادة صياغة - Rewriting/paraphrasing
    ANALYSIS_AND_EXPLANATION = "analysis_and_explanation"  # تحليل وشرح - Analysis with explanation
    GENERATION = "generation"  # توليد - Text generation
    SUMMARIZATION = "summarization"  # تلخيص - Summarization
    TRANSLATION = "translation"  # ترجمة - Translation
    CLASSIFICATION = "classification"  # تصنيف - Classification
    EXTRACTION = "extraction"  # استخراج - Information extraction
    COMPARISON = "comparison"  # مقارنة - Comparison
    EVALUATION = "evaluation"  # تقييم - Evaluation/critique
    CREATION = "creation"  # إنشاء - Creative creation
    INSTRUCTION_FOLLOWING = "instruction_following"  # اتباع التعليمات - Follow complex instructions
    CHAIN_OF_THOUGHT = "chain_of_thought"  # سلسلة التفكير - Step-by-step reasoning


@dataclass
class TrainingExample:
    """
    A single training example for Arabic LLM fine-tuning.
    
    This schema is compatible with CIDAR and other Arabic instruction datasets,
    with additional metadata for role-based training.
    """
    
    # Core fields (required)
    instruction: str  # The instruction/prompt in Arabic
    input: str  # Input text/context (can be empty)
    output: str  # Expected output/response
    
    # Role and skills
    role: Role = Role.TUTOR
    skills: List[Skill] = field(default_factory=list)
    level: Level = Level.INTERMEDIATE
    
    # Context
    domain: Domain = Domain.EDUCATION
    style: Style = Style.FUSHА_CLASSICAL
    task_type: TaskType = TaskType.EXPLANATION
    
    # Metadata
    difficulty: int = 2  # 1-5 scale
    source: str = "manual"  # Source of the example
    tags: List[str] = field(default_factory=list)
    
    # Auto-generated fields
    id: Optional[str] = None
    created_at: Optional[str] = None
    book_id: Optional[int] = None  # Source book ID from Shamela
    book_title: Optional[str] = None  # Source book title
    author_id: Optional[int] = None  # Author ID
    author_name: Optional[str] = None  # Author name
    
    def __post_init__(self):
        """Generate ID and timestamp if not provided"""
        if self.id is None:
            self.id = self._generate_id()
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content"""
        content = f"{self.instruction}{self.input}{self.output}"
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:12]
        role_prefix = self.role.value[:3]
        return f"ar-{role_prefix}-{hash_hex}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "role": self.role.value,
            "skills": [s.value for s in self.skills],
            "level": self.level.value,
            "domain": self.domain.value,
            "style": self.style.value,
            "task_type": self.task_type.value,
            "difficulty": self.difficulty,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "author_id": self.author_id,
            "author_name": self.author_name,
        }
    
    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        """Create from dictionary"""
        # Convert string enums back to enum types
        data = data.copy()
        data["role"] = Role(data["role"])
        data["skills"] = [Skill(s) for s in data.get("skills", [])]
        data["level"] = Level(data.get("level", "intermediate"))
        data["domain"] = Domain(data.get("domain", "education"))
        data["style"] = Style(data.get("style", "fusha_classical"))
        data["task_type"] = TaskType(data.get("task_type", "explanation"))
        return cls(**data)


@dataclass
class DatasetConfig:
    """
    Configuration for dataset generation - Balygh (بليغ) v2.0.
    
    Supports generation of 100K training examples across 29 roles and 76 skills,
    optimized for QLoRA fine-tuning of Qwen2.5-7B-Instruct.
    
    Role Distribution Strategy:
    - High Priority (🔴): Core roles for production use (47%)
    - Medium Priority (🟡): Important specialized roles (33%)
    - Low Priority (🟢): Niche roles for breadth (20%)
    """

    # ────────────────────────────────────────────────────────────────────────
    # Role Distribution - 29 Roles (must sum to 1.0)
    # Target: 100,000 total examples
    # ────────────────────────────────────────────────────────────────────────
    role_distribution: dict = field(default_factory=lambda: {
        # 🔴 High Priority - Production Core (47%)
        "rag_assistant": 0.20,        # 20,000 examples - RAG Q&A with citations
        "tutor": 0.15,                # 15,000 examples - Language teaching
        "fatwa_assistant_safe": 0.12, # 12,000 examples - Safe Islamic rulings
        
        # 🟡 Medium Priority - Specialized (33%)
        "edtech_tutor": 0.10,         # 10,000 examples - Educational technology
        "proofreader": 0.08,          # 8,000 examples - Grammar correction
        "muhaddith": 0.06,            # 6,000 examples - Hadith sciences
        "mufassir": 0.05,             # 5,000 examples - Quranic exegesis
        "dialect_handling_egy": 0.05, # 5,000 examples - Egyptian dialect
        "faqih": 0.04,                # 4,000 examples - Islamic jurisprudence
        
        # 🟢 Low Priority - Breadth (20%)
        "summarizer_ar": 0.04,        # 4,000 examples - Summarization
        "legal_arabic_drafting": 0.03,# 3,000 examples - Legal drafting
        "translator_ar": 0.03,        # 3,000 examples - Translation
        "tool_caller_ar": 0.03,       # 3,000 examples - Function calling
        "dataengineer_ar": 0.03,      # 3,000 examples - Data structuring
        "poet": 0.02,                 # 2,000 examples - Poetry
        "muhhaqiq": 0.02,             # 2,000 examples - Text investigation
        "aqeedah_specialist": 0.02,   # 2,000 examples - Creed
        "historian": 0.02,            # 2,000 examples - Islamic history
        "adab_specialist": 0.02,      # 2,000 examples - Literature
        "sufi": 0.01,                 # 1,000 examples - Sufism
        "logician": 0.01,             # 1,000 examples - Logic
        "physician": 0.01,            # 1,000 examples - Islamic medicine
        "genealogist": 0.01,          # 1,000 examples - Genealogy
        "geographer": 0.01,           # 1,000 examples - Geography
        "quran_reciter": 0.01,        # 1,000 examples - Quran recitation
        "assistant_general": 0.01,    # 1,000 examples - General assistant
    })

    # ────────────────────────────────────────────────────────────────────────
    # Skill Distribution Within Each Role
    # Each role's skills must sum to 1.0
    # ────────────────────────────────────────────────────────────────────────
    skill_distribution: dict = field(default_factory=lambda: {
        # Core Linguistic
        "tutor": {"nahw": 0.35, "balagha": 0.25, "sarf": 0.15, "qa": 0.15, "explanation": 0.10},
        "proofreader": {"orthography": 0.35, "nahw": 0.30, "style_editing": 0.20, "error_analysis_ar": 0.15},
        "poet": {"poetry": 0.50, "balagha": 0.25, "adab": 0.15, "literary_criticism": 0.10},
        "muhhaqiq": {"heritage": 0.35, "nahw": 0.25, "balagha": 0.20, "manuscripts": 0.20},
        "assistant_general": {"qa": 0.40, "explanation": 0.30, "simplification_ar": 0.30},
        
        # Islamic Sciences
        "faqih": {"fiqh": 0.35, "usul_fiqh": 0.25, "comparative_fiqh": 0.20, "fatwa": 0.20},
        "muhaddith": {"hadith": 0.40, "hadith_mustalah": 0.30, "seerah": 0.20, "manuscripts": 0.10},
        "mufassir": {"tafsir": 0.40, "quran_sciences": 0.30, "balagha": 0.20, "semantics": 0.10},
        "aqeedah_specialist": {"aqeedah": 0.40, "sects": 0.25, "comparative_religions": 0.20, "islamic_philosophy": 0.15},
        "sufi": {"tasawwuf": 0.50, "adab": 0.25, "islamic_philosophy": 0.15, "heritage": 0.10},
        "historian": {"islamic_history": 0.40, "islamic_civilization": 0.25, "heritage": 0.20, "manuscripts": 0.15},
        "genealogist": {"islamic_history": 0.50, "heritage": 0.30, "manuscripts": 0.20},
        "geographer": {"arabic_geography": 0.45, "islamic_civilization": 0.30, "islamic_history": 0.25},
        "physician": {"islamic_medicine": 0.50, "medical_arabic": 0.30, "heritage": 0.20},
        "logician": {"islamic_philosophy": 0.40, "usul_fiqh": 0.30, "semantics": 0.20, "analysis": 0.10},
        
        # Modern/Tech
        "rag_assistant": {
            "rag_grounded_answering": 0.35,
            "rag_retrieval": 0.25,
            "citation_extraction": 0.20,
            "qa": 0.15,
            "analysis": 0.05
        },
        "edtech_tutor": {
            "curriculum_aligned_ar": 0.30,
            "assessment_design": 0.25,
            "explanation": 0.20,
            "qa_generation": 0.15,
            "simplification_ar": 0.10
        },
        "dataengineer_ar": {
            "data_structuring": 0.30,
            "structured_output_ar": 0.25,
            "named_entity_ar": 0.20,
            "extraction": 0.15,
            "document_parsing": 0.10
        },
        "fatwa_assistant_safe": {
            "fatwa": 0.35,
            "fiqh": 0.25,
            "comparative_fiqh": 0.20,
            "usul_fiqh": 0.15,
            "quran_sciences": 0.05
        },
        "tool_caller_ar": {
            "function_calling_ar": 0.50,
            "structured_output_ar": 0.30,
            "data_structuring": 0.20
        },
        
        # Literature & Specialized
        "adab_specialist": {"adab": 0.35, "literary_criticism": 0.30, "poetry": 0.20, "heritage": 0.15},
        "quran_reciter": {"qiraat": 0.50, "quran_sciences": 0.30, "phonology": 0.20},
        "legal_arabic_drafting": {
            "legal_arabic": 0.35,
            "fiqh": 0.25,
            "judicial": 0.20,
            "structured_output_ar": 0.15,
            "data_structuring": 0.05
        },
        
        # Dialect & Language
        "dialect_handling_egy": {
            "dialect_egy": 0.40,
            "dialect_msa": 0.30,
            "translation_ar_en": 0.20,
            "simplification_ar": 0.10
        },
        "summarizer_ar": {
            "summarization": 0.45,
            "analysis": 0.25,
            "simplification_ar": 0.20,
            "structured_output_ar": 0.10
        },
        "translator_ar": {
            "translation_ar_en": 0.50,
            "semantics": 0.25,
            "lexicography": 0.15,
            "dialect_msa": 0.10
        },
    })

    # ────────────────────────────────────────────────────────────────────────
    # Level Distribution - 4 Levels (must sum to 1.0)
    # Curriculum learning: start easy, progress to hard
    # ────────────────────────────────────────────────────────────────────────
    level_distribution: dict = field(default_factory=lambda: {
        "beginner": 0.25,      # 25% - Basic concepts
        "intermediate": 0.40,  # 40% - Standard tasks
        "advanced": 0.25,      # 25% - Complex analysis
        "specialist": 0.10,    # 10% - Expert-level
    })

    # ────────────────────────────────────────────────────────────────────────
    # Domain Distribution (must sum to 1.0)
    # ────────────────────────────────────────────────────────────────────────
    domain_distribution: dict = field(default_factory=lambda: {
        "islamic_studies": 0.35,   # 35% - Islamic sciences
        "education": 0.20,         # 20% - Educational content
        "heritage": 0.15,          # 15% - Classical heritage
        "literature": 0.10,        # 10% - Literature
        "linguistics": 0.08,       # 8% - Linguistic sciences
        "general": 0.05,           # 5% - General knowledge
        "law": 0.03,               # 3% - Legal Arabic
        "medicine": 0.02,          # 2% - Medical Arabic
        "technology": 0.01,        # 1% - Technical Arabic
        "business": 0.01,          # 1% - Business Arabic
    })

    # ────────────────────────────────────────────────────────────────────────
    # Target dataset size
    # Recommended: 50K-100K for QLoRA on 7B model
    # ────────────────────────────────────────────────────────────────────────
    target_examples: int = 100000

    # ────────────────────────────────────────────────────────────────────────
    # Source categories from Shamela and other datasets
    # ────────────────────────────────────────────────────────────────────────
    source_categories: List[str] = field(default_factory=lambda: [
        # Core Islamic Sciences
        "التفسير",                    # Quranic exegesis (270 books)
        "كتب السنة",                  # Hadith collections (1,226 books)
        "الفقه الحنفي",               # Hanafi fiqh
        "الفقه المالكي",              # Maliki fiqh
        "الفقه الشافعي",              # Shafi'i fiqh
        "الفقه الحنبلي",              # Hanbali fiqh
        "العقيدة",                    # Creed/aqeedah
        "التصوف",                     # Sufism
        
        # Linguistic Sciences
        "كتب اللغة",                  # Language books (~400 books)
        "النحو",                      # Grammar
        "البلاغة",                    # Rhetoric
        "التجويد والقراءات",          # Quranic recitations
        "معاجم اللغة",                # Dictionaries
        
        # Literature & Heritage
        "الأدب",                      # Literature (415 books)
        "الشعر",                      # Poetry (~200 books)
        "التراجم والطبقات",           # Biographical dictionaries
        "التاريخ",                    # History
        "الرحلات",                    # Travel literature
        
        # Specialized Sciences
        "الطب والصيدلة",              # Medicine and pharmacy
        "الفلك",                      # Astronomy
        "الرياضيات",                  # Mathematics
        "الفلسفة",                    # Philosophy
        "المنطق",                     # Logic
    ])


@dataclass
class DatasetStatistics:
    """Statistics about a generated dataset"""
    
    total_examples: int = 0
    role_counts: dict = field(default_factory=dict)
    skill_counts: dict = field(default_factory=dict)
    level_counts: dict = field(default_factory=dict)
    domain_counts: dict = field(default_factory=dict)
    source_books: int = 0
    avg_difficulty: float = 0.0
    avg_instruction_length: float = 0.0
    avg_input_length: float = 0.0
    avg_output_length: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "role_counts": self.role_counts,
            "skill_counts": self.skill_counts,
            "level_counts": self.level_counts,
            "domain_counts": self.domain_counts,
            "source_books": self.source_books,
            "avg_difficulty": round(self.avg_difficulty, 2),
            "avg_instruction_length": round(self.avg_instruction_length, 1),
            "avg_input_length": round(self.avg_input_length, 1),
            "avg_output_length": round(self.avg_output_length, 1),
        }


def validate_example(example: TrainingExample) -> List[str]:
    """
    Validate a training example and return list of errors.
    
    Validates:
    1. Required fields (instruction, output)
    2. Arabic content ratio
    3. Difficulty range (1-5)
    4. Role-skill compatibility
    5. Level validity
    6. Domain validity

    Args:
        example: TrainingExample to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Required fields
    if not example.instruction or not example.instruction.strip():
        errors.append("Instruction is required and cannot be empty")
    if not example.output or not example.output.strip():
        errors.append("Output is required and cannot be empty")
    
    # Instruction length check (should be meaningful)
    if len(example.instruction.strip()) < 10:
        errors.append("Instruction should be at least 10 characters")
    
    # Output length check (should be substantial)
    if len(example.output.strip()) < 20:
        errors.append("Output should be at least 20 characters")

    # Arabic content check (at least some Arabic characters)
    arabic_chars = sum(1 for c in example.instruction + example.output if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 10:
        errors.append("Content should contain Arabic text (minimum 10 Arabic characters)")

    # Difficulty range
    if not 1 <= example.difficulty <= 5:
        errors.append("Difficulty must be between 1 and 5")

    # Skills should match role - Comprehensive mapping for all 29 roles
    role_skill_map = {
        # Core Linguistic (5)
        "tutor": ["nahw", "balagha", "sarf", "qa", "phonology", "semantics", "orthography", "explanation", "analysis"],
        "proofreader": ["orthography", "nahw", "style_editing", "error_analysis_ar", "semantics"],
        "poet": ["poetry", "balagha", "adab", "literary_criticism", "heritage"],
        "muhhaqiq": ["heritage", "nahw", "balagha", "manuscripts", "adab", "literary_criticism"],
        "assistant_general": ["nahw", "qa", "explanation", "simplification_ar", "general"],
        
        # Islamic Sciences (10)
        "faqih": ["fiqh", "usul_fiqh", "comparative_fiqh", "fatwa", "judicial", "quran_sciences"],
        "muhaddith": ["hadith", "hadith_mustalah", "seerah", "islamic_history", "manuscripts"],
        "mufassir": ["tafsir", "quran_sciences", "balagha", "semantics", "lexicography"],
        "aqeedah_specialist": ["aqeedah", "sects", "comparative_religions", "islamic_philosophy", "logic"],
        "sufi": ["tasawwuf", "adab", "literary_criticism", "islamic_philosophy", "heritage"],
        "historian": ["islamic_history", "islamic_civilization", "heritage", "manuscripts", "arabic_geography"],
        "genealogist": ["islamic_history", "heritage", "manuscripts"],
        "geographer": ["arabic_geography", "islamic_civilization", "islamic_history"],
        "physician": ["islamic_medicine", "medical_arabic", "heritage", "manuscripts"],
        "logician": ["islamic_philosophy", "usul_fiqh", "semantics", "analysis"],
        
        # Modern/Tech (5)
        "rag_assistant": ["rag_retrieval", "rag_grounded_answering", "citation_extraction", "qa", "analysis"],
        "edtech_tutor": ["curriculum_aligned_ar", "assessment_design", "explanation", "qa_generation", "simplification_ar"],
        "dataengineer_ar": ["data_structuring", "structured_output_ar", "named_entity_ar", "extraction", "document_parsing"],
        "fatwa_assistant_safe": ["fatwa", "fiqh", "comparative_fiqh", "usul_fiqh", "quran_sciences"],
        "tool_caller_ar": ["function_calling_ar", "structured_output_ar", "data_structuring"],
        
        # Literature & Specialized (3)
        "adab_specialist": ["adab", "literary_criticism", "poetry", "heritage", "balagha"],
        "quran_reciter": ["qiraat", "quran_sciences", "phonology", "tajweed"],
        "legal_arabic_drafting": ["legal_arabic", "fiqh", "judicial", "structured_output_ar", "data_structuring"],
        
        # Dialect & Language (3)
        "dialect_handling_egy": ["dialect_egy", "dialect_msa", "translation_ar_en", "simplification_ar"],
        "summarizer_ar": ["summarization", "analysis", "simplification_ar", "structured_output_ar"],
        "translator_ar": ["translation_ar_en", "semantics", "lexicography", "dialect_msa"],
    }
    
    valid_skills = role_skill_map.get(example.role.value, [])
    for skill in example.skills:
        if skill.value not in valid_skills:
            errors.append(f"Skill '{skill.value}' not valid for role '{example.role.value}' - Valid skills: {valid_skills}")
    
    # Level validation
    valid_levels = ["beginner", "intermediate", "advanced", "specialist"]
    if example.level.value not in valid_levels:
        errors.append(f"Level must be one of: {valid_levels}")
    
    # Domain validation
    valid_domains = [
        "education", "business", "academic", "islamic_studies", "general",
        "heritage", "literature", "linguistics", "law", "medicine",
        "technology", "media"
    ]
    if example.domain.value not in valid_domains:
        errors.append(f"Domain must be one of: {valid_domains}")
    
    # Style validation
    valid_styles = [
        "fusha_classical", "fusha_modern", "quranic", "hadith",
        "literary", "academic_formal", "dialect_egyptian",
        "dialect_levantine", "dialect_gulf", "mixed"
    ]
    if example.style.value not in valid_styles:
        errors.append(f"Style must be one of: {valid_styles}")
    
    # Task type validation
    valid_task_types = [
        "explanation", "qa", "correction", "rewrite", "analysis_and_explanation",
        "generation", "summarization", "translation", "classification",
        "extraction", "comparison", "evaluation", "creation",
        "instruction_following", "chain_of_thought"
    ]
    if example.task_type.value not in valid_task_types:
        errors.append(f"Task type must be one of: {valid_task_types}")

    return errors


def write_jsonl(examples: List[TrainingExample], output_path: str) -> int:
    """
    Write training examples to JSONL file.
    
    Args:
        examples: List of TrainingExample objects
        output_path: Path to output JSONL file
        
    Returns:
        Number of examples written
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(example.to_json(ensure_ascii=False) + '\n')
    return len(examples)


def read_jsonl(input_path: str) -> List[TrainingExample]:
    """
    Read training examples from JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        List of TrainingExample objects
    """
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(TrainingExample.from_dict(data))
    return examples


def compute_statistics(examples: List[TrainingExample]) -> DatasetStatistics:
    """Compute statistics for a dataset"""
    stats = DatasetStatistics(total_examples=len(examples))
    
    if not examples:
        return stats
    
    # Count roles, skills, levels, domains
    role_counts = {}
    skill_counts = {}
    level_counts = {}
    domain_counts = {}
    source_books = set()
    total_difficulty = 0
    total_instruction_len = 0
    total_input_len = 0
    total_output_len = 0
    
    for ex in examples:
        # Role counts
        role_counts[ex.role.value] = role_counts.get(ex.role.value, 0) + 1
        
        # Skill counts
        for skill in ex.skills:
            skill_counts[skill.value] = skill_counts.get(skill.value, 0) + 1
        
        # Level counts
        level_counts[ex.level.value] = level_counts.get(ex.level.value, 0) + 1
        
        # Domain counts
        domain_counts[ex.domain.value] = domain_counts.get(ex.domain.value, 0) + 1
        
        # Source books
        if ex.book_id:
            source_books.add(ex.book_id)
        
        # Aggregations
        total_difficulty += ex.difficulty
        total_instruction_len += len(ex.instruction)
        total_input_len += len(ex.input)
        total_output_len += len(ex.output)
    
    stats.role_counts = role_counts
    stats.skill_counts = skill_counts
    stats.level_counts = level_counts
    stats.domain_counts = domain_counts
    stats.source_books = len(source_books)
    stats.avg_difficulty = total_difficulty / len(examples)
    stats.avg_instruction_length = total_instruction_len / len(examples)
    stats.avg_input_length = total_input_len / len(examples)
    stats.avg_output_length = total_output_len / len(examples)
    
    return stats


# Example usage and testing
if __name__ == "__main__":
    # Create a sample training example
    example = TrainingExample(
        instruction="أعرب الجملة التالية ثم وضّح الصورة البلاغية فيها باختصار لطالب مبتدئ.",
        input="الكتابُ صديقٌ لا يخونُ صاحِبَه.",
        output="أولاً: الإعراب: الكتابُ: مبتدأ مرفوع وعلامة رفعه الضمة الظاهرة. صديقٌ: خبر مرفوع... ثانياً: البلاغة: في تشبيه الكتاب بالصديق تشبيه بليغ يدل على الأنس...",
        role=Role.TUTOR,
        skills=[Skill.NAHW, Skill.BALAGHA],
        level=Level.BEGINNER,
        domain=Domain.EDUCATION,
        style=Style.FUSHА_MODERN,
        task_type=TaskType.ANALYSIS_AND_EXPLANATION,
        difficulty=1,
        source="synthetic_manual",
        tags=["i3rab", "tashbih"],
    )
    
    print("Sample Training Example:")
    print(example.to_json(ensure_ascii=False, indent=2))
    
    # Validate
    errors = validate_example(example)
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ Example is valid!")

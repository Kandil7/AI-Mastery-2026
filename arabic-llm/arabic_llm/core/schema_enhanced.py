"""
Enhanced Schema for Arabic LLM Fine-Tuning

Comprehensive data schema with expanded roles and skills based on 
analysis of 8,423 Shamela books across 41 categories.

This schema supports the full spectrum of Islamic and Arabic sciences
available in the dataset.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum
import json
import hashlib
from datetime import datetime


# =============================================================================
# EXPANDED ROLES (15 total)
# =============================================================================

class Role(Enum):
    """
    Comprehensive roles for Arabic LLM covering all Islamic and Arabic sciences.
    
    Primary Roles (Linguistic):
    - tutor, proofreader, poet, muhhaqiq, assistant_general
    
    Secondary Roles (Islamic Sciences):
    - faqih (jurist), muhaddith (hadith scholar), mufassir (quran exegete)
    - aqeedah_specialist (theologian), sufi (spiritual guide)
    
    Tertiary Roles (Specialized Knowledge):
    - historian, genealogist, geographer, physician, logician, adab_specialist
    """
    
    # Primary Linguistic Roles (Original 5)
    TUTOR = "tutor"                      # معلم اللغة - Language teacher
    PROOFREADER = "proofreader"          # المصحح اللغوي - Grammar corrector
    POET = "poet"                        # الشاعر - Poetry composer
    MUHHAQIQ = "muhhaqiq"                # المحقق - Text investigator
    ASSISTANT_GENERAL = "assistant_general"  # المساعد العام - General assistant
    
    # Islamic Sciences Roles (5 new)
    FAQIH = "faqih"                      # الفقيه - Jurist (Islamic law)
    MUHADDITH = "muhaddith"              # المحدث - Hadith scholar
    MUFASSIR = "mufassir"                # المفسر - Quran exegete
    AQEEDAH_SPECIALIST = "aqeedah_specialist"  # متخصص العقيدة - Theologian
    SUFI = "sufi"                        # الصوفي - Spiritual guide
    
    # Specialized Knowledge Roles (5 new)
    HISTORIAN = "historian"              # المؤرخ - Historian
    GENEALOGIST = "genealogist"          # النسّاب - Genealogist
    GEOGRAPHER = "geographer"            # الجغرافي - Geographer/traveler
    PHYSICIAN = "physician"              # الطبيب - Classical physician
    LOGICIAN = "logician"                # المنطقي - Logician
    
    # Literature & Ethics Roles (2 new)
    ADAB_SPECIALIST = "adab_specialist"  # متخصص الأدب - Literature/ethics specialist
    QURAN_RECITER = "quran_reciter"      # القارئ - Quran recitation specialist


# =============================================================================
# EXPANDED SKILLS (40+ total)
# =============================================================================

class Skill(Enum):
    """
    Comprehensive skills covering all Islamic and Arabic sciences.
    
    Organized by domain:
    - Linguistic Sciences (8)
    - Islamic Sciences (12)
    - Literature & Poetry (6)
    - Historical Sciences (5)
    - Rational Sciences (4)
    - Other Specialized (5)
    """
    
    # --- Linguistic Sciences (8) ---
    NAHW = "nahw"                          # النحو - Grammar
    SARF = "sarf"                          # الصرف - Morphology
    BALAGHA = "balagha"                    # البلاغة - Rhetoric
    ORTHOGRAPHY = "orthography"            # الإملاء - Spelling
    PHONOLOGY = "phonology"                # الأصوات - Phonetics
    SEMANTICS = "semantics"                # الدلالة - Semantics
    LEXICOGRAPHY = "lexicography"          # المعاجم - Lexicography
    QIRAAT = "qiraat"                      # القراءات - Quranic recitations
    
    # --- Islamic Sciences (12) ---
    FIQH = "fiqh"                          # الفقه - Jurisprudence
    USUL_FIQH = "usul_fiqh"                # أصول الفقه - Legal theory
    HADITH = "hadith"                      # الحديث - Prophetic traditions
    HADITH_MUSTALAH = "hadith_mustalah"   # مصطلح الحديث - Hadith terminology
    TAFSIR = "tafsir"                      # التفسير - Quranic exegesis
    AQEEDAH = "aqeedah"                    # العقيدة - Theology/creed
    SECTS = "sects"                        # الفرق - Islamic sects
    TASAWWUF = "tasawwuf"                  # التصوف - Sufism
    ZAKAT = "zakat"                        # الزكاة - Alms calculation
    INHERITANCE = "inheritance"            # الفرائض - Inheritance law
    FATWA = "fatwa"                        # الفتاوى - Legal rulings
    JUDICIAL = "judicial"                  # القضاء - Judicial system
    
    # --- Literature & Poetry (6) ---
    POETRY = "poetry"                      # الشعر - Poetry composition
    PROSODY = "prosody"                    # العروض - Poetry meters
    ADAB = "adab"                          # الأدب - Literature/ethics
    LITERARY_CRITICISM = "literary_criticism"  # النقد الأدبي
    RHETORIC_ANALYSIS = "rhetoric_analysis"  # التحليل البلاغي
    CALLIGRAPHY = "calligraphy"            # الخط - Calligraphy
    
    # --- Historical Sciences (5) ---
    HISTORY = "history"                    # التاريخ - History
    BIOGRAPHY = "biography"                # التراجم - Biography
    GENEALOGY = "genealogy"                # الأنساب - Genealogy
    GEOGRAPHY = "geography"                # الجغرافيا - Geography
    TRAVEL = "travel"                      # الرحلات - Travel literature
    
    # --- Rational Sciences (4) ---
    LOGIC = "logic"                        # المنطق - Logic
    PHILOSOPHY = "philosophy"              # الفلسفة - Philosophy
    DEBATE = "debate"                      # الجدل - Dialectics
    ARGUMENTATION = "argumentation"        # الاستدلال - Argumentation
    
    # --- Other Specialized (5) ---
    MEDICINE = "medicine"                  # الطب - Classical medicine
    QURAN_SCIENCES = "quran_sciences"      # علوم القرآن - Quranic sciences
    TAJWID = "tajwid"                      # التجويد - Quranic elocution
    QA = "qa"                              # أسئلة وأجوبة - Q&A
    STYLE_EDITING = "style_editing"        # تحرير الأسلوب - Style editing
    
    # --- Heritage & Verification (2) ---
    HERITAGE = "heritage"                  # التراث - Classical heritage
    MANUSCRIPT = "manuscript"              # المخطوطات - Manuscript studies


# =============================================================================
# EXPANDED LEVELS
# =============================================================================

class Level(Enum):
    """Proficiency levels with traditional Islamic education stages"""
    BEGINNER = "beginner"              # مبتدئ - Elementary
    INTERMEDIATE = "intermediate"      # متوسط - Intermediate
    ADVANCED = "advanced"              # متقدم - Advanced
    SPECIALIST = "specialist"          # متخصص - Specialist
    SCHOLAR = "scholar"                # عالم - Scholar level


# =============================================================================
# EXPANDED DOMAINS
# =============================================================================

class Domain(Enum):
    """Knowledge domains covering all Islamic and Arabic sciences"""
    LINGUISTICS = "linguistics"            # اللسانيات
    ISLAMIC_STUDIES = "islamic_studies"    # الدراسات الإسلامية
    LITERATURE = "literature"              # الأدب
    HISTORY = "history"                    # التاريخ
    PHILOSOPHY = "philosophy"              # الفلسفة
    SCIENCE = "science"                    # العلوم
    MEDICINE = "medicine"                  # الطب
    LAW = "law"                            # القانون
    EDUCATION = "education"                # التعليم
    HERITAGE = "heritage"                  # التراث
    GENERAL = "general"                    # عام


# =============================================================================
# EXPANDED STYLES
# =============================================================================

class Style(Enum):
    """Language and textual styles"""
    FUSHА_CLASSICAL = "fusha_classical"       # فصحى تراثية - Classical
    FUSHА_MODERN = "fusha_modern"             # فصحى حديثة - Modern Standard
    QURANIC = "quranic"                       # قرآني - Quranic
    HADITH = "hadith"                         # حديثي - Hadith style
    POETIC = "poetic"                         # شعري - Poetic
    SCHOLARLY = "scholarly"                   # علمي - Scholarly
    LEGAL = "legal"                           # قانوني - Legal
    SUFI = "sufi"                             # صوفي - Sufi style
    DIALECT_EGYPTIAN = "dialect_egyptian"     # عامية مصرية
    DIALECT_LEVANTINE = "dialect_levantine"   # عامية شامية
    DIALECT_GULF = "dialect_gulf"             # عامية خليجية
    MIXED = "mixed"                           # مختلط


# =============================================================================
# EXPANDED TASK TYPES
# =============================================================================

class TaskType(Enum):
    """Comprehensive task types for all roles"""
    
    # Educational Tasks
    EXPLANATION = "explanation"                  # شرح
    TEACHING = "teaching"                        # تعليم
    QA = "qa"                                    # سؤال وجواب
    DRILL = "drill"                              # تمرين
    ASSESSMENT = "assessment"                    # تقييم
    
    # Analytical Tasks
    ANALYSIS = "analysis"                        # تحليل
    ANALYSIS_AND_EXPLANATION = "analysis_and_explanation"  # تحليل وشرح
    COMPARISON = "comparison"                    # مقارنة
    CLASSIFICATION = "classification"            # تصنيف
    VERIFICATION = "verification"                # تحقيق
    
    # Creative Tasks
    GENERATION = "generation"                    # توليد
    COMPOSITION = "composition"                  # تأليف
    REWRITE = "rewrite"                          # إعادة صياغة
    SUMMARIZATION = "summarization"              # تلخيص
    TRANSLATION = "translation"                  # ترجمة
    
    # Correction Tasks
    CORRECTION = "correction"                    # تصحيح
    PROOFREADING = "proofreading"                # مراجعة
    EDITING = "editing"                          # تحرير
    
    # Specialized Tasks
    FATWA = "fatwa"                              # إفتاء
    JUDGMENT = "judgment"                        # حكم
    DIAGNOSIS = "diagnosis"                      # تشخيص
    PRESCRIPTION = "prescription"                # وصفة
    RECITATION = "recitation"                    # تلاوة
    MEMORIZATION = "memorization"                # حفظ


# =============================================================================
# MAIN TRAINING EXAMPLE CLASS
# =============================================================================

@dataclass
class TrainingExample:
    """
    Enhanced training example with comprehensive metadata.
    
    Supports all 15 roles and 40+ skills for complete coverage
    of Islamic and Arabic sciences.
    """
    
    # Core fields (required)
    instruction: str  # The instruction/prompt in Arabic
    input: str  # Input text/context (can be empty)
    output: str  # Expected output/response
    
    # Role and skills (expanded)
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
    
    # Book metadata (for extracted books)
    book_id: Optional[int] = None
    book_title: Optional[str] = None
    book_category: Optional[str] = None
    author_id: Optional[int] = None
    author_name: Optional[str] = None
    author_death_year: Optional[int] = None  # Hijri year
    
    # Time period classification
    time_period: Optional[str] = None  # e.g., "classical", "medieval", "ottoman"
    century_hijri: Optional[int] = None  # Islamic century
    
    # Quality markers
    verified: bool = False  # Human verified
    quality_score: float = 1.0  # 0.0-1.0 quality score
    
    # Auto-generated fields
    id: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID and timestamp if not provided"""
        if self.id is None:
            self.id = self._generate_id()
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content and role"""
        content = f"{self.role.value}:{self.instruction[:50]}:{self.input[:50]}"
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:12]
        role_prefix = self.role.value[:4]
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
            "book_id": self.book_id,
            "book_title": self.book_title,
            "book_category": self.book_category,
            "author_name": self.author_name,
            "author_death_year": self.author_death_year,
            "time_period": self.time_period,
            "verified": self.verified,
            "quality_score": self.quality_score,
            "created_at": self.created_at,
        }
    
    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        """Create from dictionary"""
        data = data.copy()
        data["role"] = Role(data["role"])
        data["skills"] = [Skill(s) for s in data.get("skills", [])]
        data["level"] = Level(data.get("level", "intermediate"))
        data["domain"] = Domain(data.get("domain", "education"))
        data["style"] = Style(data.get("style", "fusha_classical"))
        data["task_type"] = TaskType(data.get("task_type", "explanation"))
        return cls(**data)


# =============================================================================
# ROLE-SKILL MAPPING
# =============================================================================

ROLE_SKILL_MAP = {
    # Primary Linguistic Roles
    Role.TUTOR: [
        Skill.NAHW, Skill.BALAGHA, Skill.SARF, Skill.ORTHOGRAPHY,
        Skill.SEMANTICS, Skill.LEXICOGRAPHY, Skill.QA,
    ],
    Role.PROOFREADER: [
        Skill.ORTHOGRAPHY, Skill.NAHW, Skill.STYLE_EDITING,
        Skill.PHONOLOGY, Skill.BALAGHA,
    ],
    Role.POET: [
        Skill.POETRY, Skill.PROSODY, Skill.BALAGHA,
        Skill.LITERARY_CRITICISM, Skill.RHETORIC_ANALYSIS,
    ],
    Role.MUHHAQIQ: [
        Skill.HERITAGE, Skill.MANUSCRIPT, Skill.NAHW,
        Skill.BALAGHA, Skill.HADITH_MUSTALAH,
    ],
    Role.ASSISTANT_GENERAL: [
        Skill.NAHW, Skill.QA, Skill.ISLAMIC_STUDIES, Skill.ADAB,
    ],
    
    # Islamic Sciences Roles
    Role.FAQIH: [
        Skill.FIQH, Skill.USUL_FIQH, Skill.FATWA, Skill.JUDICIAL,
        Skill.INHERITANCE, Skill.ZAKAT,
    ],
    Role.MUHADDITH: [
        Skill.HADITH, Skill.HADITH_MUSTALAH, Skill.ISLAMIC_STUDIES,
        Skill.VERIFICATION, Skill.GENEALOGY,
    ],
    Role.MUFASSIR: [
        Skill.TAFSIR, Skill.QURAN_SCIENCES, Skill.QIRAAT,
        Skill.TAJWID, Skill.BALAGHA,
    ],
    Role.AQEEDAH_SPECIALIST: [
        Skill.AQEEDAH, Skill.SECTS, Skill.PHILOSOPHY,
        Skill.LOGIC, Skill.DEBATE,
    ],
    Role.SUFI: [
        Skill.TASAWWUF, Skill.ADAB, Skill.QURAN_SCIENCES,
        Skill.PHILOSOPHY,
    ],
    
    # Specialized Knowledge Roles
    Role.HISTORIAN: [
        Skill.HISTORY, Skill.BIOGRAPHY, Skill.GENEALOGY,
        Skill.GEOGRAPHY, Skill.HERITAGE,
    ],
    Role.GENEALOGIST: [
        Skill.GENEALOGY, Skill.HISTORY, Skill.BIOGRAPHY,
        Skill.ARABIC, Skill.HERITAGE,
    ],
    Role.GEOGRAPHER: [
        Skill.GEOGRAPHY, Skill.TRAVEL, Skill.HISTORY,
        Skill.CULTURE,
    ],
    Role.PHYSICIAN: [
        Skill.MEDICINE, Skill.ARABIC, Skill.PHARMACY,
        Skill.ANATOMY,
    ],
    Role.LOGICIAN: [
        Skill.LOGIC, Skill.PHILOSOPHY, Skill.ARGUMENTATION,
        Skill.DEBATE, Skill.KALAM,
    ],
    
    # Literature & Ethics Roles
    Role.ADAB_SPECIALIST: [
        Skill.ADAB, Skill.LITERARY_CRITICISM, Skill.BALAGHA,
        Skill.POETRY, Skill.RHETORIC_ANALYSIS,
    ],
    Role.QURAN_RECITER: [
        Skill.TAJWID, Skill.QIRAAT, Skill.QURAN_SCIENCES,
        Skill.PHONOLOGY,
    ],
}


# =============================================================================
# CATEGORY-ROLE MAPPING (Based on 41 Shamela categories)
# =============================================================================

CATEGORY_ROLE_MAP = {
    # Linguistic categories
    "كتب اللغة": [Role.TUTOR, Role.MUHHAQIQ, Role.ADAB_SPECIALIST],
    "النحو والصرف": [Role.TUTOR, Role.PROOFREADER, Role.MUHHAQIQ],
    "البلاغة": [Role.TUTOR, Role.POET, Role.ADAB_SPECIALIST],
    "الدواوين الشعرية": [Role.POET, Role.ADAB_SPECIALIST],
    "العروض والقوافي": [Role.POET, Role.TUTOR],
    "الغريب والمعاجم": [Role.TUTOR, Role.MUHHAQIQ],
    
    # Quranic categories
    "التفسير": [Role.MUFASSIR, Role.TUTOR],
    "علوم القرآن وأصول التفسير": [Role.MUFASSIR, Role.QURAN_RECITER],
    "التجويد والقراءات": [Role.QURAN_RECITER, Role.MUFASSIR],
    
    # Hadith categories
    "كتب السنة": [Role.MUHADDITH, Role.FAQIH],
    "شروح الحديث": [Role.MUHADDITH, Role.TUTOR],
    "التخريج والأطراف": [Role.MUHADDITH, Role.MUHHAQIQ],
    "العلل والسؤلات الحديثية": [Role.MUHADDITH, Role.MUHHAQIQ],
    "علوم الحديث": [Role.MUHADDITH, Role.TUTOR],
    
    # Fiqh categories
    "الفقه الحنفي": [Role.FAQIH],
    "الفقه المالكي": [Role.FAQIH],
    "الفقه الشافعي": [Role.FAQIH],
    "الفقه الحنبلي": [Role.FAQIH],
    "الفقه العام": [Role.FAQIH],
    "مسائل فقهية": [Role.FAQIH],
    "أصول الفقه": [Role.FAQIH],
    "علوم الفقه والقواعد الفقهية": [Role.FAQIH],
    "الفتاوى": [Role.FAQIH],
    "الفرائض والوصايا": [Role.FAQIH],
    "السياسة الشرعية والقضاء": [Role.FAQIH, Role.JUDGE],
    
    # Aqeedah categories
    "العقيدة": [Role.AQEEDAH_SPECIALIST],
    "الفرق والردود": [Role.AQEEDAH_SPECIALIST, Role.MUHADDITH],
    
    # Sufism category
    "الرقائق والآداب والأذكار": [Role.SUFI, Role.ADAB_SPECIALIST],
    
    # Historical categories
    "التاريخ": [Role.HISTORIAN],
    "التراجم والطبقات": [Role.HISTORIAN, Role.GENEALOGIST],
    "الأنساب": [Role.GENEALOGIST],
    "البلدان والرحلات": [Role.GEOGRAPHER, Role.HISTORIAN],
    "السيرة النبوية": [Role.HISTORIAN, Role.MUHADDITH],
    
    # Literature categories
    "الأدب": [Role.ADAB_SPECIALIST, Role.TUTOR],
    "الجوامع": [Role.TUTOR, Role.ADAB_SPECIALIST],
    
    # Rational sciences
    "المنطق": [Role.LOGICIAN],
    
    # Other
    "الطب": [Role.PHYSICIAN],
    "فهارس الكتب والأدلة": [Role.MUHHAQIQ, Role.LIBRARIAN],
    "كتب عامة": [Role.ASSISTANT_GENERAL],
    "علوم أخرى": [Role.TUTOR],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_role_for_category(category: str) -> List[Role]:
    """Get appropriate roles for a book category"""
    return CATEGORY_ROLE_MAP.get(category, [Role.TUTOR, Role.ASSISTANT_GENERAL])


def get_skills_for_role(role: Role) -> List[Skill]:
    """Get all skills associated with a role"""
    return ROLE_SKILL_MAP.get(role, [Skill.QA])


def validate_example(example: TrainingExample) -> List[str]:
    """
    Validate a training example and return list of errors.
    
    Enhanced validation for expanded roles and skills.
    """
    errors = []
    
    # Required fields
    if not example.instruction or not example.instruction.strip():
        errors.append("Instruction is required")
    if not example.output or not example.output.strip():
        errors.append("Output is required")
    
    # Arabic content check
    arabic_chars = sum(1 for c in example.instruction + example.output 
                       if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 10:
        errors.append("Content should contain Arabic text")
    
    # Difficulty range
    if not 1 <= example.difficulty <= 5:
        errors.append("Difficulty must be between 1 and 5")
    
    # Role-skill compatibility
    valid_skills = ROLE_SKILL_MAP.get(example.role, [])
    for skill in example.skills:
        if skill not in valid_skills:
            errors.append(f"Skill '{skill.value}' not valid for role '{example.role.value}'")
    
    # Quality score range
    if not 0.0 <= example.quality_score <= 1.0:
        errors.append("Quality score must be between 0.0 and 1.0")
    
    return errors


def compute_statistics(examples: List[TrainingExample]) -> dict:
    """Compute comprehensive statistics for a dataset"""
    if not examples:
        return {"total": 0}
    
    stats = {
        "total": len(examples),
        "by_role": {},
        "by_skill": {},
        "by_level": {},
        "by_domain": {},
        "by_category": {},
        "by_time_period": {},
        "avg_difficulty": 0,
        "avg_quality": 0,
        "verified_count": 0,
    }
    
    total_difficulty = 0
    total_quality = 0
    
    for ex in examples:
        # Role counts
        role_key = ex.role.value
        stats["by_role"][role_key] = stats["by_role"].get(role_key, 0) + 1
        
        # Skill counts
        for skill in ex.skills:
            skill_key = skill.value
            stats["by_skill"][skill_key] = stats["by_skill"].get(skill_key, 0) + 1
        
        # Level counts
        level_key = ex.level.value
        stats["by_level"][level_key] = stats["by_level"].get(level_key, 0) + 1
        
        # Domain counts
        domain_key = ex.domain.value
        stats["by_domain"][domain_key] = stats["by_domain"].get(domain_key, 0) + 1
        
        # Category counts
        if ex.book_category:
            stats["by_category"][ex.book_category] = stats["by_category"].get(ex.book_category, 0) + 1
        
        # Time period counts
        if ex.time_period:
            stats["by_time_period"][ex.time_period] = stats["by_time_period"].get(ex.time_period, 0) + 1
        
        # Aggregations
        total_difficulty += ex.difficulty
        total_quality += ex.quality_score
        if ex.verified:
            stats["verified_count"] += 1
    
    stats["avg_difficulty"] = round(total_difficulty / len(examples), 2)
    stats["avg_quality"] = round(total_quality / len(examples), 2)
    
    return stats


# =============================================================================
# EXAMPLE TEMPLATES FOR NEW ROLES
# =============================================================================

NEW_ROLE_EXAMPLES = {
    Role.FAQIH: {
        "instruction": "ما حكم الصلاة في الثوب الذي أصابته نجاسة غير معفو عنها؟",
        "output": "الحكم: لا تصح الصلاة في الثوب الذي أصابته نجاسة غير معفو عنها...",
        "skills": [Skill.FIQH],
        "level": Level.INTERMEDIATE,
    },
    Role.MUHADDITH: {
        "instruction": "بيّن درجة هذا الحديث: «إنما الأعمال بالنيات»",
        "output": "الحديث: صحيح متفق عليه. رواه البخاري ومسلم...",
        "skills": [Skill.HADITH, Skill.HADITH_MUSTALAH],
        "level": Level.ADVANCED,
    },
    Role.MUFASSIR: {
        "instruction": "فسر قوله تعالى: ﴿بسم الله الرحمن الرحيم﴾",
        "output": "البسملة: هي افتتاحية القرآن الكريم...",
        "skills": [Skill.TAFSIR, Skill.QURAN_SCIENCES],
        "level": Level.INTERMEDIATE,
    },
    Role.HISTORIAN: {
        "instruction": "متى كانت غزوة بدر وما نتائجها؟",
        "output": "غزوة بدر: وقعت في رمضان سنة 2 هـ...",
        "skills": [Skill.HISTORY],
        "level": Level.INTERMEDIATE,
    },
    Role.PHYSICIAN: {
        "instruction": "ما علاج الصداع حسب الطب النبوي؟",
        "output": "العلاج النبوي للصداع: الحجامة، الشربة بالعسل...",
        "skills": [Skill.MEDICINE],
        "level": Level.ADVANCED,
    },
}


# Export all symbols
__all__ = [
    "Role",
    "Skill",
    "Level",
    "Domain",
    "Style",
    "TaskType",
    "TrainingExample",
    "ROLE_SKILL_MAP",
    "CATEGORY_ROLE_MAP",
    "get_role_for_category",
    "get_skills_for_role",
    "validate_example",
    "compute_statistics",
]

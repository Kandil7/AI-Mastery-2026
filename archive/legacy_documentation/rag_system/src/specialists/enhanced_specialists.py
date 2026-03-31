"""
Enhanced Islamic Specialists System

Based on comprehensive analysis of 41 categories in the dataset.
Extends the base specialist system with specialized configs for every domain.

Categories from dataset:
1. العقيدة (Aqeedah)
2. الفرق والردود (Sects)
3. التفسير (Tafsir)
4. علوم القرآن وأصول التفسير (Quran Sciences)
5. التجويد والقراءات (Qiraat)
6. كتب السنة (Hadith Collections)
7. شروح الحديث (Hadith Commentaries)
8. التخريج والأطراف (Annotation)
9. العلل والسؤلات الحديثية (Hadith Issues)
10. علوم الحديث (Hadith Sciences)
11. أصول الفقه (Usul al-Fiqh)
12. علوم الفقه والقواعد الفقهية (Fiqh Principles)
13. المنطق (Logic)
14. الفقه الحنفي (Hanafi Fiqh)
15. الفقه المالكي (Maliki Fiqh)
16. الفقه الشافعي (Shafi'i Fiqh)
17. الفقه الحنبلي (Hanbali Fiqh)
18. الفقه العام (General Fiqh)
19. مسائل فقهية (Fiqh Issues)
20. السياسة الشرعية والقضاء (Islamic Politics)
21. الفرائض والوصايا (Inheritance)
22. الفتاوى (Fatwas)
23. الرقائق والآداب والأذكار (Spirituality)
24. السيرة النبوية (Prophetic Biography)
25. التاريخ (History)
26. التراجم والطبقات (Biographies)
27. الأنساب (Genealogy)
28. البلدان والرحلات (Geography)
29. كتب اللغة (Language Books)
30. الغريب والمعاجم (Lexicography)
31. النحو والصرف (Grammar)
32. الأدب (Literature)
33. العروض والقوافي (Poetry)
34. الدواوين الشعرية (Poetry Collections)
35. البلاغة (Rhetoric)
36. الجوامع (Compendiums)
37. فهارس الكتب والأدلة (Indexes)
38. الطب (Medicine)
39. كتب عامة (General)
40. علوم أخرى (Other Sciences)
"""

from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class IslamicKnowledgeDomain(Enum):
    """All 41 knowledge domains from the dataset."""

    # Core Religious Sciences (1-12)
    AQEEDAH = "aqeedah"  # العقيدة
    SECTS_AND_POLEMIC = "seects"  # الفرق والردود
    QURAN_TAFSIR = "tafsir"  # التفسير
    QURAN_SCIENCES = "quran_sciences"  # علوم القرآن
    QIRAAT_AND_TAJWEED = "qiraat"  # التجويد والقراءات
    HADITH_COLLECTIONS = "hadith_collections"  # كتب السنة
    HADITH_COMMENTARIES = "hadith_shrouh"  # شروح الحديث
    HADITH_ANNOTATIONS = "hadith_annotation"  # التخريج والأطراف
    HADITH_ISSUES = "hadith_issues"  # العلل والسؤلات
    HADITH_SCIENCES = "hadith_sciences"  # علوم الحديث
    USUL_AL_FIQH = "usul"  # أصول الفقه
    FIQH_PRINCIPLES = "fiqh_principles"  # علوم الفقه

    # Logic
    LOGIC = "logic"  # المنطق

    # Four Madhhabs (14-17)
    FIQH_HANAFI = "hanafi"  # الفقه الحنفي
    FIQH_MALIKI = "maliki"  # الفقه المالكي
    FIQH_SHAFII = "shafii"  # الفقه الشافعي
    FIQH_HANBALI = "hanbali"  # الفقه الحنبلي

    # General Fiqh (18-22)
    FIQH_GENERAL = "fiqh_general"  # الفقه العام
    FIQH_ISSUES = "fiqh_issues"  # مسائل فقهية
    ISLAMIC_POLITICS = "politics"  # السياسة الشرعية
    INHERITANCE = "inheritance"  # الفرائض
    FATWAS = "fatwas"  # الفتاوى

    # Spirituality (23-24)
    SPIRITUALITY = "spirituality"  # الرقائق
    PROPHETIC_BIOGRAPHY = "seerah"  # السيرة النبوية

    # History (25-28)
    HISTORY = "history"  # التاريخ
    BIOGRAPHIES = "biographies"  # التراجم والطبقات
    GENEALOGY = "genealogy"  # الأنساب
    GEOGRAPHY = "geography"  # البلدان والرحلات

    # Language (29-35)
    LANGUAGE_BOOKS = "language"  # كتب اللغة
    LEXICOGRAPHY = "lexicography"  # الغريب والمعاجم
    GRAMMAR = "grammar"  # النحو والصرف
    LITERATURE = "literature"  # الأدب
    POETRY_METER = "poetry"  # العروض والقوافي
    POETRY_COLLECTIONS = "diwans"  # الدواوين
    RHETORIC = "rhetoric"  # البلاغة

    # Reference (36-37)
    COMPENDIUMS = "compendiums"  # الجوامع
    INDEXES = "indexes"  # فهارس

    # Sciences (38-40)
    MEDICINE = "medicine"  # الطب
    GENERAL = "general"  # كتب عامة
    OTHER_SCIENCES = "other"  # علوم أخرى


@dataclass
class DomainSpec:
    """Complete specification for a knowledge domain."""

    domain: IslamicKnowledgeDomain
    name_arabic: str
    name_english: str

    # Dataset categories this domain covers
    categories: List[str]

    # Priority source types
    priority_sources: List[str]

    # Query patterns that trigger this domain
    trigger_keywords: List[str]

    # Chunking strategy
    chunking_strategy: str  # "semantic", "fixed", "by_chapter"
    chunk_size: int
    overlap: int

    # Domain-specific generation settings
    generation_config: Dict[str, Any]

    # Evaluation criteria
    eval_metrics: List[str]


# Complete domain specifications
DOMAIN_SPECS: Dict[IslamicKnowledgeDomain, DomainSpec] = {
    # ============ Core Religious Sciences ============
    IslamicKnowledgeDomain.AQEEDAH: DomainSpec(
        domain=IslamicKnowledgeDomain.AQEEDAH,
        name_arabic="العقيدة",
        name_english="Islamic Theology",
        categories=["العقيدة"],
        priority_sources=[
            "التوحيد - ابن تيمية",
            "العقيدة الواسطية",
            "الشرح الممتع",
            "التوحيد - محمد بن عبد الوهاب",
        ],
        trigger_keywords=[
            "توحيد",
            "عقيدة",
            "صفات الله",
            "قدر",
            "إيمان",
            "شرك",
            "كفر",
            "نفاق",
            "ريب",
            "سلف",
            "أهل سنة",
        ],
        chunking_strategy="semantic",
        chunk_size=1024,
        overlap=100,
        generation_config={
            "style": "academic",
            "evidence_required": True,
            "citation_style": "traditional",
            "avoid_innovation": True,
        },
        eval_metrics=["accuracy", "orthodoxy", "evidence_quality"],
    ),
    IslamicKnowledgeDomain.SECTS_AND_POLEMIC: DomainSpec(
        domain=IslamicKnowledgeDomain.SECTS_AND_POLEMIC,
        name_arabic="الفرق والردود",
        name_english="Islamic Sects and Polemics",
        categories=["الفرق والردود"],
        priority_sources=[
            "الفرق بين الفرق - البغدادي",
            "الملل والنحل - الشهرستاني",
            "كشف المحجوب - Hull",
        ],
        trigger_keywords=[
            "فرق",
            "مذهب",
            "شيعه",
            "خوارج",
            "معتزله",
            " phenotypic",
            "جهميه",
            "قدرية",
            "مرجئه",
            "صوفيه",
            "وهابيه",
        ],
        chunking_strategy="by_sect",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "polemical",
            "evidence_required": True,
            "neutral_tone": True,
        },
        eval_metrics=["accuracy", "comprehensiveness", "balance"],
    ),
    IslamicKnowledgeDomain.QURAN_TAFSIR: DomainSpec(
        domain=IslamicKnowledgeDomain.QURAN_TAFSIR,
        name_arabic="التفسير",
        name_english="Quranic Exegesis",
        categories=["التفسير"],
        priority_sources=[
            "تفسير ابن كثير",
            "تفسير القرطبي",
            "تفسير الطبري",
            "تفسير السعدي",
            "تفسير البياني",
        ],
        trigger_keywords=[
            "آية",
            "سورة",
            "تفسير",
            "مكي",
            "مدني",
            "ناسخ",
            "منسوخ",
            "سبب نزول",
            "ترتيب",
            "أحكام",
        ],
        chunking_strategy="by_verse",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "tafsir_classical",
            "cite_sources": True,
            "include_qiraat": True,
        },
        eval_metrics=["accuracy", "scholarly_tradition", "clarity"],
    ),
    IslamicKnowledgeDomain.QURAN_SCIENCES: DomainSpec(
        domain=IslamicKnowledgeDomain.QURAN_SCIENCES,
        name_arabic="علوم القرآن",
        name_english="Quranic Sciences",
        categories=["علوم القرآن وأصول التفسير"],
        priority_sources=[
            "مقدمة في علوم القرآن",
            "البرهان في علوم القرآن",
            "البيان في غريب القرآن",
        ],
        trigger_keywords=[
            "ناسخ",
            "منسوخ",
            "محكم",
            "متشابه",
            "عام",
            "خاص",
            "مطلق",
            "مقيد",
            " unambiguous",
            "أحكام",
            "تشريع",
        ],
        chunking_strategy="semantic",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "academic",
            "technical_terms": True,
        },
        eval_metrics=["accuracy", "technical_correctness"],
    ),
    IslamicKnowledgeDomain.QIRAAT_AND_TAJWEED: DomainSpec(
        domain=IslamicKnowledgeDomain.QIRAAT_AND_TAJWEED,
        name_arabic="القراءات والتجويد",
        name_english="Quranic Recitations",
        categories=["التجويد والقراءات"],
        priority_sources=[
            "الحجة في القراءات السبع",
            "النشر في القراءات العشر",
            "التمهيد في التجويد",
        ],
        trigger_keywords=[
            "قراءة",
            "قارئ",
            "روايت",
            " طريق",
            "خرق",
            "همزه",
            "مد",
            "قصر",
            "إدغام",
            "إخفاء",
            "إظهار",
        ],
        chunking_strategy="by_recitation",
        chunk_size=256,
        overlap=30,
        generation_config={
            "style": "technical",
            "transcription": "arabic",
            "explain_rules": True,
        },
        eval_metrics=["accuracy", "technical_precision"],
    ),
    IslamicKnowledgeDomain.HADITH_COLLECTIONS: DomainSpec(
        domain=IslamicKnowledgeDomain.HADITH_COLLECTIONS,
        name_arabic="كتب السنة",
        name_english="Hadith Collections",
        categories=["كتب السنة"],
        priority_sources=[
            "صحيح البخاري",
            "صحيح مسلم",
            "سنن الترمذي",
            "سنن أبي داود",
            "سنن النسائي",
            "سنن ابن ماجه",
            "مسند أحمد",
        ],
        trigger_keywords=[
            "حديث",
            "صحيح",
            "حسن",
            "ضعيف",
            "موضوع",
            "معلول",
            "إسناد",
            "راوي",
            "مدلس",
            "متروك",
            "كذاب",
        ],
        chunking_strategy="by_hadith",
        chunk_size=384,
        overlap=30,
        generation_config={
            "style": "hadith_scientific",
            "include_chain": True,
            "grade_required": True,
        },
        eval_metrics=["authenticity_accuracy", "grade_correctness"],
    ),
    IslamicKnowledgeDomain.HADITH_COMMENTARIES: DomainSpec(
        domain=IslamicKnowledgeDomain.HADITH_COMMENTARIES,
        name_arabic="شروح الحديث",
        name_english="Hadith Commentaries",
        categories=["شروح الحديث"],
        priority_sources=[
            "فتح الباري",
            "شرح مسلم",
            "الترغيب والترهيب",
            "رياض الصالحين",
        ],
        trigger_keywords=[
            "شرح",
            "بيان",
            "توضيح",
            "فائدة",
            "درر",
            "عقبة",
            "مشكل",
            "مشكل",
            "محكم",
        ],
        chunking_strategy="by_theme",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "commentary",
            "explain_vocabulary": True,
            "extract_lessons": True,
        },
        eval_metrics=["scholarly_depth", "practical_lessons"],
    ),
    IslamicKnowledgeDomain.HADITH_ANNOTATIONS: DomainSpec(
        domain=IslamicKnowledgeDomain.HADITH_ANNOTATIONS,
        name_arabic="التخريج والأطراف",
        name_english="Hadith Annotation and Indexing",
        categories=["التخريج والأطراف"],
        priority_sources=[
            "المحلى",
            "الدراية",
            "كشف الاستار",
        ],
        trigger_keywords=[
            "تخريج",
            "طرد",
            "بعيد",
            " غريب",
            "إسناد",
            "تابع",
            "شاب",
            "ثقة",
            "ضعيف",
        ],
        chunking_strategy="fixed",
        chunk_size=256,
        overlap=20,
        generation_config={
            "style": "technical",
            "source_attribution": True,
        },
        eval_metrics=["source_accuracy"],
    ),
    IslamicKnowledgeDomain.HADITH_ISSUES: DomainSpec(
        domain=IslamicKnowledgeDomain.HADITH_ISSUES,
        name_arabic="العلل والسؤلات",
        name_english="Hadith Issues and Questions",
        categories=["العلل والسؤلات الحديثية"],
        priority_sources=[
            "العلل - ابن أبي حاتم",
            "السؤلات - النسائي",
        ],
        trigger_keywords=[
            "علة",
            "سؤال",
            " إشكال",
            "نظر",
            "جواب",
            "شذوذ",
            "التباس",
            "اختلاف",
        ],
        chunking_strategy="by_issue",
        chunk_size=384,
        overlap=30,
        generation_config={
            "style": "qa_format",
            "evidence_based": True,
        },
        eval_metrics=["analytical_depth"],
    ),
    IslamicKnowledgeDomain.HADITH_SCIENCES: DomainSpec(
        domain=IslamicKnowledgeDomain.HADITH_SCIENCES,
        name_arabic="علوم الحديث",
        name_english="Hadith Sciences",
        categories=["علوم الحديث"],
        priority_sources=[
            "مقدمة ابن الصلاح",
            " تدريب الراوي",
            "الكفاية",
            "الخلاصة",
        ],
        trigger_keywords=[
            "تصحيح",
            "تضعيف",
            "جرح",
            "تعديل",
            "عدالة",
            "ضبط",
            "حفظ",
            "ملك",
            " سماع",
            "قراءة",
        ],
        chunking_strategy="semantic",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "textbook",
            "technical": True,
        },
        eval_metrics=["technical_correctness"],
    ),
    IslamicKnowledgeDomain.USUL_AL_FIQH: DomainSpec(
        domain=IslamicKnowledgeDomain.USUL_AL_FIQH,
        name_arabic="أصول الفقه",
        name_english="Principles of Jurisprudence",
        categories=["أصول الفقه"],
        priority_sources=[
            "العدة",
            "المحصول",
            "التبصرة",
            "فتح الوصول",
        ],
        trigger_keywords=[
            "دليل",
            "حجة",
            "ظاهر",
            "باطن",
            " عام",
            "خاص",
            "مطلق",
            "مقيد",
            "نسخ",
            "حكم",
            "شرط",
            "سبب",
        ],
        chunking_strategy="by_principle",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "academic",
            "framework": "usuli",
            "evidence_required": True,
        },
        eval_metrics=["logical_coherence", "evidence_quality"],
    ),
    IslamicKnowledgeDomain.FIQH_PRINCIPLES: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_PRINCIPLES,
        name_arabic="علوم الفقه والقواعد",
        name_english="Fiqh Principles and Maxims",
        categories=["علوم الفقه والقواعد الفقهية"],
        priority_sources=[
            "القواعد والفتاوى",
            "الدرر البهية",
            "الفروق",
            "المبسوط",
        ],
        trigger_keywords=[
            "قاعدة",
            "فرق",
            "مذهب",
            "شاذ",
            "راجح",
            "مرجوح",
            "نادر",
            "شائع",
            "محل خلاف",
        ],
        chunking_strategy="by_maxim",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "principles_based",
            "cite_applications": True,
        },
        eval_metrics=["comprehensiveness", "practical_utility"],
    ),
    # ============ Logic ============
    IslamicKnowledgeDomain.LOGIC: DomainSpec(
        domain=IslamicKnowledgeDomain.LOGIC,
        name_arabic="المنطق",
        name_english="Islamic Logic",
        categories=["المنطق"],
        priority_sources=[
            "ال-orgonon",
            "الإشارات",
            "المباحثات",
        ],
        trigger_keywords=[
            "قياس",
            "قياس",
            "حد",
            "رسم",
            "نقيض",
            "سالب",
            "موجب",
            "كلي",
            "جزئي",
            "حمل",
        ],
        chunking_strategy="by_argument",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "logical",
            "form_structure": True,
        },
        eval_metrics=["logical_correctness"],
    ),
    # ============ Four Madhhabs ============
    IslamicKnowledgeDomain.FIQH_HANAFI: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_HANAFI,
        name_arabic="الفقه الحنفي",
        name_english="Hanafi Jurisprudence",
        categories=["الفقه الحنفي"],
        priority_sources=[
            "الهداية",
            "الفتاوى الهندية",
            "بدائع الصنائع",
            "الخراج",
        ],
        trigger_keywords=[
            "مذهب",
            "حنفي",
            "Abu Hanifa",
            "أبو حنيفة",
            "قول",
            "راجح",
            "مرجوح",
            "فتوى",
        ],
        chunking_strategy="by_topic",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "madhhab",
            "madhhab": "hanafi",
            "evidence_required": True,
        },
        eval_metrics=["madhhab_accuracy", "evidence_quality"],
    ),
    IslamicKnowledgeDomain.FIQH_MALIKI: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_MALIKI,
        name_arabic="الفقه المالكي",
        name_english="Maliki Jurisprudence",
        categories=["الفقه المالكي"],
        priority_sources=[
            "الموطأ",
            "المدونة",
            "الذخيرة",
            "الاستخرج",
        ],
        trigger_keywords=[
            "مذهب",
            "مالكي",
            "مالك",
            "Medina",
            "قول",
            "راجح",
            "شاذ",
            "فتوى",
        ],
        chunking_strategy="by_topic",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "madhhab",
            "madhhab": "maliki",
            "evidence_required": True,
        },
        eval_metrics=["madhhab_accuracy", "evidence_quality"],
    ),
    IslamicKnowledgeDomain.FIQH_SHAFII: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_SHAFII,
        name_arabic="الفقه الشافعي",
        name_english="Shafi'i Jurisprudence",
        categories=["الفقه الشافعي"],
        priority_sources=[
            "الأم",
            "المهذب",
            "الوسيط",
            "مغني المحتاج",
        ],
        trigger_keywords=[
            "مذهب",
            "شافعي",
            "شافع",
            "Ash-Shafi'i",
            "قول",
            "راجح",
            "فتوى",
            "مسألة",
        ],
        chunking_strategy="by_topic",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "madhhab",
            "madhhab": "shafii",
            "evidence_required": True,
        },
        eval_metrics=["madhhab_accuracy", "evidence_quality"],
    ),
    IslamicKnowledgeDomain.FIQH_HANBALI: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_HANBALI,
        name_arabic="الفقه الحنبلي",
        name_english="Hanbali Jurisprudence",
        categories=["الفقه الحنبلي"],
        priority_sources=[
            "المبدع",
            "كشاف القناع",
            "الروض المربع",
            "زاد المستقنع",
        ],
        trigger_keywords=[
            "مذهب",
            "حنبلي",
            "أحمد",
            "Ahmad",
            "قول",
            "راجح",
            "فتوى",
            "مسألة",
        ],
        chunking_strategy="by_topic",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "madhhab",
            "madhhab": "hanbali",
            "evidence_required": True,
        },
        eval_metrics=["madhhab_accuracy", "evidence_quality"],
    ),
    # ============ General Fiqh ============
    IslamicKnowledgeDomain.FIQH_GENERAL: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_GENERAL,
        name_arabic="الفقه العام",
        name_english="General Islamic Jurisprudence",
        categories=["الفقه العام"],
        priority_sources=[
            "المغني",
            "الشرح الكبير",
            "الموسوعة الفقهية",
        ],
        trigger_keywords=[
            "حكم",
            "شرع",
            "حل",
            "حرام",
            "واجب",
            "مستحب",
            "مباح",
            " مكروه",
            "نفل",
        ],
        chunking_strategy="by_topic",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "fiqh_general",
            "madhhab_comparison": True,
        },
        eval_metrics=["comprehensiveness", "accuracy"],
    ),
    IslamicKnowledgeDomain.FIQH_ISSUES: DomainSpec(
        domain=IslamicKnowledgeDomain.FIQH_ISSUES,
        name_arabic="المسائل الفقهية",
        name_english="Fiqh Issues and Questions",
        categories=["مسائل فقهية"],
        priority_sources=[
            "المسائل",
            "المسائل الفقهية",
            "الإنصاف",
        ],
        trigger_keywords=[
            "مسألة",
            "نازله",
            "فتوى",
            "استفتاء",
            "حكم",
            "هل",
            "يجوز",
            "يفعل",
        ],
        chunking_strategy="by_case",
        chunk_size=512,
        overlap=30,
        generation_config={
            "style": "qa_fiqh",
            "issue_analysis": True,
        },
        eval_metrics=["practical_utility"],
    ),
    IslamicKnowledgeDomain.ISLAMIC_POLITICS: DomainSpec(
        domain=IslamicKnowledgeDomain.ISLAMIC_POLITICS,
        name_arabic="السياسة الشرعية",
        name_english="Islamic Political Theory",
        categories=["السياسة الشرعية والقضاء"],
        priority_sources=[
            "السياسة الشرعية",
            "الأحكام السلطانية",
            "الخراج",
        ],
        trigger_keywords=[
            "حاكم",
            "سلطان",
            "أمير",
            "قضاء",
            "حكم",
            "ولاية",
            "بيعة",
            "خلافة",
            "إمامة",
        ],
        chunking_strategy="by_issue",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "political_theory",
            "historical_context": True,
        },
        eval_metrics=["analytical_depth"],
    ),
    IslamicKnowledgeDomain.INHERITANCE: DomainSpec(
        domain=IslamicKnowledgeDomain.INHERITANCE,
        name_arabic="الفرائض والوصايا",
        name_english="Inheritance Law",
        categories=["الفرائض والوصايا"],
        priority_sources=[
            "الفرائض",
            "كشاف القناع - قسم الفرائض",
            "التمهيد",
        ],
        trigger_keywords=[
            "ميراث",
            "تركة",
            "ورثة",
            "إرث",
            "فرض",
            "سهم",
            "عصب",
            "حجب",
            "رد",
            "مفروض",
        ],
        chunking_strategy="by_case",
        chunk_size=384,
        overlap=30,
        generation_config={
            "style": "calculation",
            "show_math": True,
        },
        eval_metrics=["calculation_accuracy"],
    ),
    IslamicKnowledgeDomain.FATWAS: DomainSpec(
        domain=IslamicKnowledgeDomain.FATWAS,
        name_arabic="الفتاوى",
        name_english="Islamic Fatwas",
        categories=["الفتاوى"],
        priority_sources=[
            "الفتاوى الكبرى",
            "الفتاوى المجموعة",
            "الفتاوى الهندية",
        ],
        trigger_keywords=[
            "فتوى",
            "سؤال",
            "استفتا",
            "سأل",
            "سائلة",
            "سؤول",
            "أجاب",
            "فصل",
            "حكم",
        ],
        chunking_strategy="by_fatwa",
        chunk_size=512,
        overlap=30,
        generation_config={
            "style": "fatwa",
            "disclaimer_required": True,
        },
        eval_metrics=["clarity", "practical_utility"],
    ),
    # ============ Spirituality ============
    IslamicKnowledgeDomain.SPIRITUALITY: DomainSpec(
        domain=IslamicKnowledgeDomain.SPIRITUALITY,
        name_arabic="الرقائق والآداب",
        name_english="Islamic Spirituality and Ethics",
        categories=["الرقائق والآداب والأذكار"],
        priority_sources=[
            "الرقائق",
            "إحياء علوم الدين",
            "منهاج العابدين",
            "مكارم الأخلاق",
        ],
        trigger_keywords=[
            "رقائق",
            "أدب",
            "خلق",
            "تهذيب",
            "تربية",
            "مجاهدة",
            "قربة",
            "إخلاص",
            "صبر",
            "شكر",
        ],
        chunking_strategy="by_theme",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "spiritual",
            "practical_advice": True,
            "stories_included": True,
        },
        eval_metrics=["spiritual_depth", "practical_impact"],
    ),
    IslamicKnowledgeDomain.PROPHETIC_BIOGRAPHY: DomainSpec(
        domain=IslamicKnowledgeDomain.PROPHETIC_BIOGRAPHY,
        name_arabic="السيرة النبوية",
        name_english="Prophetic Biography",
        categories=["السيرة النبوية"],
        priority_sources=[
            "السيرة النبوية - ابن هشام",
            "الطبقات الكبرى",
            "السيرة الحلبية",
        ],
        trigger_keywords=[
            "سيرة",
            "نبوية",
            "رسول",
            "صلى الله عليه وسلم",
            "غزوة",
            "فتح",
            "هجرة",
            "بعثة",
            "وفاة",
        ],
        chunking_strategy="by_event",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "narrative",
            "chronological": True,
            "include_lessons": True,
        },
        eval_metrics=["historical_accuracy", "educational_value"],
    ),
    # ============ History ============
    IslamicKnowledgeDomain.HISTORY: DomainSpec(
        domain=IslamicKnowledgeDomain.HISTORY,
        name_arabic="التاريخ",
        name_english="Islamic History",
        categories=["التاريخ"],
        priority_sources=[
            "تاريخ الطبري",
            "تاريخ ابن كثير",
            "الكامل في التاريخ",
        ],
        trigger_keywords=[
            "تاريخ",
            "حكم",
            "عصر",
            "دولة",
            "خلافة",
            "فتح",
            "غزوة",
            "معركة",
            "حرب",
            "سلام",
        ],
        chunking_strategy="by_period",
        chunk_size=1024,
        overlap=100,
        generation_config={
            "style": "narrative",
            "chronological": True,
            "sources_required": True,
        },
        eval_metrics=["accuracy", "balance"],
    ),
    IslamicKnowledgeDomain.BIOGRAPHIES: DomainSpec(
        domain=IslamicKnowledgeDomain.BIOGRAPHIES,
        name_arabic="التراجم والطبقات",
        name_english="Biographies and Tabaqat",
        categories=["التراجم والطبقات"],
        priority_sources=[
            "الطبقات الكبرى",
            "سير أعلام النبلاء",
            "العبر",
        ],
        trigger_keywords=[
            "ترجمة",
            "سير",
            "طبقة",
            "عالم",
            "محدث",
            "فقيه",
            "مفسر",
            "مؤرخ",
            "شاعر",
            "شيخ",
        ],
        chunking_strategy="by_person",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "biographical",
            "include_works": True,
            "include_teachers": True,
        },
        eval_metrics=["comprehensiveness", "accuracy"],
    ),
    IslamicKnowledgeDomain.GENEALOGY: DomainSpec(
        domain=IslamicKnowledgeDomain.GENEALOGY,
        name_arabic="الأنساب",
        name_english="Arabian Genealogy",
        categories=["الأنساب"],
        priority_sources=[
            "الأنساب",
            "جمهرة النسب",
            "نسب قريش",
        ],
        trigger_keywords=[
            "نسب",
            "قبيلة",
            "عشيرة",
            "بيت",
            "قوم",
            "عبد",
            "بنو",
            "أولاد",
            "ذرية",
            "سليل",
        ],
        chunking_strategy="by_lineage",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "genealogical",
            "tree_format": True,
        },
        eval_metrics=["accuracy"],
    ),
    IslamicKnowledgeDomain.GEOGRAPHY: DomainSpec(
        domain=IslamicKnowledgeDomain.GEOGRAPHY,
        name_arabic="البلدان والرحلات",
        name_english="Islamic Geography and Travel",
        categories=["البلدان والرحلات"],
        priority_sources=[
            "معجم البلدان",
            "البلدان",
            "الرحلة",
        ],
        trigger_keywords=[
            "بلدان",
            "مدينة",
            "قرية",
            "دولة",
            "جبل",
            "نهر",
            "بحر",
            "جزيرة",
            "وادي",
            "رحلة",
        ],
        chunking_strategy="by_place",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "geographical",
            "include_history": True,
        },
        eval_metrics=["accuracy", "detail"],
    ),
    # ============ Language ============
    IslamicKnowledgeDomain.LANGUAGE_BOOKS: DomainSpec(
        domain=IslamicKnowledgeDomain.LANGUAGE_BOOKS,
        name_arabic="كتب اللغة",
        name_english="Arabic Language Books",
        categories=["كتب اللغة"],
        priority_sources=[
            "الصحاح",
            "اللسان",
            "القاموس",
        ],
        trigger_keywords=[
            "لغة",
            "كلمة",
            "معنى",
            "استعمل",
            "استخدم",
            "استعمل",
            "اشتق",
            "أصل",
        ],
        chunking_strategy="by_entry",
        chunk_size=256,
        overlap=20,
        generation_config={
            "style": "lexical",
            "etymology_required": True,
        },
        eval_metrics=["accuracy"],
    ),
    IslamicKnowledgeDomain.LEXICOGRAPHY: DomainSpec(
        domain=IslamicKnowledgeDomain.LEXICOGRAPHY,
        name_arabic="الغريب والمعاجم",
        name_english="Rare Words and Dictionaries",
        categories=["الغريب والمعاجم"],
        priority_sources=[
            "غريب القرآن",
            "غريب الحديث",
            "المعاجم",
        ],
        trigger_keywords=[
            "غريب",
            "نادر",
            "عجيب",
            "فريد",
            "مهمل",
            "شاذ",
            "مطرد",
            "غير",
            "أعجمي",
        ],
        chunking_strategy="by_word",
        chunk_size=192,
        overlap=20,
        generation_config={
            "style": "dictionary",
            "multiple_meanings": True,
        },
        eval_metrics=["accuracy", "comprehensiveness"],
    ),
    IslamicKnowledgeDomain.GRAMMAR: DomainSpec(
        domain=IslamicKnowledgeDomain.GRAMMAR,
        name_arabic="النحو والصرف",
        name_english="Arabic Grammar",
        categories=["النحو والصرف"],
        priority_sources=[
            "الكتاب - سيبويه",
            "المفصل",
            "الآجرومية",
        ],
        trigger_keywords=[
            "نحو",
            "صرف",
            "إعراب",
            "بناء",
            "مرفوع",
            "منصوب",
            "مجرور",
            "مفعول",
            "فاعل",
            "مبتدأ",
        ],
        chunking_strategy="by_rule",
        chunk_size=384,
        overlap=30,
        generation_config={
            "style": "grammatical",
            "examples_required": True,
        },
        eval_metrics=["technical_correctness"],
    ),
    IslamicKnowledgeDomain.LITERATURE: DomainSpec(
        domain=IslamicKnowledgeDomain.LITERATURE,
        name_arabic="الأدب",
        name_english="Arabic Literature",
        categories=["الأدب"],
        priority_sources=[
            "المقامات",
            "البيان والتبيين",
            "عيون الأخبار",
        ],
        trigger_keywords=[
            "أدب",
            "نثر",
            "نثرة",
            "مقالة",
            "رسالة",
            "خطبة",
            "كلام",
            "فصاحة",
            "بلاغة",
        ],
        chunking_strategy="by_text",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "literary",
            "analysis_included": True,
        },
        eval_metrics=["analytical_depth"],
    ),
    IslamicKnowledgeDomain.POETRY_METER: DomainSpec(
        domain=IslamicKnowledgeDomain.POETRY_METER,
        name_arabic="العروض والقوافي",
        name_english="Arabic Poetry Meter",
        categories=["العروض والقوافي"],
        priority_sources=[
            "التفعيل",
            "القوافي",
            "شرح كافية",
        ],
        trigger_keywords=[
            "عروض",
            "تفعيل",
            "بحر",
            "روي",
            "قافية",
            "روي",
            "شطر",
            "بيت",
            "قصيدة",
            "موزون",
        ],
        chunking_strategy="by_verse",
        chunk_size=256,
        overlap=30,
        generation_config={
            "style": "technical_poetry",
            "meter_analysis": True,
        },
        eval_metrics=["technical_accuracy"],
    ),
    IslamicKnowledgeDomain.POETRY_COLLECTIONS: DomainSpec(
        domain=IslamicKnowledgeDomain.POETRY_COLLECTIONS,
        name_arabic="الدواوين",
        name_english="Poetry Collections",
        categories=["الدواوين الشعرية"],
        priority_sources=[
            "ديوان امرئ القيس",
            "ديوان المتنبي",
            "ديوان الحكماء",
        ],
        trigger_keywords=[
            "ديوان",
            "شعر",
            "شاعر",
            "قصيدة",
            "غزل",
            "مدح",
            "هجو",
            "رثاء",
            "وصف",
            "فخر",
        ],
        chunking_strategy="by_poem",
        chunk_size=384,
        overlap=30,
        generation_config={
            "style": "poetic",
            "include_poem": True,
            "explain_difficult": True,
        },
        eval_metrics=["appreciation", "analysis"],
    ),
    IslamicKnowledgeDomain.RHETORIC: DomainSpec(
        domain=IslamicKnowledgeDomain.RHETORIC,
        name_arabic="البلاغة",
        name_english="Arabic Rhetoric",
        categories=["البلاغة"],
        priority_sources=[
            "البيان والتبيين",
            "المعاني",
            "البيان",
        ],
        trigger_keywords=[
            "بلاغة",
            "تشبيه",
            "استعارة",
            "كناية",
            "طباق",
            "جناس",
            "حسن التقسيم",
            "الترصيع",
        ],
        chunking_strategy="by_device",
        chunk_size=384,
        overlap=30,
        generation_config={
            "style": "rhetorical",
            "examples_required": True,
        },
        eval_metrics=["analysis_accuracy"],
    ),
    # ============ Reference ============
    IslamicKnowledgeDomain.COMPENDIUMS: DomainSpec(
        domain=IslamicKnowledgeDomain.COMPENDIUMS,
        name_arabic="الجوامع",
        name_english="Compendiums and Encyclopedias",
        categories=["الجوامع"],
        priority_sources=[
            "الموسوعة",
            "البلغة",
            "المستدرك",
        ],
        trigger_keywords=[
            "جامع",
            "موسوعة",
            "مختصر",
            "ملخص",
            "مجموع",
            " compendium",
            "encyclopedia",
        ],
        chunking_strategy="by_topic",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "summary",
            "comprehensive": True,
        },
        eval_metrics=["comprehensiveness", "accuracy"],
    ),
    IslamicKnowledgeDomain.INDEXES: DomainSpec(
        domain=IslamicKnowledgeDomain.INDEXES,
        name_arabic="فهارس الكتب",
        name_english="Book Indexes and Catalogs",
        categories=["فهارس الكتب والأدلة"],
        priority_sources=[
            "فهرس",
            "كشف الظنون",
            "كشفSources",
        ],
        trigger_keywords=[
            "فهرس",
            "دليل",
            "كشف",
            "عرض",
            "موسوعه",
            "مؤلف",
            "عنوان",
            "موضوع",
        ],
        chunking_strategy="by_entry",
        chunk_size=256,
        overlap=20,
        generation_config={
            "style": "catalog",
            "full_info": True,
        },
        eval_metrics=["accuracy"],
    ),
    # ============ Sciences ============
    IslamicKnowledgeDomain.MEDICINE: DomainSpec(
        domain=IslamicKnowledgeDomain.MEDICINE,
        name_arabic="الطب",
        name_english="Islamic Medicine",
        categories=["الطب"],
        priority_sources=[
            "القانون في الطب",
            "الطب النبوي",
            "الأدوية",
        ],
        trigger_keywords=[
            "طب",
            "دواء",
            "علاج",
            "مرض",
            "-doctor",
            "إسعاف",
            " وقاية",
            "تداوي",
        ],
        chunking_strategy="by_condition",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "medical",
            "safety_warning": True,
        },
        eval_metrics=["accuracy", "safety"],
    ),
    IslamicKnowledgeDomain.GENERAL: DomainSpec(
        domain=IslamicKnowledgeDomain.GENERAL,
        name_arabic="كتب عامة",
        name_english="General Books",
        categories=["كتب عامة"],
        priority_sources=[],
        trigger_keywords=[
            "عام",
            "متنوع",
            "مختلط",
            "متعدد",
        ],
        chunking_strategy="semantic",
        chunk_size=768,
        overlap=50,
        generation_config={
            "style": "general",
        },
        eval_metrics=["clarity"],
    ),
    IslamicKnowledgeDomain.OTHER_SCIENCES: DomainSpec(
        domain=IslamicKnowledgeDomain.OTHER_SCIENCES,
        name_arabic="علوم أخرى",
        name_english="Other Sciences",
        categories=["علوم أخرى"],
        priority_sources=[],
        trigger_keywords=[
            "علم",
            "بحث",
            "دراسة",
            "تحليل",
        ],
        chunking_strategy="semantic",
        chunk_size=512,
        overlap=50,
        generation_config={
            "style": "academic",
        },
        eval_metrics=["accuracy"],
    ),
}


class DomainClassifier:
    """
    Classify queries and texts into Islamic knowledge domains.

    Uses keyword matching and could be extended with ML/NLP.
    """

    def __init__(self):
        self.domain_keywords: Dict[IslamicKnowledgeDomain, Set[str]] = {}
        self._build_keyword_index()

    def _build_keyword_index(self):
        """Build keyword index from domain specs."""

        for domain, spec in DOMAIN_SPECS.items():
            keywords = set()
            for keyword in spec.trigger_keywords:
                keywords.add(keyword)
                # Add normalized versions
                keywords.add(keyword.lower())
                keywords.add(keyword.replace(" ", ""))
            self.domain_keywords[domain] = keywords

    def classify_query(self, query: str) -> List[IslamicKnowledgeDomain]:
        """Classify a query and return relevant domains."""

        query_lower = query.lower()
        scores: Dict[IslamicKnowledgeDomain, float] = defaultdict(float)

        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[domain] += 1

        # Sort by score
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return domains with matches
        if sorted_domains and sorted_domains[0][1] > 0:
            return [d for d, s in sorted_domains if s > 0]

        # Default to general
        return [IslamicKnowledgeDomain.GENERAL]

    def classify_document(
        self,
        text: str,
        category_hint: str = "",
    ) -> List[IslamicKnowledgeDomain]:
        """Classify a document based on content."""

        # Use category hint if available
        if category_hint:
            domain = self._category_to_domain(category_hint)
            if domain:
                return [domain]

        # Classify by content
        return self.classify_query(text[:500])

    def _category_to_domain(
        self,
        category: str,
    ) -> Optional[IslamicKnowledgeDomain]:
        """Map category name to domain."""

        category_map = {
            "العقيدة": IslamicKnowledgeDomain.AQEEDAH,
            "الفرق والردود": IslamicKnowledgeDomain.SECTS_AND_POLEMIC,
            "التفسير": IslamicKnowledgeDomain.QURAN_TAFSIR,
            "علوم القرآن": IslamicKnowledgeDomain.QURAN_SCIENCES,
            "التجويد": IslamicKnowledgeDomain.QIRAAT_AND_TAJWEED,
            "كتب السنة": IslamicKnowledgeDomain.HADITH_COLLECTIONS,
            "شروح الحديث": IslamicKnowledgeDomain.HADITH_COMMENTARIES,
            "علوم الحديث": IslamicKnowledgeDomain.HADITH_SCIENCES,
            "أصول الفقه": IslamicKnowledgeDomain.USUL_AL_FIQH,
            "الفقه الحنفي": IslamicKnowledgeDomain.FIQH_HANAFI,
            "الفقه المالكي": IslamicKnowledgeDomain.FIQH_MALIKI,
            "الفقه الشافعي": IslamicKnowledgeDomain.FIQH_SHAFII,
            "الفقه الحنبلي": IslamicKnowledgeDomain.FIQH_HANBALI,
            "الفقه العام": IslamicKnowledgeDomain.FIQH_GENERAL,
            "مسائل فقهية": IslamicKnowledgeDomain.FIQH_ISSUES,
            "الفتاوى": IslamicKnowledgeDomain.FATWAS,
            "الرقائق": IslamicKnowledgeDomain.SPIRITUALITY,
            "السيرة النبوية": IslamicKnowledgeDomain.PROPHETIC_BIOGRAPHY,
            "التاريخ": IslamicKnowledgeDomain.HISTORY,
            "التراجم": IslamicKnowledgeDomain.BIOGRAPHIES,
            "الأنساب": IslamicKnowledgeDomain.GENEALOGY,
            "البلدان": IslamicKnowledgeDomain.GEOGRAPHY,
            "اللغة": IslamicKnowledgeDomain.LANGUAGE_BOOKS,
            "النحو": IslamicKnowledgeDomain.GRAMMAR,
            "الأدب": IslamicKnowledgeDomain.LITERATURE,
            "الشعر": IslamicKnowledgeDomain.POETRY_COLLECTIONS,
            "البلاغة": IslamicKnowledgeDomain.RHETORIC,
            "الطب": IslamicKnowledgeDomain.MEDICINE,
        }

        return category_map.get(category)


# Factory functions
def get_domain_spec(domain: IslamicKnowledgeDomain) -> Optional[DomainSpec]:
    """Get domain specification."""
    return DOMAIN_SPECS.get(domain)


def create_domain_classifier() -> DomainClassifier:
    """Create a domain classifier."""
    return DomainClassifier()


def get_domains_for_categories(
    categories: List[str],
) -> List[IslamicKnowledgeDomain]:
    """Get domains for a list of categories."""

    classifier = create_domain_classifier()
    domains = set()

    for category in categories:
        domain = classifier._category_to_domain(category)
        if domain:
            domains.add(domain)

    return list(domains)

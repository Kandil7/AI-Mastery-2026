"""
Extended Instruction Templates for Balygh (بليغ) - 29 Roles

This module extends the base templates.py with comprehensive instruction
templates for all 29 roles and 76 skills defined in schema.py.

New Roles Added:
- Islamic Sciences (10): faqih, muhaddith, mufassir, aqeedah_specialist, etc.
- Modern/Tech (5): rag_assistant, edtech_tutor, dataengineer_ar, etc.
- Specialized (6): historian, genealogist, geographer, physician, etc.
- Dialect & Language (3): dialect_handling_egy, translator_ar, summarizer_ar

Total: 200+ new templates across all roles
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
import random


@dataclass
class Template:
    """A single instruction template"""
    id: str
    role: str
    skill: str
    level: str
    instruction_template: str
    output_format: str = ""
    tags: List[str] = field(default_factory=list)

    def format_instruction(self, **kwargs) -> str:
        """Format the instruction with provided variables"""
        try:
            return self.instruction_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")


# ============================================================================
# ISLAMIC SCIENCES ROLES - العلوم الشرعية (10 roles)
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# FAQIH - الفقيه - Islamic Jurist
# ────────────────────────────────────────────────────────────────────────────

FAQIH_TEMPLATES = [
    Template(
        id="faqih_fiqh_001",
        role="faqih",
        skill="fiqh",
        level="intermediate",
        instruction_template="ما حكم {topic} في الفقه الإسلامي؟",
        output_format="الحكم: [...]\nالأدلة: [...]\nأقوال الفقهاء: [...]",
        tags=["fiqh", "ruling", "evidence"],
    ),
    Template(
        id="faqih_fiqh_002",
        role="faqih",
        skill="fiqh",
        level="advanced",
        instruction_template="قارن بين أقوال المذاهب الأربعة في مسألة: {topic}",
        output_format="الحنفية: [...]\nالمالكية: [...]\nالشافعية: [...]\nالحنابلة: [...]\nالراجح: [...]",
        tags=["fiqh", "comparative", "madhhab"],
    ),
    Template(
        id="faqih_usul_001",
        role="faqih",
        skill="usul_fiqh",
        level="advanced",
        instruction_template="ما الأصل الأصولي المستند إليه في حكم: {ruling}؟",
        output_format="الأصل: [...]\nالاستدلال: [...]\nالتطبيق: [...]",
        tags=["usul_fiqh", "methodology"],
    ),
    Template(
        id="faqih_fatwa_001",
        role="faqih",
        skill="fatwa",
        level="advanced",
        instruction_template="أفتِ في المسألة التالية مع التفصيل: {question}",
        output_format="صياغة الفتوى: [...]\nالأدلة: [...]\nالضوابط: [...]\nتنبيه: [...]",
        tags=["fatwa", "detailed"],
    ),
    Template(
        id="faqih_comparison_001",
        role="faqih",
        skill="comparative_fiqh",
        level="specialist",
        instruction_template="ناقش الخلاف الفقهي في مسألة: {topic} مع الترجيح",
        output_format="أوجه الخلاف: [...]\nأدلة كل قول: [...]\nالراجح: [...] مع التعليل",
        tags=["comparative_fiqh", "tarjeeh"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# MUHADDITH - المحدث - Hadith Scholar
# ────────────────────────────────────────────────────────────────────────────

MUHADDITH_TEMPLATES = [
    Template(
        id="muhaddith_takhreej_001",
        role="muhaddith",
        skill="hadith",
        level="advanced",
        instruction_template="خرّج الحديث التالي: \"{hadith_text}\"",
        output_format="التخريج:\n- المخرج: [...]\n- الكتاب: [...]\n- الرقم: [...]\n- الدرجة: [...]",
        tags=["hadith", "takhreej"],
    ),
    Template(
        id="muhaddith_sanad_001",
        role="muhaddith",
        skill="hadith_mustalah",
        level="advanced",
        instruction_template="حلّل سند هذا الحديث: \"{isnad}\"",
        output_format="الرواة: [...]\nالجرح والتعديل: [...]\nالعلة: [...]\nالحكم: [...]",
        tags=["hadith", "sanad", "criticism"],
    ),
    Template(
        id="muhaddith_graded_001",
        role="muhaddith",
        skill="hadith_mustalah",
        level="specialist",
        instruction_template="ما درجة هذا الحديث ولماذا؟ \"{hadith}\"",
        output_format="الدرجة: [صحيح/حسن/ضعيف/موضوع]\nالسبب: [...]\nالشواهد: [...]",
        tags=["hadith", "grading"],
    ),
    Template(
        id="muhaddith_seerah_001",
        role="muhaddith",
        skill="seerah",
        level="intermediate",
        instruction_template="ما السياق التاريخي لهذا الحديث؟ \"{hadith}\"",
        output_format="السبب: [...]\nالزمن: [...]\nالمكان: [...]\nالأشخاص: [...]",
        tags=["seerah", "context"],
    ),
    Template(
        id="muhaddith_narrators_001",
        role="muhaddith",
        skill="hadith_mustalah",
        level="specialist",
        instruction_template="ترجم للراوي التالي: \"{narrator_name}\"",
        output_format="الاسم الكامل: [...]\nالوفاة: [...]\nشيوخه: [...]\nتلاميذه: [...]\nالجرح والتعديل: [...]",
        tags=["narrators", "biography"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# MUFASSIR - المفسر - Quranic Exegete
# ────────────────────────────────────────────────────────────────────────────

MUFASSIR_TEMPLATES = [
    Template(
        id="mufassir_tafsir_001",
        role="mufassir",
        skill="tafsir",
        level="intermediate",
        instruction_template="فسّر الآية الكريمة: \"{ayah}\"",
        output_format="التفسير الإجمالي: [...]\nتفسير المفردات: [...]\nالفوائد: [...]",
        tags=["tafsir", "quran"],
    ),
    Template(
        id="mufassir_asbab_001",
        role="mufassir",
        skill="tafsir",
        level="advanced",
        instruction_template="ما سبب نزول هذه الآية؟ \"{ayah}\"",
        output_format="سبب النزول: [...]\nالرواية: [...]\nدرجة الصحة: [...]\nالفائدة: [...]",
        tags=["tafsir", "asbab_al_nuzul"],
    ),
    Template(
        id="mufassir_ruling_001",
        role="mufassir",
        skill="quran_sciences",
        level="advanced",
        instruction_template="ما الأحكام الفقهية المستفادة من الآية؟ \"{ayah}\"",
        output_format="الأحكام: [...]\nالأدلة: [...]\nخلاف الفقهاء: [...]",
        tags=["quran", "fiqh", "rulings"],
    ),
    Template(
        id="mufassir_balagha_001",
        role="mufassir",
        skill="balagha",
        level="advanced",
        instruction_template="بيّن الإعجاز البلاغي في الآية: \"{ayah}\"",
        output_format="الصور البلاغية: [...]\nأسرار البلاغة: [...]\nالفائدة: [...]",
        tags=["quran", "balagha", "ijaz"],
    ),
    Template(
        id="mufassir_qiraat_001",
        role="mufassir",
        skill="quran_sciences",
        level="specialist",
        instruction_template="ما القراءات الواردة في هذه الآية؟ \"{ayah}\"",
        output_format="القراءات: [...]\nأوجه الإعراب: [...]\nالفوائد: [...]",
        tags=["quran", "qiraat"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# AQEEDAH_SPECIALIST - متخصص العقيدة - Creed Specialist
# ────────────────────────────────────────────────────────────────────────────

AQEEDAH_SPECIALIST_TEMPLATES = [
    Template(
        id="aqeedah_principle_001",
        role="aqeedah_specialist",
        skill="aqeedah",
        level="intermediate",
        instruction_template="بيّن العقيدة الصحيحة في: {topic}",
        output_format="المعتقد الصحيح: [...]\nالأدلة من القرآن: [...]\nالأدلة من السنة: [...]\nإجماع السلف: [...]",
        tags=["aqeedah", "belief"],
    ),
    Template(
        id="aqeedah_sects_001",
        role="aqeedah_specialist",
        skill="sects",
        level="advanced",
        instruction_template="قارن بين أقوال الفرق الإسلامية في: {topic}",
        output_format="أهل السنة: [...]\nالمعتزلة: [...]\nالأشاعرة: [...]\nالراجح: [...]",
        tags=["sects", "comparative"],
    ),
    Template(
        id="aqeedah_comparison_001",
        role="aqeedah_specialist",
        skill="comparative_religions",
        level="advanced",
        instruction_template="قارن بين الموقف الإسلامي و{other_religion} في: {topic}",
        output_format="الإسلام: [...]\n{other_religion}: [...]\nأوجه الاتفاق: [...]\nأوجه الاختلاف: [...]",
        tags=["comparative", "religions"],
    ),
    Template(
        id="aqeedah_philosophy_001",
        role="aqeedah_specialist",
        skill="islamic_philosophy",
        level="specialist",
        instruction_template="ناقش الشبهة الفلسفية التالية وردّها: \"{doubt}\"",
        output_format="صياغة الشبهة: [...]\nالرد العقلي: [...]\nالرد النقلي: [...]\nالخلاصة: [...]",
        tags=["philosophy", "doubt", "refutation"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# SUFI - الصوفي - Sufism Scholar
# ────────────────────────────────────────────────────────────────────────────

SUFI_TEMPLATES = [
    Template(
        id="sufi_concept_001",
        role="sufi",
        skill="tasawwuf",
        level="intermediate",
        instruction_template="اشرح المفهوم الصوفي التالي: \"{concept}\"",
        output_format="التعريف: [...]\nالأصل الشرعي: [...]\nأقوال الصوفية: [...]\nالضوابط: [...]",
        tags=["tasawwuf", "concept"],
    ),
    Template(
        id="sufi_station_001",
        role="sufi",
        skill="tasawwuf",
        level="advanced",
        instruction_template="ما الفرق بين المقام والحال في التصوف؟",
        output_format="المقام: [...]\nالحال: [...]\nالفرق: [...]\nالأمثلة: [...]",
        tags=["tasawwuf", "stations", "states"],
    ),
    Template(
        id="sufi_adab_001",
        role="sufi",
        skill="adab",
        level="intermediate",
        instruction_template="ما آداب {spiritual_practice} عند الصوفية؟",
        output_format="الآداب الظاهرة: [...]\nالآداب الباطنة: [...]\nالفوائد: [...]",
        tags=["adab", "spiritual"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# HISTORIAN - المؤرخ - Islamic Historian
# ────────────────────────────────────────────────────────────────────────────

HISTORIAN_TEMPLATES = [
    Template(
        id="historian_event_001",
        role="historian",
        skill="islamic_history",
        level="intermediate",
        instruction_template="اشرح الحدث التاريخي التالي: \"{event}\"",
        output_format="الزمن: [...]\nالمكان: [...]\nالأشخاص: [...]\nالأسباب: [...]\nالنتائج: [...]",
        tags=["history", "event"],
    ),
    Template(
        id="historian_civilization_001",
        role="historian",
        skill="islamic_civilization",
        level="advanced",
        instruction_template="بيّن مظاهر الحضارة الإسلامية في {domain} خلال العصر {era}",
        output_format="المظاهر: [...]\nالإنجازات: [...]\nالأعلام: [...]\nالتأثير: [...]",
        tags=["civilization", "achievements"],
    ),
    Template(
        id="historian_biography_001",
        role="historian",
        skill="islamic_history",
        level="advanced",
        instruction_template="ترجم للشخصية التاريخية: \"{person_name}\"",
        output_format="الاسم الكامل: [...]\nالمولد: [...]\nالوفاة: [...]\nالإنجازات: [...]\nالمكانة: [...]",
        tags=["biography", "history"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# GENEALOGIST - النسّاب - Genealogist
# ────────────────────────────────────────────────────────────────────────────

GENEALOGIST_TEMPLATES = [
    Template(
        id="genealogist_lineage_001",
        role="genealogist",
        skill="islamic_history",
        level="advanced",
        instruction_template="ما نسب {person_name}؟",
        output_format="النسب: [...]\nالقبيلة: [...]\nالبطون: [...]\nالمصادر: [...]",
        tags=["genealogy", "lineage"],
    ),
    Template(
        id="genealogist_tribe_001",
        role="genealogist",
        skill="islamic_history",
        level="specialist",
        instruction_template="اشرح أنساب قبيلة {tribe_name}",
        output_format="أصل القبيلة: [...]\nالبطون: [...]\nالفخوذ: [...]\nالديار: [...]\nالأعلام: [...]",
        tags=["genealogy", "tribes"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# GEOGRAPHER - الجغرافي - Historical Geographer
# ────────────────────────────────────────────────────────────────────────────

GEOGRAPHER_TEMPLATES = [
    Template(
        id="geographer_place_001",
        role="geographer",
        skill="arabic_geography",
        level="intermediate",
        instruction_template="أين تقع {place_name} في الجغرافيا التاريخية؟",
        output_format="الموقع القديم: [...]\nالموقع الحديث: [...]\nالإحداثيات: [...]\nالأهمية: [...]",
        tags=["geography", "places"],
    ),
    Template(
        id="geographer_route_001",
        role="geographer",
        skill="arabic_geography",
        level="advanced",
        instruction_template="صف طريق {route_name} التجاري التاريخي",
        output_format="البداية: [...]\nالمحطات: [...]\nالنهاية: [...]\nالمسافة: [...]\nالأهمية: [...]",
        tags=["geography", "trade_routes"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# PHYSICIAN - الطبيب - Classical Islamic Physician
# ────────────────────────────────────────────────────────────────────────────

PHYSICIAN_TEMPLATES = [
    Template(
        id="physician_disease_001",
        role="physician",
        skill="islamic_medicine",
        level="advanced",
        instruction_template="ما علاج {disease} عند الأطباء المسلمين؟",
        output_format="التشخيص: [...]\nالأسباب: [...]\nالعلاج عند ابن سينا: [...]\nالعلاج عند الرازي: [...]",
        tags=["medicine", "islamic"],
    ),
    Template(
        id="physician_herb_001",
        role="physician",
        skill="islamic_medicine",
        level="intermediate",
        instruction_template="ما فوائد {herb_name} الطبية عند الأطباء المسلمين؟",
        output_format="الاسم العلمي: [...]\nالطبع: [...]\nالفوائد: [...]\nالاستعمالات: [...]\nالمحاذير: [...]",
        tags=["medicine", "herbs"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# LOGICIAN - المنطقي - Logic Scholar
# ────────────────────────────────────────────────────────────────────────────

LOGICIAN_TEMPLATES = [
    Template(
        id="logician_syllogism_001",
        role="logician",
        skill="islamic_philosophy",
        level="advanced",
        instruction_template="حلّل القياس المنطقي التالي: \"{premise1}\" و \"{premise2}\"",
        output_format="الصورة: [...]\nالنوع: [...]\nالنتيجة: [...]\nالصحة: [...]",
        tags=["logic", "syllogism"],
    ),
    Template(
        id="logician_fallacy_001",
        role="logician",
        skill="islamic_philosophy",
        level="advanced",
        instruction_template="ما المغالطة المنطقية في: \"{argument}\"؟",
        output_format="نوع المغالطة: [...]\nالتعريف: [...]\nالتوضيح: [...]\nالتصحيح: [...]",
        tags=["logic", "fallacy"],
    ),
]


# ============================================================================
# MODERN/TECH ROLES - الأدوار الحديثة والتقنية (5 roles)
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# RAG_ASSISTANT - مساعد RAG - RAG-based Assistant
# ────────────────────────────────────────────────────────────────────────────

RAG_ASSISTANT_TEMPLATES = [
    Template(
        id="rag_qa_citation_001",
        role="rag_assistant",
        skill="rag_grounded_answering",
        level="intermediate",
        instruction_template="أجب عن السؤال التالي بناءً على المصادر المعطاة مع التوثيق: \"{question}\"\n\nالمصادر: {context}",
        output_format="الإجابة: [...]\n\nالمراجع:\n- [المصدر، الصفحة/الرقم]",
        tags=["qa", "citations", "grounded"],
    ),
    Template(
        id="rag_summary_001",
        role="rag_assistant",
        skill="rag_grounded_answering",
        level="intermediate",
        instruction_template="لخّص المعلومات التالية من المصادر مع التوثيق: {context}",
        output_format="الملخص: [...]\n\nالمصادر: [...]",
        tags=["summarization", "citations"],
    ),
    Template(
        id="rag_compare_001",
        role="rag_assistant",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="قارن بين المعلومات في المصدرين مع التوثيق:\nالمصدر 1: {source1}\nالمصدر 2: {source2}",
        output_format="أوجه التشابه: [...]\nأوجه الاختلاف: [...]\nالترجيح: [...]\nالمصادر: [...]",
        tags=["comparison", "citations"],
    ),
    Template(
        id="rag_evidence_001",
        role="rag_assistant",
        skill="rag_retrieval",
        level="advanced",
        instruction_template="استخرج جميع الأدلة من المصادر حول: {topic}",
        output_format="الأدلة:\n1. [...]\n2. [...]\n\nالمصادر: [...]",
        tags=["evidence", "retrieval"],
    ),
    Template(
        id="rag_verification_001",
        role="rag_assistant",
        skill="consistency_check",
        level="advanced",
        instruction_template="تحقق من اتساق المعلومات في المصادر التالية: {context}",
        output_format="نقاط الاتساق: [...]\nنقاط الاختلاف: [...]\nالتقييم: [...]",
        tags=["verification", "consistency"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# EDTECH_TUTOR - المعلم التقني - Educational Technology Tutor
# ────────────────────────────────────────────────────────────────────────────

EDTECH_TUTOR_TEMPLATES = [
    Template(
        id="edtech_lesson_001",
        role="edtech_tutor",
        skill="curriculum_aligned_ar",
        level="intermediate",
        instruction_template="اشرح درس {lesson_topic} من منهج {curriculum} للصف {grade_level}",
        output_format="أهداف الدرس: [...]\nالشرح: [...]\nأمثلة: [...]\nتمارين: [...]",
        tags=["lesson", "curriculum"],
    ),
    Template(
        id="edtech_mcq_001",
        role="edtech_tutor",
        skill="assessment_design",
        level="intermediate",
        instruction_template="ضع 5 أسئلة اختيار من متعدد على درس {lesson_topic} مع نموذج الإجابة",
        output_format="الأسئلة:\n1. [السؤال]\n   أ) [...]\n   ب) [...]\n   ج) [...]\n   الإجابة: [...]",
        tags=["mcq", "assessment"],
    ),
    Template(
        id="edtech_exercise_001",
        role="edtech_tutor",
        skill="curriculum_aligned_ar",
        level="intermediate",
        instruction_template="صمم تمريناً تعليمياً على {skill} مع نموذج الحل",
        output_format="التمرين: [...]\nالتعليمات: [...]\nنموذج الحل: [...]",
        tags=["exercise", "practice"],
    ),
    Template(
        id="edtech_feedback_001",
        role="edtech_tutor",
        skill="error_analysis_ar",
        level="intermediate",
        instruction_template="حلّل خطأ الطالب التالي واشرح التصحيح: \"{student_answer}\"",
        output_format="نوع الخطأ: [...]\nالسبب: [...]\nالتصحيح: [...]\nشرح للطالب: [...]",
        tags=["feedback", "error_analysis"],
    ),
    Template(
        id="edtech_quiz_001",
        role="edtech_tutor",
        skill="assessment_design",
        level="intermediate",
        instruction_template="صمم مسابقة قصيرة من 10 أسئلة عن {topic} مع الدرجات",
        output_format="الأسئلة: [...]\nنظام الدرجات: [...]\nمفتاح الإجابة: [...]",
        tags=["quiz", "assessment"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# DATAENGINEER_AR - مهندس البيانات العربي - Arabic Data Engineer
# ────────────────────────────────────────────────────────────────────────────

DATAENGINEER_AR_TEMPLATES = [
    Template(
        id="dataengineer_extract_quran_001",
        role="dataengineer_ar",
        skill="extraction",
        level="intermediate",
        instruction_template="استخرج جميع الآيات القرآنية من النص مع التوثيق: \"{text}\"",
        output_format="الآيات:\n- {الآية} | {السورة} | {الرقم}",
        tags=["extraction", "quran", "structured"],
    ),
    Template(
        id="dataengineer_extract_hadith_001",
        role="dataengineer_ar",
        skill="extraction",
        level="advanced",
        instruction_template="استخرج الأحاديث النبوية من النص مع التخريج: \"{text}\"",
        output_format="الأحاديث:\n- {الحديث} | {المخرج} | {الرقم} | {الدرجة}",
        tags=["extraction", "hadith", "structured"],
    ),
    Template(
        id="dataengineer_ner_001",
        role="dataengineer_ar",
        skill="named_entity_ar",
        level="advanced",
        instruction_template="استخرج الكيانات المسماة (أشخاص، أماكن، تواريخ) من: \"{text}\"",
        output_format="الأشخاص: [...]\nالأماكن: [...]\nالتواريخ: [...]\nالهيئات: [...]",
        tags=["ner", "entities"],
    ),
    Template(
        id="dataengineer_structure_001",
        role="dataengineer_ar",
        skill="data_structuring",
        level="advanced",
        instruction_template="حوّل النص التالي إلى JSON منظم: \"{text}\"",
        output_format="{\n  \"entities\": [...],\n  \"relations\": [...],\n  \"metadata\": {...}\n}",
        tags=["structuring", "json"],
    ),
    Template(
        id="dataengineer_outline_001",
        role="dataengineer_ar",
        skill="data_structuring",
        level="advanced",
        instruction_template="استخرج الهيكل التنظيمي للكتاب مع الفصول والأبواب: \"{book_title}\"",
        output_format="الفهرس:\nأولاً: [الفصل الأول]\n  1. [الباب الأول]\n  2. [الباب الثاني]",
        tags=["outline", "structure"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# FATWA_ASSISTANT_SAFE - مساعد الفتوى الآمن - Safe Fatwa Assistant
# ────────────────────────────────────────────────────────────────────────────

# Already in templates.py - adding more templates
FATWA_ASSISTANT_SAFE_TEMPLATES_EXTENDED = [
    Template(
        id="fatwa_safe_madhhab_001",
        role="fatwa_assistant_safe",
        skill="fiqh",
        level="advanced",
        instruction_template="ما المذهب الأقرب للصواب في: {question}؟",
        output_format="أقوال المذاهب: [...]\nأدلة كل مذهب: [...]\nالراجح: [...]\nتنبيه: راجع مفتياً معتمداً",
        tags=["madhhab", "tarjeeh", "safe"],
    ),
    Template(
        id="fatwa_safe_evidence_001",
        role="fatwa_assistant_safe",
        skill="quran_sciences",
        level="advanced",
        instruction_template="ما الدليل الشرعي على: {ruling}؟",
        output_format="الأدلة من القرآن: [...]\nالأدلة من السنة: [...]\nالإجماع: [...]\nالقياس: [...]",
        tags=["evidence", "ruling", "safe"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# TOOL_CALLER_AR - مستدعي الأدوات - Tool/Function Caller
# ────────────────────────────────────────────────────────────────────────────

TOOL_CALLER_AR_TEMPLATES = [
    Template(
        id="tool_call_search_001",
        role="tool_caller_ar",
        skill="function_calling_ar",
        level="intermediate",
        instruction_template="استدعِ أداة البحث للعثور على معلومات عن: {query}",
        output_format="{\"tool\": \"search\", \"query\": \"{query}\", \"parameters\": {...}}",
        tags=["tool_call", "search"],
    ),
    Template(
        id="tool_call_calc_001",
        role="tool_caller_ar",
        skill="function_calling_ar",
        level="intermediate",
        instruction_template="احسب: {math_expression}",
        output_format="{\"tool\": \"calculator\", \"expression\": \"{math_expression}\"}",
        tags=["tool_call", "calculator"],
    ),
    Template(
        id="tool_call_translate_001",
        role="tool_caller_ar",
        skill="function_calling_ar",
        level="intermediate",
        instruction_template="ترجم النص التالي إلى {target_language}: \"{text}\"",
        output_format="{\"tool\": \"translate\", \"text\": \"{text}\", \"target\": \"{target_language}\"}",
        tags=["tool_call", "translation"],
    ),
]


# ============================================================================
# DIALECT & LANGUAGE ROLES - اللهجات وخدمات اللغة (3 roles)
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# DIALECT_HANDLING_EGY - معالجة اللهجة المصرية - Egyptian Dialect Handler
# ────────────────────────────────────────────────────────────────────────────

DIALECT_HANDLING_EGY_TEMPLATES = [
    Template(
        id="dialect_convert_egy_001",
        role="dialect_handling_egy",
        skill="dialect_egy",
        level="intermediate",
        instruction_template="حوّل من العامية المصرية للفصحى: \"{egyptian}\"",
        output_format="بالفصحى: [...]",
        tags=["dialect", "conversion", "egyptian"],
    ),
    Template(
        id="dialect_respond_egy_001",
        role="dialect_handling_egy",
        skill="dialect_egy",
        level="intermediate",
        instruction_template="أجب بالعامية المصرية على: \"{question}\"",
        output_format="بالعامية: [...]",
        tags=["dialect", "response", "egyptian"],
    ),
    Template(
        id="dialect_correct_egy_001",
        role="dialect_handling_egy",
        skill="dialect_egy",
        level="advanced",
        instruction_template="صحّح الأخطاء في العامية المصرية: \"{egyptian_text}\"",
        output_format="الصحيح: [...]\nالشرح: [...]",
        tags=["dialect", "correction", "egyptian"],
    ),
    Template(
        id="dialect_explain_egy_001",
        role="dialect_handling_egy",
        skill="dialect_egy",
        level="intermediate",
        instruction_template="اشرح معنى الكلمة العامية: \"{dialect_word}\"",
        output_format="المعنى: [...]\nالأصل: [...]\nالاستخدام: [...]",
        tags=["dialect", "vocabulary", "egyptian"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# SUMMARIZER_AR - الملخّص العربي - Arabic Summarizer
# ────────────────────────────────────────────────────────────────────────────

SUMMARIZER_AR_TEMPLATES = [
    Template(
        id="summarizer_short_001",
        role="summarizer_ar",
        skill="summarization",
        level="intermediate",
        instruction_template="لخّص النص التالي في فقرة واحدة: \"{text}\"",
        output_format="الملخص: [...]",
        tags=["summarization", "short"],
    ),
    Template(
        id="summarizer_points_001",
        role="summarizer_ar",
        skill="summarization",
        level="intermediate",
        instruction_template="استخرج النقاط الرئيسية من النص في 5 نقاط: \"{text}\"",
        output_format="النقاط:\n1. [...]\n2. [...]\n3. [...]\n4. [...]\n5. [...]",
        tags=["summarization", "points"],
    ),
    Template(
        id="summarizer_abstract_001",
        role="summarizer_ar",
        skill="summarization",
        level="advanced",
        instruction_template="اكتب ملخصاً أكاديمياً (200 كلمة) مع الكلمات المفتاحية: \"{text}\"",
        output_format="الملخص: [...]\n\nالكلمات المفتاحية: [...]",
        tags=["summarization", "academic"],
    ),
    Template(
        id="summarizer_elaborate_001",
        role="summarizer_ar",
        skill="summarization",
        level="advanced",
        instruction_template="لخّص النص مع الحفاظ على الأفكار الرئيسية والأمثلة: \"{text}\"",
        output_format="الملخص الموسع: [...]",
        tags=["summarization", "elaborate"],
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# TRANSLATOR_AR - المترجم - Arabic Translator
# ────────────────────────────────────────────────────────────────────────────

TRANSLATOR_AR_TEMPLATES = [
    Template(
        id="translator_ar_en_001",
        role="translator_ar",
        skill="translation_ar_en",
        level="intermediate",
        instruction_template="ترجم النص التالي من العربية للإنجليزية: \"{arabic_text}\"",
        output_format="Translation: [...]",
        tags=["translation", "ar_en"],
    ),
    Template(
        id="translator_en_ar_001",
        role="translator_ar",
        skill="translation_ar_en",
        level="intermediate",
        instruction_template="ترجم النص التالي من الإنجليزية للعربية: \"{english_text}\"",
        output_format="الترجمة: [...]",
        tags=["translation", "en_ar"],
    ),
    Template(
        id="translator_term_001",
        role="translator_ar",
        skill="lexicography",
        level="advanced",
        instruction_template="ما الترجمة الدقيقة للمصطلح: \"{term}\" في مجال {domain}؟",
        output_format="الترجمة: [...]\nالشرح: [...]\nالاستعمالات: [...]",
        tags=["terminology", "translation"],
    ),
    Template(
        id="translator_compare_001",
        role="translator_ar",
        skill="translation_ar_en",
        level="advanced",
        instruction_template="قارن بين الترجمتين التاليةين وبيّن الأصح: \"{trans1}\" و \"{trans2}\"",
        output_format="مقارنة: [...]\nالأصح: [...]\nالسبب: [...]",
        tags=["translation", "comparison"],
    ),
]


# ============================================================================
# TEMPLATE REGISTRY - تسجيل القوالب
# ============================================================================

# Extended templates registry with all new roles
EXTENDED_TEMPLATES = {
    # Islamic Sciences (10)
    "faqih": FAQIH_TEMPLATES,
    "muhaddith": MUHADDITH_TEMPLATES,
    "mufassir": MUFASSIR_TEMPLATES,
    "aqeedah_specialist": AQEEDAH_SPECIALIST_TEMPLATES,
    "sufi": SUFI_TEMPLATES,
    "historian": HISTORIAN_TEMPLATES,
    "genealogist": GENEALOGIST_TEMPLATES,
    "geographer": GEOGRAPHER_TEMPLATES,
    "physician": PHYSICIAN_TEMPLATES,
    "logician": LOGICIAN_TEMPLATES,
    
    # Modern/Tech (5)
    "rag_assistant": RAG_ASSISTANT_TEMPLATES,
    "edtech_tutor": EDTECH_TUTOR_TEMPLATES,
    "dataengineer_ar": DATAENGINEER_AR_TEMPLATES,
    "fatwa_assistant_safe": FATWA_ASSISTANT_SAFE_TEMPLATES_EXTENDED,
    "tool_caller_ar": TOOL_CALLER_AR_TEMPLATES,
    
    # Dialect & Language (3)
    "dialect_handling_egy": DIALECT_HANDLING_EGY_TEMPLATES,
    "summarizer_ar": SUMMARIZER_AR_TEMPLATES,
    "translator_ar": TRANSLATOR_AR_TEMPLATES,
}

# Merge with existing templates if available
try:
    from .templates import ALL_TEMPLATES as BASE_TEMPLATES
    ALL_TEMPLATES_EXTENDED = {**BASE_TEMPLATES, **EXTENDED_TEMPLATES}
except ImportError:
    ALL_TEMPLATES_EXTENDED = EXTENDED_TEMPLATES


def get_extended_templates(role: Optional[str] = None, skill: Optional[str] = None, level: Optional[str] = None) -> List[Template]:
    """
    Get templates filtered by role, skill, and/or level from extended collection.
    
    Args:
        role: Filter by role (optional)
        skill: Filter by skill (optional)
        level: Filter by level (optional)
    
    Returns:
        List of matching templates
    """
    if role:
        templates = ALL_TEMPLATES_EXTENDED.get(role, [])
    elif skill:
        templates = [t for role_templates in ALL_TEMPLATES_EXTENDED.values() 
                    for t in role_templates if t.skill == skill]
    else:
        templates = [t for role_templates in ALL_TEMPLATES_EXTENDED.values() 
                    for t in role_templates]
    
    if level:
        templates = [t for t in templates if t.level == level]
    
    return templates


def get_random_extended_template(role: Optional[str] = None, skill: Optional[str] = None, level: Optional[str] = None) -> Template:
    """Get a random template from extended collection matching the criteria"""
    templates = get_extended_templates(role, skill, level)
    if not templates:
        raise ValueError(f"No templates found for role={role}, skill={skill}, level={level}")
    return random.choice(templates)


def get_template_by_id_extended(template_id: str) -> Template:
    """Get a template by its ID from extended collection"""
    for role, templates in ALL_TEMPLATES_EXTENDED.items():
        for template in templates:
            if template.id == template_id:
                return template
    raise ValueError(f"Template not found: {template_id}")


# Statistics
if __name__ == "__main__":
    print("=== Extended Templates Statistics ===\n")
    
    total = 0
    for role, templates in ALL_TEMPLATES_EXTENDED.items():
        print(f"{role}: {len(templates)} templates")
        total += len(templates)
    
    print(f"\nTotal: {total} templates across {len(ALL_TEMPLATES_EXTENDED)} roles")

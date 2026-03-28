"""
Instruction Templates for Arabic LLM Fine-Tuning

This module contains instruction templates for all roles and skills,
designed to generate diverse training examples from Arabic books.

Templates are organized by:
- Role (tutor, proofreader, poet, muhhaqiq, assistant_general)
- Skill (nahw, balagha, sarf, poetry, etc.)
- Level (beginner, intermediate, advanced)
- Content Type (verse, prose, hadith, poetry, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable,Optional
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
# TUTOR ROLE - معلم اللغة العربية
# ============================================================================

TUTOR_TEMPLATES = [
    # Grammar (Nahw) Templates
    Template(
        id="tutor_nahw_001",
        role="tutor",
        skill="nahw",
        level="beginner",
        instruction_template="أعرب الجملة التالية إعراباً مبسطاً لطالب مبتدئ: \"{sentence}\"",
        output_format="الإعراب: [كلمة بكلمة مع ذكر النوع والإعراب]",
        tags=["i3rab", "basic_grammar"],
    ),
    Template(
        id="tutor_nahw_002",
        role="tutor",
        skill="nahw",
        level="intermediate",
        instruction_template="أعرب الجملة التالية إعراباً مفصلاً مع بيان العامل والعللة: \"{sentence}\"",
        output_format="الإعراب المفصل: [كلمة بكلمة مع العامل والعلامة والتقدير]",
        tags=["i3rab", "detailed_grammar"],
    ),
    Template(
        id="tutor_nahw_003",
        role="tutor",
        skill="nahw",
        level="advanced",
        instruction_template="حلّل التركيب النحوي للنص التالي تحليلاً دقيقاً مع ذكر الشواهد النحوية: \"{text}\"",
        output_format="التحليل النحوي: [تحليل شامل مع الشواهد]",
        tags=["nahw", "advanced_analysis"],
    ),
    Template(
        id="tutor_nahw_004",
        role="tutor",
        skill="nahw",
        level="intermediate",
        instruction_template="بيّن نوع الجملة التالية (اسمية/فعلية) واذكر مكوناتها: \"{sentence}\"",
        output_format="نوع الجملة: [اسمية/فعلية]\nالمكونات: [الفاعل، المفعول، ...]",
        tags=["sentence_type", "grammar"],
    ),
    Template(
        id="tutor_nahw_005",
        role="tutor",
        skill="nahw",
        level="beginner",
        instruction_template="استخرج من النص التالي: المبتدأ، الخبر، الفاعل، المفعول به: \"{text}\"",
        output_format="المبتدأ: ...\nالخبر: ...\nالفاعل: ...\nالمفعول به: ...",
        tags=["grammar_extraction"],
    ),
    
    # Rhetoric (Balagha) Templates
    Template(
        id="tutor_balagha_001",
        role="tutor",
        skill="balagha",
        level="beginner",
        instruction_template="هل في الجملة التالية صورة بلاغية؟ إذا نعم، فسمّها واشرحها: \"{sentence}\"",
        output_format="نوع الصورة: [تشبيه/استعارة/كناية]\nالشرح: [توضيح مبسط]",
        tags=["balagha", "figures_of_speech"],
    ),
    Template(
        id="tutor_balagha_002",
        role="tutor",
        skill="balagha",
        level="intermediate",
        instruction_template="حدّد نوع التشبيه في قوله: \"{quote}\" واذكر أركانه",
        output_format="نوع التشبيه: [تام/مجمل/مفصل/بليغ]\nالأركان: [المشبه، المشبه به، وجه الشبه، أداة التشبيه]",
        tags=["tashbih", "balagha"],
    ),
    Template(
        id="tutor_balagha_003",
        role="tutor",
        skill="balagha",
        level="intermediate",
        instruction_template="ما نوع الاستعارة في قوله: \"{quote}\"؟ وبيّن وجه الاستعارة",
        output_format="نوع الاستعارة: [تصريحية/مكنية/تخيلية]\nوجه الاستعارة: [الشرح]",
        tags=["isti3ara", "balagha"],
    ),
    Template(
        id="tutor_balagha_004",
        role="tutor",
        skill="balagha",
        level="advanced",
        instruction_template="حلّل الصور البيانية في النص التالي تحليلاً بلاغياً: \"{text}\"",
        output_format="التحليل البلاغي: [ذكر جميع الصور مع الشرح]",
        tags=["balagha", "analysis"],
    ),
    Template(
        id="tutor_balagha_005",
        role="tutor",
        skill="balagha",
        level="intermediate",
        instruction_template="بيّن الفرق في البلاغة بين الجملتين: \"{sentence1}\" و \"{sentence2}\"",
        output_format="الجملة الأولى: [التحليل]\nالجملة الثانية: [التحليل]\nالفرق: [الوجه البلاغي]",
        tags=["balagha", "comparison"],
    ),
    
    # Combined Grammar + Rhetoric Templates
    Template(
        id="tutor_combined_001",
        role="tutor",
        skill="nahw",
        level="intermediate",
        instruction_template="أعرب الجملة التالية ثم وضّح الصورة البلاغية فيها: \"{sentence}\"",
        output_format="أولاً: الإعراب: [...]\nثانياً: البلاغة: [...]",
        tags=["i3rab", "balagha", "combined"],
    ),
    Template(
        id="tutor_combined_002",
        role="tutor",
        skill="balagha",
        level="advanced",
        instruction_template="اقرأ النص التالي، ثم: 1) أعرب الكلمات المُشار إليها، 2) اذكر الصور البلاغية: \"{text}\"",
        output_format="1) الإعراب: [...]\n2) البلاغة: [...]",
        tags=["combined", "advanced"],
    ),
    
    # Morphology (Sarf) Templates
    Template(
        id="tutor_sarf_001",
        role="tutor",
        skill="sarf",
        level="beginner",
        instruction_template="ما وزن الكلمة التالية: \"{word}\"؟",
        output_format="الوزن: [على وزن فعل]",
        tags=["sarf", "wazn"],
    ),
    Template(
        id="tutor_sarf_002",
        role="tutor",
        skill="sarf",
        level="intermediate",
        instruction_template="صرّف الفعل \"{verb}\" في المضارع والأمر مع الضبط",
        output_format="المضارع: [أفعل، تفعل، ...]\nالأمر: [افعل، ...]",
        tags=["sarf", "tasreef"],
    ),
    Template(
        id="tutor_sarf_003",
        role="tutor",
        skill="sarf",
        level="intermediate",
        instruction_template="بيّن نوع الاشتقاق في الكلمة التالية واذكر مادتها: \"{word}\"",
        output_format="المادة: [الجذر]\nنوع الاشتقاق: [مجرد/مزيد]",
        tags=["sarf", "ishtiqaq"],
    ),
    
    # Q&A Templates
    Template(
        id="tutor_qa_001",
        role="tutor",
        skill="qa",
        level="beginner",
        instruction_template="ما هو الفاعل في اللغة العربية؟ اشرح بإيجاز",
        output_format="الفاعل: [تعريف مبسط مع مثال]",
        tags=["qa", "definition"],
    ),
    Template(
        id="tutor_qa_002",
        role="tutor",
        skill="qa",
        level="intermediate",
        instruction_template="ما الفرق بين المبتدأ والفاعل؟",
        output_format="الفرق: [الفرق مع أمثلة]",
        tags=["qa", "difference"],
    ),
    Template(
        id="tutor_qa_003",
        role="tutor",
        skill="qa",
        level="advanced",
        instruction_template="اشرح بالتفصيل أنواع الخبر في الجملة الاسمية",
        output_format="أنواع الخبر: [مفرد، جملة، شبه جملة مع الأمثلة]",
        tags=["qa", "detailed"],
    ),
]

# ============================================================================
# PROOFREADER ROLE - المصحح اللغوي
# ============================================================================

PROOFREADER_TEMPLATES = [
    # Spelling Correction Templates
    Template(
        id="proof_ortho_001",
        role="proofreader",
        skill="orthography",
        level="beginner",
        instruction_template="صحّح الأخطاء الإملائية في النص التالي: \"{text}\"",
        output_format="النص المصحَّح: [النص بعد التصحيح]",
        tags=["imla", "correction"],
    ),
    Template(
        id="proof_ortho_002",
        role="proofreader",
        skill="orthography",
        level="intermediate",
        instruction_template="صحّح الأخطاء الإملائية في النص التالي مع شرح الأخطاء: \"{text}\"",
        output_format="النص المصحَّح: [...]\nشرح الأخطاء: [1) ... 2) ...]",
        tags=["imla", "correction", "explanation"],
    ),
    Template(
        id="proof_ortho_003",
        role="proofreader",
        skill="orthography",
        level="intermediate",
        instruction_template="أي الكلمتين أصح إملائياً: \"{word1}\" أم \"{word2}\"؟ ولماذا؟",
        output_format="الأصح: [الكلمة]\nالسبب: [القاعدة الإملائية]",
        tags=["imla", "comparison"],
    ),
    
    # Grammar Correction Templates
    Template(
        id="proof_nahw_001",
        role="proofreader",
        skill="nahw",
        level="intermediate",
        instruction_template="صحّح الأخطاء النحوية في الجملة التالية: \"{sentence}\"",
        output_format="الجملة المصحَّحة: [بعد التصحيح]",
        tags=["nahw", "correction"],
    ),
    Template(
        id="proof_nahw_002",
        role="proofreader",
        skill="nahw",
        level="advanced",
        instruction_template="صحّح الأخطاء النحوية في النص التالي مع بيان نوع كل خطأ: \"{text}\"",
        output_format="النص المصحَّح: [...]\nالأخطاء: [1) نوع الخطأ: ... 2) ...]",
        tags=["nahw", "correction", "analysis"],
    ),
    Template(
        id="proof_nahw_003",
        role="proofreader",
        skill="nahw",
        level="intermediate",
        instruction_template="بيّن الخطأ النحوي في الجملة التالية وصحّحه: \"{sentence}\"",
        output_format="الخطأ: [تحديد الخطأ]\nالتصحيح: [الجملة الصحيحة]\nالسبب: [القاعدة]",
        tags=["nahw", "error_identification"],
    ),
    
    # Style Editing Templates
    Template(
        id="proof_style_001",
        role="proofreader",
        skill="style_editing",
        level="intermediate",
        instruction_template="حسّن صياغة الجملة التالية مع الحفاظ على المعنى: \"{sentence}\"",
        output_format="الصياغة المحسَّنة: [النص المحسّن]",
        tags=["style", "improvement"],
    ),
    Template(
        id="proof_style_002",
        role="proofreader",
        skill="style_editing",
        level="advanced",
        instruction_template="أعد صياغة النص التالي ليكون أكثر فصاحة وبلاغة: \"{text}\"",
        output_format="النص المُعاد صياغته: [النص المحسّن]",
        tags=["style", "balagha"],
    ),
    Template(
        id="proof_style_003",
        role="proofreader",
        skill="style_editing",
        level="intermediate",
        instruction_template="اختصر النص التالي مع الحفاظ على المعنى: \"{text}\"",
        output_format="النص المختصر: [النص بعد الاختصار]",
        tags=["style", "summarization"],
    ),
]

# ============================================================================
# POET ROLE - الشاعر والناقد الأدبي
# ============================================================================

POET_TEMPLATES = [
    # Poetry Composition Templates
    Template(
        id="poet_compose_001",
        role="poet",
        skill="poetry",
        level="intermediate",
        instruction_template="انظم بيتاً من الشعر على بحر {meter} عن موضوع: {topic}",
        output_format="البيت:\n[الشعر الموزون]",
        tags=["poetry", "composition", "meter"],
    ),
    Template(
        id="poet_compose_002",
        role="poet",
        skill="poetry",
        level="advanced",
        instruction_template="انظم قصيدة قصيرة (4 أبيات) على بحر {meter} في مدح: {topic}",
        output_format="القصيدة:\n[الأبيات الأربعة]",
        tags=["poetry", "composition", "qasida"],
    ),
    Template(
        id="poet_compose_003",
        role="poet",
        skill="poetry",
        level="intermediate",
        instruction_template="أكمل البيت التالي بما يناسب المعنى والوزن: \"{partial_verse}\"",
        output_format="البيت الكامل:\n[البيت تاماً]",
        tags=["poetry", "completion"],
    ),
    
    # Poetry Criticism Templates
    Template(
        id="poet_critique_001",
        role="poet",
        skill="poetry",
        level="intermediate",
        instruction_template="حلّل البيت التالي تحليلاً عروضياً: \"{verse}\"",
        output_format="البحر: [اسم البحر]\nالتفعيلة: [التفعيلات]",
        tags=["poetry", "analysis", "arud"],
    ),
    Template(
        id="poet_critique_002",
        role="poet",
        skill="poetry",
        level="advanced",
        instruction_template="انقد القصيدة التالية نقداً أدبياً من حيث المعنى والمبنى: \"{poem}\"",
        output_format="النقد:\n- المعنى: [...]\n- المبنى: [...]\n- الصور البلاغية: [...]",
        tags=["poetry", "criticism"],
    ),
    Template(
        id="poet_critique_003",
        role="poet",
        skill="balagha",
        level="advanced",
        instruction_template="ما الصور البلاغية في قول الشاعر: \"{verse}\"؟",
        output_format="الصور البلاغية: [الشرح]",
        tags=["poetry", "balagha"],
    ),
    
    # Poetry Explanation Templates
    Template(
        id="poet_explain_001",
        role="poet",
        skill="poetry",
        level="intermediate",
        instruction_template="اشرح معنى البيت التالي بلغة معاصرة: \"{verse}\"",
        output_format="الشرح: [الشرح المبسّط]",
        tags=["poetry", "explanation"],
    ),
    Template(
        id="poet_explain_002",
        role="poet",
        skill="heritage",
        level="advanced",
        instruction_template="بيّن السياق التاريخي والأدبي لهذا البيت: \"{verse}\"",
        output_format="السياق: [الشرح التاريخي والأدبي]",
        tags=["poetry", "heritage", "context"],
    ),
]

# ============================================================================
# MUHHAQIQ ROLE - المحقق اللغوي
# ============================================================================

MUHHAQIQ_TEMPLATES = [
    # Text Verification Templates
    Template(
        id="muhhaqiq_verify_001",
        role="muhhaqiq",
        skill="heritage",
        level="advanced",
        instruction_template="اقرأ النص التراثي التالي، ثم: 1) أعرب الكلمات الرئيسة، 2) وضّح المعنى العام: \"{text}\"",
        output_format="1) الإعراب: [...]\n2) المعنى: [...]",
        tags=["heritage", "verification", "i3rab"],
    ),
    Template(
        id="muhhaqiq_verify_002",
        role="muhhaqiq",
        skill="nahw",
        level="advanced",
        instruction_template="حقّق النص التالي تحقيقاً لغوياً مع ضبط الكلمات: \"{text}\"",
        output_format="النص المحقَّق: [النص مضبوطاً]",
        tags=["tahqiq", "nahw"],
    ),
    Template(
        id="muhhaqiq_verify_003",
        role="muhhaqiq",
        skill="heritage",
        level="advanced",
        instruction_template="بيّن ما في النص التالي من ألفاظ قديمة واشرح معانيها: \"{text}\"",
        output_format="الألفاظ القديمة: [الكلمة: المعنى]",
        tags=["heritage", "vocabulary"],
    ),
    
    # Manuscript Analysis Templates
    Template(
        id="muhhaqiq_manuscript_001",
        role="muhhaqiq",
        skill="heritage",
        level="advanced",
        instruction_template="قارن بين الروايتين التين واختر الأصح لغوياً: \"{version1}\" / \"{version2}\"",
        output_format="الأصح: [الرواية]\nالسبب: [التعليل اللغوي]",
        tags=["manuscript", "comparison"],
    ),
    Template(
        id="muhhaqiq_manuscript_002",
        role="muhhaqiq",
        skill="nahw",
        level="advanced",
        instruction_template="ما الوجه الإعرابي الراجح في قوله: \"{quote}\"؟ ولماذا؟",
        output_format="الوجه الراجح: [الإعراب]\nالسبب: [التعليل]",
        tags=["nahw", "manuscript"],
    ),
    
    # Classical Text Analysis Templates
    Template(
        id="muhhaqiq_classical_001",
        role="muhhaqiq",
        skill="heritage",
        level="advanced",
        instruction_template="اشرح هذا النص التراثي شرحاً مفصلاً مع ذكر الفوائد اللغوية: \"{text}\"",
        output_format="الشرح: [...]\nالفوائد اللغوية: [...]",
        tags=["heritage", "explanation"],
    ),
    Template(
        id="muhhaqiq_classical_002",
        role="muhhaqiq",
        skill="balagha",
        level="advanced",
        instruction_template="بيّن مظاهر البلاغة في هذا النص القديم: \"{text}\"",
        output_format="مظاهر البلاغة: [الشرح]",
        tags=["heritage", "balagha"],
    ),
]

# ============================================================================
# ASSISTANT GENERAL ROLE - المساعد العام
# ============================================================================

ASSISTANT_TEMPLATES = [
    # General Q&A Templates
    Template(
        id="assistant_qa_001",
        role="assistant_general",
        skill="nahw",
        level="beginner",
        instruction_template="ما الفرق بين الجملة الاسمية والجملة الفعلية؟",
        output_format="الفرق: [الشرح مع الأمثلة]",
        tags=["qa", "basic_grammar"],
    ),
    Template(
        id="assistant_qa_002",
        role="assistant_general",
        skill="qa",
        level="beginner",
        instruction_template="ما هي أقسام الكلمة في اللغة العربية؟",
        output_format="الأقسام: [اسم، فعل، حرف مع الشرح]",
        tags=["qa", "definition"],
    ),
    Template(
        id="assistant_qa_003",
        role="assistant_general",
        skill="qa",
        level="intermediate",
        instruction_template="كيف أتميَّز بين المفعول المطلق والنائب عن المفعول المطلق؟",
        output_format="التمييز: [الشرح مع الأمثلة]",
        tags=["qa", "distinction"],
    ),
    
    # Writing Assistance Templates
    Template(
        id="assistant_write_001",
        role="assistant_general",
        skill="style_editing",
        level="intermediate",
        instruction_template="اكتب مقدمة موضوعية عن: {topic}",
        output_format="المقدمة: [النص]",
        tags=["writing", "introduction"],
    ),
    Template(
        id="assistant_write_002",
        role="assistant_general",
        skill="style_editing",
        level="intermediate",
        instruction_template="لخّص النص التالي في فقرة واحدة: \"{text}\"",
        output_format="الملخّص: [الفقرة]",
        tags=["summarization"],
    ),
]

# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

ALL_TEMPLATES = {
    "tutor": TUTOR_TEMPLATES,
    "proofreader": PROOFREADER_TEMPLATES,
    "poet": POET_TEMPLATES,
    "muhhaqiq": MUHHAQIQ_TEMPLATES,
    "assistant_general": ASSISTANT_TEMPLATES,
}

# Skill to templates mapping
SKILL_TEMPLATES = {}
for role, templates in ALL_TEMPLATES.items():
    for template in templates:
        if template.skill not in SKILL_TEMPLATES:
            SKILL_TEMPLATES[template.skill] = []
        SKILL_TEMPLATES[template.skill].append(template)


def get_templates(role: Optional[str] = None, skill: Optional[str] = None, level: Optional[str] = None) -> List[Template]:
    """
    Get templates filtered by role, skill, and/or level.

    Args:
        role: Filter by role (optional)
        skill: Filter by skill (optional)
        level: Filter by level (optional)

    Returns:
        List of matching templates
    """
    if role:
        templates = ALL_TEMPLATES.get(role, [])
    elif skill:
        templates = SKILL_TEMPLATES.get(skill, [])
    else:
        templates = [t for role_templates in ALL_TEMPLATES.values() for t in role_templates]
    
    if level:
        templates = [t for t in templates if t.level == level]
    
    return templates


def get_random_template(role: Optional[str] = None, skill: Optional[str] = None, level: Optional[str] = None) -> Template:
    """Get a random template matching the criteria"""
    templates = get_templates(role, skill, level)
    if not templates:
        raise ValueError(f"No templates found for role={role}, skill={skill}, level={level}")
    return random.choice(templates)


def get_template_by_id(template_id: str) -> Template:
    """Get a template by its ID"""
    for role, templates in ALL_TEMPLATES.items():
        for template in templates:
            if template.id == template_id:
                return template
    raise ValueError(f"Template not found: {template_id}")


# ============================================================================
# MODERN APPLICATION ROLES - الأدوار التطبيقية الحديثة
# ============================================================================

# DATAENGINEER_AR TEMPLATES - مهندس البيانات العربي
# ============================================================================

DATAENGINEER_AR_TEMPLATES = [
    # Data Extraction Templates
    Template(
        id="dataengineer_extract_001",
        role="dataengineer_ar",
        skill="rag_grounded_answering",
        level="intermediate",
        instruction_template="استخرج جميع الآيات القرآنية من النص التالي مع ذكر السورة ورقم الآية: \"{text}\"",
        output_format="قائمة الآيات:\n- الآية: [...] | السورة: [...] | الرقم: [...]",
        tags=["extraction", "quran", "structured_data"],
    ),
    Template(
        id="dataengineer_extract_002",
        role="dataengineer_ar",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="استخرج الأحاديث النبوية من النص مع توثيق المخرج ورقم الحديث: \"{text}\"",
        output_format="قائمة الأحاديث:\n- الحديث: [...] | المخرج: [...] | الرقم: [...]",
        tags=["extraction", "hadith", "structured_data"],
    ),
    Template(
        id="dataengineer_summarize_001",
        role="dataengineer_ar",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="لخّص الكتاب التالي في شكل outline منظم مع الفصول الرئيسية والفرعية: \"{book_title}\"",
        output_format="هيكل الكتاب:\nأولاً: [الفصل الأول]\n  1. [المبحث الأول]\n  2. [المبحث الثاني]",
        tags=["summarization", "structured_outline"],
    ),
    Template(
        id="dataengineer_entity_001",
        role="dataengineer_ar",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="استخرج الكيانات المسماة (الأشخاص، الأماكن، التواريخ) من النص: \"{text}\"",
        output_format="الكيانات:\n- الأشخاص: [...]\n- الأماكن: [...]\n- التواريخ: [...]",
        tags=["ner", "entity_extraction", "structured_data"],
    ),
]

# RAG_ASSISTANT TEMPLATES - مساعد RAG
# ============================================================================

RAG_ASSISTANT_TEMPLATES = [
    # Q&A with Citations
    Template(
        id="rag_qa_001",
        role="rag_assistant",
        skill="rag_grounded_answering",
        level="intermediate",
        instruction_template="أجب عن السؤال التالي بناءً على المصادر المعطاة مع ذكر المراجع: \"{question}\"\n\nالمصادر: \"{context}\"",
        output_format="الإجابة: [...]\n\nالمراجع:\n- [اسم المصدر، الصفحة/الرقم]",
        tags=["qa", "citations", "grounded"],
    ),
    Template(
        id="rag_compare_001",
        role="rag_assistant",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="قارن بين القولين التاليين مع توثيق المصادر: \"{opinion1}\" و \"{opinion2}\"",
        output_format="أوجه التشابه: [...]\nأوجه الاختلاف: [...]\nالمصادر: [...]",
        tags=["comparison", "citations", "grounded"],
    ),
    Template(
        id="rag_evidence_001",
        role="rag_assistant",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="ما الدليل على الحكم التالي من المصادر المعطاة؟ \"{ruling}\"\n\nالمصادر: \"{context}\"",
        output_format="الدليل: [...]\nنص الدليل: [...]\nالمصدر: [...]",
        tags=["evidence", "citations", "grounded"],
    ),
]

# EDTECH_TUTOR TEMPLATES - المعلم التقني التعليمي
# ============================================================================

EDTECH_TUTOR_TEMPLATES = [
    # Curriculum-Aligned Lessons
    Template(
        id="edtech_lesson_001",
        role="edtech_tutor",
        skill="curriculum_aligned_ar",
        level="intermediate",
        instruction_template="اشرح درس \"{lesson_topic}\" من منهج \"{curriculum}\" للصف \"{grade_level}\" بطريقة مبسطة",
        output_format="أهداف الدرس: [...]\nالشرح: [...]\nأمثلة: [...]",
        tags=["lesson", "curriculum", "explanation"],
    ),
    Template(
        id="edtech_mcq_001",
        role="edtech_tutor",
        skill="curriculum_aligned_ar",
        level="intermediate",
        instruction_template="ضع 5 أسئلة اختيار من متعدد على درس \"{lesson_topic}\" مع نموذج الإجابة",
        output_format="الأسئلة:\n1. [السؤال]\n   أ) [...]\n   ب) [...]\n   ج) [...]\n   الإجابة الصحيحة: [...]",
        tags=["mcq", "assessment", "curriculum"],
    ),
    Template(
        id="edtech_exercise_001",
        role="edtech_tutor",
        skill="curriculum_aligned_ar",
        level="intermediate",
        instruction_template="صمم تمريناً تعليمياً على \"{skill}\" مع نموذج الحل",
        output_format="التمرين: [...]\nتعليمات الحل: [...]\nنموذج الحل: [...]",
        tags=["exercise", "practice", "curriculum"],
    ),
    Template(
        id="edtech_feedback_001",
        role="edtech_tutor",
        skill="error_analysis_ar",
        level="intermediate",
        instruction_template="حلّل الخطأ التالي للطالب واشرح السبب وطريقة التصحيح: \"{student_answer}\"",
        output_format="نوع الخطأ: [...]\nالسبب: [...]\nالتصحيح: [...]\nشرح للطالب: [...]",
        tags=["error_analysis", "feedback", "pedagogy"],
    ),
]

# FATWA_ASSISTANT_SAFE TEMPLATES - مساعد الفتاوى الحذر
# ============================================================================

FATWA_ASSISTANT_SAFE_TEMPLATES = [
    # Safe Fatwa Guidance
    Template(
        id="fatwa_safe_001",
        role="fatwa_assistant_safe",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="لخّص أقوال المذاهب الأربعة في المسألة التالية: \"{question}\"",
        output_format="المذهب الحنفي: [...]\nالمذهب المالكي: [...]\nالمذهب الشافعي: [...]\nالمذهب الحنبلي: [...]\n\nتنبيه: هذه مجرد ملخصات وليست فتوى. راجع دار الإفتاء للفتوى الرسمية.",
        tags=["madhhab", "comparison", "safe_fatwa"],
    ),
    Template(
        id="fatwa_safe_002",
        role="fatwa_assistant_safe",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="ما الحكم الشرعي في: \"{question}\" مع ذكر أقوال العلماء",
        output_format="أقوال العلماء:\n- [...]\n- [...]\n\nالأدلة: [...]\n\nتنبيه: استشر مفتياً معتمداً للحالة الشخصية.",
        tags=["ruling", "scholars", "safe_fatwa"],
    ),
    Template(
        id="fatwa_safe_003",
        role="fatwa_assistant_safe",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="أرشدني لمصادر الفتوى الرسمية في مسألة: \"{question}\"",
        output_format="المصادر الرسمية:\n- دار الإفتاء المصرية: [...]\n- الموقع الإلكتروني: [...]\n- رقم الهاتف: [...]\n\nتنبيه: هذه المصادر الرسمية المعتمدة.",
        tags=["guidance", "official_sources", "safe_fatwa"],
    ),
    Template(
        id="fatwa_safe_004",
        role="fatwa_assistant_safe",
        skill="rag_grounded_answering",
        level="advanced",
        instruction_template="هل هذه المسألة من مسائل الاجتهاد أم الإجماع؟ \"{question}\"",
        output_format="نوع المسألة: [اجتهاد/إجماع]\nالقول الراجح: [...]\nتنبيه: المسائل الاجتهادية فيها سعة، راجع مفتياً.",
        tags=["ijtihad", "ijma", "safe_fatwa"],
    ),
]

# ERROR_ANALYSIS_AR TEMPLATES - تحليل الأخطاء العربية
# ============================================================================

ERROR_ANALYSIS_AR_TEMPLATES = [
    # Error Analysis Templates
    Template(
        id="error_grammar_001",
        role="proofreader",
        skill="error_analysis_ar",
        level="intermediate",
        instruction_template="حلّل الأخطاء النحوية في الجملة التالية واشرح السبب وصحّحها: \"{sentence}\"",
        output_format="الأخطاء:\n1. الخطأ: [...]\n   النوع: [...]\n   السبب: [...]\n   التصحيح: [...]",
        tags=["error_analysis", "grammar", "correction"],
    ),
    Template(
        id="error_spelling_001",
        role="proofreader",
        skill="error_analysis_ar",
        level="beginner",
        instruction_template="صحّح الأخطاء الإملائية في النص التالي مع شرح القاعدة: \"{text}\"",
        output_format="الأخطاء المصححة:\n- الخطأ: [...] → الصواب: [...]\nالقاعدة: [...]",
        tags=["error_analysis", "spelling", "correction"],
    ),
    Template(
        id="error_style_001",
        role="proofreader",
        skill="error_analysis_ar",
        level="advanced",
        instruction_template="قيّم الأسلوب التالي واقترح تحسينات: \"{text}\"",
        output_format="نقاط القوة: [...]\nنقاط الضعف: [...]\nالتحسينات المقترحة: [...]",
        tags=["error_analysis", "style", "improvement"],
    ),
]

# DIALECT_HANDLING_EGY TEMPLATES - اللهجة المصرية
# ============================================================================

DIALECT_HANDLING_EGY_TEMPLATES = [
    # Dialect to MSA Conversion
    Template(
        id="dialect_convert_001",
        role="assistant_general",
        skill="dialect_handling_egy",
        level="intermediate",
        instruction_template="حوّل الجملة التالية من العامية المصرية للفصحى: \"{egyptian_dialect}\"",
        output_format="بالفصحى: [...]",
        tags=["dialect_conversion", "egyptian", "msa"],
    ),
    Template(
        id="dialect_understand_001",
        role="assistant_general",
        skill="dialect_handling_egy",
        level="intermediate",
        instruction_template="افهم المعنى المقصود من الجملة العامية التالية: \"{egyptian_dialect}\"",
        output_format="المعنى: [...]\nالكلمات الصعبة: [...]",
        tags=["dialect_understanding", "egyptian"],
    ),
    Template(
        id="dialect_respond_001",
        role="assistant_general",
        skill="dialect_handling_egy",
        level="advanced",
        instruction_template="أجب بالعامية المصرية على السؤال التالي مع الحفاظ على دقة المحتوى: \"{question}\"",
        output_format="بالعامية: [...]",
        tags=["dialect_response", "egyptian", "customer_support"],
    ),
]

# LEGAL_ARABIC_DRAFTING TEMPLATES - الصياغة القانونية العربية
# ============================================================================

LEGAL_ARABIC_DRAFTING_TEMPLATES = [
    # Legal Document Drafting
    Template(
        id="legal_letter_001",
        role="dataengineer_ar",
        skill="legal_arabic_drafting",
        level="advanced",
        instruction_template="اصغِ خطاباً رسمياً لـ \"{recipient}\" بخصوص: \"{subject}\"",
        output_format="السادة/ [...]\nتحية طيبة،\nالموضوع: [...]\nنود إفادتكم بأن: [...]\nوتفضلوا بقبول فائق الاحترام.",
        tags=["legal_drafting", "official_letter", "formal"],
    ),
    Template(
        id="legal_complaint_001",
        role="dataengineer_ar",
        skill="legal_arabic_drafting",
        level="advanced",
        instruction_template="اصغِ شكوى رسمية لـ \"{authority}\" حول: \"{issue}\"",
        output_format="إلى: [...]\nالموضوع: شكوى بخصوص [...]\nالوقائع: [...]\nالطلبات: [...]\nمقدم الشكوى: [...]",
        tags=["legal_drafting", "complaint", "formal"],
    ),
    Template(
        id="legal_contract_001",
        role="dataengineer_ar",
        skill="legal_arabic_drafting",
        level="advanced",
        instruction_template="اصغِ مسودة عقد مبسطة لـ \"{contract_type}\" بين طرفين",
        output_format="عقد [...]\nبين: [...]\nوبين: [...]\nالمادة الأولى: موضوع العقد\nالمادة الثانية: الالتزامات\n...",
        tags=["legal_drafting", "contract", "formal"],
    ),
]

# Add all new templates to ALL_TEMPLATES dictionary
ALL_TEMPLATES["dataengineer_ar"] = DATAENGINEER_AR_TEMPLATES
ALL_TEMPLATES["rag_assistant"] = RAG_ASSISTANT_TEMPLATES
ALL_TEMPLATES["edtech_tutor"] = EDTECH_TUTOR_TEMPLATES
ALL_TEMPLATES["fatwa_assistant_safe"] = FATWA_ASSISTANT_SAFE_TEMPLATES
ALL_TEMPLATES["error_analysis_ar"] = ERROR_ANALYSIS_AR_TEMPLATES
ALL_TEMPLATES["dialect_handling_egy"] = DIALECT_HANDLING_EGY_TEMPLATES
ALL_TEMPLATES["legal_arabic_drafting"] = LEGAL_ARABIC_DRAFTING_TEMPLATES

# Also add to skill-based retrieval
# Note: error_analysis_ar and rag_grounded_answering are skills, not roles
# So we need to add them to the skill-based retrieval in get_templates()


# ============================================================================
# AGENT SYSTEM PROMPTS - prompts للوكلاء الذكيين
# ============================================================================
# These are system prompts for AI agents using different roles
# Each prompt defines the agent's personality, capabilities, and constraints

@dataclass
class AgentSystemPrompt:
    """System prompt for AI agent"""
    role: str
    system_prompt: str
    greeting: str
    constraints: List[str]
    examples: List[str]


# ============================================================================
# MODERN APPLICATION AGENTS - وكلاء الأدوار التطبيقية
# ============================================================================

DATAENGINEER_AR_AGENT = AgentSystemPrompt(
    role="dataengineer_ar",
    system_prompt="""أنت مهندس بيانات عربي متخصص في تحويل النصوص العربية الخام إلى بيانات مهيكلة.

مهامك:
- استخراج الآيات القرآنية مع ذكر السورة ورقم الآية
- استخراج الأحاديث النبوية مع توثيق المخرج ورقم الحديث
- تلخيص الكتب العربية في شكل outline منظم
- استخراج الكيانات المسماة (الأشخاص، الأماكن، التواريخ)
- تحويل النصوص إلى JSON منظم

أسلوبك:
- دقيق ومنظم
- توثيق كامل للمصادر
- هيكلية واضحة""",
    greeting="أهلاً بك! أنا مهندس البيانات العربي. سأساعدك في تحويل النصوص العربية إلى بيانات مهيكلة ومنظمة.",
    constraints=[
        "لا تستخرج آيات أو أحاديث بدون توثيق",
        "تأكد من دقة المراجع قبل الإخراج",
        "استخدم تنسيق JSON منظم",
        "احفظ التسلسل الهرمي للبيانات",
    ],
    examples=[
        "المستخدم: استخرج الآيات من هذا النص\nالوكيل: تم استخراج 5 آيات مع التوثيق...",
        "المستخدم: لخص هذا الكتاب في outline\nالوكيل: هيكل الكتاب: أولاً: الفصل الأول...",
    ]
)

RAG_ASSISTANT_AGENT = AgentSystemPrompt(
    role="rag_assistant",
    system_prompt="""أنت مساعد RAG متخصص في الإجابة المعتمدة على مصادر موثوقة.

مهامك:
- الإجابة عن الأسئلة بناءً على المصادر المعطاة
- ذكر المراجع والاستشهادات بوضوح
- مقارنة الآراء المختلفة مع توثيق المصادر
- تقديم إجابات grounded في النصوص

أسلوبك:
- دقيق في التوثيق
- متوازن في العرض
- واضح في الاستشهاد""",
    greeting="أهلاً بك! أنا مساعد RAG. أجيب على أسئلتك بناءً على مصادر موثقة مع ذكر المراجع.",
    constraints=[
        "لا تجيب بدون مصادر",
        "اذكر كل المراجع المستخدمة",
        "وضح إذا كانت المصادر مختلفة",
        "لا تضيف معلومات من خارج المصادر",
    ],
    examples=[
        "المستخدم: ما حكم الصلاة؟\nالوكيل: الإجابة: [...] المصادر: [القرآن، البخاري...]",
        "المستخدم: قارن بين القولين\nالوكيل: أوجه التشابه: [...] أوجه الاختلاف: [...] المصادر: [...]",
    ]
)

EDTECH_TUTOR_AGENT = AgentSystemPrompt(
    role="edtech_tutor",
    system_prompt="""أنت معلم عربي متخصص في مناهج اللغة العربية الحديثة.

مهامك:
- شرح دروس النحو والصرف والبلاغة بطريقة مبسطة
- وضع أسئلة اختيار من متعدد مع نموذج الإجابة
- تصميم تمارين تعليمية مناسبة للمستوى
- تحليل أخطاء الطلاب وشرح التصحيح

أسلوبك:
- مبسط وواضح
- مشجع للطلاب
- منهجي ومنظم""",
    greeting="أهلاً بك يا طالب العلم! أنا معلمك العربي. سأساعدك في فهم دروس اللغة العربية بطريقة سهلة.",
    constraints=[
        "استخدم لغة مناسبة للمستوى",
        "قدم أمثلة توضيحية",
        "شجع الطالب على الفهم لا الحفظ",
        "وضح الأخطاء الشائعة",
    ],
    examples=[
        "المستخدم: اشرح درس المبتدأ والخبر\nالوكيل: أهداف الدرس: [...] الشرح: [...] أمثلة: [...]",
        "المستخدم: ضع 5 أسئلة على الدرس\nالوكيل: الأسئلة: 1. [...] الإجابة: [...]",
    ]
)

FATWA_ASSISTANT_SAFE_AGENT = AgentSystemPrompt(
    role="fatwa_assistant_safe",
    system_prompt="""أنت مساعد فتاوى حذر ومتزن، تلتزم بأقصى درجات الحذر والأمان.

مهامك:
- تلخيص أقوال المذاهب الأربعة في المسائل الفقهية
- ذكر أقوال العلماء مع التوثيق
- إرشاد المستخدمين لمصادر الفتوى الرسمية
- التوضيح أنك لست بديلاً عن المفتي المعتمد

أسلوبك:
- حذر ومتزن
- موثق للمصادر
- محترم للمذاهب المختلفة
- واضح في التحذيرات""",
    greeting="أهلاً بك! أنا مساعد فتاوى آمن. أقدم لك معلومات فقهية موثقة، لكنني لست بديلاً عن المفتي المعتمد.",
    constraints=[
        "لا تفتِ بشكل قاطع",
        "اذكر دائماً أن هذا ليس فتوى رسمية",
        "أشر لمصادر الفتوى الرسمية",
        "اذكر المذاهب المختلفة",
        "تحذير من الاعتماد الكامل على النموذج",
    ],
    examples=[
        "المستخدم: ما حكم كذا؟\nالوكيل: أقوال العلماء: [...] تنبيه: استشر مفتياً معتمداً...",
        "المستخدم: لخص أقوال المذاهب\nالوكيل: الحنفي: [...] المالكي: [...] الشافعي: [...] الحنبلي: [...]",
    ]
)

ERROR_ANALYSIS_AR_AGENT = AgentSystemPrompt(
    role="proofreader",
    system_prompt="""أنت محلل أخطاء لغوية عربي متخصص في تحليل وتصحيح الأخطاء.

مهامك:
- تحليل الأخطاء النحوية والصرفية والإملائية
- شرح سبب الخطأ بشكل واضح
- تقديم التصحيح الصحيح
- تقديم نصائح لتجنب الخطأ في المستقبل

أسلوبك:
- تعليمي وواضح
- صبور ومفصل
- عملي ومفيد""",
    greeting="أهلاً بك! أنا محلل الأخطاء اللغوية. سأساعدك في تحليل وتصحيح أخطائك العربية مع الشرح.",
    constraints=[
        "اشرح سبب كل خطأ",
        "قدم التصحيح الصحيح",
        "قدم نصائح للوقاية",
        "كن مشجعاً لا ناقداً",
    ],
    examples=[
        "المستخدم: صحّح: ذهبوا إلى المسجد\nالوكيل: الخطأ: [...] السبب: [...] التصحيح: [...]",
    ]
)

DIALECT_HANDLING_EGY_AGENT = AgentSystemPrompt(
    role="assistant_general",
    system_prompt="""أنت مساعد متخصص في اللهجة المصرية، تفهم العامية وتتقن التحويل للفصحى والعكس.

مهامك:
- تحويل العامية المصرية إلى فصحى
- فهم المعنى المقصود من الجمل العامية
- الرد بالعامية المصرية مع الحفاظ على دقة المحتوى
- شرح الفروق بين العامية والفصحى

أسلوبك:
- ودود وطبيعي
- يفهم الثقافة المصرية
- يحافظ على الدقة اللغوية""",
    greeting="أهلاً بيك! أنا مساعد اللهجة المصرية. بفهم عاميتك وبقدر أرد بالفصحى أو بالعامية زي ما تحب.",
    constraints=[
        "احترم اللهجة المصرية",
        "لا تسخر من العامية",
        "حافظ على الدقة في التحويل",
        "اشرح الفروق عند الحاجة",
    ],
    examples=[
        "المستخدم: ايه ده بالعامية؟\nالوكيل: ده معناه: [...] بالفصحى: [...]",
        "المستخدم: رد بالعامية\nالوكيل: بالعامية: [...] مع الحفاظ على الدقة",
    ]
)

LEGAL_ARABIC_DRAFTING_AGENT = AgentSystemPrompt(
    role="dataengineer_ar",
    system_prompt="""أنت مساعد صياغة قانونية وإدارية عربي متخصص.

مهامك:
- صياغة خطابات رسمية لجهات حكومية
- كتابة شكاوى رسمية
- صياغة عقود مبسطة
- مراجعة الصياغة القانونية

أسلوبك:
- رسمي ومنضبط
- دقيق في المصطلحات
- واضح في الصياغة""",
    greeting="أهلاً بك! أنا مساعد الصياغة القانونية. سأساعدك في صياغة الخطابات والعقود الرسمية.",
    constraints=[
        "استخدم اللغة الرسمية",
        "التزم بالصياغة القانونية",
        "وضح أن هذه مسودة وليست وثيقة نهائية",
        "أنصح بمراجعة مختص قانوني",
    ],
    examples=[
        "المستخدم: اكتب خطاب رسمي\nالوكيل: السادة/ [...] تحية طيبة، الموضوع: [...]",
        "المستخدم: اصغِ عقد بسيط\nالوكيل: عقد [...] بين: [...] المادة الأولى: [...]",
    ]
)

# ============================================================================
# AGENT PROMPTS REGISTRY - سجل وكلاء النظام
# ============================================================================

AGENT_PROMPTS = {
    "dataengineer_ar": DATAENGINEER_AR_AGENT,
    "rag_assistant": RAG_ASSISTANT_AGENT,
    "edtech_tutor": EDTECH_TUTOR_AGENT,
    "fatwa_assistant_safe": FATWA_ASSISTANT_SAFE_AGENT,
    "proofreader": ERROR_ANALYSIS_AR_AGENT,
    "assistant_general": DIALECT_HANDLING_EGY_AGENT,
    "dataengineer_ar_legal": LEGAL_ARABIC_DRAFTING_AGENT,
}


def get_agent_prompt(role: str, skill: Optional[str] = None) -> AgentSystemPrompt:
    """
    Get agent system prompt for a role.
    
    Args:
        role: Role name
        skill: Optional skill name for specialized agents
        
    Returns:
        AgentSystemPrompt object
    """
    # Handle skill-based agents
    if skill:
        special_key = f"{role}_{skill}"
        if special_key in AGENT_PROMPTS:
            return AGENT_PROMPTS[special_key]
    
    # Default to role-based agent
    return AGENT_PROMPTS.get(role, DATAENGINEER_AR_AGENT)


def format_agent_message(agent: AgentSystemPrompt, user_message: str) -> str:
    """
    Format complete agent message with system prompt.
    
    Args:
        agent: AgentSystemPrompt object
        user_message: User's message
        
    Returns:
        Formatted message string
    """
    return f"""<system>
{agent.system_prompt}
</system>

<greeting>
{agent.greeting}
</greeting>

<constraints>
{chr(10).join(f'- {c}' for c in agent.constraints)}
</constraints>

<user_message>
{user_message}
</user_message>"""


# Poetry meters for template filling
POETRY_METERS = [
    "الطويل", "المديد", "البسيط", "الوافر", "الكامل",
    "الهزج", "الرجز", "الرمل", "السريع", "المنسرح",
    "الخفيف", "المضارع", "المقتضب", "المجتث", "المتقارب", "المتدارك"
]

# Common topics for poetry
POETRY_TOPICS = [
    "طلب العلم", "الحكمة", "مدح الرسول", "الوصف", "الرثاء",
    "الفخر", "الغزل العفيف", "الزهد", "الشوق للوطن", "الصبر"
]


if __name__ == "__main__":
    # Test template retrieval
    print("=== Template Statistics ===\n")
    
    for role, templates in ALL_TEMPLATES.items():
        print(f"{role}: {len(templates)} templates")
        skill_counts = {}
        for t in templates:
            skill_counts[t.skill] = skill_counts.get(t.skill, 0) + 1
        for skill, count in skill_counts.items():
            print(f"  - {skill}: {count}")
    
    print("\n=== Sample Templates ===\n")
    
    # Show sample templates
    sample = get_random_template(role="tutor", skill="nahw")
    print(f"ID: {sample.id}")
    print(f"Instruction: {sample.instruction_template}")
    print(f"Output Format: {sample.output_format}")

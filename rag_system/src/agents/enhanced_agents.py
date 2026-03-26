"""
Enhanced Agent System for Islamic Literature RAG

Based on comprehensive analysis of 40+ Islamic knowledge categories in the dataset.
Extends the base agent system with specialized roles for every domain.

New Roles:
1. Quranic Researcher - التفسير and related
2. Hadith Specialist - علوم الحديث
3. Fiqh Scholar (4 madhhabs) - الفقه
4. Aqeedah Theologian - العقيدة
5. Usul al-Fiqh Expert - أصول الفقه
6. Islamic Historian - التاريخ
7. Biographer - التراجم والطبقات
8. Geographer - البلدان والرحلات
9. Arabic Linguist - اللغة العربية
10. Literature Analyst - الأدب
11. Poet Analysis - الدواوين
12. Qiraat Specialist - القراءات
13. Inheritance Expert - الفرائض
14. Fatwa Researcher - الفتاوى
15. Logic Expert - المنطق
16. Medical Scholar - الطب (Islamic medicine)
17. Spirituality Guide - الرقائق
18. Sects Analyst - الفرق والردود
19. Index/Catalog Expert - فهارس الكتب
20. General Researcher - كتب عامة
"""

from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class EnhancedAgentRole(Enum):
    """Extended agent roles for all Islamic knowledge domains."""

    # Core Religious Sciences
    QURAN_RESEARCHER = "quran_researcher"  # التفسير and related
    HADITH_SPECIALIST = "hadith_specialist"  # علوم الحديث
    FIQH_SCHOLAR = "fiqh_scholar"  # General Fiqh
    HANAFI_SCHOLAR = "hanafi_scholar"  # الفقه الحنفي
    MALIKI_SCHOLAR = "maliki_scholar"  # الفقه المالكي
    SHAFII_SCHOLAR = "shafii_scholar"  # الفقه الشافعي
    HANBALI_SCHOLAR = "hanbali_scholar"  # الفقه الحنبلي
    AQEEDAH_THEOLOGIAN = "aqeedah_theologian"  # العقيدة
    USUL_EXPERT = "usul_expert"  # أصول الفقه

    # Language & Literature
    ARABIC_LINGUIST = "arabic_linguist"  # اللغة العربية
    GRAMMAR_ANALYST = "grammar_analyst"  # النحو والصرف
    LEXICOGRAPHER = "lexicographer"  # الغريب والمعاجم
    LITERATURE_ANALYST = "literature_analyst"  # الأدب
    POETRY_ANALYST = "poetry_analyst"  # الدواوين الشعرية
    RHETORIC_EXPERT = "rhetoric_expert"  # البلاغة

    # History & Biography
    ISLAMIC_HISTORIAN = "islamic_historian"  # التاريخ
    BIOGRAPHER = "biographer"  # التراجم والطبقات
    GENEALOGIST = "genealogist"  # الأنساب
    GEOGRAPHER = "geographer"  # البلدان والرحلات

    # Specialized Sciences
    QIRAAT_SPECIALIST = "qiraat_specialist"  # التجويد والقراءات
    INHERITANCE_EXPERT = "inheritance_expert"  # الفرائض
    FATWA_RESEARCHER = "fatwa_researcher"  # الفتاوى
    LOGIC_EXPERT = "logic_expert"  # المنطق
    MEDICAL_SCHOLAR = "medical_scholar"  # الطب

    # Spirituality & Ethics
    SPIRITUALITY_GUIDE = "spirituality_guide"  # الرقائق والآداب
    ETHICS_TEACHER = "ethics_teacher"  # الأذكار

    # Sects & Theology
    SECTS_ANALYST = "sects_analyst"  # الفرق والردود

    # Reference & Catalogs
    INDEX_EXPERT = "index_expert"  # فهارس الكتب والأدلة
    REFERENCE_LIBRARIAN = "reference_librarian"  # الجوامع

    # General
    GENERAL_RESEARCHER = "general_researcher"  # كتب عامة


@dataclass
class DomainCategory:
    """Maps agent role to dataset categories."""

    role: EnhancedAgentRole
    dataset_categories: List[str]
    primary_category: str
    expertise_arabic: str
    description: str

    # Retrieval parameters
    default_top_k: int = 5
    chunk_size: int = 768

    # Specialized prompts
    system_prompt: str = ""
    query_instructions: str = ""


# Complete domain mapping based on 40 categories in dataset
DOMAIN_MAPPING: Dict[EnhancedAgentRole, DomainCategory] = {
    # ============ Core Religious Sciences ============
    EnhancedAgentRole.QURAN_RESEARCHER: DomainCategory(
        role=EnhancedAgentRole.QURAN_RESEARCHER,
        dataset_categories=[
            "التفسير",
            "علوم القرآن وأصول التفسير",
            "التجويد والقراءات",
        ],
        primary_category="التفسير",
        expertise_arabic="مفسر قرآن",
        description="Research Quranic exegesis, tafsir, and Quranic sciences",
        system_prompt="""أنت متخصص في علوم القرآن والتفسير.
لديك معرفة عميقة بكتب التفسير الكلاسيكية (ابن كثير، القرطبي، الطبري، السعدي).
تستطيع:
- تفسير الآيات القرآنية بأسلوب المفسرين
- ذكر الأقوال المختلفة في التفسير
- الاستشهاد بأقوال المفسرين المعتمدين
- توضيح Reasons for revelation (سبب النزول)
- بيان связи بين الآيات""",
    ),
    EnhancedAgentRole.HADITH_SPECIALIST: DomainCategory(
        role=EnhancedAgentRole.HADITH_SPECIALIST,
        dataset_categories=[
            "كتب السنة",
            "شروح الحديث",
            "التخريج والأطراف",
            "العلل والسؤلات الحديثية",
            "علوم الحديث",
        ],
        primary_category="كتب السنة",
        expertise_arabic="محدث",
        description="Analyze hadith collections, chains, and grades",
        system_prompt="""أنت متخصص في علوم الحديث.
لديك معرفة بكتب السنة (صحيح البخاري، صحيح مسلم، السنن الأربعة).
تستطيع:
- تحديد درجة الحديث (صحيح، حسن، ضعيف)
- تحليل الإسناد
- معرفة المدلسين والرواة
- بيان علل الحديث
- ذكر التخريج""",
    ),
    EnhancedAgentRole.FIQH_SCHOLAR: DomainCategory(
        role=EnhancedAgentRole.FIQH_SCHOLAR,
        dataset_categories=[
            "الفقه العام",
            "مسائل فقهية",
            "أصول الفقه",
            "علوم الفقه والقواعد الفقهية",
        ],
        primary_category="الفقه العام",
        expertise_arabic="فقيه",
        description="General Islamic jurisprudence and rulings",
        system_prompt="""أنت فقيه مسلم.
تعرف أحكام الفقه الإسلامي من مصادره الأصلية.
تستطيع:
- استخراج الأحكام من الأدلة
- بيان أقوال المذاهب الأربعة
- ذكر القول الراجح مع الدليل
- بيان القواعد الفقهية""",
    ),
    EnhancedAgentRole.HANAFI_SCHOLAR: DomainCategory(
        role=EnhancedAgentRole.HANAFI_SCHOLAR,
        dataset_categories=["الفقه الحنفي"],
        primary_category="الفقه الحنفي",
        expertise_arabic="حنفي",
        description="Hanafi school of jurisprudence",
        system_prompt="""أنت فقيه على مذهب أبي حنيفة النعمان.
تعرف كتب الفقه الحنفي (الهداية، البدائع، الفتاوي الهندية).
تستطيع:
- بيان حكم المسألة على مذهب الحنفية
- التفرقة بين القول الراجح والمرجوح
- الاستدلال بأدلة المذهب""",
    ),
    EnhancedAgentRole.MALIKI_SCHOLAR: DomainCategory(
        role=EnhancedAgentRole.MALIKI_SCHOLAR,
        dataset_categories=["الفقه المالكي"],
        primary_category="الفقه المالكي",
        expertise_arabic="مالكي",
        description="Maliki school of jurisprudence",
        system_prompt="""أنت فقيه على مذهب مالك بن أنس.
تعرف كتب الفقه المالكي (الموطأ، المدونة).
تستطيع:
- بيان حكم المسألة على مذهب المالكية
- التفرقة بين القول الراجح والمرجوح
- الاستدلال بأدلة المذهب""",
    ),
    EnhancedAgentRole.SHAFII_SCHOLAR: DomainCategory(
        role=EnhancedAgentRole.SHAFII_SCHOLAR,
        dataset_categories=["الفقه الشافعي"],
        primary_category="الفقه الشافعي",
        expertise_arabic="شافعي",
        description="Shafi'i school of jurisprudence",
        system_prompt="""أنت فقيه على مذهب محمد بن إدريس الشافعي.
تعرف كتب الفقه الشافعي (المهذب، الوسيط).
تستطيع:
- بيان حكم المسألة على مذهب الشافعية
- التفرقة بين القول الراجح والمرجوح
- الاستدلال بأدلة المذهب""",
    ),
    EnhancedAgentRole.HANBALI_SCHOLAR: DomainCategory(
        role=EnhancedAgentRole.HANBALI_SCHOLAR,
        dataset_categories=["الفقه الحنبلي"],
        primary_category="الفقه الحنبلي",
        expertise_arabic="حنبلي",
        description="Hanbali school of jurisprudence",
        system_prompt="""أنت فقيه على مذهب أحمد بن حنبل.
تعرف كتب الفقه الحنبلي (المبدع، كشاف القناع).
تستطيع:
- بيان حكم المسألة على مذهب الحنابلة
- التفرقة بين القول الراجح والمرجوح
- الاستدلال بأدلة المذهب""",
    ),
    EnhancedAgentRole.AQEEDAH_THEOLOGIAN: DomainCategory(
        role=EnhancedAgentRole.AQEEDAH_THEOLOGIAN,
        dataset_categories=["العقيدة", "الفرق والردود"],
        primary_category="العقيدة",
        expertise_arabic="متكلم",
        description="Islamic theology and creed",
        system_prompt="""أنت متخصص في العقيدة الإسلامية.
تعرف أصول الإيمان والتوحيد وصفات الله.
تستطيع:
- بيان عقيدة أهل السنة والجماعة
- الرد على الشبهات العقدية
- الاستدلال بالنصوص من القرآن والسنة
- بيان صفات الله دون تأويل""",
    ),
    EnhancedAgentRole.USUL_EXPERT: DomainCategory(
        role=EnhancedAgentRole.USUL_EXPERT,
        dataset_categories=["أصول الفقه"],
        primary_category="أصول الفقه",
        expertise_arabic="أصولي",
        description="Principles of Islamic jurisprudence (Usul al-Fiqh)",
        system_prompt="""أنت متخصص في أصول الفقه.
تعرف أدلة الفقه وأنواعها (الكتاب، السنة، الإجماع، القياس).
تستطيع:
- بيان الأدلة الشرعية
- تطبيق القواعد الأصولية
- حل الإشكالات""",
    ),
    # ============ Language & Literature ============
    EnhancedAgentRole.ARABIC_LINGUIST: DomainCategory(
        role=EnhancedAgentRole.ARABIC_LINGUIST,
        dataset_categories=["اللغة العربية"],
        primary_category="اللغة العربية",
        expertise_arabic="لغوي",
        description="Arabic language and linguistics",
        system_prompt="""أنت لغوي متخصص في اللغة العربية.
تعرف أصوات العربية وصرفها.
تستطيع:
- تحليل بنية الكلمة
- بيان الاشتقاق
- شرح المعنى اللغوي""",
    ),
    EnhancedAgentRole.GRAMMAR_ANALYST: DomainCategory(
        role=EnhancedAgentRole.GRAMMAR_ANALYST,
        dataset_categories=["النحو والصرف"],
        primary_category="النحو والصرف",
        expertise_arabic="نحوي",
        description="Arabic grammar and morphology",
        system_prompt="""أنت نحوي متخصص في النحو والصرف.
تعرف قواعد العربية وإعرابها.
تستطيع:
- إعراب الجملة العربية
- بيان الإعراب والبناء
- شرح الظواهر النحوية والصرفية""",
    ),
    EnhancedAgentRole.LEXICOGRAPHER: DomainCategory(
        role=EnhancedAgentRole.LEXICOGRAPHER,
        dataset_categories=["الغريب والمعاجم"],
        primary_category="الغريب والمعاجم",
        expertise_arabic="مُعجمي",
        description="Rare words and dictionaries",
        system_prompt="""أنت متخصص في الغريب والمعاجم.
تعرف المعاني اللغوية والنادرة.
تستطيع:
- شرح الكلمات الغريبة
- تتبع الاشتقاق
- ذكر المرادف والمقابل""",
    ),
    EnhancedAgentRole.LITERATURE_ANALYST: DomainCategory(
        role=EnhancedAgentRole.LITERATURE_ANALYST,
        dataset_categories=["الأدب"],
        primary_category="الأدب",
        expertise_arabic="أديب",
        description="Arabic literature analysis",
        system_prompt="""أنت أديب متخصص في الأدب العربي.
تعرف أساليب الأدب وفنونه.
تستطيع:
- تحليل النصوص الأدبية
- بيان الأساليب البلاغية
- شرح الدلالات""",
    ),
    EnhancedAgentRole.POETRY_ANALYST: DomainCategory(
        role=EnhancedAgentRole.POETRY_ANALYST,
        dataset_categories=["الدواوين الشعرية", "العروض والقوافي"],
        primary_category="الدواوين الشعرية",
        expertise_arabic="عروضي",
        description="Arabic poetry and meter",
        system_prompt="""أنت متخصص في الشعر العربي والعروض.
تعرف بحور الشعر وقوافيه.
تستطيع:
- تحليل البحر الشعري
- بيان القافية والروي
- شرح القصيدة""",
    ),
    EnhancedAgentRole.RHETORIC_EXPERT: DomainCategory(
        role=EnhancedAgentRole.RHETORIC_EXPERT,
        dataset_categories=["البلاغة"],
        primary_category="البلاغة",
        expertise_arabic="بلاغي",
        description="Arabic rhetoric and stylistics",
        system_prompt="""أنت بلاغي متخصص في علوم البلاغة.
تعرف البيان والمعاني والبيان.
تستطيع:
- تحليل التشبيهات والاستعارات
- بيان المحسنات البديعية
- شرح الأساليب البلاغية""",
    ),
    # ============ History & Biography ============
    EnhancedAgentRole.ISLAMIC_HISTORIAN: DomainCategory(
        role=EnhancedAgentRole.ISLAMIC_HISTORIAN,
        dataset_categories=["التاريخ"],
        primary_category="التاريخ",
        expertise_arabic="مؤرخ",
        description="Islamic history",
        system_prompt="""أنت مؤرخ إسلامي.
تعرف أحداث التاريخ الإسلامي وتفسيرها.
تستطيع:
- سرد الأحداث التاريخية
- تحليل الأسباب والنتائج
- الاستشهاد بالمصادر التاريخية""",
    ),
    EnhancedAgentRole.BIOGRAPHER: DomainCategory(
        role=EnhancedAgentRole.BIOGRAPHER,
        dataset_categories=["التراجم والطبقات"],
        primary_category="التراجم والطبقات",
        expertise_arabic="مترجم",
        description="Biographies of scholars and historical figures",
        system_prompt="""أنت متخصص في التراجم والطبقات.
تعرف سير العلماء والمحدثين والفقهاء.
تستطيع:
- سرد ترجمة العالم
- بيان مؤلفاته
- بيانTeachers and students""",
    ),
    EnhancedAgentRole.GENEALOGIST: DomainCategory(
        role=EnhancedAgentRole.GENEALOGIST,
        dataset_categories=["الأنساب"],
        primary_category="الأنساب",
        expertis_arabic="نسابة",
        description="Arabian genealogy and lineages",
        system_prompt="""أنت نسابة متخصص في الأنساب.
تعرف نسب العرب وقبائلهم.
تستطيع:
- توضيح النسب
- بيان القبيلة والعشيرة
- تتبع النسل""",
    ),
    EnhancedAgentRole.GEOGRAPHER: DomainCategory(
        role=EnhancedAgentRole.GEOGRAPHER,
        dataset_categories=["البلدان والرحلات"],
        primary_category="البلدان والرحلات",
        expertise_arabic="جغرافي",
        description="Islamic geography and travel literature",
        system_prompt="""أنت جغرافي متخصص في البلدان الإسلامية.
تعرف المدن والبلدان ومعالمها.
تستطيع:
- وصف المكان
- بيان الأهمية التاريخية
- تتبع الرحلات""",
    ),
    # ============ Specialized Sciences ============
    EnhancedAgentRole.QIRAAT_SPECIALIST: DomainCategory(
        role=EnhancedAgentRole.QIRAAT_SPECIALIST,
        dataset_categories=["التجويد والقراءات"],
        primary_category="التجويد والقراءات",
        expertis_arabic="قارئ",
        description="Quranic recitations and tajweed",
        system_prompt="""أنت متخصص في القراءات والتجويد.
تعرف القراءات السبع والعشر.
تستطيع:
- بيان القراءات المختلفة
- تطبيق أحكام التجويد
- توضيح البيوت""",
    ),
    EnhancedAgentRole.INHERITANCE_EXPERT: DomainCategory(
        role=EnhancedAgentRole.INHERITANCE_EXPERT,
        dataset_categories=["الفرائض والوصايا"],
        primary_category="الفرائض والوصايا",
        expertise_arabic="فارض",
        description="Islamic inheritance law (Fara'id)",
        system_prompt="""أنت متخصص في الفرائض والوصايا.
تعرف أحكام المواريث.
تستطيع:
- توزيع التركة على الورثة
- حل المسائل الإرثية
- بيان أحكام الوصية""",
    ),
    EnhancedAgentRole.FATWA_RESEARCHER: DomainCategory(
        role=EnhancedAgentRole.FATWA_RESEARCHER,
        dataset_categories=["الفتاوى"],
        primary_category="الفتاوى",
        expertise_arabic="مفتي",
        description="Research Islamic fatwas",
        system_prompt="""أنت باحث في الفتاوى.
تعرف فتاوى العلماء المعاصرين والقديمين.
تستطيع:
- جمع الفتاوى في المسألة
- بيان الاتفاق والخلاف
- ذكر القول الراجح""",
    ),
    EnhancedAgentRole.LOGIC_EXPERT: DomainCategory(
        role=EnhancedAgentRole.LOGIC_EXPERT,
        dataset_categories=["المنطق"],
        primary_category="المنطق",
        expertise_arabic="منطقي",
        description="Islamic logic (Mantiq)",
        system_prompt="""أنت متخصص في المنطق.
تعرف أدوات التفكير الصحيح.
تستطيع:
- تحليل البرهان
- تحديد المغالطات
- بناء الحجة""",
    ),
    EnhancedAgentRole.MEDICAL_SCHOLAR: DomainCategory(
        role=EnhancedAgentRole.MEDICAL_SCHOLAR,
        dataset_categories=["الطب"],
        primary_category="الطب",
        expertise_arabic="طبيب",
        description="Islamic medicine and healthcare",
        system_prompt="""أنت متخصص في طب الإسلام.
تعرف الطب النبوي والأعشاب.
تستطيع:
- بيان العلاج النبوي
- شرحMedicine from classical sources
- نصائح صحية""",
    ),
    # ============ Spirituality & Ethics ============
    EnhancedAgentRole.SPIRITUALITY_GUIDE: DomainCategory(
        role=EnhancedAgentRole.SPIRITUALITY_GUIDE,
        dataset_categories=["الرقائق والآداب والأذكار"],
        primary_category="الرقائق والآداب والأذكار",
        expertise_arabic="واعظ",
        description="Islamic spirituality and moral development",
        system_prompt="""أنت رقائق وأخلاق.
تعرف آداب الإسلام والرقائق.
تستطيع:
- تقديم المواعظ
- بيان الآداب الإسلامية
- ذكر الأدعية والأذكار""",
    ),
    EnhancedAgentRole.ETHICS_TEACHER: DomainCategory(
        role=EnhancedAgentRole.ETHICS_TEACHER,
        dataset_categories=["الرقائق والآداب والأذكار"],
        primary_category="الرقائق والآداب والأذكار",
        expertise_arabic="مربٍ",
        description="Teach Islamic ethics and Remembrances",
        system_prompt="""أنت مربٍ ومعلم أخلاق.
تعرف آداب السلوك الإسلامي.
تستطيع:
- تعليم الأخلاق الحميدة
- بيان الآداب الشرعية
- تذكير بالأذكار""",
    ),
    # ============ Sects & Theology ============
    EnhancedAgentRole.SECTS_ANALYST: DomainCategory(
        role=EnhancedAgentRole.SECTS_ANALIST,
        dataset_categories=["الفرق والردود"],
        primary_category="الفرق والردود",
        expertise_arabic="مُردِّد",
        description="Analyze Islamic sects and respond to deviant groups",
        system_prompt="""أنت متخصص في الفرق والردود.
تعرف المذاهب الدينية المخالفة.
تستطيع:
- بيان ضلال المبتدعة
- الرد على الشبهات
- تحديدالفرق""",
    ),
    # ============ Reference & Catalogs ============
    EnhancedAgentRole.INDEX_EXPERT: DomainCategory(
        role=EnhancedAgentRole.INDEX_EXPERT,
        dataset_categories=["فهارس الكتب والأدلة"],
        primary_category="فهارس الكتب والأدلة",
        expertise_arabic="مُفهرس",
        description="Book indexes and catalogs",
        system_prompt="""أنت متخصص في فهارس الكتب.
تعرف أسماء الكتب ومؤلفيها.
تستطيع:
- تحديد الكتاب من وصفه
- بيان المؤلف والمؤلفات
- ربطالكتب ببعض""",
    ),
    EnhancedAgentRole.REFERENCE_LIBRARIAN: DomainCategory(
        role=EnhancedAgentRole.REFERENCE_LIBRARIAN,
        dataset_categories=["الجوامع"],
        primary_category="الجوامع",
        expertise_arabic="أمين مكتبة",
        description="Reference librarian for Islamic literature",
        system_prompt="""أنت أمين مكتبة متخصص.
تعرف مصادر المعلومات الإسلامية.
تستطيع:
- توجيه الباحث للمصادر المناسبة
- تلخيص المحتويات
- ربط الموضوعات""",
    ),
    # ============ General ============
    EnhancedAgentRole.GENERAL_RESEARCHER: DomainCategory(
        role=EnhancedAgentRole.GENERAL_RESEARCHER,
        dataset_categories=["كتب عامة", "علوم أخرى"],
        primary_category="كتب عامة",
        expertise_arabic="باحث",
        description="General research on Islamic topics",
        system_prompt="""أنت باحث إسلامي عام.
تعرف موضوعات متنوعة من العلوم الإسلامية.
تستطيع:
- البحث في موضوعات مختلفة
- الجمع بين التخصصات
- التوضيح للجميع""",
    ),
}


class EnhancedAgentSystem:
    """
    Enhanced agent system with specialized roles for all Islamic knowledge domains.

    Extends the base agent system with:
    - 30+ specialized roles matching 40+ dataset categories
    - Domain-aware query routing
    - Category-based retrieval filtering
    - Specialized prompts per domain
    """

    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline
        self.agents: Dict[EnhancedAgentRole, Any] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all specialized agents."""

        for role, domain_config in DOMAIN_MAPPING.items():
            self.agents[role] = self._create_agent_for_domain(role, domain_config)

    def _create_agent_for_domain(
        self,
        role: EnhancedAgentRole,
        config: DomainCategory,
    ) -> Any:
        """Create agent for specific domain."""

        return {
            "role": role,
            "config": config,
            "categories": config.dataset_categories,
            "prompt": config.system_prompt,
        }

    def get_agent_for_category(self, category: str) -> Optional[EnhancedAgentRole]:
        """Find appropriate agent role for a category."""

        for role, config in DOMAIN_MAPPING.items():
            if category in config.dataset_categories:
                return role

        return None

    def get_agents_for_query(self, query: str) -> List[EnhancedAgentRole]:
        """Determine which agents are relevant for a query."""

        relevant_agents = []

        # Simple keyword matching (would use NLP in production)
        query_lower = query.lower()

        # Map keywords to roles
        keyword_map = {
            "تفسير": [EnhancedAgentRole.QURAN_RESEARCHER],
            "آية": [EnhancedAgentRole.QURAN_RESEARCHER],
            "حديث": [EnhancedAgentRole.HADITH_SPECIALIST],
            "صحيح": [EnhancedAgentRole.HADITH_SPECIALIST],
            "فقه": [EnhancedAgentRole.FIQH_SCHOLAR],
            "حكم": [EnhancedAgentRole.FIQH_SCHOLAR],
            "عقيدة": [EnhancedAgentRole.AQEEDAH_THEOLOGIAN],
            "توحيد": [EnhancedAgentRole.AQEEDAH_THEOLOGIAN],
            "نحو": [EnhancedAgentRole.GRAMMAR_ANALYST],
            "صرف": [EnhancedAgentRole.GRAMMAR_ANALYST],
            "شعر": [EnhancedAgentRole.POETRY_ANALYST],
            "أدب": [EnhancedAgentRole.LITERATURE_ANALYST],
            "تاريخ": [EnhancedAgentRole.ISLAMIC_HISTORIAN],
            "ترجمة": [EnhancedAgentRole.BIOGRAPHER],
            "أنساب": [EnhancedAgentRole.GENEALOGIST],
            "بلدان": [EnhancedAgentRole.GEOGRAPHER],
            "قراءة": [EnhancedAgentRole.QIRAAT_SPECIALIST],
            "تجويد": [EnhancedAgentRole.QIRAAT_SPECIALIST],
            "فرائض": [EnhancedAgentRole.INHERITANCE_EXPERT],
            "إرث": [EnhancedAgentRole.INHERITANCE_EXPERT],
            "فتوى": [EnhancedAgentRole.FATWA_RESEARCHER],
            "منطق": [EnhancedAgentRole.LOGIC_EXPERT],
            "طب": [EnhancedAgentRole.MEDICAL_SCHOLAR],
            "رقائق": [EnhancedAgentRole.SPIRITUALITY_GUIDE],
            "أذكار": [EnhancedAgentRole.ETHICS_TEACHER],
            "فرق": [EnhancedAgentRole.SECTS_ANALYST],
            "ردود": [EnhancedAgentRole.SECTS_ANALYST],
            "فهرس": [EnhancedAgentRole.INDEX_EXPERT],
            "أصول": [EnhancedAgentRole.USUL_EXPERT],
        }

        for keyword, roles in keyword_map.items():
            if keyword in query_lower:
                for role in roles:
                    if role not in relevant_agents:
                        relevant_agents.append(role)

        # Default to general researcher if no matches
        if not relevant_agents:
            relevant_agents = [EnhancedAgentRole.GENERAL_RESEARCHER]

        return relevant_agents

    async def query_with_routing(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query with automatic agent routing."""

        if not self.pipeline:
            return {"error": "Pipeline not initialized"}

        # Determine relevant agents
        relevant_roles = self.get_agents_for_query(query)

        # Collect results from relevant agents
        results = {}

        for role in relevant_roles:
            agent = self.agents.get(role)
            if agent:
                # Get category filter
                categories = agent["categories"]

                # Query with filter
                result = await self.pipeline.query(
                    query,
                    top_k=top_k,
                    filters={"category": {"$in": categories}},
                )

                results[role.value] = {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "domain": agent["config"].primary_category,
                }

        # Synthesize results
        synthesis = self._synthesize_results(query, results)

        return {
            "query": query,
            "results": results,
            "synthesis": synthesis,
            "used_agents": [r.value for r in relevant_roles],
        }

    def _synthesize_results(
        self,
        query: str,
        results: Dict[str, Any],
    ) -> str:
        """Synthesize results from multiple agents."""

        synthesis = f"## السؤال: {query}\n\n"
        synthesis += "### النتائج من الخبراء المتخصصين:\n\n"

        for agent_name, result in results.items():
            synthesis += f"**{agent_name}** ({result.get('domain', '')}):\n"
            synthesis += f"{result.get('answer', '')[:300]}...\n\n"

        return synthesis

    def get_all_roles(self) -> List[EnhancedAgentRole]:
        """Get list of all available roles."""
        return list(DOMAIN_MAPPING.keys())

    def get_role_info(self, role: EnhancedAgentRole) -> Optional[Dict[str, Any]]:
        """Get information about a specific role."""

        config = DOMAIN_MAPPING.get(role)
        if not config:
            return None

        return {
            "role": role.value,
            "name": config.expertise_arabic,
            "categories": config.dataset_categories,
            "primary": config.primary_category,
            "description": config.description,
        }


# Factory function
def create_enhanced_agent(
    role: str,
    pipeline: Any = None,
) -> Optional[Any]:
    """Create an enhanced agent for a specific role."""

    role_map = {
        # Core Religious
        "quran": EnhancedAgentRole.QURAN_RESEARCHER,
        "hadith": EnhancedAgentRole.HADITH_SPECIALIST,
        "fiqh": EnhancedAgentRole.FIQH_SCHOLAR,
        "hanafi": EnhancedAgentRole.HANAFI_SCHOLAR,
        "maliki": EnhancedAgentRole.MALIKI_SCHOLAR,
        "shafii": EnhancedAgentRole.SHAFII_SCHOLAR,
        "hanbali": EnhancedAgentRole.HANBALI_SCHOLAR,
        "aqeedah": EnhancedAgentRole.AQEEDAH_THEOLOGIAN,
        "usul": EnhancedAgentRole.USUL_EXPERT,
        # Language
        "linguist": EnhancedAgentRole.ARABIC_LINGUIST,
        "grammar": EnhancedAgentRole.GRAMMAR_ANALYST,
        "lexicon": EnhancedAgentRole.LEXICOGRAPHER,
        "literature": EnhancedAgentRole.LITERATURE_ANALYST,
        "poetry": EnhancedAgentRole.POETRY_ANALYST,
        "rhetoric": EnhancedAgentRole.RHETORIC_EXPERT,
        # History
        "history": EnhancedAgentRole.ISLAMIC_HISTORIAN,
        "biography": EnhancedAgentRole.BIOGRAPHER,
        "genealogy": EnhancedAgentRole.GENEALOGIST,
        "geography": EnhancedAgentRole.GEOGRAPHER,
        # Specialized
        "qiraat": EnhancedAgentRole.QIRAAT_SPECIALIST,
        "inheritance": EnhancedAgentRole.INHERITANCE_EXPERT,
        "fatwa": EnhancedAgentRole.FATWA_RESEARCHER,
        "logic": EnhancedAgentRole.LOGIC_EXPERT,
        "medicine": EnhancedAgentRole.MEDICAL_SCHOLAR,
        # Spirituality
        "spirituality": EnhancedAgentRole.SPIRITUALITY_GUIDE,
        "ethics": EnhancedAgentRole.ETHICS_TEACHER,
        # Sects
        "seects": EnhancedAgentRole.SECTS_ANALYST,
        # Reference
        "index": EnhancedAgentRole.INDEX_EXPERT,
        "library": EnhancedAgentRole.REFERENCE_LIBRARIAN,
        # General
        "general": EnhancedAgentRole.GENERAL_RESEARCHER,
    }

    role_enum = role_map.get(role.lower())

    if not role_enum:
        return None

    system = EnhancedAgentSystem(pipeline)
    return system.agents.get(role_enum)


def create_enhanced_system(pipeline: Any = None) -> EnhancedAgentSystem:
    """Create the complete enhanced agent system."""

    return EnhancedAgentSystem(pipeline)

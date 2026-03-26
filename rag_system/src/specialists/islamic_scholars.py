"""
Islamic Scholar Specialist System

Advanced domain-specific capabilities for Islamic scholarship.
Each specialist is optimized for a specific category of knowledge.

Categories:
1. Tafsir (Quranic Exegesis) - التفسير
2. Hadith Studies - علوم الحديث
3. Fiqh (Jurisprudence) - الفقه
4. Aqeedah (Theology) - العقيدة
5. Arabic Language - اللغة العربية
6. Islamic History - التاريخ الإسلامي
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class IslamicDomain(Enum):
    """Islamic knowledge domains."""

    QURAN = "quran"  # التفسير and related
    HADITH = "hadith"  # الحديث and related
    FIQH_HANAFI = "fiqh_hanafi"  # الفقه الحنفي
    FIQH_MALIKI = "fiqh_maliki"  # الفقه المالكي
    FIQH_SHAFII = "fiqh_shafii"  # الفقه الشافعي
    FIQH_HANBALI = "fiqh_hanbali"  # الفقه الحنبلي
    FIQH_COMPARATIVE = "fiqh_comparative"  # Comparative Fiqh
    AQEEDAH = "aqeedah"  # العقيدة
    ARABIC_LANGUAGE = "arabic"  # اللغة العربية
    HISTORY = "history"  # التاريخ
    SPIRITUALITY = "spirituality"  # الرقائق
    LITERATURE = "literature"  # الأدب


@dataclass
class DomainConfig:
    """Configuration for a specialized domain."""

    domain: IslamicDomain
    name_ar: str
    name_en: str
    categories: List[str]
    priority_sources: List[str] = field(default_factory=list)
    specialized_prompts: Dict[str, str] = field(default_factory=dict)
    chunking_strategy: str = "semantic"  # or "fixed"
    chunk_size: int = 768


# Domain-specific configurations
DOMAIN_CONFIGS = {
    IslamicDomain.QURAN: DomainConfig(
        domain=IslamicDomain.QURAN,
        name_ar="التفسير وعلوم القرآن",
        name_en="Quranic Exegesis",
        categories=["التفسير", "علوم القرآن وأصول التفسير", "التجويد والقراءات"],
        specialized_prompts={
            "tafsir_style": "أورد الآية القرآنية أولاً ثم أشرحها بأسلوب المفسرين الكلاسيكيين",
            "tafsir_source": "أستشهد بأقوال المفسرين مثل ابن كثير والقرطبي والسعدي",
            "qiraat": "أذكر القراءات المختلفة للآية عند الحاجة",
        },
        chunking_strategy="semantic",
        chunk_size=1024,
    ),
    IslamicDomain.HADITH: DomainConfig(
        domain=IslamicDomain.HADITH,
        name_ar="علوم الحديث",
        name_en="Hadith Sciences",
        categories=[
            "كتب السنة",
            "شروح الحديث",
            "التخريج والأطراف",
            "العلل والسؤلات الحديثية",
            "علوم الحديث",
        ],
        specialized_prompts={
            "chain": "أبدأ بالإسناد ثم أذكر المتن",
            "grades": "أذكر درجة الحديث (صحيح، حسن، ضعيف) ومحدداته",
            "sources": "أستشهد بصحيح البخاري ومسلم والكتب الستة",
        },
        chunking_strategy="fixed",
        chunk_size=512,
    ),
    IslamicDomain.FIQH_HANAFI: DomainConfig(
        domain=IslamicDomain.FIQH_HANAFI,
        name_ar="الفقه الحنفي",
        name_en="Hanafi Jurisprudence",
        categories=["الفقه الحنفي"],
        specialized_prompts={
            "madhhab": "أجيب according to Hanafi jurisprudence",
            "sources": "أستشهد بمذهب أبي حنيفة وأصحابه",
            "granular": "أفصل بين القول الراجح والمرجوح",
        },
        chunk_size=768,
    ),
    IslamicDomain.FIQH_MALIKI: DomainConfig(
        domain=IslamicDomain.FIQH_MALIKI,
        name_ar="الفقه المالكي",
        name_en="Maliki Jurisprudence",
        categories=["الفقه المالكي"],
        specialized_prompts={
            "madhhab": "أجيب according to Maliki jurisprudence",
            "sources": "أستشهد بموطأ مالك ومدونة Cologne",
        },
        chunk_size=768,
    ),
    IslamicDomain.FIQH_SHAFII: DomainConfig(
        domain=IslamicDomain.FIQH_SHAFII,
        name_ar="الفقه الشافعي",
        name_en="Shafi'i Jurisprudence",
        categories=["الفقه الشافعي"],
        specialized_prompts={
            "madhhab": "أجيب according to Shafi'i jurisprudence",
            "sources": "أستشهد بأمهات الكتب الشافعية",
        },
        chunk_size=768,
    ),
    IslamicDomain.FIQH_HANBALI: DomainConfig(
        domain=IslamicDomain.FIQH_HANBALI,
        name_ar="الفقه الحنبلي",
        name_en="Hanbali Jurisprudence",
        categories=["الفقه الحنبلي"],
        specialized_prompts={
            "madhhab": "أجيب according to Hanbali jurisprudence",
            "sources": "أستشهد بمجموع الفتاوى والفتاوى الهندية",
        },
        chunk_size=768,
    ),
    IslamicDomain.FIQH_COMPARATIVE: DomainConfig(
        domain=IslamicDomain.FIQH_COMPARATIVE,
        name_ar="الفقه المقارن",
        name_en="Comparative Islamic Jurisprudence",
        categories=[
            "الفقه العام",
            "مسائل فقهية",
            "أصول الفقه",
            "علوم الفقه والقواعد الفقهية",
        ],
        specialized_prompts={
            "compare": "أقارن بين آراء المذاهب الأربعة",
            "evidence": "أستدل بالأدلة من القرآن والسنة",
            "conclusion": "أذكر القول الراجح مع توضيح الخلاف",
        },
        chunking_strategy="semantic",
        chunk_size=1024,
    ),
    IslamicDomain.AQEEDAH: DomainConfig(
        domain=IslamicDomain.AQEEDAH,
        name_ar="العقيدة",
        name_en="Islamic Theology",
        categories=["العقيدة", "الفرق والردود"],
        specialized_prompts={
            "salafi": "أجيب according to Salafi methodology",
            "proofs": "أستدل بالنصوص من القرآن والسنة",
            "attributes": "أثبت صفات الله كما جاءت без تأويل",
            "avoid": "أتجنب الجدل الكلامي المبتدع",
        },
        chunking_strategy="semantic",
        chunk_size=768,
    ),
    IslamicDomain.ARABIC_LANGUAGE: DomainConfig(
        domain=IslamicDomain.ARABIC_LANGUAGE,
        name_ar="اللغة العربية",
        name_en="Arabic Language",
        categories=["اللغة العربية", "النحو والصرف", "الغريب والمعاجم", "كتب اللغة"],
        specialized_prompts={
            "grammar": "أشرح القواعد النحوية والصرفية",
            "examples": "أذكر أمثلة من القرآن والشعر",
            "derivations": "أبين الاشتقاق والمأخذ",
        },
        chunking_strategy="semantic",
        chunk_size=512,
    ),
    IslamicDomain.HISTORY: DomainConfig(
        domain=IslamicDomain.HISTORY,
        name_ar="التاريخ الإسلامي",
        name_en="Islamic History",
        categories=["التاريخ", "التراجم والطبقات", "الأنساب", "البلدان والرحلات"],
        specialized_prompts={
            "timeline": "أرتب الأحداث زمنياً",
            "sources": "أستشهد بمؤرخي الإسلام",
            "context": "أوفر السياق التاريخي للأحداث",
        },
        chunking_strategy="semantic",
        chunk_size=1024,
    ),
    IslamicDomain.SPIRITUALITY: DomainConfig(
        domain=IslamicDomain.SPIRITUALITY,
        name_ar="الرقائق والآداب",
        name_en="Islamic Spirituality",
        categories=["الرقائق والآداب والأذكار", "السيرة النبوية"],
        specialized_prompts={
            "spiritual": "أذكر الحكم والموعظة",
            "practical": "أ给出 Practical advice للسلوك",
            "dhikr": "أذكر الأدعية والأذكار المناسبة",
        },
        chunking_strategy="semantic",
        chunk_size=768,
    ),
    IslamicDomain.LITERATURE: DomainConfig(
        domain=IslamicDomain.LITERATURE,
        name_ar="الأدب العربي",
        name_en="Arabic Literature",
        categories=["الأدب", "العروض والقوافي", "الدواوين الشعرية", "البلاغة"],
        specialized_prompts={
            "analysis": "أحلل النصوص الأدبية",
            "poetry": "أشرح القصائد مع ذكر بحورها",
            "rhetoric": "أبيان البلاغة والأساليب",
        },
        chunking_strategy="semantic",
        chunk_size=1024,
    ),
}


class IslamicScholar:
    """
    Domain-specific Islamic scholar specialist.

    Each instance is specialized for a specific domain of Islamic knowledge
    with optimized prompts, retrieval, and processing.
    """

    def __init__(self, domain: IslamicDomain, base_pipeline: Any = None):
        self.domain = domain
        self.config = DOMAIN_CONFIGS.get(domain)
        self.base_pipeline = base_pipeline

        if not self.config:
            raise ValueError(f"Unknown domain: {domain}")

    def get_system_prompt(self) -> str:
        """Get specialized system prompt for this domain."""

        base_prompt = """أنت باحث متخصص في مجال {domain_ar} من العلوم الإسلامية.
لديك معرفة عميقة بالنصوص الكلاسيكية والمصادر الأصيلة.

عند الإجابة:
{domain_instructions}

مهم:
- استخدم المصادر الأساسية في هذا المجال
- استشهد بالكتب والمؤلفين المعتمدين
- حافظ على الدقة العلمية
- اعترف بما لا تعرفه""".format(
            domain_ar=self.config.name_ar,
            domain_instructions="\n".join(
                [f"- {inst}" for inst in self.config.specialized_prompts.values()]
            ),
        )

        return base_prompt

    def get_filter(self) -> Dict[str, Any]:
        """Get metadata filter for this domain."""

        return {"category": {"$in": self.config.categories}}

    async def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query with domain specialization."""

        if not self.base_pipeline:
            raise RuntimeError("Base pipeline not set")

        # Apply domain-specific filter
        result = await self.base_pipeline.query(
            question=question,
            top_k=top_k,
            filters=self.get_filter(),
        )

        # Add domain metadata
        result.domain = self.config.name_ar

        return result

    def format_for_madhhab(
        self,
        book_title: str,
        author: str,
    ) -> Dict[str, str]:
        """Format citation in madhhab style."""

        return {
            "book": book_title,
            "author": author,
            "madhhab": self.config.name_ar,
        }


class ComparativeFiqhScholar(IslamicScholar):
    """
    Specialized scholar for comparative fiqh analysis.

    Can compare rulings across the four madhhabs and identify consensus.
    """

    def __init__(self, base_pipeline: Any = None):
        super().__init__(IslamicDomain.FIQH_COMPARATIVE, base_pipeline)
        self.madhhab_scholars = {
            "hanafi": IslamicScholar(IslamicDomain.FIQH_HANAFI, base_pipeline),
            "maliki": IslamicScholar(IslamicDomain.FIQH_MALIKI, base_pipeline),
            "shafii": IslamicScholar(IslamicDomain.FIQH_SHAFII, base_pipeline),
            "hanbali": IslamicScholar(IslamicDomain.FIQH_HANBALI, base_pipeline),
        }

    async def query_with_comparison(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """Query with full madhhab comparison."""

        results = {}

        # Get results from each madhhab
        for madhhab, scholar in self.madhhab_scholars.items():
            try:
                result = await scholar.query(question, top_k=2)
                results[madhhab] = {
                    "answer": result.answer,
                    "sources": result.sources,
                }
            except Exception as e:
                results[madhhab] = {"error": str(e)}

        # Analyze consensus
        consensus = self._analyze_consensus(results)

        return {
            "question": question,
            "madhhab_results": results,
            "consensus": consensus,
            "analysis": self._generate_analysis(results, consensus),
        }

    def _analyze_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus (Ijma) among madhhab."""

        answers = []
        for madhhab, result in results.items():
            if "error" not in result:
                answers.append(madhhab)

        if len(answers) == 4:
            return {
                "type": "full_consensus",
                "message": "الإجماع - جميع المذاهب الأربعة على رأي واحد",
            }
        elif len(answers) >= 3:
            return {
                "type": "majority",
                "message": f"الأغلبية ({len(answers)} من 4) على رأي واحد",
            }
        else:
            return {
                "type": "disagreement",
                "message": "خلاف بين المذاهب",
            }

    def _generate_analysis(
        self,
        results: Dict[str, Any],
        consensus: Dict[str, Any],
    ) -> str:
        """Generate comparative analysis text."""

        analysis = "## Comparison Analysis\n\n"
        analysis += f"**Consensus:** {consensus['message']}\n\n"

        for madhhab, result in results.items():
            if "error" not in result:
                analysis += f"### {madhhab.upper()}\n"
                analysis += f"{result['answer'][:200]}...\n\n"

        return analysis


class ChainOfScholarship:
    """
    Chain-of-thought reasoning for Islamic scholarship.

    Follows the methodology of classical scholars:
    1. Identify the issue
    2. Present evidence (Quran, Sunnah, Ijma, Qiyas)
    3. Present different opinions
    4. Conclude with strongest view
    """

    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline

    async def reason(
        self,
        question: str,
        include_evidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning on Islamic question.

        Steps:
        1. Analyze the question
        2. Retrieve relevant evidence
        3. Present scholarly opinions
        4. Form conclusion
        """

        steps = []

        # Step 1: Question Analysis
        steps.append(
            {
                "step": "analysis",
                "title": "تحليل السؤال",
                "content": self._analyze_question(question),
            }
        )

        # Step 2: Evidence Retrieval
        if include_evidence:
            evidence = await self._retrieve_evidence(question)
            steps.append(
                {
                    "step": "evidence",
                    "title": "الأدلة",
                    "content": evidence,
                }
            )

        # Step 3: Scholarly Opinions
        opinions = await self._retrieve_opinions(question)
        steps.append(
            {
                "step": "opinions",
                "title": "آراء العلماء",
                "content": opinions,
            }
        )

        # Step 4: Conclusion
        conclusion = self._form_conclusion(steps)
        steps.append(
            {
                "step": "conclusion",
                "title": "الخلاصة",
                "content": conclusion,
            }
        )

        return {
            "question": question,
            "reasoning_chain": steps,
            "final_answer": conclusion,
        }

    def _analyze_question(self, question: str) -> str:
        """Analyze the question to identify topic and type."""

        topic = "unknown"
        question_type = "factual"

        # Simple keyword-based analysis
        if any(word in question for word in ["حكم", "هل", "يجوز"]):
            question_type = "ruling"
        elif any(word in question for word in ["ما", "تعريف"]):
            question_type = "definitional"
        elif any(word in question for word in ["من", "who"]):
            question_type = "biographical"

        return f"Type: {question_type}\nTopic extraction needed"

    async def _retrieve_evidence(self, question: str) -> Dict[str, Any]:
        """Retrieve relevant evidence from Quran and Hadith."""

        if not self.pipeline:
            return {"quran_verses": [], "hadith": []}

        # Search for Quranic evidence
        quran_results = await self.pipeline.query(
            f"آيات قرآنية related to: {question}",
            top_k=3,
        )

        # Search for Hadith evidence
        hadith_results = await self.pipeline.query(
            f"أحاديث related to: {question}",
            top_k=3,
        )

        return {
            "quran_verses": [
                {
                    "text": r.sources[0]["book_title"] if r.sources else "",
                    "source": "Quran",
                }
                for r in [quran_results]
            ],
            "hadith": hadith_results.sources,
        }

    async def _retrieve_opinions(self, question: str) -> Dict[str, Any]:
        """Retrieve different scholarly opinions."""

        if not self.pipeline:
            return {"opinions": []}

        # Search for scholarly opinions
        results = await self.pipeline.query(
            f"آراء العلماء في: {question}",
            top_k=5,
        )

        return {
            "opinions": [
                {
                    "source": s["book_title"],
                    "author": s["author"],
                    "opinion": s.get("content_preview", "")[:200],
                }
                for s in results.sources
            ]
        }

    def _form_conclusion(self, steps: List[Dict[str, Any]]) -> str:
        """Form final conclusion based on reasoning chain."""

        # Extract key points from steps
        evidence = next((s for s in steps if s["step"] == "evidence"), None)
        opinions = next((s for s in steps if s["step"] == "opinions"), None)

        conclusion = "## Conclusion\n\n"

        if evidence:
            conclusion += "Based on evidence from Quran and Sunnah, "

        if opinions:
            conclusion += "and after reviewing scholarly opinions, "

        conclusion += "the preferred view is as follows:\n\n"
        conclusion += "[Detailed ruling based on the evidence]"

        return conclusion


class CitationBuilder:
    """
    Build Islamic-style citations (Isnad-style).

    Traditional format:
    - Book > Author > Category > Chain
    """

    @staticmethod
    def build_citation(
        book_title: str,
        author: str,
        category: str,
        page: Optional[str] = None,
    ) -> str:
        """Build a proper Islamic scholarly citation."""

        citation = f"**{book_title}**"
        citation += f" - تأليف: {author}"

        if category:
            citation += f"\n*التصنيف: {category}*"

        if page:
            citation += f"\n*صفحة: {page}*"

        return citation

    @staticmethod
    def build_hadith_citation(
        book: str,
        hadith_number: int,
        narrator: Optional[str] = None,
    ) -> str:
        """Build hadith citation in traditional format."""

        citation = f"**{book}** ({hadith_number})"

        if narrator:
            citation += f"\n*رواه: {narrator}*"

        return citation

    @staticmethod
    def build_fiqh_citation(
        book: str,
        chapter: str,
        ruling: str,
        madhhab: Optional[str] = None,
    ) -> str:
        """Build fiqh ruling citation."""

        citation = f"**{book}**"
        citation += f"\n*الباب: {chapter}*"
        citation += f"\n*الحكم: {ruling}*"

        if madhhab:
            citation += f"\n*المذهب: {madhhab}*"

        return citation


# Factory function
def create_islamic_scholar(
    domain: str,
    base_pipeline: Any = None,
) -> IslamicScholar:
    """Create a specialized Islamic scholar."""

    domain_map = {
        "quran": IslamicDomain.QURAN,
        "hadith": IslamicDomain.HADITH,
        "fiqh": IslamicDomain.FIQH_COMPARATIVE,
        "fiqh_hanafi": IslamicDomain.FIQH_HANAFI,
        "fiqh_maliki": IslamicDomain.FIQH_MALIKI,
        "fiqh_shafii": IslamicDomain.FIQH_SHAFII,
        "fiqh_hanbali": IslamicDomain.FIQH_HANBALI,
        "aqeedah": IslamicDomain.AQEEDAH,
        "arabic": IslamicDomain.ARABIC_LANGUAGE,
        "history": IslamicDomain.HISTORY,
        "spirituality": IslamicDomain.SPIRITUALITY,
        "literature": IslamicDomain.LITERATURE,
    }

    domain_enum = domain_map.get(domain.lower())

    if not domain_enum:
        raise ValueError(f"Unknown domain: {domain}")

    return IslamicScholar(domain_enum, base_pipeline)


def create_comparative_fiqh_scholar(
    base_pipeline: Any = None,
) -> ComparativeFiqhScholar:
    """Create a comparative fiqh scholar."""

    return ComparativeFiqhScholar(base_pipeline)

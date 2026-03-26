"""
Arabic LLM Fine-Tuning Data Schema

This module defines the JSONL schema for Arabic language model training data,
following the specification from LLM_Arabic_plan.md

Schema supports 5 roles:
- tutor: Language teacher (نحو، بلاغة، تعليم)
- proofreader: Grammar corrector (تصحيح لغوي)
- poet: Poetry composition (شعر، نظم، نقد)
- muhhaqiq: Text investigator (تحقيق، تراث، نصوص قديمة)
- assistant_general: General Arabic assistant
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum
import json
import hashlib
import uuid
from datetime import datetime


class Role(Enum):
    """Training example roles"""
    TUTOR = "tutor"
    PROOFREADER = "proofreader"
    POET = "poet"
    MUHHAQIQ = "muhhaqiq"
    ASSISTANT_GENERAL = "assistant_general"


class Skill(Enum):
    """Linguistic skills"""
    NAHW = "nahw"  # نحو - Grammar
    SARF = "sarf"  # صرف - Morphology
    BALAGHA = "balagha"  # بلاغة - Rhetoric
    ORTHOGRAPHY = "orthography"  # إملاء - Spelling
    QA = "qa"  # أسئلة وأجوبة
    STYLE_EDITING = "style_editing"  # تحرير الأسلوب
    POETRY = "poetry"  # شعر
    HERITAGE = "heritage"  # تراث


class Level(Enum):
    """Proficiency levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Domain(Enum):
    """Content domains"""
    EDUCATION = "education"
    BUSINESS = "business"
    ACADEMIC = "academic"
    ISLAMIC_STUDIES = "islamic_studies"
    GENERAL = "general"
    HERITAGE = "heritage"
    LITERATURE = "literature"


class Style(Enum):
    """Language styles"""
    FUSHА_CLASSICAL = "fusha_classical"  # فصحى تراثية
    FUSHА_MODERN = "fusha_modern"  # فصحى حديثة
    DIALECT_EGYPTIAN = "dialect_egyptian"  # عامية مصرية
    DIALECT_LEVANTINE = "dialect_levantine"  # عامية شامية
    DIALECT_GULF = "dialect_gulf"  # عامية خليجية
    MIXED = "mixed"  # مختلط


class TaskType(Enum):
    """Task types"""
    EXPLANATION = "explanation"  # شرح
    QA = "qa"  # سؤال وجواب
    CORRECTION = "correction"  # تصحيح
    REWRITE = "rewrite"  # إعادة صياغة
    ANALYSIS_AND_EXPLANATION = "analysis_and_explanation"  # تحليل وشرح
    GENERATION = "generation"  # توليد
    SUMMARIZATION = "summarization"  # تلخيص
    TRANSLATION = "translation"  # ترجمة
    CLASSIFICATION = "classification"  # تصنيف


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
    """Configuration for dataset generation"""
    
    # Role distribution (should sum to 1.0)
    role_distribution: dict = field(default_factory=lambda: {
        "tutor": 0.40,
        "proofreader": 0.25,
        "poet": 0.15,
        "muhhaqiq": 0.15,
        "assistant_general": 0.05,
    })
    
    # Skill distribution within roles
    skill_distribution: dict = field(default_factory=lambda: {
        "tutor": {"nahw": 0.4, "balagha": 0.3, "sarf": 0.2, "qa": 0.1},
        "proofreader": {"orthography": 0.4, "nahw": 0.4, "style_editing": 0.2},
        "poet": {"poetry": 0.7, "balagha": 0.3},
        "muhhaqiq": {"heritage": 0.4, "nahw": 0.3, "balagha": 0.3},
        "assistant_general": {"nahw": 0.5, "qa": 0.5},
    })
    
    # Level distribution
    level_distribution: dict = field(default_factory=lambda: {
        "beginner": 0.3,
        "intermediate": 0.5,
        "advanced": 0.2,
    })
    
    # Target dataset size
    target_examples: int = 50000
    
    # Source categories from Shamela
    source_categories: List[str] = field(default_factory=lambda: [
        "كتب اللغة",  # Language books
        "التفسير",  # Quranic exegesis
        "كتب السنة",  # Hadith
        "الأدب",  # Literature
        "الشعر",  # Poetry
        "البلاغة",  # Rhetoric
        "النحو",  # Grammar
        "التجويد والقراءات",  # Quranic recitation
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
    
    # Arabic content check (at least some Arabic characters)
    arabic_chars = sum(1 for c in example.instruction + example.output if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 10:
        errors.append("Content should contain Arabic text")
    
    # Difficulty range
    if not 1 <= example.difficulty <= 5:
        errors.append("Difficulty must be between 1 and 5")
    
    # Skills should match role
    role_skill_map = {
        "tutor": ["nahw", "balagha", "sarf", "qa"],
        "proofreader": ["orthography", "nahw", "style_editing"],
        "poet": ["poetry", "balagha"],
        "muhhaqiq": ["heritage", "nahw", "balagha"],
        "assistant_general": ["nahw", "qa"],
    }
    valid_skills = role_skill_map.get(example.role.value, [])
    for skill in example.skills:
        if skill.value not in valid_skills:
            errors.append(f"Skill '{skill.value}' not valid for role '{example.role.value}'")
    
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

"""
Tests for Arabic LLM Schema Module

Test data models, validation, and utilities.
"""

import pytest
from arabic_llm.core import (
    TrainingExample,
    Role,
    Skill,
    Level,
    validate_example,
)


class TestTrainingExample:
    """Test TrainingExample dataclass"""

    def test_create_example(self):
        """Test creating a valid training example"""
        example = TrainingExample(
            instruction="أعرب الجملة",
            input="العلمُ نورٌ",
            output="العلمُ: مبتدأ",
            role=Role.TUTOR,
            skills=[Skill.NAHW],
            level=Level.INTERMEDIATE,
        )

        assert example.role == Role.TUTOR
        assert Skill.NAHW in example.skills
        assert example.level == Level.INTERMEDIATE

    def test_example_with_book_metadata(self):
        """Test example with book metadata"""
        example = TrainingExample(
            instruction="أعرب",
            input="نص",
            output="إعراب",
            role=Role.TUTOR,
            skills=[Skill.NAHW],
            level=Level.BEGINNER,
            book_id=10018,
            book_title="النحو الواضح",
            author_name="محمود أبو سريع",
        )

        assert example.book_id == 10018
        assert example.book_title == "النحو الواضح"


class TestValidation:
    """Test example validation"""

    def test_valid_example(self):
        """Test validation of valid example"""
        example = TrainingExample(
            instruction="أعرب",
            input="جملة",
            output="إعراب",
            role=Role.TUTOR,
            skills=[Skill.NAHW],
            level=Level.BEGINNER,
        )

        errors = validate_example(example)
        assert len(errors) == 0

    def test_invalid_example_missing_instruction(self):
        """Test validation fails without instruction"""
        example = TrainingExample(
            instruction="",
            input="جملة",
            output="إعراب",
            role=Role.TUTOR,
            skills=[Skill.NAHW],
            level=Level.BEGINNER,
        )

        errors = validate_example(example)
        assert len(errors) > 0
        assert "instruction" in str(errors)

    def test_invalid_example_missing_output(self):
        """Test validation fails without output"""
        example = TrainingExample(
            instruction="أعرب",
            input="جملة",
            output="",
            role=Role.TUTOR,
            skills=[Skill.NAHW],
            level=Level.BEGINNER,
        )

        errors = validate_example(example)
        assert len(errors) > 0
        assert "output" in str(errors)


class TestRoleEnum:
    """Test Role enumeration"""

    def test_role_values(self):
        """Test role enum values"""
        assert Role.TUTOR.value == "tutor"
        assert Role.PROOFREADER.value == "proofreader"
        assert Role.POET.value == "poet"
        assert Role.MUHHAQIQ.value == "muhhaqiq"
        assert Role.ASSISTANT_GENERAL.value == "assistant_general"

    def test_all_roles_present(self):
        """Test all roles are present"""
        roles = [role.value for role in Role]
        assert len(roles) == 5
        assert "tutor" in roles
        assert "proofreader" in roles
        assert "poet" in roles
        assert "muhhaqiq" in roles
        assert "assistant_general" in roles


class TestSkillEnum:
    """Test Skill enumeration"""

    def test_skill_values(self):
        """Test skill enum values"""
        assert Skill.NAHW.value == "nahw"
        assert Skill.SARF.value == "sarf"
        assert Skill.BALAGHA.value == "balagha"
        assert Skill.POETRY.value == "poetry"

    def test_all_core_skills_present(self):
        """Test core skills are present"""
        skills = [skill.value for skill in Skill]
        assert "nahw" in skills
        assert "sarf" in skills
        assert "balagha" in skills
        assert "orthography" in skills
        assert "poetry" in skills

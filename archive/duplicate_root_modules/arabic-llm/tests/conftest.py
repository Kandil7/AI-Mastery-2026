"""
Arabic LLM Test Suite

Pytest configuration and fixtures for testing.
"""

import pytest
import sys
from pathlib import Path

# Add arabic_llm to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_arabic_text():
    """Sample Arabic text for testing"""
    return "العلمُ نورٌ والجهلُ ظلامٌ."


@pytest.fixture
def sample_training_example():
    """Sample training example"""
    return {
        "instruction": "أعرب الجملة التالية",
        "input": "العلمُ نورٌ",
        "output": "العلمُ: مبتدأ مرفوع",
        "role": "tutor",
        "skills": ["nahw"],
        "level": "intermediate",
    }


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        "role_distribution": {
            "tutor": 0.35,
            "proofreader": 0.25,
            "poet": 0.20,
            "muhhaqiq": 0.15,
            "assistant_general": 0.05,
        },
        "level_distribution": {
            "beginner": 0.30,
            "intermediate": 0.50,
            "advanced": 0.20,
        },
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

"""
Arabic LLM Package - Version Information

Version management for the Arabic LLM project.
Follows semantic versioning (MAJOR.MINOR.PATCH)
"""

__version__ = "2.0.0"
__version_info__ = (2, 0, 0)
__author__ = "Arabic LLM Project Team"
__email__ = "arabic-llm@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Arabic LLM Project"


def get_version() -> str:
    """Get version string"""
    return __version__


def get_version_info() -> tuple:
    """Get version tuple"""
    return __version_info__

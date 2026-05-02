"""
ML Learning Module - Setup Configuration
=========================================

Author: ML Learning Module
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read the README
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="ml-learning-module",
    version="1.0.0",
    author="AI-Mastery-2026",
    author_email="contact@aimastery2026.com",
    description="A comprehensive machine learning educational module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-mastery-2026/ml-learning-module",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="machine learning deep learning nlp computer vision education",
    project_urls={
        "Documentation": "https://github.com/ai-mastery-2026/ml-learning-module#readme",
        "Source": "https://github.com/ai-mastery-2026/ml-learning-module",
        "Tracker": "https://github.com/ai-mastery-2026/ml-learning-module/issues",
    },
) if __name__ == "__main__" else None

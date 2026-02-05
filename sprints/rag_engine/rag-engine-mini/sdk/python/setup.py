from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-engine",
    version="1.0.0",
    author="RAG Engine Team",
    author_email="team@ragengine.ai",
    description="Official Python SDK for RAG Engine Mini",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rag-engine-mini",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: AsyncIO",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-engine=rag_engine.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="rag, llm, ai, embeddings, search, documents, chatbot",
    project_urls={
        "Bug Reports": "https://github.com/your-org/rag-engine-mini/issues",
        "Source": "https://github.com/your-org/rag-engine-mini/tree/main/sdk/python",
        "Documentation": "https://rag-engine.readthedocs.io/",
    },
)

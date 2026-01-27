from setuptools import setup, find_packages

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "AI-Mastery-2026: The Ultimate AI Engineering Toolkit"

# Define extra dependencies for focused installs
EXTRAS = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.18.0",
        "black>=22.0.0",
        "mypy>=0.950",
        "flake8>=4.0.0",
        "isort>=5.10.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.15.0",
    ],
    "viz": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
    ],
    "ml": [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "torchdiffeq>=0.2.0",
        "tensorflow>=2.9.0",
        "keras>=2.9.0",
        "numba>=0.55.0",
    ],
    "llm": [
        "transformers>=4.20.0",
        "tokenizers>=0.12.0",
        "langchain>=0.0.200",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "sentence-transformers>=2.2.0",
        "rank-bm25>=0.2.2",
        "accelerate>=0.12.0",
        "datasets>=2.0.0",
    ],
    "vector": [
        "faiss-cpu>=1.7.0",
        "hnswlib>=0.7.0",
        "pgvector>=0.2.5",
        "qdrant-client>=1.8.0",
        "weaviate-client>=3.25.0",
    ],
    "db": [
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=1.4.0",
    ],
    "eval": [
        "ragas>=0.1.0",
    ],
    "ui": [
        "streamlit>=1.10.0",
    ],
    "prod": [
        "gunicorn>=20.1.0",
        "prometheus-client>=0.14.0",
    ],
}

# Aggregate 'all' extra
_all_deps = []
for _deps in EXTRAS.values():
    _all_deps.extend(_deps)
EXTRAS["all"] = sorted(set(_all_deps))

setup(
    name="ai-mastery-2026",
    version="0.1.0",
    description="A comprehensive AI Engineer Toolkit built from first principles (Week 1-17)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kandil7",
    author_email="medokandeal7@gmail.com",
    url="https://github.com/Kandil7/AI-Mastery-2026",
    project_urls={
        "Documentation": "https://github.com/Kandil7/AI-Mastery-2026",
        "Source": "https://github.com/Kandil7/AI-Mastery-2026",
        "Issues": "https://github.com/Kandil7/AI-Mastery-2026/issues",
    },
    keywords="ai, machine learning, deep learning, llm, rag, mlops, fastapi",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.10",
    
    # Core Dependencies (Essential for src/core and src/ml)
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.78.0",
        "uvicorn[standard]>=0.17.0",
        "pydantic>=1.9.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "python-dotenv>=0.20.0",
        "joblib>=1.1.0",
    ],
    
    extras_require=EXTRAS,
    
    entry_points={
        "console_scripts": [
            "ai-train=scripts.train_save_models:main",
            "ai-benchmark=scripts.run_benchmarks:main",
            "ai-app=src.production.api:start_server",  # Added entry point for API
        ],
    },
    include_package_data=True,
)

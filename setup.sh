#!/bin/bash
# AI-Mastery-2026 Setup Script
# ============================

set -e

echo "🧠 AI-Mastery-2026 Setup"
echo "========================"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔨 Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed
mkdir -p notebooks
mkdir -p case_studies
mkdir -p benchmarks

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v --tb=short || echo "⚠️ Some tests may fail on first run"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate  (Linux/Mac)"
echo "  .venv\\Scripts\\activate    (Windows)"
echo ""
echo "Quick start:"
echo "  pytest tests/ -v            # Run tests"
echo "  python -m src.core          # Test imports"
echo ""

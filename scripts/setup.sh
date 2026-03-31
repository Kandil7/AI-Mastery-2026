#!/bin/bash
# Setup Script for AI-Mastery-2026
# =================================
# Sets up the development environment.
#
# Usage:
#   ./scripts/setup.sh              # Full setup
#   ./scripts/setup.sh --minimal    # Minimal setup (core only)
#   ./scripts/setup.sh --dev        # Development setup (default)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=================================="
echo "AI-Mastery-2026 Setup"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

echo -e "${BLUE}Checking Python version...${NC}"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"

# Parse arguments
INSTALL_MODE="dev"
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            INSTALL_MODE="minimal"
            shift
            ;;
        --dev)
            INSTALL_MODE="dev"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

case $INSTALL_MODE in
    minimal)
        echo "Installing core dependencies only..."
        pip install -e "."
        ;;
    dev)
        echo "Installing development dependencies..."
        pip install -e ".[dev]"
        ;;
esac

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Install pre-commit hooks
echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
pre-commit install
echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
echo ""

# Run initial tests
echo -e "${YELLOW}Running smoke tests...${NC}"
python -c "from src.core import Adam; from src.ml import NeuralNetwork; print('Import test passed')"
echo -e "${GREEN}✓ Smoke tests passed${NC}"
echo ""

# Summary
echo "=================================="
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run tests: ./scripts/test.sh"
echo "  3. Start coding!"
echo "=================================="

#!/bin/bash

# Setup script for AI-Mastery-2026
# Creates virtual environment and installs dependencies

set -e  # Exit on any error

echo "Setting up AI-Mastery-2026 environment..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name ai-mastery-2026 --display-name "AI-Mastery-2026"

echo "Setup complete!"
echo "To activate the environment in the future, run: source .venv/bin/activate"
echo "Or on Windows: .venv\Scripts\activate"
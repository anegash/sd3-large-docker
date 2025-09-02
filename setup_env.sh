#!/bin/bash

# SD3 Large API Environment Setup Script (Poetry Version)
set -e

echo "🚀 Setting up SD3 Large API environment with Poetry..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed."
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "⚠️  Please restart your terminal or run: source ~/.bashrc"
    echo "Then run this script again."
    exit 1
fi

echo "✅ Poetry found: $(poetry --version)"

# Check if Python 3.10+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.10+ is required. Found: Python $python_version"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

echo "✅ Python $python_version found"

# Install dependencies with Poetry (creates virtual environment automatically)
echo "📦 Installing dependencies with Poetry..."
poetry install --only main

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Poetry has automatically created and configured a virtual environment."
echo ""
echo "To run commands in the Poetry environment:"
echo "  poetry run python main.py          # Run the server"
echo "  poetry shell                       # Activate shell in environment"
echo "  poetry install --with dev          # Install development dependencies"
echo ""
echo "To start the server:"
echo "  poetry run python main.py"
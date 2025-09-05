#!/bin/bash
# SD3 Large API Environment Setup Script
set -e

echo "🚀 Setting up SD3 Large API environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "⚠️  Please restart terminal or run: source ~/.bashrc"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install --only main

echo "🎉 Setup complete!"
echo "Run: poetry run python main.py"
#!/usr/bin/env bash
# ==========================================================
# Bash Script to Setup Environment and Install Essential Libraries for CDFT
# ==========================================================

# Stop on error
set -e

# --- Environment setup ---
ENV_NAME="myenv"
PYTHON=${PYTHON:-python3}

echo "🔧 Checking for Python installation..."
if ! command -v $PYTHON &>/dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ before continuing."
    exit 1
fi

# --- Create virtual environment if it doesn't exist ---
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment: $ENV_NAME"
    $PYTHON -m venv "$ENV_NAME"
fi

# --- Activate virtual environment ---
echo "🚀 Activating environment..."
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

# --- Upgrade pip ---
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# --- Install essential dependencies ---
echo "📚 Installing essential libraries..."
pip install numpy json5 matplotlib pynufft scipy pyfftw sympy

echo "✅ Essential environment setup complete! To activate, run: source $ENV_NAME/bin/activate"


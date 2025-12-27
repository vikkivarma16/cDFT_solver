#!/usr/bin/env bash
# ==========================================================
# Local Installer: Setup Environment, Run Essential Installer, and Install cdft_package
# ==========================================================

# Stop on error
set -e

# --- Environment setup ---
ENV_NAME="myenv"
PYTHON=${PYTHON:-python3}

echo "🔧 Checking for Python installation..."
if ! command -v $PYTHON &>/dev/null; then
    echo "❌ Python not found. Please install Python 3.8+."
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
source "$ENV_NAME/bin/activate" || { echo "❌ Failed to activate virtual environment"; exit 1; }

# --- Upgrade pip ---
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# --- Install essential libraries ---
echo "📚 Installing essential libraries..."
pip install numpy json5 matplotlib pynufft scipy pyfftw sympy

# --- Run installer_essential.sh if present ---
if [ -f installer_essential.sh ]; then
    echo "🔧 Running installer_essential.sh..."
    chmod +x installer_essential.sh
    ./installer_essential.sh
fi

# --- Install cdft_package in editable mode ---
echo "📦 Installing cdft_package (editable mode)..."
pip install -e .

echo "✅ Environment setup complete! Activate it with: source $ENV_NAME/bin/activate"


#!/usr/bin/env bash
# ==========================================================
# Bash Script to Setup Environment and Run CDFT Example
# ==========================================================

# Stop on error
set -e

# --- Environment setup ---
ENV_NAME="cdft_env"
PYTHON=${PYTHON:-python3}

echo "🔧 Checking for Python installation..."
if ! command -v $PYTHON &>/dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ before continuing."
    exit 1
fi

# --- Create and activate virtual environment ---
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment: $ENV_NAME"
    $PYTHON -m venv "$ENV_NAME"
fi

echo "🚀 Activating environment..."
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

# --- Upgrade pip ---
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# --- Install dependencies ---
echo "📚 Installing required libraries..."
pip install numpy json5 matplotlib pynufft scipy pyfftw sympy

# --- Install your cdft_package ---
echo "📦 Installing cdft_package..."
pip install -e .

# --- Run example script ---
# Replace cdft_package.example_executor with your actual module path if needed
echo "▶️ Running CDFT example executor..."

python3 -m cdft_package.executor_dft_main executor_isochor.in

echo "✅ Execution complete."


#!/usr/bin/env bash
# ==========================================================
# Bash Script to Commit, Push, and Reinstall cDFT Solver
# ==========================================================

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


# --- Install from GitHub ---
GIT_URL="https://github.com/vikkivarma16/cDFT_solver.git"
echo "📦 Installing cdft_package from GitHub..."
pip install git+$GIT_URL

echo "✅ Done! Environment activated and package installed."


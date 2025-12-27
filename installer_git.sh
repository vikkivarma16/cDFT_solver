#!/usr/bin/env bash
# ==========================================================
# Bash Script to Commit, Push, and Reinstall cDFT Solver
# ==========================================================

# Stop on error
set -e

# --- Git commit & push ---
echo "🔧 Adding all changes..."
git add .

echo "✏️ Committing changes..."
git commit -m "update_data_$(date '+%Y-%m-%d_%H-%M-%S')" || echo "No changes to commit."


echo "🚀 Pushing to remote..."
git push -u origin main

# --- Activate Python environment ---
ENV_PATH=~/myenv
if [ ! -d "$ENV_PATH" ]; then
    echo "❌ Environment $ENV_PATH does not exist."
    exit 1
fi

echo "🚀 Activating environment..."
# shellcheck disable=SC1091
source "$ENV_PATH/bin/activate"

# --- Uninstall existing package ---
echo "🗑️ Uninstalling existing cdft_solver package if it exists..."
pip uninstall -y cdft_solver || echo "No existing package found."

# --- Install from GitHub ---
GIT_URL="https://github.com/vikkivarma16/cDFT_solver.git"
echo "📦 Installing cdft_package from GitHub..."
pip install git+$GIT_URL

echo "✅ Done! Environment activated and package installed."


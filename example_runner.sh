#!/usr/bin/env bash
# ==========================================================
# Bash Script to Setup Environment and Run CDFT Example
# ==========================================================

# Stop on error
set -e

echo "🚀 Activating environment..."
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"


# --- Run example script ---
# Replace cdft_package.example_executor with your actual module path if needed
echo "▶️ Running CDFT example executor..."

python3 -m cdft_solver.executor_dft_main executor_isochor.in

echo "✅ Execution complete."

#!/usr/bin/env bash
# One-command setup: venv + dependencies + test suite.
set -e
cd "$(dirname "$0")"

echo "[1/3] Creating virtual environment..."
python3 -m venv .venv
echo "[2/3] Installing dependencies..."
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "[3/3] Running unit tests..."
.venv/bin/python -m pytest tests/ -m "not integration"

echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Run the pipeline with:          streamlit run app.py"

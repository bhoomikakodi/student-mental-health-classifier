#!/usr/bin/env bash
# Sets up the virtual environment, installs dependencies, trains the models
# and launches the Streamlit app.
# Usage: ./run.sh   (macOS/Linux, or Git Bash/WSL on Windows)
set -euo pipefail

# Always run from the project root, regardless of where the script is called from
cd "$(dirname "${BASH_SOURCE[0]}")"

# Pick the Python launcher: python3 on macOS/Linux, python on Windows
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# 2. Create virtual environment
if [ ! -d venv ]; then
    "$PYTHON" -m venv venv
fi

# Activate based on OS: venv/bin/activate (macOS/Linux) or venv/Scripts/activate (Windows)
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv/Scripts/activate ]; then
    source venv/Scripts/activate
else
    echo "Could not find venv activation script (venv/bin/activate or venv/Scripts/activate)." >&2
    exit 1
fi

# 3. Install dependencies
python -m pip install -r requirements.txt
#    (Notebook users: register the venv as a kernel and select it in Jupyter/VS Code)
python -m pip install ipykernel && python -m ipykernel install --user --name student-mental-health-venv

# 4. Train all models and export artifacts
python src/train.py

# 5. Launch interactive Streamlit application
python -m streamlit run app.py

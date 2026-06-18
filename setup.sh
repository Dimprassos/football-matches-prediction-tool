#!/bin/bash
set -e

# Define colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}Setting up the Football Prediction Tool${NC}"
echo -e "${CYAN}==========================================${NC}"

# Must run from the project root (where requirements.txt lives).
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found. Run this script from the project root.${NC}"
    exit 1
fi

# Find a Python interpreter.
PYTHON=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+ and re-run.${NC}"
    exit 1
fi
echo -e "\nUsing Python interpreter: ${PYTHON}"

VENV_PY=".venv/bin/python"

# (Re)create the virtual environment if it is missing OR broken (e.g. a previous
# attempt left a .venv without pip). This makes re-runs self-healing.
NEED_CREATE=1
if [ -x "$VENV_PY" ] && "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    NEED_CREATE=0
    echo -e "${GREEN}Virtual environment .venv already exists and is healthy.${NC}"
elif [ -e ".venv" ]; then
    echo -e "${YELLOW}Existing .venv is missing pip; recreating it ...${NC}"
fi
if [ "$NEED_CREATE" -eq 1 ]; then
    rm -rf .venv
    echo -e "${YELLOW}Creating virtual environment .venv ...${NC}"
    "$PYTHON" -m venv .venv
fi

echo -e "\nUpgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip

echo -e "\nInstalling dependencies from requirements.txt ..."
"$VENV_PY" -m pip install -r requirements.txt

echo -e "\n${CYAN}==========================================${NC}"
echo -e "${GREEN}Setup complete.${NC}"
echo -e "\nActivate the environment with:  ${YELLOW}source .venv/bin/activate${NC}"
echo -e "Then run:"
echo -e "  ${YELLOW}python scripts/main.py${NC}      # train the models once (a few minutes)"
echo -e "  ${YELLOW}streamlit run app.py${NC}        # launch the interactive tool"
echo -e "${CYAN}==========================================${NC}"

#!/bin/bash

# Define colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}Setting up the Football Prediction Tool${NC}"
echo -e "${CYAN}==========================================${NC}"

echo -e "\nChecking for existing virtual environment (.venv)..."
if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Creating virtual environment .venv ...${NC}"
    python3 -m venv .venv
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

echo "Activating the virtual environment..."
source .venv/bin/activate

echo -e "\nUpgrading pip to the latest version..."
python3 -m pip install --upgrade pip

echo -e "\nInstalling dependencies from requirements.txt..."
python3 -m pip install -r requirements.txt

echo -e "\n${CYAN}==========================================${NC}"
echo -e "${GREEN}Setup complete. The virtual environment (.venv) is active.${NC}"
echo -e "\nNext steps:"
echo -e "  ${YELLOW}1. Train the models once (a few minutes):  python scripts/main.py${NC}"
echo -e "  ${YELLOW}2. Launch the interactive tool:            streamlit run app.py${NC}"
echo -e "${CYAN}==========================================${NC}"

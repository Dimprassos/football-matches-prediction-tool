Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setting up the Football Prediction Tool" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nChecking for existing virtual environment (.venv)..."
if (-not (Test-Path ".venv\Scripts\activate.ps1")) {
    Write-Host "Creating virtual environment .venv ..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

Write-Host "`nActivating the virtual environment..."
. .\.venv\Scripts\activate.ps1

Write-Host "`nUpgrading pip to the latest version..."
python -m pip install --upgrade pip

Write-Host "`nInstalling dependencies from requirements.txt..."
python -m pip install -r requirements.txt

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Setup complete. The virtual environment (.venv) is active." -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Train the models once (a few minutes):  python scripts/main.py" -ForegroundColor Yellow
Write-Host "  2. Launch the interactive tool:            streamlit run app.py" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setting up the Football Prediction Tool" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Must run from the project root (where requirements.txt lives).
if (-not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found. Run this script from the project root." -ForegroundColor Red
    exit 1
}

# Find a Python launcher: prefer `python`, fall back to the Windows `py` launcher.
$pythonCmd = $null
foreach ($cand in @("python", "py")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) { $pythonCmd = $cand; break }
}
if (-not $pythonCmd) {
    Write-Host "ERROR: Python was not found on PATH. Install Python 3.10+ and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "`nUsing Python launcher: $pythonCmd"

$venvPython = ".\.venv\Scripts\python.exe"

# Create the virtual environment if it does not exist.
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment .venv ..." -ForegroundColor Yellow
    & $pythonCmd -m venv .venv
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path $venvPython))) {
        Write-Host "ERROR: failed to create the virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment .venv already exists." -ForegroundColor Green
}

# Make sure pip is available inside the venv. Some Python installs create venvs
# without pip; bootstrap it with ensurepip in that case.
& $venvPython -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip not found in .venv; bootstrapping with ensurepip ..." -ForegroundColor Yellow
    & $venvPython -m ensurepip --upgrade
    & $venvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: could not bootstrap pip in the virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Call the venv's Python directly. This avoids depending on activate.ps1, which
# the PowerShell execution policy can block on a fresh machine.
Write-Host "`nUpgrading pip ..."
& $venvPython -m pip install --upgrade pip

Write-Host "`nInstalling dependencies from requirements.txt (torch is large, please wait) ..."
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: dependency installation failed (see the pip output above)." -ForegroundColor Red
    exit 1
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "`nTo activate the environment in this shell:"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "(If activation is blocked, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)"
Write-Host "`nThen run:"
Write-Host "  python scripts/main.py      # train the models once (a few minutes)" -ForegroundColor Yellow
Write-Host "  streamlit run app.py        # launch the interactive tool" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

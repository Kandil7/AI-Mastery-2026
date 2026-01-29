param(
    [ValidateSet("auto", "conda", "venv")]
    [string]$EnvManager = "auto",
    [string]$Cuda = "",
    [switch]$Jupyter,
    [switch]$Test,
    [switch]$Minimal
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$EnvName = "ai-mastery-2026"
$EnvFile = "environment.full.yml"
$VenvPath = ".venv"

function Has-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-PythonCommand {
    if (Has-Command "python") {
        return "python"
    }
    if (Has-Command "py") {
        $py310 = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $py310) {
            return "py -3.10"
        }
        return "py"
    }
    throw "Python not found in PATH. Install Python 3.10+ or add it to PATH."
}

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

if ($EnvManager -eq "auto") {
    $EnvManager = if (Has-Command "conda") { "conda" } else { "venv" }
}

if ($EnvManager -eq "conda" -and -not (Has-Command "conda")) {
    throw "conda not found in PATH. Install Miniconda/Anaconda or use -EnvManager venv."
}

if ($EnvManager -eq "conda") {
    Write-Step "Preparing conda environment '$EnvName'"

    $envExists = (conda env list | Select-String -Quiet " $EnvName ")
    if ($envExists) {
        conda env update -n $EnvName -f $EnvFile --prune
    } else {
        conda env create -n $EnvName -f $EnvFile
    }

    if ($Cuda) {
        Write-Step "Installing CUDA PyTorch build (pytorch-cuda=$Cuda)"
        conda install -n $EnvName pytorch torchvision torchaudio pytorch-cuda=$Cuda -c pytorch -c nvidia -y
    }

    if ($Jupyter) {
        Write-Step "Registering Jupyter kernel"
        conda run -n $EnvName python -m ipykernel install --user --name $EnvName --display-name "AI-Mastery-2026"
    }

    if ($Test) {
        Write-Step "Running smoke test"
        conda run -n $EnvName python -c "import numpy, torch, fastapi; print('deps ok')"
    }

    Write-Host "Done. Activate with: conda activate $EnvName"
    exit 0
}

Write-Step "Preparing venv at '$VenvPath'"

if (-not (Test-Path $VenvPath)) {
    $PythonCmd = Get-PythonCommand
    Invoke-Expression "$PythonCmd -m venv $VenvPath"
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Venv python not found at $VenvPython. Verify Python installation."
}
& $VenvPython -m pip install --upgrade pip

if ($Minimal) {
    & $VenvPython -m pip install -r requirements-minimal.txt
} else {
    & $VenvPython -m pip install -r requirements.txt
}

if ($Cuda) {
    Write-Host "CUDA requested, but venv install is not automated. Install a matching torch wheel manually."
}

if ($Jupyter) {
    & $VenvPython -m ipykernel install --user --name $EnvName --display-name "AI-Mastery-2026"
}

if ($Test) {
    & $VenvPython -c "import numpy, torch, fastapi; print('deps ok')"
}

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"

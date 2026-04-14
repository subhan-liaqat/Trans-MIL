param(
    [string]$EnvName = "transmil",
    [string]$PythonVersion = "3.9"
)

$ErrorActionPreference = "Stop"

function Has-Command($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

if (Has-Command "conda") {
    Write-Host "Using conda environment '$EnvName' (Python $PythonVersion)..."
    conda create -n $EnvName python=$PythonVersion -y
    conda run -n $EnvName conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -y
    conda run -n $EnvName python -m pip install --upgrade pip
    conda run -n $EnvName python -m pip install -r requirements.txt
    Write-Host "Setup complete with conda."
    Write-Host "Run with: .\scripts\run_train_windows.ps1 -Stage train -Fold 0 -Gpu 0"
    exit 0
}

if (-not (Has-Command "uv")) {
    throw "Neither conda nor uv was found. Install Miniconda or uv first."
}

$venvDir = ".venv-transmil"
$pyExe = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Conda not found. Using uv-managed local venv at '$venvDir'..."
uv python install $PythonVersion
uv venv --python $PythonVersion --seed $venvDir

Write-Host "Installing PyTorch stack (CPU) for compatibility..."
Invoke-Checked $pyExe @("-m", "pip", "install", "--upgrade", "pip<24.1")
Invoke-Checked $pyExe @("-m", "pip", "install", "setuptools", "wheel")
Invoke-Checked $pyExe @("-m", "pip", "install", "torch==1.10.1", "torchvision==0.11.2", "torchaudio==0.10.1")

Write-Host "Installing runtime dependencies with Windows/Python3.9 compatible versions..."
Invoke-Checked $pyExe @(
    "-m", "pip", "install",
    "addict==2.2.1",
    "einops==0.3.0",
    "matplotlib==3.5.1",
    "numpy==1.23.5",
    "nystrom-attention==0.0.9",
    "omegaconf==2.2.3",
    "opencv-python==4.8.1.78",
    "opencv-python-headless==4.8.1.78",
    "pandas==1.5.3",
    "Pillow==9.5.0",
    "pytorch-lightning==1.5.9",
    "PyYAML==6.0.2",
    "scipy==1.11.4",
    "torchmetrics==0.6.0"
)
Invoke-Checked $pyExe @("-m", "pip", "install", "pytorch-toolbelt==0.5.0", "--no-deps")

# Ensure core torch stack remains pinned for this codebase.
Invoke-Checked $pyExe @("-m", "pip", "install", "torch==1.10.1", "torchvision==0.11.2", "torchaudio==0.10.1")

Write-Host "Setup complete with uv + venv."
Write-Host "Run with: .\scripts\run_train_windows.ps1 -Stage train -Fold 0 -Gpu 0"

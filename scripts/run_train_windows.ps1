param(
    [ValidateSet("train", "test")]
    [string]$Stage = "train",
    [int]$Fold = 0,
    [int]$Gpu = 0,
    [string]$Config = "Camelyon/TransMIL.windows.yaml",
    [string]$EnvName = "transmil"
)

$ErrorActionPreference = "Stop"

function Has-Command($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "Running: python train.py --stage='$Stage' --config='$Config' --gpus=$Gpu --fold=$Fold"
if (Has-Command "conda") {
    conda run -n $EnvName python train.py --stage="$Stage" --config="$Config" --gpus=$Gpu --fold=$Fold
    exit 0
}

$venvPython = ".venv-transmil\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No conda and no local venv found. Run .\scripts\setup_windows.ps1 first."
}

& $venvPython train.py --stage="$Stage" --config="$Config" --gpus=$Gpu --fold=$Fold

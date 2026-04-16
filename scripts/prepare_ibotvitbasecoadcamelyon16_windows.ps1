param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot
)

$ErrorActionPreference = "Stop"
$venvPython = ".venv-transmil\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No local venv found. Run .\scripts\setup_windows.ps1 first."
}

& $venvPython scripts/prepare_ibotvitbasecoadcamelyon16.py --dataset_root $DatasetRoot

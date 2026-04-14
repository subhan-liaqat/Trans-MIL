param(
    [Parameter(Mandatory = $true)]
    [string]$RawSlidesDir,
    [string]$SlideExt = ".tif",
    [string]$FoldCsv = "dataset_csv/camelyon16/fold0.csv",
    [string]$ClamDir = "tools/CLAM",
    [string]$PatchOutDir = "Camelyon16/clam_results",
    [string]$FeatDir = "Camelyon16",
    [string]$Preset = "bwh_biopsy.csv",
    [int]$PatchSize = 256,
    [int]$BatchSize = 256
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDir = "")
    if ($WorkingDir -ne "") {
        Push-Location $WorkingDir
    }
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        if ($WorkingDir -ne "") {
            Pop-Location
        }
    }
}

$py = ".venv-transmil\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Missing local env. Run .\scripts\setup_windows.ps1 first."
}

if (-not (Test-Path $RawSlidesDir)) {
    throw "RawSlidesDir does not exist: $RawSlidesDir"
}
if (-not (Test-Path $FoldCsv)) {
    throw "Fold CSV does not exist: $FoldCsv"
}

if (-not (Test-Path $ClamDir)) {
    Write-Host "CLAM not found. Cloning into $ClamDir ..."
    git clone https://github.com/mahmoodlab/CLAM $ClamDir
}

$patchScript = Join-Path $ClamDir "create_patches_fp.py"
$featScript = Join-Path $ClamDir "extract_features_fp.py"
if (-not (Test-Path $patchScript)) { throw "Missing CLAM script: $patchScript" }
if (-not (Test-Path $featScript)) { throw "Missing CLAM script: $featScript" }

Write-Host "Installing CLAM preprocessing dependencies in local env..."
Invoke-Checked $py @("-m", "pip", "install", "h5py", "openslide-python", "openslide-bin", "timm", "Pillow")

$clamCsv = Join-Path $FeatDir "clam_slides.csv"
Write-Host "Generating CLAM slide CSV: $clamCsv"
Invoke-Checked $py @("scripts/generate_clam_slide_csv.py", "--fold_csv", $FoldCsv, "--output_csv", $clamCsv)

Write-Host "Running CLAM patching..."
Invoke-Checked $py @(
    "create_patches_fp.py",
    "--source", $RawSlidesDir,
    "--save_dir", $PatchOutDir,
    "--patch_size", "$PatchSize",
    "--preset", $Preset,
    "--seg", "--patch", "--stitch"
) $ClamDir

Write-Host "Running CLAM feature extraction..."
Invoke-Checked $py @(
    "extract_features_fp.py",
    "--data_h5_dir", (Resolve-Path $PatchOutDir).Path,
    "--data_slide_dir", (Resolve-Path $RawSlidesDir).Path,
    "--csv_path", (Resolve-Path $clamCsv).Path,
    "--feat_dir", (Resolve-Path $FeatDir).Path,
    "--batch_size", "$BatchSize",
    "--slide_ext", $SlideExt
) $ClamDir

Write-Host "Feature build complete. Expected output:"
Write-Host " - $FeatDir\pt_files\*.pt"

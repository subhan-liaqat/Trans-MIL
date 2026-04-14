# TransMIL Implementation Guide (Windows First, Colab Ready)

This workspace is set up from the official TransMIL codebase:

- Repository: [szc19990412/TransMIL](https://github.com/szc19990412/TransMIL)

Use this guide to get a stable local baseline first, then port the same workflow to Google Colab.

## 1) Local Baseline Setup (Windows)

From this folder (`TransMIL`), run:

```powershell
.\scripts\setup_windows.ps1
```

This creates a conda environment named `transmil` and installs compatible dependencies.

## 2) Validate Data Layout

TransMIL expects:

- Feature tensors in: `Camelyon16/pt_files/`
- Split CSV in: `dataset_csv/camelyon16/fold0.csv`

Each feature file should be named `<slide_id>.pt`, where `<slide_id>` matches IDs used in CSV columns (`train`, `val`, `test`).

## 2.1) Build Features From Raw WSIs

If you only have raw slide files and no `.pt` features yet, run:

```powershell
.\scripts\build_camelyon16_features_windows.ps1 -RawSlidesDir "D:\path\to\camelyon16\slides" -SlideExt ".tif"
```

What this script does:

- Uses `tools/CLAM` preprocessing scripts.
- Generates `Camelyon16/clam_slides.csv` from your fold split CSV.
- Runs patching to `Camelyon16/clam_results/patches`.
- Extracts features to `Camelyon16/pt_files`.

After it finishes, you can start TransMIL training directly.

## 3) Train / Test Commands

Run training:

```powershell
.\scripts\run_train_windows.ps1 -Stage train -Fold 0 -Gpu 0 -Config Camelyon/TransMIL.yaml
```

Run testing:

```powershell
.\scripts\run_train_windows.ps1 -Stage test -Fold 0 -Gpu 0 -Config Camelyon/TransMIL.yaml
```

## 4) Recommended First Run

- Use one fold (`fold0`) only.
- Start with your real features but short training duration (or temporarily reduce `General.epochs` in config).
- Confirm logs and checkpoints are saved without runtime errors.

## 5) Colab Migration Later

When moving to Colab:

- Use Python 3.7/3.8 runtime if possible (or pin packages carefully).
- Clone this same repo.
- Recreate folder paths exactly (`Camelyon16/pt_files`, `dataset_csv/camelyon16`).
- Mount Google Drive and point data paths to Drive if needed.

The main goal is keeping path conventions unchanged so behavior matches your local baseline.

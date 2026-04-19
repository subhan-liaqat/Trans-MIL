# CAMELYON16 Pipeline Guide

This repository is now aligned to one workflow:

1. Download pre-extracted MIL features from `torchmil/Camelyon16_MIL`.
2. Train and test TransMIL directly from those `.npy` bags.

## Recommended Strategy

If your goal is to train TransMIL on Google Colab, this is the correct strategy.

- It avoids raw whole-slide image preprocessing.
- It avoids OpenSlide, CLAM, and very large storage requirements.
- It gives you MIL-ready slide features that fit this repository with minimal setup.

## Data Source Recommendation

Use the Hugging Face dataset `torchmil/Camelyon16_MIL`.

- It already contains MIL-style slide features.
- It exposes a `features_resnet50.tar.gz` archive that matches this repository's default 1024-dimensional input size.
- The downloaded files are one bag per slide and can be flattened into `Camelyon16/pt_files/` as `.npy` files.
- The ResNet50 feature archive is large, about 26.7 GB, but still far more practical than raw-slide preprocessing on Colab.

- `owkin/camelyon16-features`: useful, but it uses Phikon-derived features and a different dataset packaging. It is not the simplest drop-in path for this repo.
- `forcewithme/camelyon-normal6` on Kaggle: only a tiny subset, not suitable for training this repository end to end.
- Baidu Pan mirrors: valid as mirrors, but inconvenient unless you already rely on Baidu.
- Raw official CAMELYON16 mirrors: useful only if you later decide to build a separate preprocessing pipeline outside this repo.

## Expected Training Inputs

Training in this repo needs:

- a feature directory with one file per slide
- a split CSV at `dataset_csv/camelyon16/fold0.csv`

The split CSV is already included in the repo. The only missing part is the feature directory.

By default, training looks under:

```text
Camelyon16/pt_files/
```

and expects:

- `slide_id.npy`

## Repository Layout After Download

After running the downloader, the important files look like this:

```text
Camelyon16/
  pt_files/
    normal_001.npy
    normal_002.npy
    ...
    tumor_111.npy
  torchmil_splits.csv
dataset_csv/
  camelyon16/
    fold0.csv
```

`fold0.csv` is the split file this repository uses for training and testing.

## Run The Pipeline

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/download_camelyon16_torchmil.py --feature-set resnet50
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
python train.py --stage test --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

What that download script does:

- downloads `features_resnet50.tar.gz` from `torchmil/Camelyon16_MIL`
- extracts every slide feature file into `Camelyon16/pt_files/`
- keeps the original slide IDs such as `normal_001.npy` and `tumor_111.npy`

Those names line up with the provided `fold0.csv`.

## Colab Notes

For Colab, the practical path is:

1. Clone this repo.
2. Run `bash scripts/colab_setup.sh`.
3. Download the Hugging Face ResNet50 features.
4. Train from `Camelyon16/pt_files/`.

No CLAM or raw-slide preprocessing is required for this workflow.

## Feature Dimensionality

This repo now exposes `Model.in_dim` in the YAML config.

- `features_resnet50.tar.gz`: 1024 dim
- `features_resnet50_bt.tar.gz`: 1024 dim
- `features_UNI.tar.gz`: 1024 dim

If you switch to a different feature source with a different output dimension, update `Model.in_dim` in the YAML to match.

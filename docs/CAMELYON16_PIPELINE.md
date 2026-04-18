# CAMELYON16 Pipeline Guide

This repository now supports two practical workflows:

1. Quick-start on Colab with pre-extracted features.
2. Full raw-slide preprocessing with CLAM.

## Recommended Strategy

If your goal is to train TransMIL on Google Colab, use the quick-start workflow first.

- It avoids downloading roughly 900 GB of raw whole-slide images.
- It avoids OpenSlide and CLAM preprocessing on Colab.
- It gives you MIL-ready slide features that fit this repository after a small amount of preparation.

Use the full CLAM workflow only if you explicitly need raw-slide patching, tissue segmentation, or your own feature extraction run.

## Data Source Recommendation

### Best source for quick-start training

Use the Hugging Face dataset `torchmil/Camelyon16_MIL`.

- It already contains MIL-style slide features.
- It exposes a `features_resnet50.tar.gz` archive that matches this repository's default 1024-dimensional input size.
- The downloaded files are one bag per slide and can be flattened into `Camelyon16/pt_files/` as `.npy` files.
- The ResNet50 feature archive is large, about 26.7 GB, but still far more practical than raw-slide preprocessing on Colab.

### Best source for full raw-slide preprocessing

Use the official CAMELYON16 mirrors, preferably the AWS Open Data bucket or the official Grand Challenge mirrors.

- AWS Open Data is the cleanest public raw-data source.
- The official CAMELYON16 README in the bucket states that the dataset contains 399 WSIs in TIFF format plus annotations and masks.
- CLAM can preprocess those raw slides into the `pt_files/` that this repository expects.

### Sources that are not the best default here

- `owkin/camelyon16-features`: useful, but it uses Phikon-derived features and a different dataset packaging. It is not the simplest drop-in path for this repo.
- `forcewithme/camelyon-normal6` on Kaggle: only a tiny subset, not suitable for training this repository end to end.
- Baidu Pan mirrors: valid as mirrors, but inconvenient unless you already rely on Baidu.

## Expected Training Inputs

Training in this repo needs:

- a feature directory with one file per slide
- a split CSV at `dataset_csv/camelyon16/fold0.csv`

The split CSV is already included in the repo. The only missing part is the feature directory.

By default, training looks under:

```text
Camelyon16/pt_files/
```

and accepts either:

- `slide_id.pt`
- `slide_id.npy`

## Quick-Start Workflow

From the repository root:

```bash
python scripts/download_camelyon16_torchmil.py --feature-set resnet50
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
python train.py --stage test --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

What that download script does:

- downloads `features_resnet50.tar.gz` from `torchmil/Camelyon16_MIL`
- extracts every slide feature file into `Camelyon16/pt_files/`
- keeps the original slide IDs such as `normal_001.npy` and `tumor_111.npy`

Those names line up with the provided `fold0.csv`.

## Full Raw-Slide Workflow

### 1. Get the official CAMELYON16 raw slides

You can use the official AWS Open Data bucket or the official Grand Challenge mirrors.

The raw data is large, so this is usually not a Colab-friendly path.

### 2. Bootstrap CLAM and preprocess the slides

From the repository root:

```bash
python scripts/run_clam_preprocessing.py \
  --raw-slide-dir /path/to/camelyon16/raw_slides \
  --slide-ext .tif \
  --model-name resnet50_trunc
```

This wrapper will:

- clone CLAM into `third_party/CLAM/` if needed
- build a `slide_id` CSV for CLAM
- run `create_patches_fp.py`
- run `extract_features_fp.py`
- write slide features into `Camelyon16/pt_files/`

After that, the regular TransMIL command works unchanged:

```bash
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

## Colab Notes

For Colab, the practical path is:

1. Clone this repo.
2. Run `bash scripts/colab_setup.sh`.
3. Download the Hugging Face ResNet50 features.
4. Train from `Camelyon16/pt_files/`.

If you try to preprocess raw CAMELYON16 slides directly on Colab, you will usually run into storage and runtime limits before training even starts.

## Feature Dimensionality

This repo now exposes `Model.in_dim` in the YAML config.

- `resnet50_trunc` features: 1024 dim
- `uni_v1` features: 1024 dim

Those two are the safest CLAM extractors for this project.

If you switch to another encoder with a different output dimension, update `Model.in_dim` in the YAML to match.

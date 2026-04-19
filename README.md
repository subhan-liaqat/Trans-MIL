# TransMIL

Transformer-based correlated multiple instance learning for whole-slide image classification.

This fork is set up for one practical pipeline:

- download CAMELYON16 MIL-ready features from `torchmil/Camelyon16_MIL`
- train directly from slide-level `.npy` bags
- run cleanly on newer Python environments such as Google Colab

## What This Repo Needs to Train

The training code does not consume raw whole-slide images directly. It consumes one feature file per slide plus a split CSV.

The included split file is:

```text
dataset_csv/camelyon16/fold0.csv
```

The expected feature directory is:

```text
Camelyon16/pt_files/
```

Each slide feature file is expected to be:

- `slide_id.npy`

## Recommended Dataset Source

Use the MIL-ready CAMELYON16 feature release from Hugging Face:

- dataset: [torchmil/Camelyon16_MIL](https://huggingface.co/datasets/torchmil/Camelyon16_MIL)
- feature archive: `features_resnet50.tar.gz`
- reason: it already matches this repo's default `1024`-dimensional TransMIL input

This is the best fit for Google Colab and the only workflow this repo is now optimized for.

## Colab Setup

On Colab:

```bash
bash scripts/colab_setup.sh
python scripts/download_camelyon16_torchmil.py --feature-set resnet50 --download-splits
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
python train.py --stage test --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

## Local Or Colab Workflow

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/download_camelyon16_torchmil.py --feature-set resnet50 --download-splits
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
python train.py --stage test --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

The downloader script:

- downloads `features_resnet50.tar.gz`
- extracts all slide bags into `Camelyon16/pt_files/`
- keeps the original filenames such as `normal_001.npy`
- optionally downloads the original `torchmil` split file for reference

## Notes

- The included training split file remains `dataset_csv/camelyon16/fold0.csv`.
- The feature directory still uses the historical name `Camelyon16/pt_files/`, even though the files are `.npy`.
- CLAM/raw-slide preprocessing has been removed from the repo path to keep the project focused.

## Detailed Guide

See [docs/CAMELYON16_PIPELINE.md](docs/CAMELYON16_PIPELINE.md) for:

- the chosen dataset source
- expected directory layout
- the exact `torchmil`-based training flow
- Colab-specific notes

## Citation

```tex
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
```

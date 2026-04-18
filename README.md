# TransMIL

Transformer-based correlated multiple instance learning for whole-slide image classification.

This fork is set up to be much easier to run end to end:

- it can train directly from slide-level `.pt` or `.npy` feature files
- it includes a quick-start downloader for CAMELYON16 MIL features from Hugging Face
- it includes a CLAM wrapper for raw-slide patching and feature extraction
- it is patched to be friendlier to newer Python environments such as Google Colab

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

Each slide feature file can now be either:

- `slide_id.pt`
- `slide_id.npy`

## Fastest Way to Get It Running

Use the MIL-ready CAMELYON16 feature release from Hugging Face:

```bash
python scripts/download_camelyon16_torchmil.py --feature-set resnet50
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
python train.py --stage test --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

That path is the best fit for Google Colab.

## Raw-Slide End-to-End Workflow

If you want the full preprocessing pipeline from raw CAMELYON16 whole-slide images:

```bash
python scripts/run_clam_preprocessing.py \
  --raw-slide-dir /path/to/raw_camelyon16_slides \
  --slide-ext .tif \
  --model-name resnet50_trunc
```

This will bootstrap CLAM under `third_party/CLAM`, run patching and feature extraction, and place the resulting slide features where TransMIL expects them.

## Colab Setup

On Colab:

```bash
bash scripts/colab_setup.sh
python scripts/download_camelyon16_torchmil.py --feature-set resnet50
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
```

## Detailed Guide

See [docs/CAMELYON16_PIPELINE.md](docs/CAMELYON16_PIPELINE.md) for:

- which dataset source to use
- which options are official vs convenient
- why CLAM is needed only for raw-slide preprocessing
- how to choose between quick-start and full end-to-end execution

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

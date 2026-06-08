#!/usr/bin/env python3
"""Preflight checks for Google Colab + Google Drive torchmil dataset."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from addict import Dict


def _load_camel_data():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / 'datasets' / 'camel_data.py'
    spec = importlib.util.spec_from_file_location('camel_data', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CamelData


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='Camelyon/TransMIL_colab.yaml')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as fh:
        cfg = Dict(yaml.safe_load(fh))

    required_scripts = [
        Path('scripts/configure_colab_data.py'),
        Path('utils/hilbert_sort.py'),
        Path('models/TransMIL.py'),
    ]
    for script in required_scripts:
        if not script.is_file():
            raise FileNotFoundError(
                f'Missing {script}. Your cloned repo is outdated. '
                'Push the latest code to GitHub or copy your local repo to Colab.'
            )

    camel_data_src = Path('datasets/camel_data.py').read_text(encoding='utf-8')
    if 'split_format' not in camel_data_src or 'torchmil' not in camel_data_src:
        raise RuntimeError(
            'datasets/camel_data.py does not include torchmil split support. '
            'Clone/pull the latest Trans-MIL repository before training.'
        )

    splits_path = Path(cfg.Data.splits_csv)
    features_dir = Path(cfg.Data.data_dir)
    coords_dir = Path(cfg.Data.coords_dir)
    labels_dir = Path(cfg.Data.labels_dir)

    for path in [splits_path, features_dir, coords_dir, labels_dir]:
        if not path.exists():
            raise FileNotFoundError(f'Required dataset path not found: {path}')

    splits = pd.read_csv(splits_path)
    print('splits.csv columns:', list(splits.columns))
    split_col = 'split' if 'split' in splits.columns else splits.columns[1]
    print('split counts:\n', splits[split_col].astype(str).str.lower().value_counts())

    bag_col = 'bag_name' if 'bag_name' in splits.columns else splits.columns[0]
    bag_name = str(splits.iloc[0][bag_col]).removesuffix('.npy')
    feature_path = features_dir / f'{bag_name}.npy'
    coords_path = coords_dir / f'{bag_name}.npy'
    label_path = labels_dir / f'{bag_name}.npy'

    features = np.load(feature_path)
    coords = np.load(coords_path)
    label = np.load(label_path)
    print(f'sample bag: {bag_name}')
    print(f'feature shape: {features.shape}')
    print(f'coords shape:  {coords.shape}')
    print(f'label value:   {label}')

    if features.ndim != 2:
        raise ValueError(f'Expected features shape [n, d], got {features.shape}')
    if int(cfg.Model.in_dim) != features.shape[1]:
        raise ValueError(
            f'Model in_dim={cfg.Model.in_dim} but features have dim {features.shape[1]}. '
            'Set Model.in_dim to match your feature files.'
        )

    CamelData = _load_camel_data()
    ds = CamelData(dataset_cfg=cfg.Data, state='train')
    features_t, coords_t, label_t = ds[0]
    print(f'dataloader sample: features={tuple(features_t.shape)}, coords={tuple(coords_t.shape)}, label={label_t}')
    print('Colab dataset preflight passed.')


if __name__ == '__main__':
    main()

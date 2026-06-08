#!/usr/bin/env python3
"""Quick sanity checks for the Hilbert-sort TransMIL pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import importlib.util

def _load_transmil_module():
    module_path = _REPO_ROOT / 'models' / 'TransMIL.py'
    spec = importlib.util.spec_from_file_location('TransMIL', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

TransMIL = _load_transmil_module().TransMIL
from utils.hilbert_sort import coords_are_valid, hilbert_sort_indices, pseudo_grid_coords


def test_model_without_coords() -> None:
    data = torch.randn(1, 128, 1024)
    model = TransMIL(n_classes=2, in_dim=1024)
    out = model(data=data)
    assert out['logits'].shape == (1, 2)


def test_model_with_coords() -> None:
    data = torch.randn(1, 128, 1024)
    coords = torch.from_numpy(pseudo_grid_coords(128).astype(np.float32) * 512.0).unsqueeze(0)
    model = TransMIL(n_classes=2, in_dim=1024)
    out = model(data=data, coords=coords)
    assert out['logits'].shape == (1, 2)


def test_model_with_empty_coords() -> None:
    data = torch.randn(1, 64, 1024)
    coords = torch.empty(1, 0, 2)
    model = TransMIL(n_classes=2, in_dim=1024)
    out = model(data=data, coords=coords)
    assert out['logits'].shape == (1, 2)


def _load_camel_data_class():
    module_path = _REPO_ROOT / 'datasets' / 'camel_data.py'
    spec = importlib.util.spec_from_file_location('camel_data', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CamelData


def test_dataset_unpack_format() -> None:
    import yaml
    from addict import Dict

    CamelData = _load_camel_data_class()

    with (_REPO_ROOT / 'Camelyon' / 'TransMIL.yaml').open('r', encoding='utf-8') as fh:
        cfg = Dict(yaml.load(fh, Loader=yaml.Loader))
    cfg.Data.fold = 0
    ds = CamelData(dataset_cfg=cfg.Data, state='test')
    try:
        features, coords, label = ds[0]
    except FileNotFoundError as exc:
        print(f'Skipping dataset sample check (features not downloaded yet): {exc}')
        return

    assert features.ndim == 2
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert isinstance(label, int)


def main() -> None:
    test_model_without_coords()
    test_model_with_coords()
    test_model_with_empty_coords()
    test_dataset_unpack_format()

    order = hilbert_sort_indices(
        coords=pseudo_grid_coords(16),
        patch_size=512,
    )
    assert len(order) == 16
    assert coords_are_valid(pseudo_grid_coords(16), 16)
    assert not coords_are_valid(torch.empty(0, 2), 16)

    print('All Hilbert pipeline checks passed.')


if __name__ == '__main__':
    main()

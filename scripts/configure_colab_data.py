#!/usr/bin/env python3
"""Write a Colab-ready YAML config pointing at a Google Drive torchmil dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        required=True,
        type=Path,
        help='Folder containing coords/, features_resnet50/, labels/, and splits.csv',
    )
    parser.add_argument(
        '--template',
        default=Path('Camelyon/TransMIL_colab.yaml'),
        type=Path,
        help='YAML template to patch.',
    )
    parser.add_argument(
        '--output',
        default=Path('Camelyon/TransMIL_colab.yaml'),
        type=Path,
        help='Generated config path used by train.py on Colab.',
    )
    return parser.parse_args()


def _count_npy_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.glob('*.npy')))


def _first_existing(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def resolve_torchmil_root(data_root: Path) -> dict[str, Path]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'Dataset root is not a directory: {data_root}')

    search_roots = [data_root]
    nested = data_root / 'patches_512'
    if nested.is_dir():
        search_roots.append(nested)

    features_dir = None
    coords_dir = None
    labels_dir = None
    splits_csv = None

    for root in search_roots:
        features_dir = features_dir or _first_existing(
            root,
            ['features_resnet50', 'features/features_resnet50', 'features_resnet50_bt'],
        )
        coords_dir = coords_dir or _first_existing(root, ['coords'])
        labels_dir = labels_dir or _first_existing(root, ['labels'])
        splits_csv = splits_csv or _first_existing(root, ['splits.csv'])

    missing = []
    if features_dir is None:
        missing.append('features_resnet50/')
    if coords_dir is None:
        missing.append('coords/')
    if labels_dir is None:
        missing.append('labels/')
    if splits_csv is None:
        missing.append('splits.csv')

    if missing:
        contents = sorted(p.name for p in data_root.iterdir()) if data_root.is_dir() else []
        raise FileNotFoundError(
            'Could not resolve torchmil dataset layout under '
            f'{data_root}. Missing: {missing}. Top-level contents: {contents}'
        )

    feature_count = _count_npy_files(features_dir)
    coords_count = _count_npy_files(coords_dir)
    labels_count = _count_npy_files(labels_dir)
    print(f'Using dataset root: {data_root}')
    print(f'features_dir: {features_dir} ({feature_count} .npy files)')
    print(f'coords_dir:   {coords_dir} ({coords_count} .npy files)')
    print(f'labels_dir:   {labels_dir} ({labels_count} .npy files)')
    print(f'splits_csv:   {splits_csv}')

    if feature_count == 0:
        raise FileNotFoundError(f'No .npy feature files found in {features_dir}')

    return {
        'features_dir': features_dir,
        'coords_dir': coords_dir,
        'labels_dir': labels_dir,
        'splits_csv': splits_csv,
    }


def main() -> None:
    args = parse_args()
    paths = resolve_torchmil_root(args.data_root)

    with args.template.open('r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    cfg['Data']['splits_csv'] = str(paths['splits_csv'])
    cfg['Data']['data_dir'] = str(paths['features_dir'])
    cfg['Data']['coords_dir'] = str(paths['coords_dir'])
    cfg['Data']['labels_dir'] = str(paths['labels_dir'])
    cfg['Data']['train_dataloader']['num_workers'] = 0
    cfg['Data']['test_dataloader']['num_workers'] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print(f'Wrote config: {args.output.resolve()}')


if __name__ == '__main__':
    main()

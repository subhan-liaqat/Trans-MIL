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
        default=Path('Camelyon/TransMIL_colab.local.yaml'),
        type=Path,
        help='Generated config path used by train.py on Colab.',
    )
    return parser.parse_args()


def _count_npy_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.glob('*.npy')))


def validate_torchmil_root(data_root: Path) -> None:
    required = {
        'features_resnet50': data_root / 'features_resnet50',
        'coords': data_root / 'coords',
        'labels': data_root / 'labels',
        'splits.csv': data_root / 'splits.csv',
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Dataset root is missing required items: '
            f'{missing}. Expected layout from torchmil/Camelyon16_MIL.'
        )

    feature_count = _count_npy_files(required['features_resnet50'])
    coords_count = _count_npy_files(required['coords'])
    labels_count = _count_npy_files(required['labels'])
    print(f'Found features: {feature_count}, coords: {coords_count}, labels: {labels_count}')
    if feature_count == 0:
        raise FileNotFoundError(f'No .npy feature files found in {required["features_resnet50"]}')


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    validate_torchmil_root(data_root)

    with args.template.open('r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    cfg['Data']['splits_csv'] = str(data_root / 'splits.csv')
    cfg['Data']['data_dir'] = str(data_root / 'features_resnet50')
    cfg['Data']['coords_dir'] = str(data_root / 'coords')
    cfg['Data']['labels_dir'] = str(data_root / 'labels')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print(f'Wrote config: {args.output.resolve()}')


if __name__ == '__main__':
    main()

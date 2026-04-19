#!/usr/bin/env python3
"""Download MIL-ready CAMELYON16 features from Hugging Face into this repo."""

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


ARCHIVES = {
    'resnet50': {
        'url': 'https://huggingface.co/datasets/torchmil/Camelyon16_MIL/resolve/main/dataset/patches_512/features/features_resnet50.tar.gz?download=true',
        'expected_dim': 1024,
    },
    'resnet50_bt': {
        'url': 'https://huggingface.co/datasets/torchmil/Camelyon16_MIL/resolve/main/dataset/patches_512/features/features_resnet50_bt.tar.gz?download=true',
        'expected_dim': 1024,
    },
    'uni': {
        'url': 'https://huggingface.co/datasets/torchmil/Camelyon16_MIL/resolve/main/dataset/patches_512/features/features_UNI.tar.gz?download=true',
        'expected_dim': 1024,
    },
}

SPLITS_URL = (
    'https://huggingface.co/datasets/torchmil/Camelyon16_MIL/resolve/main/'
    'dataset/splits.csv?download=true'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--feature-set',
        default='resnet50',
        choices=sorted(ARCHIVES.keys()),
        help='Feature archive to download. TransMIL uses 1024-d inputs, so resnet50 is the safest default.',
    )
    parser.add_argument(
        '--repo-root',
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help='Path to the TransMIL repository root.',
    )
    parser.add_argument(
        '--archive-dir',
        default=None,
        type=Path,
        help='Directory where the downloaded tar.gz archive should be cached.',
    )
    parser.add_argument(
        '--feature-dir',
        default=None,
        type=Path,
        help='Directory where extracted slide feature files should be stored.',
    )
    parser.add_argument(
        '--keep-archive',
        action='store_true',
        help='Keep the downloaded tar.gz file after extraction.',
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Re-download the archive even if it already exists locally.',
    )
    parser.add_argument(
        '--download-splits',
        action='store_true',
        help='Also download torchmil splits.csv for reference.',
    )
    return parser.parse_args()


def download_file(url, destination, force_download=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force_download:
        destination.unlink(missing_ok=True)

    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'

    print(f'Downloading {url}')
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        resumed = response.getheader('Content-Range') is not None
        total_size = response.getheader('Content-Length')
        total_size = int(total_size) if total_size is not None else None
        if total_size is not None and resumed:
            total_size += existing_size

        if existing_size > 0 and not resumed:
            existing_size = 0

        mode = 'ab' if resumed and existing_size > 0 else 'wb'
        downloaded = existing_size
        with destination.open(mode) as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total_size is None:
                    print(f'  downloaded {downloaded / (1024 ** 3):.2f} GB', end='\r')
                else:
                    progress = 100.0 * downloaded / total_size
                    print(
                        f'  downloaded {downloaded / (1024 ** 3):.2f} / '
                        f'{total_size / (1024 ** 3):.2f} GB ({progress:.1f}%)',
                        end='\r',
                    )

    print()

    print(f'Saved archive to {destination}')
    return destination


def extract_npy_members(archive_path, feature_dir):
    feature_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive_path, mode='r:gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.npy'):
                continue

            target_path = feature_dir / Path(member.name).name
            if target_path.exists():
                extracted += 1
                continue

            source_fh = tar.extractfile(member)
            if source_fh is None:
                continue

            with source_fh, target_path.open('wb') as target_fh:
                shutil.copyfileobj(source_fh, target_fh)
            extracted += 1

    if extracted == 0:
        raise RuntimeError(f'No .npy feature files were extracted from {archive_path}')

    print(f'Feature directory ready: {feature_dir}')
    print(f'Found or extracted {extracted} slide feature files.')


def maybe_download_splits(repo_root):
    splits_path = repo_root / 'Camelyon16' / 'torchmil_splits.csv'
    download_file(SPLITS_URL, splits_path, force_download=False)
    print(f'Saved reference split file to {splits_path}')


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    archive_dir = (args.archive_dir or repo_root / 'downloads').resolve()
    feature_dir = (args.feature_dir or repo_root / 'Camelyon16' / 'pt_files').resolve()

    archive_name = f'camelyon16_{args.feature_set}.tar.gz'
    archive_path = archive_dir / archive_name
    archive_path = download_file(
        ARCHIVES[args.feature_set]['url'],
        archive_path,
        force_download=args.force_download,
    )
    extract_npy_members(archive_path, feature_dir)

    if args.download_splits:
        maybe_download_splits(repo_root)

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)
        print(f'Removed cached archive: {archive_path}')

    print()
    print('Next step:')
    print('  python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0')


if __name__ == '__main__':
    main()

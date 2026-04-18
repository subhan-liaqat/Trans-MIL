#!/usr/bin/env python3
"""Bootstrap CLAM and run raw-slide patching plus feature extraction."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


CLAM_REPO_URL = 'https://github.com/mahmoodlab/CLAM.git'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--raw-slide-dir',
        required=True,
        type=Path,
        help='Directory containing raw CAMELYON16 whole-slide images (.tif or .svs).',
    )
    parser.add_argument(
        '--slide-ext',
        default='.tif',
        help='Slide filename extension to scan for when building the CLAM slide list.',
    )
    parser.add_argument(
        '--clam-dir',
        default=Path(__file__).resolve().parents[1] / 'third_party' / 'CLAM',
        type=Path,
        help='Where the CLAM repository should live.',
    )
    parser.add_argument(
        '--patch-root',
        default=Path(__file__).resolve().parents[1] / 'Camelyon16' / 'clam_patching',
        type=Path,
        help='Directory where CLAM should write coordinate .h5 files, masks, and stitches.',
    )
    parser.add_argument(
        '--feature-root',
        default=Path(__file__).resolve().parents[1] / 'Camelyon16',
        type=Path,
        help='Directory where CLAM should write pt_files/ and h5_files/.',
    )
    parser.add_argument(
        '--patch-size',
        default=256,
        type=int,
        help='Patch size passed to create_patches_fp.py.',
    )
    parser.add_argument(
        '--batch-size',
        default=256,
        type=int,
        help='Feature extraction batch size passed to extract_features_fp.py.',
    )
    parser.add_argument(
        '--model-name',
        default='resnet50_trunc',
        choices=['resnet50_trunc', 'uni_v1', 'conch_v1'],
        help='CLAM encoder. TransMIL defaults to 1024-d features, so resnet50_trunc or uni_v1 are recommended.',
    )
    parser.add_argument(
        '--preset',
        default=None,
        help='Optional CLAM segmentation preset, for example bwh_biopsy.csv.',
    )
    parser.add_argument(
        '--target-patch-size',
        default=224,
        type=int,
        help='Target patch size passed to CLAM feature extraction.',
    )
    parser.add_argument(
        '--no-stitch',
        action='store_true',
        help='Skip stitched overview image generation during patching.',
    )
    parser.add_argument(
        '--skip-clone',
        action='store_true',
        help='Do not clone CLAM automatically. Useful when the repo already exists.',
    )
    return parser.parse_args()


def run_command(command, cwd=None):
    printable = ' '.join(str(part) for part in command)
    print(f'Running: {printable}')
    subprocess.run(command, cwd=cwd, check=True)


def ensure_clam_repo(clam_dir, skip_clone):
    if clam_dir.exists():
        return

    if skip_clone:
        raise FileNotFoundError(f'CLAM repo not found at {clam_dir}')

    clam_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(['git', 'clone', '--depth', '1', CLAM_REPO_URL, str(clam_dir)])


def build_slide_id_csv(raw_slide_dir, slide_ext, output_csv):
    slide_paths = sorted(raw_slide_dir.glob(f'*{slide_ext}'))
    if not slide_paths:
        raise FileNotFoundError(
            f'No slide files ending with "{slide_ext}" were found in {raw_slide_dir}'
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['slide_id'])
        writer.writeheader()
        for slide_path in slide_paths:
            writer.writerow({'slide_id': slide_path.stem})

    print(f'Wrote slide list with {len(slide_paths)} entries to {output_csv}')
    return output_csv


def main():
    args = parse_args()

    raw_slide_dir = args.raw_slide_dir.resolve()
    clam_dir = args.clam_dir.resolve()
    patch_root = args.patch_root.resolve()
    feature_root = args.feature_root.resolve()
    slide_list_csv = patch_root / 'slide_ids.csv'

    ensure_clam_repo(clam_dir, skip_clone=args.skip_clone)
    build_slide_id_csv(raw_slide_dir, args.slide_ext, slide_list_csv)

    patch_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    patch_command = [
        sys.executable,
        'create_patches_fp.py',
        '--source', str(raw_slide_dir),
        '--save_dir', str(patch_root),
        '--patch_size', str(args.patch_size),
        '--seg',
        '--patch',
    ]
    if not args.no_stitch:
        patch_command.append('--stitch')
    if args.preset:
        patch_command.extend(['--preset', args.preset])

    feature_command = [
        sys.executable,
        'extract_features_fp.py',
        '--data_h5_dir', str(patch_root),
        '--data_slide_dir', str(raw_slide_dir),
        '--csv_path', str(slide_list_csv),
        '--feat_dir', str(feature_root),
        '--slide_ext', args.slide_ext,
        '--batch_size', str(args.batch_size),
        '--model_name', args.model_name,
        '--target_patch_size', str(args.target_patch_size),
    ]

    run_command(patch_command, cwd=clam_dir)
    run_command(feature_command, cwd=clam_dir)

    print()
    print(f'CLAM preprocessing finished. Slide features should now be under {feature_root / "pt_files"}')
    print('You can train TransMIL with:')
    print('  py train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0')


if __name__ == '__main__':
    main()

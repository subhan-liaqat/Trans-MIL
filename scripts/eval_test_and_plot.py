#!/usr/bin/env python3
"""
Load a checkpoint, run the test split, save per-slide probabilities, and write
ROC / PR / confusion-matrix figures (Colab-friendly).

Example (Colab, from repo root):

  !python scripts/eval_test_and_plot.py \\
      --config Camelyon/TransMIL.yaml \\
      --ckpt logs/Camelyon/TransMIL/fold0/epoch=07-val_loss=0.4929.ckpt \\
      --fold 0 --gpus 0

On Colab, Hugging Face's ``datasets`` package shadows this repo's ``datasets/``
folder. We never ``import datasets`` here: ``CamelData`` is loaded from
``datasets/camel_data.py`` via ``importlib`` so the name clash cannot occur.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _load_camel_data_class():
    """Load CamelData without importing the ``datasets`` package (HF conflict on Colab)."""
    path = _REPO_ROOT / "datasets" / "camel_data.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected {path}")
    spec = importlib.util.spec_from_file_location("transmil_camel_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CamelData


CamelData = _load_camel_data_class()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc as sk_auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from models import ModelInterface
from utils.utils import read_yaml


def resolve_log_path(cfg):
    log_path = cfg.General.log_path
    log_name = Path(cfg.config).parent
    version_name = Path(cfg.config).name[:-5]
    return Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'


def find_metrics_csv(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted(log_dir.rglob('metrics.csv'), key=lambda p: p.stat().st_mtime, reverse=True)


def plot_training_curves(metrics_path: Path, out_path: Path) -> bool:
    mdf = pd.read_csv(metrics_path)
    if mdf.empty:
        return False

    fig, ax1 = plt.subplots(figsize=(7, 4))
    plotted = False

    if 'val_loss' in mdf.columns:
        s = mdf['val_loss'].dropna()
        if not s.empty:
            x = mdf.loc[s.index, 'epoch'] if 'epoch' in mdf.columns else s.index.to_numpy()
            ax1.plot(x, s.to_numpy(), color='C0', label='val_loss')
            ax1.set_xlabel('epoch' if 'epoch' in mdf.columns else 'row')
            ax1.set_ylabel('val_loss', color='C0')
            ax1.tick_params(axis='y', labelcolor='C0')
            plotted = True

    if 'auc' in mdf.columns:
        ax2 = ax1.twinx() if plotted else ax1
        s = mdf['auc'].dropna()
        if not s.empty:
            x = mdf.loc[s.index, 'epoch'] if 'epoch' in mdf.columns else s.index.to_numpy()
            ax2.plot(x, s.to_numpy(), color='C1', label='val auc')
            ax2.set_ylabel('auc', color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    fig.suptitle(f'Training curves (from {metrics_path.name})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='Camelyon/TransMIL.yaml')
    p.add_argument('--ckpt', required=True, help='Path to a .ckpt file')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--gpus', default='0', help='Use "cpu" for CPU inference.')
    p.add_argument(
        '--out-dir',
        default=None,
        help='Output directory (default: same folder as the checkpoint).',
    )
    p.add_argument(
        '--positive-class',
        type=int,
        default=1,
        help='Index in Y_prob used as the positive score (tumor=1 for binary CAMELYON).',
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg.config = args.config
    cfg.Data.fold = args.fold

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_dir = Path(args.out_dir).resolve() if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = resolve_log_path(cfg)

    use_cuda = args.gpus.strip().lower() not in {'cpu', 'none', ''} and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    model = ModelInterface.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        data=cfg.Data,
        log=log_path,
    )
    model.eval()
    model.to(device)

    n_classes = int(model.n_classes)
    if n_classes != 2:
        raise ValueError(
            f'This script expects binary classification (n_classes=2); got {n_classes}.'
        )

    pos_idx = args.positive_class
    if pos_idx < 0 or pos_idx >= n_classes:
        raise ValueError(f'positive-class must be in [0, {n_classes - 1}]')

    test_ds = CamelData(dataset_cfg=cfg.Data, state='test')

    slide_ids: list[str] = []
    y_true: list[int] = []
    prob_pos: list[float] = []

    with torch.inference_mode():
        for idx in range(len(test_ds)):
            slide_id = str(test_ds.data.iloc[idx])
            label = int(test_ds.label.iloc[idx])
            features, _ = test_ds[idx]
            features = features.unsqueeze(0).to(device)
            out = model.model(data=features)
            probs = out['Y_prob'].squeeze(0).detach().float().cpu().numpy()
            score = float(probs[pos_idx])

            slide_ids.append(slide_id)
            y_true.append(label)
            prob_pos.append(score)

    pred_csv = out_dir / 'test_predictions.csv'
    pd.DataFrame(
        {'slide_id': slide_ids, 'y_true': y_true, 'prob_positive': prob_pos}
    ).to_csv(pred_csv, index=False)
    print(f'Wrote {pred_csv}')

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(prob_pos, dtype=float)

    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = sk_auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)

    y_hat = (s >= 0.5).astype(int)
    cm = confusion_matrix(y, y_hat)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].set_title('ROC (test)')
    axes[0].legend(loc='lower right')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].plot(recall, precision, lw=2, label=f'AP = {ap:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision–recall (test)')
    axes[1].legend(loc='lower left')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    im = axes[2].imshow(cm, cmap='Blues')
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(['Pred 0', 'Pred 1'])
    axes[2].set_yticklabels(['True 0', 'True 1'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, int(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    axes[2].set_title('Confusion @ threshold 0.5')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(ckpt_path.name, fontsize=10, y=1.02)
    fig.tight_layout()
    fig_path = out_dir / 'test_roc_pr_confusion.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_path}')

    metrics_files = find_metrics_csv(log_path)
    if metrics_files:
        curves_path = out_dir / 'training_val_curves.png'
        if plot_training_curves(metrics_files[0], curves_path):
            print(f'Wrote {curves_path} (from {metrics_files[0]})')
        else:
            print(f'Skipped training curves: could not plot from {metrics_files[0]}')
    else:
        print(f'No metrics.csv found under {log_path}; skipped training curves plot.')

    print('Done.')
    print('Absolute paths (use with IPython.display.Image in Colab):')
    print(f'  {pred_csv.resolve()}')
    print(f'  {fig_path.resolve()}')
    curves_png = out_dir / 'training_val_curves.png'
    if curves_png.is_file():
        print(f'  {curves_png.resolve()}')


if __name__ == '__main__':
    main()

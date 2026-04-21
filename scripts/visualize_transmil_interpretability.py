#!/usr/bin/env python3
"""
Create TransMIL interpretability visualizations from test-slide feature bags.

Outputs:
- per-slide token importance vector (gradient-based)
- per-slide square-grid token heatmap
- optional XY-coordinate scatter heatmap (if coords are provided)
- optional top-k patch gallery (if patch manifest is provided)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _load_camel_data_class():
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

from models import ModelInterface
from utils.utils import read_yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="Camelyon/TransMIL.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--gpus", default="0", help='Use "cpu" to force CPU.')
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-slides", type=int, default=20)
    p.add_argument("--target-class", type=int, default=1)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument(
        "--coords-dir",
        default=None,
        help="Optional directory containing per-slide coords .npy files with shape [n_tokens, 2].",
    )
    p.add_argument(
        "--patch-manifest",
        default=None,
        help=(
            "Optional CSV with columns: slide_id,patch_index,image_path. "
            "Used to render top-k patch gallery."
        ),
    )
    return p.parse_args()


def resolve_log_path(cfg):
    log_path = cfg.General.log_path
    log_name = Path(cfg.config).parent
    version_name = Path(cfg.config).name[:-5]
    return Path(log_path) / log_name / version_name / f"fold{cfg.Data.fold}"


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_v = float(scores.min())
    max_v = float(scores.max())
    if max_v <= min_v:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


def compute_token_importance(model, features: torch.Tensor, target_class: int) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    x = features.clone().detach().requires_grad_(True)
    out = model.model(data=x)
    logits = out["logits"]
    if target_class < 0 or target_class >= logits.shape[1]:
        raise ValueError(f"target-class must be in [0, {logits.shape[1] - 1}]")
    score = logits[:, target_class].sum()
    score.backward()
    grads = x.grad.detach().float().cpu().numpy().squeeze(0)  # [n_tokens, feat_dim]
    token_importance = np.linalg.norm(grads, axis=1)
    return normalize_scores(token_importance)


def plot_grid_heatmap(scores: np.ndarray, out_path: Path, title: str):
    n = scores.shape[0]
    side = int(np.ceil(np.sqrt(n)))
    padded = np.pad(scores, (0, side * side - n), mode="constant", constant_values=0.0)
    grid = padded.reshape(side, side)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="inferno")
    ax.set_title(title)
    ax.set_xlabel("Token X")
    ax.set_ylabel("Token Y")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coords_heatmap(scores: np.ndarray, coords_path: Path, out_path: Path, title: str) -> bool:
    if not coords_path.is_file():
        return False
    coords = np.load(coords_path)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return False
    n = min(scores.shape[0], coords.shape[0])
    if n == 0:
        return False
    xy = coords[:n]
    val = scores[:n]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap="inferno", s=8, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_topk_gallery(scores: np.ndarray, slide_id: str, manifest_df: pd.DataFrame, top_k: int, out_path: Path) -> bool:
    rows = manifest_df[manifest_df["slide_id"] == slide_id]
    if rows.empty:
        return False
    rows = rows.sort_values("patch_index")
    patch_indices = rows["patch_index"].astype(int).to_numpy()
    image_paths = rows["image_path"].astype(str).to_numpy()
    candidates = [(idx, image_paths[i]) for i, idx in enumerate(patch_indices) if 0 <= idx < len(scores)]
    if not candidates:
        return False

    ranked = sorted(candidates, key=lambda t: float(scores[t[0]]), reverse=True)[:top_k]
    if not ranked:
        return False

    cols = int(np.ceil(np.sqrt(len(ranked))))
    rows_n = int(np.ceil(len(ranked) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.6 * cols, 2.6 * rows_n))
    axes = np.atleast_1d(axes).reshape(rows_n, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (patch_idx, image_path) in zip(axes.ravel(), ranked):
        path = Path(image_path)
        if not path.is_file():
            continue
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(f"idx={patch_idx}, s={scores[patch_idx]:.3f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Top-{len(ranked)} patches: {slide_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg.config = args.config
    cfg.Data.fold = args.fold

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else ckpt_path.parent / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_dir = Path(args.coords_dir).resolve() if args.coords_dir else None
    manifest_df = None
    if args.patch_manifest:
        manifest_df = pd.read_csv(args.patch_manifest)
        required_cols = {"slide_id", "patch_index", "image_path"}
        missing = required_cols - set(manifest_df.columns)
        if missing:
            raise ValueError(f"patch-manifest missing columns: {sorted(missing)}")

    log_path = resolve_log_path(cfg)
    use_cuda = args.gpus.strip().lower() not in {"cpu", "none", ""} and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = ModelInterface.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        data=cfg.Data,
        log=log_path,
    )
    model.eval()
    model.to(device)

    ds = CamelData(dataset_cfg=cfg.Data, state="test")
    max_slides = min(args.max_slides, len(ds))

    summary_rows = []
    for idx in range(max_slides):
        slide_id = str(ds.data.iloc[idx])
        label = int(ds.label.iloc[idx])
        features, _ = ds[idx]
        x = features.unsqueeze(0).to(device)

        with torch.enable_grad():
            scores = compute_token_importance(model, x, args.target_class)

        npy_path = out_dir / f"{slide_id}_token_importance.npy"
        np.save(npy_path, scores)

        grid_path = out_dir / f"{slide_id}_grid_heatmap.png"
        plot_grid_heatmap(scores, grid_path, f"{slide_id} token importance")

        coords_png = ""
        if coords_dir is not None:
            cpath = coords_dir / f"{slide_id}.npy"
            coords_out = out_dir / f"{slide_id}_coords_heatmap.png"
            if plot_coords_heatmap(scores, cpath, coords_out, f"{slide_id} coordinate heatmap"):
                coords_png = str(coords_out.resolve())

        gallery_png = ""
        if manifest_df is not None:
            gallery_out = out_dir / f"{slide_id}_topk_gallery.png"
            if plot_topk_gallery(scores, slide_id, manifest_df, args.top_k, gallery_out):
                gallery_png = str(gallery_out.resolve())

        summary_rows.append(
            {
                "slide_id": slide_id,
                "label": label,
                "n_tokens": int(scores.shape[0]),
                "importance_npy": str(npy_path.resolve()),
                "grid_heatmap_png": str(grid_path.resolve()),
                "coords_heatmap_png": coords_png,
                "topk_gallery_png": gallery_png,
            }
        )

        print(f"[{idx + 1}/{max_slides}] Wrote interpretability assets for {slide_id}")

    summary_csv = out_dir / "interpretability_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

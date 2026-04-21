#!/usr/bin/env python3
"""
Plot ablation and convergence comparisons.

1) Ablation:
   Input CSV must contain columns: experiment, auc
   Optional columns: dataset, params_m

2) Convergence:
   Compare one metric (default: auc) from multiple Lightning metrics.csv files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation-csv", default=None, help="CSV with columns experiment,auc[,dataset,params_m].")
    p.add_argument("--metrics-csvs", nargs="*", default=None, help="One or more Lightning metrics.csv paths.")
    p.add_argument(
        "--metrics-labels",
        nargs="*",
        default=None,
        help="Optional labels for --metrics-csvs (same count).",
    )
    p.add_argument("--metric-name", default="auc", help="Metric column to compare for convergence.")
    p.add_argument("--out-dir", default="plots")
    return p.parse_args()


def plot_ablation(ablation_csv: Path, out_dir: Path):
    df = pd.read_csv(ablation_csv)
    required = {"experiment", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ablation CSV missing columns: {sorted(missing)}")

    if "dataset" in df.columns:
        datasets = sorted(df["dataset"].dropna().unique())
    else:
        datasets = ["all"]
        df["dataset"] = "all"

    for ds in datasets:
        part = df[df["dataset"] == ds].copy()
        part = part.sort_values("auc", ascending=False)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.bar(part["experiment"], part["auc"], color="tab:blue", alpha=0.85)
        ax.set_ylim(0, max(1.0, float(part["auc"].max()) + 0.02))
        ax.set_ylabel("AUC")
        ax.set_title(f"Ablation ({ds})")
        ax.tick_params(axis="x", rotation=25)
        for bar, auc in zip(bars, part["auc"]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.003, f"{auc:.4f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"ablation_{ds}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


def _extract_metric_curve(csv_path: Path, metric_name: str):
    df = pd.read_csv(csv_path)
    if metric_name not in df.columns:
        return None, None
    metric = df[metric_name].dropna()
    if metric.empty:
        return None, None
    if "epoch" in df.columns:
        x = df.loc[metric.index, "epoch"].to_numpy()
    else:
        x = metric.index.to_numpy()
    return x, metric.to_numpy()


def plot_convergence(metrics_csvs: list[Path], labels: list[str] | None, metric_name: str, out_dir: Path):
    if labels is not None and len(labels) != len(metrics_csvs):
        raise ValueError("--metrics-labels must match --metrics-csvs count.")
    if labels is None:
        labels = [p.parent.name for p in metrics_csvs]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    for csv_path, label in zip(metrics_csvs, labels):
        x, y = _extract_metric_curve(csv_path, metric_name)
        if x is None:
            print(f"Skipped {csv_path}: no usable '{metric_name}' values")
            continue
        ax.plot(x, y, lw=2, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        raise RuntimeError(f"No convergence curves could be plotted for metric '{metric_name}'.")

    ax.set_xlabel("epoch" if metric_name != "step" else "step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Convergence comparison ({metric_name})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"convergence_compare_{metric_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.ablation_csv and not args.metrics_csvs:
        raise ValueError("Provide at least one of: --ablation-csv or --metrics-csvs")

    if args.ablation_csv:
        plot_ablation(Path(args.ablation_csv).resolve(), out_dir)

    if args.metrics_csvs:
        csv_paths = [Path(p).resolve() for p in args.metrics_csvs]
        labels = args.metrics_labels if args.metrics_labels else None
        plot_convergence(csv_paths, labels, args.metric_name, out_dir)


if __name__ == "__main__":
    main()

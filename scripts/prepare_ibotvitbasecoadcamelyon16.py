from pathlib import Path
import argparse
import random
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


POSITIVE_HINTS = ("tumor", "tumour", "positive", "metastasis", "metastases", "malignant")
NEGATIVE_HINTS = ("normal", "negative", "benign")


def infer_label(path: Path) -> Optional[int]:
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if any(h in p for h in POSITIVE_HINTS):
            return 1
    for p in parts:
        if any(h in p for h in NEGATIVE_HINTS):
            return 0
    # CAMELYON test slides are often named test_XXX and are unlabeled.
    if any("test_" in p for p in parts):
        return None
    raise ValueError(f"Cannot infer class label from path: {path}")


def normalize_slide_id(raw_name: str) -> str:
    slide_id = raw_name
    slide_id = re.sub(r"__features$", "", slide_id, flags=re.IGNORECASE)
    slide_id = re.sub(r"\.(tif|tiff|svs|ndpi)$", "", slide_id, flags=re.IGNORECASE)
    slide_id = slide_id.replace(" ", "_")
    return slide_id


def split_classwise(ids: List[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def pad_rows(values: List[str], labels: List[int], max_len: int):
    out_vals = values + [None] * (max_len - len(values))
    out_lbls = labels + [None] * (max_len - len(labels))
    return out_vals, out_lbls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--output_feature_dir", default="IbotViTBaseCOADCamelyon16/pt_files", type=str)
    parser.add_argument("--output_csv", default="dataset_csv/ibotvitbasecoadcamelyon16/fold0.csv", type=str)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--val_ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=2021, type=int)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    feature_out = Path(args.output_feature_dir)
    csv_out = Path(args.output_csv)
    feature_out.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(dataset_root.rglob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy feature files found in {dataset_root}")

    id_to_label = {}
    skipped_unlabeled = 0
    for npy_path in npy_files:
        label = infer_label(npy_path)
        if label is None:
            skipped_unlabeled += 1
            continue
        # In this Kaggle dataset each feature file is named "features.npy"
        # under a per-slide directory like "Tumor_001.tif/features.npy".
        slide_id = normalize_slide_id(npy_path.parent.name)
        arr = np.load(npy_path)
        tensor = torch.from_numpy(arr).float()
        torch.save(tensor, feature_out / f"{slide_id}.pt")
        id_to_label[slide_id] = label

    class0 = sorted([sid for sid, y in id_to_label.items() if y == 0])
    class1 = sorted([sid for sid, y in id_to_label.items() if y == 1])
    if not class0 or not class1:
        raise RuntimeError(
            "Need both classes for classification. Could not find both normal and tumor feature files."
        )

    train0, val0, test0 = split_classwise(class0, args.train_ratio, args.val_ratio, args.seed)
    train1, val1, test1 = split_classwise(class1, args.train_ratio, args.val_ratio, args.seed + 1)

    train_ids = train0 + train1
    train_lbl = [0] * len(train0) + [1] * len(train1)
    val_ids = val0 + val1
    val_lbl = [0] * len(val0) + [1] * len(val1)
    test_ids = test0 + test1
    test_lbl = [0] * len(test0) + [1] * len(test1)

    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    train_ids, train_lbl = pad_rows(train_ids, train_lbl, max_len)
    val_ids, val_lbl = pad_rows(val_ids, val_lbl, max_len)
    test_ids, test_lbl = pad_rows(test_ids, test_lbl, max_len)

    df = pd.DataFrame(
        {
            "train": train_ids,
            "train_label": train_lbl,
            "val": val_ids,
            "val_label": val_lbl,
            "test": test_ids,
            "test_label": test_lbl,
        }
    )
    df.to_csv(csv_out)

    print(f"Processed {len(id_to_label)} feature files from {dataset_root}")
    print(f"Skipped unlabeled test features: {skipped_unlabeled}")
    print(f"Saved PT features to: {feature_out}")
    print(f"Saved fold CSV to: {csv_out}")
    print(f"Class counts: normal={len(class0)}, tumor={len(class1)}")


if __name__ == "__main__":
    main()

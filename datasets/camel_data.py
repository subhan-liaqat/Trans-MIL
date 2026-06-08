import random
import torch
import pandas as pd
import numpy as np
from pathlib import Path

import torch.utils.data as data


class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = Path(self.dataset_cfg.data_dir)
        self.coords_dir = getattr(self.dataset_cfg, 'coords_dir', None)
        if self.coords_dir:
            self.coords_dir = Path(self.coords_dir)
        self.labels_dir = getattr(self.dataset_cfg, 'labels_dir', None)
        if self.labels_dir:
            self.labels_dir = Path(self.labels_dir)
        self.patch_size = getattr(self.dataset_cfg, 'patch_size', 512)
        self.split_format = getattr(self.dataset_cfg, 'split_format', 'fold_csv')
        self.feature_exts = self._resolve_feature_exts()
        self.shuffle = self.dataset_cfg.data_shuffle

        if self.split_format == 'torchmil':
            self._init_torchmil_splits(state)
        else:
            self._init_fold_csv_splits(state)

    def _init_fold_csv_splits(self, state):
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

    def _init_torchmil_splits(self, state):
        splits_path = Path(self.dataset_cfg.splits_csv)
        if not splits_path.is_file():
            raise FileNotFoundError(f'splits.csv not found: {splits_path}')

        splits_df = pd.read_csv(splits_path)
        bag_col, split_col = self._resolve_torchmil_columns(splits_df)
        splits_df[split_col] = splits_df[split_col].astype(str).str.strip().str.lower()

        train_df = splits_df[splits_df[split_col] == 'train'].copy()
        test_df = splits_df[splits_df[split_col] == 'test'].copy()

        if train_df.empty:
            raise ValueError('No rows with split == "train" found in splits.csv')
        if test_df.empty:
            raise ValueError('No rows with split == "test" found in splits.csv')

        val_ratio = float(getattr(self.dataset_cfg, 'val_ratio', 0.1))
        split_seed = int(getattr(self.dataset_cfg, 'split_seed', 2021))
        train_bags = train_df[bag_col].astype(str).tolist()

        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(len(train_bags))
        n_val = max(1, int(round(len(train_bags) * val_ratio)))
        val_idx = set(perm[:n_val].tolist())

        partition = {
            'train': [train_bags[i] for i in range(len(train_bags)) if i not in val_idx],
            'val': [train_bags[i] for i in range(len(train_bags)) if i in val_idx],
            'test': test_df[bag_col].astype(str).tolist(),
        }

        if state not in partition:
            raise ValueError(f'Unsupported dataset state: {state}')

        bag_names = partition[state]
        labels = [self._load_bag_label(bag_name) for bag_name in bag_names]
        self.data = pd.Series(bag_names, dtype=object)
        self.label = pd.Series(labels, dtype=int)

    def _resolve_torchmil_columns(self, splits_df):
        normalized = {str(col).strip().lower(): col for col in splits_df.columns}

        bag_col = normalized.get('bag_name') or normalized.get('slide_id') or normalized.get('wsi')
        split_col = normalized.get('split') or normalized.get('partition')

        if bag_col is None or split_col is None:
            raise ValueError(
                'splits.csv must contain bag_name (or slide_id/wsi) and split (or partition) columns. '
                f'Found columns: {list(splits_df.columns)}'
            )
        return bag_col, split_col

    def _resolve_feature_exts(self):
        if hasattr(self.dataset_cfg, 'feature_exts') and self.dataset_cfg.feature_exts:
            return list(self.dataset_cfg.feature_exts)

        feature_ext = getattr(self.dataset_cfg, 'feature_ext', None)
        if feature_ext:
            return [feature_ext]

        return ['.npy']

    def _load_bag_label(self, slide_id):
        if self.labels_dir is None:
            raise ValueError(
                f'labels_dir is required for torchmil split format (slide: {slide_id}).'
            )

        label_path = self.labels_dir / f'{slide_id}.npy'
        if not label_path.is_file():
            raise FileNotFoundError(f'Label file not found: {label_path}')

        value = np.load(label_path)
        return int(np.asarray(value).reshape(-1)[0])

    def _load_features(self, slide_id):
        for feature_ext in self.feature_exts:
            full_path = self.feature_dir / f'{slide_id}{feature_ext}'
            if not full_path.exists():
                continue

            if feature_ext == '.pt':
                features = torch.load(full_path, map_location='cpu')
            elif feature_ext == '.npy':
                features = torch.from_numpy(np.load(full_path))
            else:
                raise ValueError(f'Unsupported feature extension: {feature_ext}')

            if not torch.is_tensor(features):
                features = torch.as_tensor(features)

            return features.float()

        searched_paths = [str(self.feature_dir / f'{slide_id}{ext}') for ext in self.feature_exts]
        raise FileNotFoundError(
            f'Could not find features for slide "{slide_id}". Looked for: {searched_paths}'
        )

    def _load_coords(self, slide_id, n_tokens):
        if not self.coords_dir:
            return None

        coords_path = self.coords_dir / f'{slide_id}.npy'
        if not coords_path.exists():
            return None

        coords = np.load(coords_path)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f'Expected coords shape [n_tokens, 2] for slide "{slide_id}", got {coords.shape}'
            )

        n = min(n_tokens, coords.shape[0])
        coords = coords[:n].astype(np.float32)
        return torch.from_numpy(coords)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = str(self.data.iloc[idx])
        label = int(self.label.iloc[idx])
        features = self._load_features(slide_id)
        coords = self._load_coords(slide_id, features.shape[0])
        if coords is not None and coords.shape[0] != features.shape[0]:
            n = min(features.shape[0], coords.shape[0])
            features = features[:n]
            coords = coords[:n]

        if self.shuffle:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
            if coords is not None:
                coords = coords[index]

        if coords is None:
            coords = torch.empty(0, 2, dtype=torch.float32)

        return features, coords, label

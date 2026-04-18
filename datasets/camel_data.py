import random
import torch
import pandas as pd
import numpy as np
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader


class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)
        self.feature_exts = self._resolve_feature_exts()

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

    def _resolve_feature_exts(self):
        if hasattr(self.dataset_cfg, 'feature_exts') and self.dataset_cfg.feature_exts:
            return list(self.dataset_cfg.feature_exts)

        feature_ext = getattr(self.dataset_cfg, 'feature_ext', None)
        if feature_ext:
            return [feature_ext]

        return ['.pt', '.npy']

    def _load_features(self, slide_id):
        for feature_ext in self.feature_exts:
            full_path = Path(self.feature_dir) / f'{slide_id}{feature_ext}'
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

        searched_paths = [str(Path(self.feature_dir) / f'{slide_id}{ext}') for ext in self.feature_exts]
        raise FileNotFoundError(
            f'Could not find features for slide "{slide_id}". Looked for: {searched_paths}'
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data.iloc[idx]
        label = int(self.label.iloc[idx])
        features = self._load_features(slide_id)

        #----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]


        return features, label


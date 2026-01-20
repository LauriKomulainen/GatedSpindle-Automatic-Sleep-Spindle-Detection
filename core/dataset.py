# data_preprocess/dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging
from paths import SELECTED_DATASET
from signal_processing.transforms import compute_input_channels, RandomAugment1D

if SELECTED_DATASET == "DREAMS":
    from configs.dreams_config import (
        DATA_PARAMS
    )
elif SELECTED_DATASET == "MASS":
    from configs.mass_config import (
        DATA_PARAMS
    )

log = logging.getLogger(__name__)


class SpindleDataset(Dataset):
    def __init__(self, x_1d_path, y_1d_path, augment=False):
        self.x_1d_path = x_1d_path
        self.y_path = y_1d_path
        self.x_mmap = np.load(x_1d_path, mmap_mode='r')
        self.y_mmap = np.load(y_1d_path, mmap_mode='r')
        self.length = self.x_mmap.shape[0]
        self.fs = DATA_PARAMS['fs']
        self.augment = augment
        self.augmentor = RandomAugment1D(p=0.5)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raw_signal = np.array(self.x_mmap[idx], dtype=np.float32)
        signal_tensor = compute_input_channels(raw_signal, self.fs)

        if self.augment:
            signal_tensor = self.augmentor(signal_tensor)

        # Label loading
        mask_1d = np.array(self.y_mmap[idx], dtype=np.float32)
        mask_tensor = torch.tensor(mask_1d, dtype=torch.float32)

        # Global label
        has_spindle = 1.0 if np.max(mask_1d) > 0.5 else 0.0
        label_tensor = torch.tensor(has_spindle, dtype=torch.float32).unsqueeze(0)

        return signal_tensor, mask_tensor, label_tensor


def get_dataloaders(processed_data_dir: str, batch_size: int, train_subject_ids: list, val_subject_ids: list,
                    test_subject_ids: list):

    log.info(f"Training data: {train_subject_ids}")
    log.info(f"Validation data: {val_subject_ids}")
    log.info(f"Test data: {test_subject_ids}")

    datasets = {}
    all_subjects = list(set(train_subject_ids + val_subject_ids + test_subject_ids))

    for subject_id in all_subjects:
        x_1d_path = os.path.join(processed_data_dir, f"{subject_id}_X_1D.npy")
        y_1d_path = os.path.join(processed_data_dir, f"{subject_id}_Y_1D.npy")

        if not (os.path.exists(x_1d_path) and os.path.exists(y_1d_path)):
            log.warning(f"Files not found for {subject_id} in {processed_data_dir}")
            continue

        datasets[subject_id] = SpindleDataset(x_1d_path, y_1d_path, augment=False)

    train_list = []
    for sid in train_subject_ids:
        if sid in datasets:
            ds = SpindleDataset(datasets[sid].x_1d_path, datasets[sid].y_path, augment=True)
            train_list.append(ds)

    train_ds = ConcatDataset(train_list) if train_list else []
    val_ds = ConcatDataset([datasets[sid] for sid in val_subject_ids if sid in datasets]) if val_subject_ids else []
    test_ds = ConcatDataset([datasets[sid] for sid in test_subject_ids if sid in datasets]) if test_subject_ids else []

    common = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True}

    train_shuffle = True if len(train_ds) > 0 else False

    return (DataLoader(train_ds, shuffle=train_shuffle, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common))
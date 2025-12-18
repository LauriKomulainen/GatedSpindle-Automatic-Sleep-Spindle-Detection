# data_preprocess/dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import logging
import random
from scipy.signal import butter, filtfilt
from configs.dreams_config import DATA_PARAMS, METRIC_PARAMS
from data_preprocess.normalization import normalize_data

log = logging.getLogger(__name__)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


class RandomAugment1D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        if random.random() < self.p:
            gain = random.uniform(0.8, 1.2)
            signal = signal * gain
            noise_level = random.uniform(0.0, 0.02)
            noise = torch.randn_like(signal) * noise_level
            signal = signal + noise
        return signal


class SpindleDataset(Dataset):
    def __init__(self, x_1d_path, y_1d_path, seq_len=1, augment=False):
        self.x_1d_path = x_1d_path
        self.y_path = y_1d_path
        self.x_mmap = np.load(x_1d_path, mmap_mode='r')
        self.y_mmap = np.load(y_1d_path, mmap_mode='r')
        self.length = self.x_mmap.shape[0]
        self.fs = 200.0
        self.augment = augment
        self.augmentor = RandomAugment1D(p=0.5)
        self.use_instance_norm = DATA_PARAMS.get('use_instance_norm', True)

        # Haetaan taajuusrajat configista
        self.low_f = METRIC_PARAMS['spindle_freq_low']
        self.high_f = METRIC_PARAMS['spindle_freq_high']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Channel 1: Raw EEG
        raw_signal = np.array(self.x_mmap[idx], dtype=np.float32)
        ch1 = torch.tensor(raw_signal, dtype=torch.float32).unsqueeze(0)

        # Channel 2: Sigma Filtered
        sigma_signal = butter_bandpass_filter(raw_signal, self.low_f, self.high_f, self.fs, order=4)

        if self.use_instance_norm:
            sigma_signal = normalize_data(sigma_signal)

        ch2 = torch.tensor(sigma_signal.copy(), dtype=torch.float32).unsqueeze(0)

        signal_tensor = torch.cat([ch1, ch2], dim=0)

        if self.augment:
            signal_tensor = self.augmentor(signal_tensor)

        # --- LABEL LOADING (HARD LABELS) ---
        mask_1d = np.array(self.y_mmap[idx], dtype=np.float32)
        mask_tensor = torch.tensor(mask_1d, dtype=torch.float32)

        # Global label
        has_spindle = 1.0 if np.max(mask_1d) > 0.5 else 0.0
        label_tensor = torch.tensor(has_spindle, dtype=torch.float32).unsqueeze(0)

        return signal_tensor, mask_tensor, label_tensor


def get_dataloaders(processed_data_dir: str, batch_size: int, train_subject_ids: list, val_subject_ids: list,
                    test_subject_ids: list, use_fraction: float = 1.0):
    if not os.path.exists(processed_data_dir):
        log.error(f"CRITICAL: Data directory not found: {processed_data_dir}")
        empty = DataLoader([], batch_size=batch_size)
        return empty, empty, empty

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

    if use_fraction < 1.0 and len(train_ds) > 0:
        def get_subset(ds):
            if len(ds) == 0: return ds
            indices = random.sample(range(len(ds)), int(len(ds) * use_fraction))
            return Subset(ds, indices)

        train_ds = get_subset(train_ds)
        val_ds = get_subset(val_ds)
        test_ds = get_subset(test_ds)

    common = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True}

    train_shuffle = True if len(train_ds) > 0 else False

    return (DataLoader(train_ds, shuffle=train_shuffle, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common))
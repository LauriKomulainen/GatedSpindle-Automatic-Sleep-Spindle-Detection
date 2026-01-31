# core/dataset.py

import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from signal_processing.transforms import compute_input_channels, RandomAugment1D
from configs.config_loader import DATA_PARAMS

log = logging.getLogger(__name__)


class SpindleDataset(Dataset):
    """Dataset for sleep spindle detection with optional augmentation."""

    def __init__(self, x_path: str, y_path: str, augment: bool = False):
        self.x_path = x_path
        self.y_path = y_path
        self.x_mmap = np.load(x_path, mmap_mode="r")
        self.y_mmap = np.load(y_path, mmap_mode="r")
        self.length = self.x_mmap.shape[0]
        self.fs = DATA_PARAMS["fs"]
        self.augment = augment
        self.augmentor = RandomAugment1D(p=0.5) if augment else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load and transform signal
        raw_signal = np.array(self.x_mmap[idx], dtype=np.float32)
        signal = compute_input_channels(raw_signal, self.fs)

        if self.augmentor:
            signal = self.augmentor(signal)

        # Load mask and create global label
        mask = np.array(self.y_mmap[idx], dtype=np.float32)
        has_spindle = float(mask.max() > 0.5)

        return (
            signal,
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor([has_spindle], dtype=torch.float32),
        )


def _get_data_paths(data_dir: str, subject_id: str) -> tuple[str, str]:
    """Return (x_path, y_path) for a subject."""
    return (
        os.path.join(data_dir, f"{subject_id}_X_1D.npy"),
        os.path.join(data_dir, f"{subject_id}_Y_1D.npy"),
    )


def _filter_valid_subjects(data_dir: str, subject_ids: list) -> list:
    """Return only subjects that have both X and Y files."""
    valid = []
    for sid in subject_ids:
        x_path, y_path = _get_data_paths(data_dir, sid)
        if os.path.exists(x_path) and os.path.exists(y_path):
            valid.append(sid)
    return valid


def get_dataloaders(
    processed_data_dir: str,
    batch_size: int,
    train_subject_ids: list,
    val_subject_ids: list,
    test_subject_ids: list,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    # Filter to valid subjects only
    all_requested = set(train_subject_ids + val_subject_ids + test_subject_ids)
    valid_subjects = set(_filter_valid_subjects(processed_data_dir, all_requested))
    missing = all_requested - valid_subjects

    if missing:
        mode = DATA_PARAMS.get("scorer_mode", "UNKNOWN")
        log.info(f"Filtered out {len(missing)} subjects (Scorer Mode: {mode}): {sorted(missing)}")

    # Update lists with valid subjects only
    train_ids = [s for s in train_subject_ids if s in valid_subjects]
    val_ids = [s for s in val_subject_ids if s in valid_subjects]
    test_ids = [s for s in test_subject_ids if s in valid_subjects]

    log.info(f"Dataset split - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Helper to create dataset for a subject
    def make_dataset(sid: str, augment: bool = False) -> SpindleDataset:
        x_path, y_path = _get_data_paths(processed_data_dir, sid)
        return SpindleDataset(x_path, y_path, augment=augment)

    # Build datasets
    train_ds = ConcatDataset([make_dataset(s, augment=True) for s in train_ids]) if train_ids else []
    val_ds = ConcatDataset([make_dataset(s) for s in val_ids]) if val_ids else []
    test_ds = ConcatDataset([make_dataset(s) for s in test_ids]) if test_ids else []

    loader_kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": True}

    return (
        DataLoader(train_ds, shuffle=bool(train_ds), **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )
# core/dataset.py

"""
Dataset module for sleep spindle detection.

Supports runtime scorer mode selection: the correct Y mask file is loaded
based on DATA_PARAMS['scorer_mode'] without needing to rebuild the dataset.

File naming convention:
    {subject_id}_X_1D.npy       - Signal windows (always present)
    {subject_id}_Y_E1.npy       - Expert 1 masks (MASS only)
    {subject_id}_Y_E2.npy       - Expert 2 masks (MASS, 15/19 subjects)
    {subject_id}_Y_UNION.npy    - Merged E1+E2 masks (MASS only)
    {subject_id}_Y_1D.npy       - Default/legacy masks (always present)
"""

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

        # Load mask and create global label
        mask = np.array(self.y_mmap[idx], dtype=np.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Apply augmentation to BOTH signal and mask jointly
        if self.augmentor:
            signal, mask = self.augmentor(signal, mask)

        has_spindle = float(mask.max() > 0.5)

        return (
            signal,
            mask,
            torch.tensor([has_spindle], dtype=torch.float32),
        )


def _get_data_paths(data_dir: str, subject_id: str, scorer_mode: str = None) -> tuple[str, str]:
    """
    Return (x_path, y_path) for a subject.

    For MASS dataset with scorer-specific masks:
    - Tries {id}_Y_{scorer_mode}.npy first (e.g., _Y_E1.npy)
    - Falls back to {id}_Y_1D.npy if scorer-specific file not found

    Args:
        data_dir: Directory containing processed .npy files
        subject_id: Subject identifier (e.g., '01-02-0001')
        scorer_mode: One of 'E1', 'E2', 'UNION', or None for default
    """
    x_path = os.path.join(data_dir, f"{subject_id}_X_1D.npy")

    # Try scorer-specific Y file first
    if scorer_mode and scorer_mode in ('E1', 'E2', 'UNION'):
        y_path_scorer = os.path.join(data_dir, f"{subject_id}_Y_{scorer_mode}.npy")
        if os.path.exists(y_path_scorer):
            return x_path, y_path_scorer

    # Fall back to default
    y_path = os.path.join(data_dir, f"{subject_id}_Y_1D.npy")
    return x_path, y_path


def _filter_valid_subjects(data_dir: str, subject_ids: list, scorer_mode: str = None) -> list:
    """
    Return only subjects that have both X and Y files for the given scorer mode.

    This is where E2-mode correctly filters out subjects without E2 annotations,
    WITHOUT requiring a rebuild of the dataset.
    """
    valid = []
    for sid in subject_ids:
        x_path, y_path = _get_data_paths(data_dir, sid, scorer_mode)
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
    """
    Create train, validation, and test dataloaders.

    Automatically selects the correct Y mask file based on
    DATA_PARAMS['scorer_mode']. No dataset rebuild needed.
    """
    scorer_mode = DATA_PARAMS.get("scorer_mode", None)

    # Filter to valid subjects only
    all_requested = set(train_subject_ids + val_subject_ids + test_subject_ids)
    valid_subjects = set(_filter_valid_subjects(processed_data_dir, all_requested, scorer_mode))
    missing = all_requested - valid_subjects

    if missing:
        log.info(
            f"Filtered out {len(missing)} subjects "
            f"(Scorer Mode: {scorer_mode or 'default'}): {sorted(missing)}"
        )

    # Update lists with valid subjects only
    train_ids = [s for s in train_subject_ids if s in valid_subjects]
    val_ids = [s for s in val_subject_ids if s in valid_subjects]
    test_ids = [s for s in test_subject_ids if s in valid_subjects]

    log.info(
        f"Dataset split (scorer={scorer_mode or 'default'}) - "
        f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}"
    )

    # Helper to create dataset for a subject
    def make_dataset(sid: str, augment: bool = False) -> SpindleDataset:
        x_path, y_path = _get_data_paths(processed_data_dir, sid, scorer_mode)
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
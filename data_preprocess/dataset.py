# data_preprocess/dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import logging
import random

log = logging.getLogger(__name__)


class SpindleDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Hakee 3 kuvan sekvenssin (X), mutta vain
    keskimmäisen kuvan maskin (Y).
    """

    def __init__(self, x_img_path, y_img_path, x_1d_path, seq_len=3):
        self.x_data = np.load(x_img_path).astype(np.float32)
        self.y_data = np.load(y_img_path).astype(np.float32)
        self.x_1d_data = np.load(x_1d_path).astype(np.float32)
        self.seq_len = seq_len
        self.padding = self.seq_len // 2  # 3 -> 1

        assert len(self.x_data) == len(self.y_data) == len(self.x_1d_data), "Data file lengths do not match!"
        log.info(f"Loaded {len(self.x_data)} data triplets from {os.path.basename(x_img_path)}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        # --- Rakenna sekvenssi ---
        indices = []
        for i in range(self.seq_len):
            current_idx = idx - self.padding + i
            # Reunapeilaus
            if current_idx < 0:
                current_idx = abs(current_idx)
            elif current_idx >= len(self.x_data):
                current_idx = (len(self.x_data) - 1) - (current_idx - len(self.x_data))
            indices.append(current_idx)

        x_img_seq = [torch.from_numpy(self.x_data[i]) for i in indices]

        x_img_tensor = torch.stack(x_img_seq, dim=0)

        # Palauta VAIN keskimmäinen maski ja 1D-signaali
        y_img_tensor = torch.from_numpy(self.y_data[idx])
        x_1d_tensor = torch.from_numpy(self.x_1d_data[idx])

        return (
            x_img_tensor,  # (S, C, H, W)
            y_img_tensor,  # (C, H, W)
            x_1d_tensor  # (L)
        )


def get_dataloaders(processed_data_dir: str,
                    batch_size: int,
                    train_subject_ids: list,
                    val_subject_ids: list,
                    test_subject_ids: list,
                    use_fraction: float = 1.0):

    log.info(f"Training data: {train_subject_ids}")
    log.info(f"Validation data: {val_subject_ids}")
    log.info(f"Test data: {test_subject_ids}")

    datasets = {}
    all_subjects_in_use = list(set(train_subject_ids + val_subject_ids + test_subject_ids))

    for subject_id in all_subjects_in_use:
        x_img_path = os.path.join(processed_data_dir, f"{subject_id}_X_images.npy")
        y_img_path = os.path.join(processed_data_dir, f"{subject_id}_Y_images.npy")
        x_1d_path = os.path.join(processed_data_dir, f"{subject_id}_X_1D.npy")

        if not (os.path.exists(x_img_path) and os.path.exists(y_img_path) and os.path.exists(x_1d_path)):
            log.warning(f"Skipping {subject_id}, required files (X_images, Y_images, X_1D) not found.")
            continue

        datasets[subject_id] = SpindleDataset(x_img_path, y_img_path, x_1d_path, seq_len=3)

    train_ds = ConcatDataset([datasets[sid] for sid in train_subject_ids if sid in datasets])
    val_ds = ConcatDataset([datasets[sid] for sid in val_subject_ids if sid in datasets])
    test_ds = ConcatDataset([datasets[sid] for sid in test_subject_ids if sid in datasets])

    if use_fraction < 1.0:
        log.warning(f"Using only {use_fraction * 100:.0f}% of data for fast testing.")

        def get_subset(dataset):
            total_size = len(dataset)
            subset_size = int(total_size * use_fraction)
            indices = random.sample(range(total_size), subset_size)
            return Subset(dataset, indices)

        train_ds = get_subset(train_ds)
        val_ds = get_subset(val_ds)
        test_ds = get_subset(test_ds)

    log.info(f"Total training images: {len(train_ds)}")
    log.info(f"Total validation images: {len(val_ds)}")
    log.info(f"Total test images: {len(test_ds)}")

    common_loader_params = {
        'batch_size': batch_size,
        'num_workers': 8,
        'pin_memory': False,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_params)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_params)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_params)

    return train_loader, val_loader, test_loader
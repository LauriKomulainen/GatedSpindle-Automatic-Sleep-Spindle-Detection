# data_preprocess/dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging

log = logging.getLogger(__name__)


class SpindleDataset(Dataset):
    """
    Custom PyTorch Dataset, loads 3 files:
    1. 2D CWT Image (X_image)
    2. 2D CWT Mask (Y_image)
    3. 1D Signal Window (X_1D)
    """

    def __init__(self, x_img_path, y_img_path, x_1d_path):
        self.x_data = np.load(x_img_path).astype(np.float32)
        self.y_data = np.load(y_img_path).astype(np.float32)
        self.x_1d_data = np.load(x_1d_path).astype(np.float32)
        assert len(self.x_data) == len(self.y_data) == len(self.x_1d_data), "Data file lengths do not match!"
        log.info(f"Loaded {len(self.x_data)} data triplets from {os.path.basename(x_img_path)}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_data[idx]),
            torch.from_numpy(self.y_data[idx]),
            torch.from_numpy(self.x_1d_data[idx])
        )


def get_dataloaders(processed_data_dir: str,
                    batch_size: int,
                    train_subject_ids: list,
                    val_subject_ids: list,
                    test_subject_ids: list):
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

        datasets[subject_id] = SpindleDataset(x_img_path, y_img_path, x_1d_path)

    train_ds = ConcatDataset([datasets[sid] for sid in train_subject_ids if sid in datasets])
    val_ds = ConcatDataset([datasets[sid] for sid in val_subject_ids if sid in datasets])
    test_ds = ConcatDataset([datasets[sid] for sid in test_subject_ids if sid in datasets])

    log.info(f"Total training images: {len(train_ds)}")
    log.info(f"Total validation images: {len(val_ds)}")
    log.info(f"Total test images: {len(test_ds)}")

    common_loader_params = {
        'batch_size': batch_size,
        'num_workers': 0,  # Set to 0 for stability on macOS/MPS
        'pin_memory': False,  # Disabled as it's not supported on MPS
    }

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_params)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_params)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_params)

    return train_loader, val_loader, test_loader
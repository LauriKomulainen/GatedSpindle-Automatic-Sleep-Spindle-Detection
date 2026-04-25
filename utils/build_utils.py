# utils/build_utils.py

"""
Shared helpers for dataset build scripts (build_mass_dataset.py and
build_dreams_dataset.py).
"""

import logging
import shutil
import numpy as np
import paths

log = logging.getLogger(__name__)


def prepare_directories() -> tuple:
    """Prepare output directories, cleaning if they exist.

    Returns (processed_data_dir, plots_dir).
    """
    dirs = [paths.PROCESSED_DATA_DIR, paths.PLOTS_DIR]
    for d in dirs:
        if d.exists():
            log.info(f"Cleaning: {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    return tuple(dirs)


def create_spindle_mask(annotations: list, signal_length: int, fs: float) -> np.ndarray:
    """Create binary mask from spindle annotations."""
    mask = np.zeros(signal_length, dtype=np.float32)
    for annot in annotations:
        if "spindle" not in annot["description"].lower():
            continue
        start = int(annot["onset"] * fs)
        end = min(int(start + annot["duration"] * fs), signal_length)
        if start < end:
            mask[start:end] = 1.0
    return mask


def create_stage_mask(
    hypnogram: np.ndarray,
    signal_length: int,
    fs: float,
    included_stages,
    hypnogram_resolution_sec: float,
) -> np.ndarray:
    """Create mask for valid sleep stages."""
    mask = np.zeros(signal_length, dtype=np.float32)
    samples_per_epoch = int(hypnogram_resolution_sec * fs)

    for i, stage in enumerate(hypnogram):
        start = i * samples_per_epoch
        if start >= signal_length:
            break
        end = min(start + samples_per_epoch, signal_length)
        if stage in included_stages:
            mask[start:end] = 1.0
    return mask


def create_stage_codes(
    hypnogram: np.ndarray,
    signal_length: int,
    fs: float,
    hypnogram_resolution_sec: float,
) -> np.ndarray:
    """Create per-sample stage code array from hypnogram.

    Returns array of single-character stage codes (e.g. 'W', '1', '2', '3', 'R'
    for MASS or '0'-'5' for DREAMS), one per signal sample. Samples outside
    any scored epoch get '?'.
    """
    codes = np.full(signal_length, '?', dtype='<U1')
    samples_per_epoch = int(hypnogram_resolution_sec * fs)

    for i, stage in enumerate(hypnogram):
        start = i * samples_per_epoch
        if start >= signal_length:
            break
        end = min(start + samples_per_epoch, signal_length)
        codes[start:end] = stage
    return codes
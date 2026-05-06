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


def count_spindles_per_stage(
    spindle_mask: np.ndarray,
    hypnogram: np.ndarray,
    fs: float,
    hypnogram_resolution_sec: float,
) -> dict:
    """Count spindle events per sleep stage.

    A spindle event is assigned to the stage that contains the majority of its
    samples. This way each event is counted exactly once even when it straddles
    a stage boundary.

    Returns a dict mapping stage label (str) -> spindle count (int).
    Stage labels are taken directly from the hypnogram values, converted to str.
    """
    from scipy.ndimage import label

    labeled, n_events = label(spindle_mask > 0)
    if n_events == 0:
        return {}

    stage_codes = create_stage_codes(
        hypnogram, len(spindle_mask), fs, hypnogram_resolution_sec
    )

    counts: dict = {}
    for event_id in range(1, n_events + 1):
        event_indices = np.where(labeled == event_id)[0]
        if len(event_indices) == 0:
            continue
        # Majority stage of the event's samples.
        event_stages = stage_codes[event_indices]
        unique, freq = np.unique(event_stages, return_counts=True)
        majority_stage = str(unique[freq.argmax()])
        counts[majority_stage] = counts.get(majority_stage, 0) + 1

    return counts


def format_stage_counts(counts: dict, total: int) -> str:
    """Format a per-stage spindle count dict as a compact human-readable string.

    Example: 'N2=104 (77.6%), N3=28 (20.9%), N1=2 (1.5%)' — sorted by count
    descending. Stage labels are passed through unchanged (caller decides on
    naming; '2' stays '2', 'W' stays 'W').
    """
    if total == 0 or not counts:
        return "(no spindles)"
    parts = []
    for stage, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total
        parts.append(f"{stage}={n} ({pct:.1f}%)")
    return ", ".join(parts)
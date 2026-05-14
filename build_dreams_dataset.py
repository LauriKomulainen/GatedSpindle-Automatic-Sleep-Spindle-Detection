# build_dreams_dataset.py

import json
import logging
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
import paths
from utils.logger import setup_logging
from utils.build_utils import (
    prepare_directories, create_spindle_mask, create_stage_mask, create_stage_codes,
    count_spindles_per_stage, format_stage_counts,
)
from signal_processing import bandpassfilter, normalization
from configs.dreams_config import DATA_PARAMS
from data_loaders import dreams_loader

setup_logging("build_dreams_dataset.log")
log = logging.getLogger(__name__)
log.info("Dataset: DREAMS")

# Extract parameters from config
LOWCUT = DATA_PARAMS["lowcut"]
HIGHCUT = DATA_PARAMS["highcut"]
FILTER_ORDER = DATA_PARAMS["filter_order"]
WINDOW_SEC = DATA_PARAMS["window_sec"]
OVERLAP_SEC = DATA_PARAMS["overlap_sec"]
USE_INSTANCE_NORM = DATA_PARAMS["use_instance_norm"]
INCLUDED_STAGES = DATA_PARAMS["included_stages"]
HYPNOGRAM_RESOLUTION_SEC = DATA_PARAMS["hypnogram_resolution_sec"]


def get_scorer_annotations(patient_file_group: dict, sfreq: float) -> tuple:
    """Extract sleep spindle annotations from expert scorers (used for subject_stats.json)."""
    scorer1_events, scorer2_events = [], []

    for ann_file in patient_file_group.get("annotation_files", []):
        mne_annotations = dreams_loader._load_dreams_annotations_txt(ann_file, sfreq)
        if not mne_annotations:
            continue

        events = list(zip(mne_annotations.onset, mne_annotations.duration))
        filename = str(ann_file.name).lower()
        if "scoring1" in filename:
            scorer1_events.extend(events)
        elif "scoring2" in filename:
            scorer2_events.extend(events)

    return scorer1_events, scorer2_events


def segment_data(raw, hypnogram: np.ndarray) -> tuple:
    """
    Segment continuous signal into overlapping windows with stage filtering.
    Uses midpoint rule: window is kept if its center point falls within valid sleep stages.

    Returns:
        tuple: (x_windows, y_masks, n_total_spindles, n_kept_spindles, n_pos_windows, per_stage)
    """
    fs = raw.info["sfreq"]
    signal = raw.get_data()[0]
    signal_length = len(signal)

    window_samples = int(WINDOW_SEC * fs)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * fs)

    # Create spindle mask from annotations
    spindle_mask = create_spindle_mask(raw.annotations, signal_length, fs)

    # Count total spindles (before stage filtering)
    _, n_total = label(spindle_mask > 0)

    # Calculate n_kept: spindles that fall within valid sleep stages
    if hypnogram is not None:
        stage_mask = create_stage_mask(
            hypnogram, signal_length, fs, INCLUDED_STAGES, HYPNOGRAM_RESOLUTION_SEC
        )
        filtered_spindle_mask = spindle_mask * stage_mask
        _, n_kept = label(filtered_spindle_mask > 0)

        n_lost = n_total - n_kept
        stages_str = "/".join(f"N{s}" if str(s).isdigit() else str(s) for s in INCLUDED_STAGES)
        log.info(f"  Spindle events: Raw Union: {n_total}. In {stages_str} stages: {n_kept}. Lost: {n_lost}")

        # Per-stage breakdown of all annotated spindles. Lets us quantify
        # how many spindles each candidate filter (N2 only, N2+N3, etc.)
        # would retain or drop on this dataset.
        per_stage = count_spindles_per_stage(
            spindle_mask, hypnogram, fs, HYPNOGRAM_RESOLUTION_SEC
        )
        log.info(f"  Spindles by stage: {format_stage_counts(per_stage, n_total)}")
    else:
        n_kept = n_total
        per_stage = {}

    use_hypno = hypnogram is not None
    x_windows, y_masks = [], []

    # Statistics for logging
    kept_midpoint = 0

    # Sliding window extraction with midpoint rule
    for start in range(0, signal_length - window_samples, step_samples):
        end = start + window_samples

        if use_hypno:
            # Midpoint rule: check if the center of the window is in valid stage
            midpoint_sec = (start + window_samples / 2) / fs
            mid_idx = int(midpoint_sec / HYPNOGRAM_RESOLUTION_SEC)

            if mid_idx >= len(hypnogram):
                break

            if hypnogram[mid_idx] not in INCLUDED_STAGES:
                continue

            kept_midpoint += 1

        # Extract window
        window = signal[start:end]

        # Apply per-window normalization if enabled
        if USE_INSTANCE_NORM:
            window = normalization.normalize_data(window)

        x_windows.append(window)
        y_masks.append(spindle_mask[start:end])

    # Log window segmentation statistics
    if use_hypno:
        log.info(f" Windows segmentation: Total={kept_midpoint} (midpoint rule)")

    x_windows = np.array(x_windows, dtype=np.float32)
    y_masks = np.array(y_masks, dtype=np.float32)

    # Count positive windows (windows containing ≥1 spindle sample)
    if len(y_masks) > 0:
        has_spindle_per_window = (y_masks.max(axis=1) > 0.5)
        n_pos_windows = int(has_spindle_per_window.sum())
    else:
        n_pos_windows = 0

    return x_windows, y_masks, n_total, n_kept, n_pos_windows, per_stage


def segment_data_full(raw, hypnogram: np.ndarray) -> tuple:
    """
    Segment continuous signal into overlapping windows WITHOUT stage filtering.

    Model predicts the entire recording, and
    events are filtered by the stage_mask afterwards.

    Returns:
        tuple: (x_windows_full, y_masks_full, stage_mask_full, stage_codes_full)
    """
    fs = raw.info["sfreq"]
    signal = raw.get_data()[0]
    signal_length = len(signal)

    window_samples = int(WINDOW_SEC * fs)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * fs)

    spindle_mask = create_spindle_mask(raw.annotations, signal_length, fs)

    if hypnogram is not None:
        stage_mask_full = create_stage_mask(
            hypnogram, signal_length, fs, INCLUDED_STAGES, HYPNOGRAM_RESOLUTION_SEC
        )
        stage_codes_full = create_stage_codes(
            hypnogram, signal_length, fs, HYPNOGRAM_RESOLUTION_SEC
        )
    else:
        stage_mask_full = np.ones(signal_length, dtype=np.float32)
        stage_codes_full = np.full(signal_length, '?', dtype='<U1')

    x_windows_full = []
    y_masks_full = []

    # Sliding window across the ENTIRE signal — no stage filtering
    for start in range(0, signal_length - window_samples, step_samples):
        end = start + window_samples
        window = signal[start:end]
        if USE_INSTANCE_NORM:
            window = normalization.normalize_data(window)
        x_windows_full.append(window)
        y_masks_full.append(spindle_mask[start:end])

    x_windows_full = np.array(x_windows_full, dtype=np.float32)
    y_masks_full = np.array(y_masks_full, dtype=np.float32)

    log.info(
        f"  FULL segmentation (no stage filter): {len(x_windows_full)} windows, "
        f"signal_length={signal_length} samples ({signal_length/fs:.1f}s)"
    )

    return x_windows_full, y_masks_full, stage_mask_full, stage_codes_full


def _process_patient(patient_file_group: dict, processed_dir: Path) -> tuple | None:
    """Process a single patient's data."""
    patient_id = patient_file_group["id"]
    log.info(f"Processing patient: {patient_id}")

    # Load raw data
    raw = dreams_loader.load_dreams_patient_data(patient_file_group)
    if raw is None:
        log.warning(f"Failed to load data for {patient_id}")
        return None

    fs = raw.info["sfreq"]
    scorer1_events, scorer2_events = get_scorer_annotations(patient_file_group, fs)

    # Apply bandpass filter
    filtered = bandpassfilter.apply_bandpass_filter(
        raw.get_data()[0], fs, LOWCUT, HIGHCUT, FILTER_ORDER
    )

    # Apply normalization if needed
    if not USE_INSTANCE_NORM:
        filtered = normalization.normalize_data(filtered)
    raw._data[0] = filtered

    # Load hypnogram and segment
    hypnogram = dreams_loader.load_dreams_hypnogram(patient_file_group)
    if hypnogram is None:
        log.warning(f"  No hypnogram for {patient_id}, skipping stage filtering")

    x_windows, y_masks, n_total, n_kept, n_pos_windows, per_stage = segment_data(
        raw, hypnogram
    )

    if len(x_windows) == 0:
        log.warning(f"No valid windows for {patient_id}")
        return None

    log.info(f"Final shapes: X={x_windows.shape}, Y={y_masks.shape}")

    n_total_windows = len(x_windows)
    pos_ratio = n_pos_windows / n_total_windows if n_total_windows > 0 else 0.0
    log.info(
        f"  Windows: {n_pos_windows}/{n_total_windows} positive "
        f"({pos_ratio:.1%}), spindles kept: {n_kept}"
    )

    # Save processed arrays
    np.save(processed_dir / f"{patient_id}_X_1D.npy", x_windows)
    np.save(processed_dir / f"{patient_id}_Y_1D.npy", y_masks)
    log.info(f"  Saved: {patient_id}_X_1D.npy, {patient_id}_Y_1D.npy")

    stats = {
        "id": patient_id,
        "s1": len(scorer1_events),
        "s2": len(scorer2_events),
        "union": n_total,
        "kept": n_kept,
        "n_windows": len(x_windows),
        "per_stage": per_stage,
    }

    # FULL data for inference (no stage filtering at window level;
    # stage_mask is applied to events after prediction)
    x_full, y_full, stage_mask_full, stage_codes_full = segment_data_full(raw, hypnogram)

    np.save(processed_dir / f"{patient_id}_X_FULL.npy", x_full)
    np.save(processed_dir / f"{patient_id}_Y_FULL.npy", y_full)
    np.save(processed_dir / f"{patient_id}_STAGE_MASK_FULL.npy", stage_mask_full)
    np.save(processed_dir / f"{patient_id}_STAGE_CODES_FULL.npy", stage_codes_full)
    log.info(
        f"  Saved: {patient_id}_X_FULL.npy ({len(x_full)} windows), "
        f"Y_FULL.npy, STAGE_MASK_FULL.npy, STAGE_CODES_FULL.npy "
        f"({len(stage_mask_full)} samples)"
    )

    return stats, n_pos_windows


def main():
    log.info(f"Starting preprocessing for: DREAMS")
    log.info(f"Raw data: {paths.RAW_DREAMS_DATA_DIR}")
    log.info(f"Output: {paths.PROCESSED_DATA_DIR}")

    start_time = time.time()
    processed_dir, _ = prepare_directories()

    patient_list = dreams_loader.find_dreams_data_files(paths.RAW_DREAMS_DATA_DIR)
    if not patient_list:
        log.error(f"No valid data files found in {paths.RAW_DREAMS_DATA_DIR}")
        return

    log.info(f"Found {len(patient_list)} patients")

    results = [r for p in patient_list if (r := _process_patient(p, processed_dir))]
    stats = [s for s, _ in results]
    pos_windows_per_subject = [pw for _, pw in results]

    if stats:
        output = {
            "dataset": "DREAMS",
            "subjects": stats,
        }
        with open(processed_dir / "subject_stats.json", "w") as f:
            json.dump(output, f, indent=2)

        total_windows = sum(s["n_windows"] for s in stats)
        total_pos_windows = sum(pos_windows_per_subject)
        total_spindles = sum(s["kept"] for s in stats)
        pos_ratio = total_pos_windows / total_windows if total_windows > 0 else 0.0

        log.info(f"Summary: {len(stats)} patients, {total_windows} windows")
        log.info(
            f"  Total spindles in valid stages: {total_spindles} | "
            f"Positive windows: {total_pos_windows}/{total_windows} ({pos_ratio:.1%})"
        )

        # Aggregate per-stage spindle counts across all subjects.
        # This shows where annotated spindles actually live in the dataset and
        # quantifies how much would be lost if the stage filter were narrowed
        agg_per_stage: dict = {}
        for s in stats:
            for stage, n in s.get("per_stage", {}).items():
                agg_per_stage[stage] = agg_per_stage.get(stage, 0) + n
        agg_total = sum(agg_per_stage.values())
        if agg_total > 0:
            log.info(
                f"  Spindles by stage (all subjects, n={agg_total}): "
                f"{format_stage_counts(agg_per_stage, agg_total)}"
            )

    log.info(f"Complete. Time: {time.time() - start_time:.2f}s")
    log.info("To generate visualization plots, run: python -m utils.signal_visualization --dataset dreams")


if __name__ == "__main__":
    main()
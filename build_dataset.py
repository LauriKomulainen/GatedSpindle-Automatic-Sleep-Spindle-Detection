# build_dataset.py

"""
Dataset Builder for Sleep Spindle Detection
============================================
Preprocesses PSG data for training sleep spindle detection models.
Supports DREAMS and MASS datasets through a unified interface.

Pipeline:
1. Discover and load raw EDF files
2. Apply bandpass filtering and normalization
3. Extract sleep spindle annotations from expert scorers
4. Segment continuous signals into overlapping windows
5. Filter windows by sleep stage (N2/N3)
6. Save processed data as NumPy arrays

Usage:
    python build_dataset.py
"""

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pyedflib
from scipy.ndimage import label

import paths
from utils.logger import setup_logging
from signal_processing import bandpassfilter, normalization
from utils.signal_visualization import save_model_input_examples, plot_eeg_trace
from configs.config_loader import DATA_PARAMS, SELECTED_DATASET

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

# Extract parameters from config
LOWCUT = DATA_PARAMS["lowcut"]
HIGHCUT = DATA_PARAMS["highcut"]
FILTER_ORDER = DATA_PARAMS["filter_order"]
WINDOW_SEC = DATA_PARAMS["window_sec"]
OVERLAP_SEC = DATA_PARAMS["overlap_sec"]
USE_INSTANCE_NORM = DATA_PARAMS["use_instance_norm"]
INCLUDED_STAGES = DATA_PARAMS["included_stages"]
HYPNOGRAM_RESOLUTION_SEC = DATA_PARAMS["hypnogram_resolution_sec"]


def _load_dataset_config() -> dict:
    """Load dataset-specific loaders and configuration."""
    if SELECTED_DATASET == "DREAMS":
        from data_loaders import dreams_loader as loader
        return {
            "data_dir": paths.RAW_DREAMS_DATA_DIR,
            "loader_module": loader,
            "find_files": loader.find_dreams_data_files,
            "load_patient": loader.load_dreams_patient_data,
            "load_hypnogram": loader.load_dreams_hypnogram,
            "viz_params": {"channel_names": ["EEG"]},
        }
    elif SELECTED_DATASET == "MASS":
        from data_loaders import mass_loader as loader
        return {
            "data_dir": paths.RAW_MASS_DATA_DIR,
            "loader_module": loader,
            "find_files": loader.find_mass_data_files,
            "load_patient": loader.load_mass_patient_data,
            "load_hypnogram": loader.load_mass_hypnogram,
            "viz_params": {"channel_names": DATA_PARAMS.get("channels", ["EEG C3-CLE"])},
        }
    else:
        raise ValueError(f"Unknown dataset: {SELECTED_DATASET}")


# Initialize dataset config
CONFIG = _load_dataset_config()
log.info(f"Dataset selected: {SELECTED_DATASET}")


def get_scorer_annotations(patient_file_group: dict, sfreq: float) -> tuple:
    """Extract sleep spindle annotations from expert scorers for visualization."""
    scorer1_events, scorer2_events = [], []

    if SELECTED_DATASET == "DREAMS":
        loader = CONFIG["loader_module"]
        for ann_file in patient_file_group.get("annotation_files", []):
            mne_annotations = loader._load_dreams_annotations_txt(ann_file, sfreq)
            if not mne_annotations:
                continue

            events = list(zip(mne_annotations.onset, mne_annotations.duration))
            filename = str(ann_file.name).lower()
            if "scoring1" in filename:
                scorer1_events.extend(events)
            elif "scoring2" in filename:
                scorer2_events.extend(events)

    elif SELECTED_DATASET == "MASS":
        try:
            with pyedflib.EdfReader(str(patient_file_group["file_eeg"])) as f_eeg:
                psg_offset = f_eeg.starttime_subsecond * 1e-7

            for file_key, event_list in [("file_marks_1", scorer1_events), ("file_marks_2", scorer2_events)]:
                marks_file = patient_file_group[file_key]
                if marks_file.exists():
                    with pyedflib.EdfReader(str(marks_file)) as f:
                        time_adj = f.starttime_subsecond * 1e-7 - psg_offset
                        onsets, durations = f.readAnnotations()[:2]
                        event_list.extend(zip(onsets + time_adj, durations))
        except Exception as e:
            log.warning(f"Could not load MASS scorer events: {e}")

    return scorer1_events, scorer2_events


def _create_spindle_mask(annotations: list, signal_length: int, fs: float) -> np.ndarray:
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


def _create_stage_mask(hypnogram: np.ndarray, signal_length: int, fs: float) -> np.ndarray:
    """Create mask for valid sleep stages."""
    mask = np.zeros(signal_length, dtype=np.float32)
    samples_per_epoch = int(HYPNOGRAM_RESOLUTION_SEC * fs)

    for i, stage in enumerate(hypnogram):
        start = i * samples_per_epoch
        if start >= signal_length:
            break
        end = min(start + samples_per_epoch, signal_length)
        if stage in INCLUDED_STAGES:
            mask[start:end] = 1.0
    return mask


def segment_data(raw, hypnogram: np.ndarray, raw_unfiltered: np.ndarray = None) -> tuple:
    """
    Segment continuous signal into overlapping windows with stage filtering.

    Returns:
        tuple: (x_windows, y_masks, n_total_spindles, n_kept_spindles, raw_windows, window_times)
    """
    fs = raw.info["sfreq"]
    signal = raw.get_data()[0]
    signal_length = len(signal)

    window_samples = int(WINDOW_SEC * fs)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * fs)

    # Create masks
    spindle_mask = _create_spindle_mask(raw.annotations, signal_length, fs)
    stage_mask = _create_stage_mask(hypnogram, signal_length, fs) if hypnogram is not None else np.ones(signal_length, dtype=np.float32)

    # Count total spindles
    _, n_total = label(spindle_mask > 0)

    x_windows, y_masks, raw_windows, window_times = [], [], [], []

    # Sliding window extraction
    start = 0
    while start + window_samples <= signal_length:
        end = start + window_samples

        # Keep window if at least 50% overlaps with valid stages
        if stage_mask[start:end].mean() >= 0.5:
            window = signal[start:end]

            # Apply per-window normalization if enabled
            if USE_INSTANCE_NORM:
                window = normalization.normalize_data(window)

            x_windows.append(window)
            y_masks.append(spindle_mask[start:end])
            window_times.append(start / fs)
            if raw_unfiltered is not None:
                raw_windows.append(raw_unfiltered[start:end])

        start += step_samples

    x_windows = np.array(x_windows, dtype=np.float32)
    y_masks = np.array(y_masks, dtype=np.float32)

    # Count spindles in kept windows
    combined_mask = np.zeros(signal_length, dtype=np.float32)
    for i, wt in enumerate(window_times):
        start_sample = int(wt * fs)
        end_sample = start_sample + window_samples
        combined_mask[start_sample:end_sample] = np.maximum(
            combined_mask[start_sample:end_sample], y_masks[i]
        )
    _, n_kept = label(combined_mask > 0)

    return (
        x_windows,
        y_masks,
        n_total,
        n_kept,
        np.array(raw_windows) if raw_windows else np.array([]),
        np.array(window_times),
    )


def _prepare_directories() -> tuple:
    """Prepare output directories, cleaning if they exist."""
    dirs = [paths.PROCESSED_DATA_DIR, paths.PLOTS_DIR]
    for d in dirs:
        if d.exists():
            log.info(f"Cleaning: {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    return tuple(dirs)


def _process_patient(patient_file_group: dict, processed_dir: Path, plots_dir: Path) -> dict | None:
    """Process a single patient's data."""
    patient_id = patient_file_group["id"]
    log.info(f"Processing patient: {patient_id}")

    # Load raw data
    raw = CONFIG["load_patient"](patient_file_group)
    if raw is None:
        log.warning(f"Failed to load data for {patient_id}")
        return None

    fs = raw.info["sfreq"]
    scorer1_events, scorer2_events = get_scorer_annotations(patient_file_group, fs)
    original_signal = raw.get_data()[0].copy()

    # Apply bandpass filter
    filtered = bandpassfilter.apply_bandpass_filter(
        raw.get_data()[0], fs, LOWCUT, HIGHCUT, FILTER_ORDER
    )

    # Generate EEG trace plot
    try:
        plot_eeg_trace(filtered, fs, scorer1_events, scorer2_events, patient_id, plots_dir)
    except Exception as e:
        log.warning(f"  EEG trace plotting failed: {e}")

    # Apply normalization if needed
    if not USE_INSTANCE_NORM:
        filtered = normalization.normalize_data(filtered)
    raw._data[0] = filtered

    # Load hypnogram and segment
    hypnogram = CONFIG["load_hypnogram"](patient_file_group)
    if hypnogram is None:
        log.warning(f"  No hypnogram for {patient_id}, skipping stage filtering")

    x_windows, y_masks, n_total, n_kept, raw_windows, window_times = segment_data(
        raw, hypnogram, original_signal
    )

    if len(x_windows) == 0:
        log.warning(f"  No valid windows for {patient_id}")
        return None

    log.info(f"  Final shapes: X={x_windows.shape}, Y={y_masks.shape}")

    # Save visualization examples
    try:
        save_model_input_examples(
            x_data=x_windows,
            y_data=y_masks,
            raw_windows=raw_windows,
            subject_id=patient_id,
            save_dir=plots_dir,
            fs=fs,
            n_examples=1,
            channel_names=CONFIG["viz_params"]["channel_names"],
            scorer1_events=scorer1_events,
            scorer2_events=scorer2_events,
            window_times=window_times,
        )
    except Exception as e:
        log.warning(f"  Could not save input examples: {e}")

    # Save processed arrays
    np.save(processed_dir / f"{patient_id}_X_1D.npy", x_windows)
    np.save(processed_dir / f"{patient_id}_Y_1D.npy", y_masks)
    log.info(f"  Saved: {patient_id}_X_1D.npy, {patient_id}_Y_1D.npy")

    return {
        "id": patient_id,
        "s1": len(scorer1_events),
        "s2": len(scorer2_events),
        "union": n_total,
        "kept": n_kept,
        "n_windows": len(x_windows),
    }


def main():
    log.info(f"Starting preprocessing for: {SELECTED_DATASET}")
    log.info(f"Raw data: {CONFIG['data_dir']}")
    log.info(f"Output: {paths.PROCESSED_DATA_DIR}")

    start_time = time.time()
    processed_dir, plots_dir = _prepare_directories()

    patient_list = CONFIG["find_files"](CONFIG["data_dir"])
    if not patient_list:
        log.error(f"No valid data files found in {CONFIG['data_dir']}")
        return

    log.info(f"Found {len(patient_list)} patients")

    stats = [s for p in patient_list if (s := _process_patient(p, processed_dir, plots_dir))]

    if stats:
        with open(processed_dir / "subject_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        total_windows = sum(s["n_windows"] for s in stats)
        total_spindles = sum(s["kept"] for s in stats)
        log.info(f"Summary: {len(stats)} patients, {total_windows} windows, {total_spindles} spindles")

    log.info(f"Complete. Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
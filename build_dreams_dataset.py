# build_dreams_dataset.py

import json
import logging
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
import paths
from utils.logger import setup_logging
from utils.build_utils import prepare_directories, create_spindle_mask, create_stage_mask
from signal_processing import bandpassfilter, normalization
from utils.signal_visualization import save_model_input_examples, plot_eeg_trace
from configs.dreams_config import DATA_PARAMS
from configs.dreams_model_config import SIGNAL_VISUALIZATION_PARAMS
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
    """Extract sleep spindle annotations from expert scorers for visualization."""
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


def segment_data(raw, hypnogram: np.ndarray, raw_unfiltered: np.ndarray = None) -> tuple:
    """
    Segment continuous signal into overlapping windows with stage filtering.
    Uses midpoint rule: window is kept if its center point falls within valid sleep stages.

    Returns:
        tuple: (x_windows, y_masks, n_total_spindles, n_kept_spindles, raw_windows, window_times)
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
    else:
        n_kept = n_total

    use_hypno = hypnogram is not None
    x_windows, y_masks, raw_windows, window_times = [], [], [], []

    # Statistics for logging
    kept_midpoint = 0
    kept_strict = 0
    discarded_mixed = 0

    # Sliding window extraction with midpoint rule
    for start in range(0, signal_length - window_samples, step_samples):
        end = start + window_samples

        if use_hypno:
            # Midpoint rule: check if the center of the window is in valid stage
            midpoint_sec = (start + window_samples / 2) / fs
            mid_idx = int(midpoint_sec / HYPNOGRAM_RESOLUTION_SEC)

            if mid_idx >= len(hypnogram):
                break

            is_valid_midpoint = hypnogram[mid_idx] in INCLUDED_STAGES

            # Check strict validity for stats (entire window in valid stages)
            start_sec = start / fs
            end_sec = (end - 1) / fs
            s_idx = int(start_sec / HYPNOGRAM_RESOLUTION_SEC)
            e_idx = int(end_sec / HYPNOGRAM_RESOLUTION_SEC)
            stages_in_window = hypnogram[s_idx: e_idx + 1]
            is_valid_strict = all(s in INCLUDED_STAGES for s in stages_in_window)

            if is_valid_midpoint:
                kept_midpoint += 1
                if not is_valid_strict:
                    discarded_mixed += 1

            if not is_valid_midpoint:
                continue

            if is_valid_strict:
                kept_strict += 1

        # Extract window
        window = signal[start:end]

        # Apply per-window normalization if enabled
        if USE_INSTANCE_NORM:
            window = normalization.normalize_data(window)

        x_windows.append(window)
        y_masks.append(spindle_mask[start:end])
        window_times.append(start / fs)

        if raw_unfiltered is not None:
            raw_windows.append(raw_unfiltered[start:end])

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

    return (
        x_windows,
        y_masks,
        n_total,
        n_kept,
        n_pos_windows,
        np.array(raw_windows) if raw_windows else np.array([]),
        np.array(window_times),
    )


def _process_patient(patient_file_group: dict, processed_dir: Path, plots_dir: Path) -> dict | None:
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
    hypnogram = dreams_loader.load_dreams_hypnogram(patient_file_group)
    if hypnogram is None:
        log.warning(f"  No hypnogram for {patient_id}, skipping stage filtering")

    x_windows, y_masks, n_total, n_kept, n_pos_windows, raw_windows, window_times = segment_data(
        raw, hypnogram,
        original_signal
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

    n_viz_examples = SIGNAL_VISUALIZATION_PARAMS.get('input_examples', None)

    # Save visualization examples
    if n_viz_examples is not None and n_viz_examples > 0:
        try:
            save_model_input_examples(
                x_data=x_windows,
                y_data=y_masks,
                raw_windows=raw_windows,
                subject_id=patient_id,
                save_dir=plots_dir,
                fs=fs,
                n_examples=n_viz_examples,
                channel_names=["EEG"],
            )
        except Exception as e:
            log.warning(f"Could not save input examples: {e}")


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
    }
    return stats, n_pos_windows


def main():
    log.info(f"Starting preprocessing for: DREAMS")
    log.info(f"Raw data: {paths.RAW_DREAMS_DATA_DIR}")
    log.info(f"Output: {paths.PROCESSED_DATA_DIR}")

    start_time = time.time()
    processed_dir, plots_dir = prepare_directories()

    patient_list = dreams_loader.find_dreams_data_files(paths.RAW_DREAMS_DATA_DIR)
    if not patient_list:
        log.error(f"No valid data files found in {paths.RAW_DREAMS_DATA_DIR}")
        return

    log.info(f"Found {len(patient_list)} patients")

    results = [r for p in patient_list if (r := _process_patient(p, processed_dir, plots_dir))]
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

    log.info(f"Complete. Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
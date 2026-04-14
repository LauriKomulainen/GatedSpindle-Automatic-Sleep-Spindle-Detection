# build_mass_data.py

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
from configs.mass_config import DATA_PARAMS
from configs.model_config import SIGNAL_VISUALIZATION_PARAMS
from data_loaders import mass_loader

setup_logging("build_mass_dataset.log")
log = logging.getLogger(__name__)
log.info("Dataset: MASS")

# Extract parameters from config
LOWCUT = DATA_PARAMS["lowcut"]
HIGHCUT = DATA_PARAMS["highcut"]
FILTER_ORDER = DATA_PARAMS["filter_order"]
WINDOW_SEC = DATA_PARAMS["window_sec"]
OVERLAP_SEC = DATA_PARAMS["overlap_sec"]
USE_INSTANCE_NORM = DATA_PARAMS["use_instance_norm"]
INCLUDED_STAGES = DATA_PARAMS["included_stages"]
HYPNOGRAM_RESOLUTION_SEC = DATA_PARAMS["hypnogram_resolution_sec"]

# All scorer modes to generate masks for
SCORER_MODES = ['E1', 'E2', 'UNION']


def get_scorer_annotations(patient_file_group: dict, sfreq: float) -> tuple:
    """Extract sleep spindle annotations from expert scorers for visualization."""
    scorer1_events, scorer2_events = [], []

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
    Uses midpoint rule: window is kept if its center point falls within valid sleep stages.

    Returns masks for all available scorer modes (E1, E2, UNION).

    Returns:
        tuple: (x_windows, y_masks_dict, n_spindle_counts, raw_windows, window_times)
            - y_masks_dict: dict mapping scorer_mode -> np.ndarray of masks
            - n_spindle_counts: dict mapping scorer_mode -> (n_total, n_kept)
    """
    fs = raw.info["sfreq"]
    signal = raw.get_data()[0]
    signal_length = len(signal)

    window_samples = int(WINDOW_SEC * fs)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * fs)

    # Determine which scorer modes to process
    scorer_modes = {}
    for mode, annots in raw.annotations_by_scorer.items():
        if len(annots) > 0 or mode == 'E1':  # Always include E1
            scorer_modes[mode] = annots

    # Create spindle masks for each scorer mode
    spindle_masks = {}
    n_spindle_counts = {}

    for mode, annots in scorer_modes.items():
        mask = _create_spindle_mask(annots, signal_length, fs)
        spindle_masks[mode] = mask

        _, n_total = label(mask > 0)

        if hypnogram is not None:
            stage_mask = _create_stage_mask(hypnogram, signal_length, fs)
            filtered_mask = mask * stage_mask
            _, n_kept = label(filtered_mask > 0)
        else:
            n_kept = n_total

        n_spindle_counts[mode] = (n_total, n_kept)

    # Log spindle counts
    stages_str = "/".join(f"N{s}" if str(s).isdigit() else str(s) for s in INCLUDED_STAGES)
    for mode, (n_total, n_kept) in n_spindle_counts.items():
        n_lost = n_total - n_kept
        log.info(f"  [{mode}] Spindles: Total={n_total}, In {stages_str}={n_kept}, Lost={n_lost}")

    use_hypno = hypnogram is not None
    x_windows = []
    y_masks_per_mode = {mode: [] for mode in spindle_masks}
    raw_windows, window_times = [], []

    # Sliding window extraction with midpoint rule
    for start in range(0, signal_length - window_samples, step_samples):
        end = start + window_samples

        if use_hypno:
            midpoint_sec = (start + window_samples / 2) / fs
            mid_idx = int(midpoint_sec / HYPNOGRAM_RESOLUTION_SEC)

            if mid_idx >= len(hypnogram):
                break

            if hypnogram[mid_idx] not in INCLUDED_STAGES:
                continue

        # Extract window
        window = signal[start:end]

        if USE_INSTANCE_NORM:
            window = normalization.normalize_data(window)

        x_windows.append(window)
        window_times.append(start / fs)

        # Extract spindle mask windows for each scorer mode
        for mode, mask in spindle_masks.items():
            y_masks_per_mode[mode].append(mask[start:end])

        if raw_unfiltered is not None:
            raw_windows.append(raw_unfiltered[start:end])

    if use_hypno:
        log.info(f"  Windows segmentation: Total={len(x_windows)} (midpoint rule)")

    x_windows = np.array(x_windows, dtype=np.float32)
    y_masks_dict = {
        mode: np.array(masks, dtype=np.float32)
        for mode, masks in y_masks_per_mode.items()
    }

    return (
        x_windows,
        y_masks_dict,
        n_spindle_counts,
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
    raw = mass_loader.load_mass_patient_data(patient_file_group)
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
    hypnogram = mass_loader.load_mass_hypnogram(patient_file_group)
    if hypnogram is None:
        log.warning(f"  No hypnogram for {patient_id}, skipping stage filtering")

    x_windows, y_masks_dict, n_spindle_counts, raw_windows, window_times = segment_data(
        raw, hypnogram, original_signal
    )

    if len(x_windows) == 0:
        log.warning(f"No valid windows for {patient_id}")
        return None

    log.info(f"Final shapes: X={x_windows.shape}")

    n_viz_examples = SIGNAL_VISUALIZATION_PARAMS.get('input_examples', None)

    # Save visualization examples (using UNION masks for visualization)
    if n_viz_examples is not None and n_viz_examples > 0:
        viz_masks = y_masks_dict.get('UNION')
        if viz_masks is not None:
            try:
                save_model_input_examples(
                    x_data=x_windows,
                    y_data=viz_masks,
                    raw_windows=raw_windows,
                    subject_id=patient_id,
                    save_dir=plots_dir,
                    fs=fs,
                    n_examples=n_viz_examples,
                    channel_names=SIGNAL_VISUALIZATION_PARAMS["channel_names"],
                )
            except Exception as e:
                log.warning(f"Could not save input examples: {e}")

    # Save X data (shared across all scorer modes)
    np.save(processed_dir / f"{patient_id}_X_1D.npy", x_windows)
    log.info(f"  Saved: {patient_id}_X_1D.npy")

    # Save Y masks for each scorer mode
    stats = {
        "id": patient_id,
        "s1": len(scorer1_events),
        "s2": len(scorer2_events),
        "n_windows": len(x_windows),
        "has_e2": patient_file_group.get('has_e2', True),
        "scorers": {},
    }

    default_scorer = DATA_PARAMS.get('scorer_mode', 'UNION')

    for mode, y_masks in y_masks_dict.items():
        # Save per-scorer masks
        y_filename = f"{patient_id}_Y_{mode}.npy"
        np.save(processed_dir / y_filename, y_masks)
        log.info(f"  Saved: {y_filename}")

        n_total, n_kept = n_spindle_counts.get(mode, (0, 0))
        stats["scorers"][mode] = {"total": n_total, "kept": n_kept}

    # Create backward-compatible Y_1D.npy (points to default scorer mode)
    default_y = y_masks_dict.get(default_scorer)
    if default_y is not None:
        np.save(processed_dir / f"{patient_id}_Y_1D.npy", default_y)
        log.info(f"  Saved: {patient_id}_Y_1D.npy (default={default_scorer})")

    # Add summary counts for default mode
    default_counts = n_spindle_counts.get(default_scorer, (0, 0))
    stats["union"] = default_counts[0]
    stats["kept"] = default_counts[1]

    return stats


def main():
    log.info(f"Starting preprocessing for: MASS")
    log.info(f"Raw data: {paths.RAW_MASS_DATA_DIR}")
    log.info(f"Output: {paths.PROCESSED_DATA_DIR}")

    log.info(f"Will generate masks for ALL scorer modes: {SCORER_MODES}")
    log.info("No need to re-run when switching scorer modes.")

    start_time = time.time()
    processed_dir, plots_dir = _prepare_directories()

    patient_list = mass_loader.find_mass_data_files(paths.RAW_MASS_DATA_DIR)
    if not patient_list:
        log.error(f"No valid data files found in {paths.RAW_MASS_DATA_DIR}")
        return

    log.info(f"Found {len(patient_list)} patients")

    stats = [s for p in patient_list if (s := _process_patient(p, processed_dir, plots_dir))]

    if stats:
        with open(processed_dir / "subject_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        total_windows = sum(s["n_windows"] for s in stats)
        log.info(f"Summary: {len(stats)} patients, {total_windows} windows")

        # Log per-scorer totals
        for mode in SCORER_MODES:
            total_spindles = sum(
                s["scorers"].get(mode, {}).get("kept", 0)
                for s in stats
            )
            if total_spindles > 0:
                log.info(f"  [{mode}] Total spindles in valid stages: {total_spindles}")

    log.info(f"Complete. Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
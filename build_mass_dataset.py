# build_mass_dataset.py

import json
import logging
import time
from pathlib import Path
import numpy as np
import pyedflib
from scipy.ndimage import label
import paths
from utils.logger import setup_logging
from utils.build_utils import (
    prepare_directories, create_spindle_mask, create_stage_mask, create_stage_codes,
    count_spindles_per_stage, format_stage_counts,
)
from signal_processing import bandpassfilter, normalization
from configs.mass_config import DATA_PARAMS
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
COUNT_SPINDLE_STATS = DATA_PARAMS.get("count_spindle_stats", False)

# All scorer modes to generate masks for
SCORER_MODES = ['E1', 'E2', 'UNION']


def get_scorer_annotations(patient_file_group: dict, sfreq: float) -> tuple:
    """Extract sleep spindle annotations from expert scorers (used for subject_stats.json)."""
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


def segment_data(raw, hypnogram: np.ndarray) -> tuple:
    """
    Segment continuous signal into overlapping windows with stage filtering.
    Uses midpoint rule: window is kept if its center point falls within valid sleep stages.

    Returns masks for all available scorer modes (E1, E2, UNION).

    Returns:
        tuple: (x_windows, y_masks_dict, n_spindle_counts, pos_window_counts, per_stage_counts)
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
    per_stage_counts = {}  # mode -> {stage_code: count}

    for mode, annots in scorer_modes.items():
        # Mask itself is always needed — it's how we get the per-window Y arrays.
        mask = create_spindle_mask(annots, signal_length, fs)
        spindle_masks[mode] = mask

        if COUNT_SPINDLE_STATS:
            _, n_total = label(mask > 0)

            if hypnogram is not None:
                stage_mask = create_stage_mask(
                    hypnogram, signal_length, fs, INCLUDED_STAGES, HYPNOGRAM_RESOLUTION_SEC
                )
                filtered_mask = mask * stage_mask
                _, n_kept = label(filtered_mask > 0)

                # Per-stage breakdown for this scorer.
                per_stage_counts[mode] = count_spindles_per_stage(
                    mask, hypnogram, fs, HYPNOGRAM_RESOLUTION_SEC
                )
            else:
                n_kept = n_total
                per_stage_counts[mode] = {}

            n_spindle_counts[mode] = (n_total, n_kept)
        else:
            # Stats disabled — record None/0 placeholders. The per-window
            # positive count is still computed later from the actual Y masks.
            n_spindle_counts[mode] = (None, None)
            per_stage_counts[mode] = {}

    # Log spindle counts (only meaningful when stats were computed)
    if COUNT_SPINDLE_STATS:
        stages_str = "/".join(f"N{s}" if str(s).isdigit() else str(s) for s in INCLUDED_STAGES)
        for mode, (n_total, n_kept) in n_spindle_counts.items():
            n_lost = n_total - n_kept
            log.info(f"  [{mode}] Spindles: Total={n_total}, In {stages_str}={n_kept}, Lost={n_lost}")
            if per_stage_counts.get(mode):
                log.info(
                    f"  [{mode}] Spindles by stage: "
                    f"{format_stage_counts(per_stage_counts[mode], n_total)}"
                )
    else:
        log.info("  Per-stage spindle stats skipped (count_spindle_stats=False).")

    use_hypno = hypnogram is not None
    x_windows = []
    y_masks_per_mode = {mode: [] for mode in spindle_masks}

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

        # Extract spindle mask windows for each scorer mode
        for mode, mask in spindle_masks.items():
            y_masks_per_mode[mode].append(mask[start:end])

    if use_hypno:
        log.info(f"  Windows segmentation: Total={len(x_windows)} (midpoint rule)")

    x_windows = np.array(x_windows, dtype=np.float32)
    y_masks_dict = {
        mode: np.array(masks, dtype=np.float32)
        for mode, masks in y_masks_per_mode.items()
    }

    # Count positive windows (windows containing ≥1 spindle sample) per scorer mode
    pos_window_counts = {}
    for mode, y_arr in y_masks_dict.items():
        if len(y_arr) > 0:
            # Window is "positive" if any sample in it is marked as spindle
            has_spindle_per_window = (y_arr.max(axis=1) > 0.5)
            pos_window_counts[mode] = int(has_spindle_per_window.sum())
        else:
            pos_window_counts[mode] = 0

    return x_windows, y_masks_dict, n_spindle_counts, pos_window_counts, per_stage_counts


def segment_data_full(raw, hypnogram: np.ndarray) -> tuple:
    """
    Segment continuous signal into overlapping windows WITHOUT stage filtering.

    Model predicts the entire recording, and
    events are filtered by the stage_mask afterwards.

    Returns:
        tuple: (x_windows_full, y_masks_full_dict, stage_mask_full, stage_codes_full)
            - x_windows_full: shape (n_windows, window_samples)
            - y_masks_full_dict: dict scorer_mode -> shape (n_windows, window_samples)
            - stage_mask_full: per-sample N2 mask, shape (signal_length,)
    """
    fs = raw.info["sfreq"]
    signal = raw.get_data()[0]
    signal_length = len(signal)

    window_samples = int(WINDOW_SEC * fs)
    step_samples = int((WINDOW_SEC - OVERLAP_SEC) * fs)

    # Determine which scorer modes to process (same logic as segment_data)
    scorer_modes = {}
    for mode, annots in raw.annotations_by_scorer.items():
        if len(annots) > 0 or mode == 'E1':
            scorer_modes[mode] = annots

    spindle_masks = {
        mode: create_spindle_mask(annots, signal_length, fs)
        for mode, annots in scorer_modes.items()
    }

    # Per-sample N2 mask for the whole signal (used at inference time to filter events)
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
    y_masks_full_per_mode = {mode: [] for mode in spindle_masks}

    # Sliding window extraction across the ENTIRE signal — no stage filtering
    for start in range(0, signal_length - window_samples, step_samples):
        end = start + window_samples

        window = signal[start:end]
        if USE_INSTANCE_NORM:
            window = normalization.normalize_data(window)

        x_windows_full.append(window)
        for mode, mask in spindle_masks.items():
            y_masks_full_per_mode[mode].append(mask[start:end])

    x_windows_full = np.array(x_windows_full, dtype=np.float32)
    y_masks_full_dict = {
        mode: np.array(masks, dtype=np.float32)
        for mode, masks in y_masks_full_per_mode.items()
    }

    log.info(
        f"  FULL segmentation (no stage filter): {len(x_windows_full)} windows, "
        f"signal_length={signal_length} samples ({signal_length/fs:.1f}s)"
    )

    return x_windows_full, y_masks_full_dict, stage_mask_full, stage_codes_full


def _process_patient(patient_file_group: dict, processed_dir: Path) -> dict | None:
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

    # Apply bandpass filter
    filtered = bandpassfilter.apply_bandpass_filter(
        raw.get_data()[0], fs, LOWCUT, HIGHCUT, FILTER_ORDER
    )

    # Apply normalization if needed
    if not USE_INSTANCE_NORM:
        filtered = normalization.normalize_data(filtered)

    raw._data[0] = filtered

    # Load hypnogram and segment
    hypnogram = mass_loader.load_mass_hypnogram(patient_file_group)
    if hypnogram is None:
        log.warning(f"  No hypnogram for {patient_id}, skipping stage filtering")

    x_windows, y_masks_dict, n_spindle_counts, pos_window_counts, per_stage_counts = segment_data(
        raw, hypnogram
    )

    if len(x_windows) == 0:
        log.warning(f"No valid windows for {patient_id}")
        return None

    log.info(f"Final shapes: X={x_windows.shape}")

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

    for mode, y_masks in y_masks_dict.items():
        # Save per-scorer masks (N2-filtered, used for training)
        y_filename = f"{patient_id}_Y_{mode}.npy"
        np.save(processed_dir / y_filename, y_masks)
        log.info(f"  Saved: {y_filename}")

        n_total, n_kept = n_spindle_counts.get(mode, (None, None))
        n_pos_windows = pos_window_counts.get(mode, 0)
        n_total_windows = len(x_windows)
        pos_ratio = n_pos_windows / n_total_windows if n_total_windows > 0 else 0.0

        stats["scorers"][mode] = {
            "total": n_total,
            "kept": n_kept,
            "pos_windows": n_pos_windows,
            "total_windows": n_total_windows,
            "pos_ratio": round(pos_ratio, 3),
            "per_stage": per_stage_counts.get(mode, {}),
        }

        if n_kept is not None:
            log.info(
                f"  [{mode}] Windows: {n_pos_windows}/{n_total_windows} positive "
                f"({pos_ratio:.1%}), spindles kept: {n_kept}"
            )
        else:
            log.info(
                f"  [{mode}] Windows: {n_pos_windows}/{n_total_windows} positive "
                f"({pos_ratio:.1%})"
            )

    # FULL data for inference (no stage filtering at window level;
    # stage_mask is applied to events after prediction)
    x_full, y_full_dict, stage_mask_full, stage_codes_full = segment_data_full(raw, hypnogram)

    np.save(processed_dir / f"{patient_id}_X_FULL.npy", x_full)
    log.info(f"  Saved: {patient_id}_X_FULL.npy ({len(x_full)} windows)")

    for mode, y_full in y_full_dict.items():
        y_full_filename = f"{patient_id}_Y_FULL_{mode}.npy"
        np.save(processed_dir / y_full_filename, y_full)
        log.info(f"  Saved: {y_full_filename}")

    np.save(processed_dir / f"{patient_id}_STAGE_MASK_FULL.npy", stage_mask_full)
    np.save(processed_dir / f"{patient_id}_STAGE_CODES_FULL.npy", stage_codes_full)
    log.info(
        f"  Saved: {patient_id}_STAGE_MASK_FULL.npy + STAGE_CODES_FULL.npy "
        f"({len(stage_mask_full)} samples)"
    )

    return stats


def main():
    log.info(f"Starting preprocessing for: MASS")
    log.info(f"Raw data: {paths.RAW_MASS_DATA_DIR}")
    log.info(f"Output: {paths.PROCESSED_DATA_DIR}")

    log.info(f"Will generate masks for ALL scorer modes: {SCORER_MODES}")
    log.info("No need to re-run when switching scorer modes.")

    start_time = time.time()
    processed_dir, _ = prepare_directories()

    patient_list = mass_loader.find_mass_data_files(paths.RAW_MASS_DATA_DIR)
    if not patient_list:
        log.error(f"No valid data files found in {paths.RAW_MASS_DATA_DIR}")
        return

    log.info(f"Found {len(patient_list)} patients")

    stats = [s for p in patient_list if (s := _process_patient(p, processed_dir))]

    if stats:
        output = {
            "dataset": "MASS",
            "subjects": stats,
        }
        with open(processed_dir / "subject_stats.json", "w") as f:
            json.dump(output, f, indent=2)

        total_windows = sum(s["n_windows"] for s in stats)
        log.info(f"Summary: {len(stats)} patients, {total_windows} windows")

        # Log per-scorer totals
        for mode in SCORER_MODES:
            total_spindles = sum(
                s["scorers"].get(mode, {}).get("kept", 0)
                for s in stats
            )
            total_pos_windows = sum(
                s["scorers"].get(mode, {}).get("pos_windows", 0)
                for s in stats
            )
            total_windows_all = sum(
                s["scorers"].get(mode, {}).get("total_windows", 0)
                for s in stats
            )
            if total_spindles > 0:
                pos_ratio = total_pos_windows / total_windows_all if total_windows_all > 0 else 0.0
                log.info(
                    f"  [{mode}] Total spindles in valid stages: {total_spindles} | "
                    f"Positive windows: {total_pos_windows}/{total_windows_all} ({pos_ratio:.1%})"
                )

            # Aggregate per-stage spindle counts across all subjects for this scorer.
            # Quantifies how many spindles each candidate stage filter would
            # retain or drop on the dataset (e.g. moving from N2+N3 to N2 only).
            agg_per_stage: dict = {}
            for s in stats:
                for stage, n in s["scorers"].get(mode, {}).get("per_stage", {}).items():
                    agg_per_stage[stage] = agg_per_stage.get(stage, 0) + n
            agg_total = sum(agg_per_stage.values())
            if agg_total > 0:
                log.info(
                    f"  [{mode}] Spindles by stage (all subjects, n={agg_total}): "
                    f"{format_stage_counts(agg_per_stage, agg_total)}"
                )

    log.info(f"Complete. Time: {time.time() - start_time:.2f}s")
    log.info("To generate visualization plots, run: python -m utils.signal_visualization --dataset mass")


if __name__ == "__main__":
    main()
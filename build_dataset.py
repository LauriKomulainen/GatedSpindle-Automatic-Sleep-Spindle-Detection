# data_handler.py

import logging
import numpy as np
import time
import shutil
import json
from scipy.ndimage import label, find_objects

# CONFIGS & UTILS
from utils.logger import setup_logging
from signal_processing import bandpassfilter, normalization
from utils.signal_visualization import save_model_input_examples, plot_eeg_trace

import paths
from paths import SELECTED_DATASET, DATA_DIRECTORY

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

# DYNAMIC IMPORT & SETUP
if SELECTED_DATASET == "DREAMS":
    log.info(f"Dataset selected: {SELECTED_DATASET}")
    from configs.dreams_config import DATA_PARAMS, SIGNAL_VISUALIZATION_PARAMS
    from data_loaders import dreams_loader as loader_module

    find_dataset_files = loader_module.find_dreams_data_files
    load_patient_data = loader_module.load_dreams_patient_data
    load_hypnograms = loader_module.load_dreams_hypnogram
    DATA_DIRECTORY = paths.RAW_DREAMS_DATA_DIR

elif SELECTED_DATASET == "MASS":
    log.info(f"Dataset selected: {SELECTED_DATASET}")
    from configs.mass_config import DATA_PARAMS, SIGNAL_VISUALIZATION_PARAMS
    from data_loaders import mass_loader as loader_module

    find_dataset_files = loader_module.find_mass_data_files
    load_patient_data = loader_module.load_mass_patient_data
    load_hypnogram = loader_module.load_mass_hypnogram
    DATA_DIRECTORY = paths.RAW_MASS_DATA_DIR
else:
    raise ValueError(f"Unknown dataset in paths.py: {SELECTED_DATASET}")

# CONSTANTS
PROCESSED_DATA_DIR = paths.PROCESSED_DATA_DIR
PLOTS_DIR = paths.PLOTS_DIR
DATA_DIRECTORY = paths.RAW_DREAMS_DATA_DIR

LOWCUT = DATA_PARAMS['lowcut']
HIGHCUT = DATA_PARAMS['highcut']
FILTER_ORDER = DATA_PARAMS['filter_order']
WINDOW_SEC = DATA_PARAMS['window_sec']
OVERLAP_SEC = DATA_PARAMS['overlap_sec']
USE_INSTANCE_NORM = DATA_PARAMS['use_instance_norm']
INCLUDED_STAGES = DATA_PARAMS['included_stages']
HYPNO_RES = DATA_PARAMS['hypnogram_resolution_sec']


def get_scorer_annotations(annotation_files, sfreq):
    """
    Extracts scorer events for visualization.
    Only functional for DREAMS txt annotations. Returns empty for MASS.
    """
    if SELECTED_DATASET != "DREAMS":
        return [], []

    scorer1_evs = []
    scorer2_evs = []

    for ann_file in annotation_files:
        # Access the helper of dreams_loader
        mne_ann = loader_module._load_dreams_annotations_txt(ann_file, sfreq)
        filename = str(ann_file.name).lower()

        events = []
        if mne_ann:
            for onset, duration in zip(mne_ann.onset, mne_ann.duration):
                events.append((onset, duration))

        if "scoring1" in filename:
            scorer1_evs.extend(events)
        elif "scoring2" in filename:
            scorer2_evs.extend(events)

    return scorer1_evs, scorer2_evs


def segment_data_with_filtering(raw, hypnogram, window_sec, overlap_sec, raw_unfiltered_array=None):
    fs = raw.info['sfreq']
    signal = raw.get_data()[0]

    # 1. Spindle Event Filtering
    vote_mask = np.zeros_like(signal, dtype=np.float32)

    # Works for both DREAMS and MASS because loaders populate raw.annotations
    for annot in raw.annotations:
        if 'spindle' in annot['description'].lower():
            start_sample = int(annot['onset'] * fs)
            end_sample = int(start_sample + (annot['duration'] * fs))
            end_sample = min(end_sample, len(vote_mask))
            if start_sample < end_sample:
                vote_mask[start_sample:end_sample] = 1.0

    _, n_total = label(vote_mask)

    if hypnogram is not None:
        valid_stage_mask = np.zeros_like(vote_mask)
        samples_per_epoch = int(HYPNO_RES * fs)

        for i, stage in enumerate(hypnogram):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            if start >= len(valid_stage_mask): break
            end = min(end, len(valid_stage_mask))

            if stage in INCLUDED_STAGES:
                valid_stage_mask[start:end] = 1.0

        # Apply filtering
        filtered_mask = vote_mask * valid_stage_mask
        _, n_kept = label(filtered_mask)

        n_lost = n_total - n_kept
        log.info(f"Spindle events: Raw Union: {n_total}. Union of N2 and N3 stages: {n_kept}. Lost spindles: {n_lost}")

        # Analysis of lost events
        if n_lost > 0:
            rejected_mask = vote_mask * (1.0 - valid_stage_mask)
            labeled_rejected, _ = label(rejected_mask)
            slices = find_objects(labeled_rejected)

            reason_counts = {}
            for sl in slices:
                mid_idx = (sl[0].start + sl[0].stop) // 2
                hypno_idx = int(mid_idx / (HYPNO_RES * fs))

                if hypno_idx < len(hypnogram):
                    stage = hypnogram[hypno_idx]
                    reason_counts[stage] = reason_counts.get(stage, 0) + 1

    # 2. Window Segmentation Logic
    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples

    all_windows = []
    all_masks = []
    all_raw_windows = []

    kept_midpoint = 0
    kept_strict = 0
    discarded_mixed = 0

    use_hypno = hypnogram is not None

    for start in range(0, len(signal) - window_samples, step_samples):
        end = start + window_samples

        if use_hypno:
            midpoint_sec = (start + window_samples / 2) / fs
            mid_idx = int(midpoint_sec / HYPNO_RES)

            if mid_idx >= len(hypnogram): break

            is_valid_midpoint = hypnogram[mid_idx] in INCLUDED_STAGES

            # Check strict validity for stats
            start_sec = start / fs
            end_sec = (end - 1) / fs
            s_idx = int(start_sec / HYPNO_RES)
            e_idx = int(end_sec / HYPNO_RES)
            stages_in_window = hypnogram[s_idx: e_idx + 1]
            is_valid_strict = all(s in INCLUDED_STAGES for s in stages_in_window)

            if is_valid_midpoint:
                kept_midpoint += 1
                if not is_valid_strict:
                    discarded_mixed += 1

            if not is_valid_midpoint:
                continue

        kept_strict += 1 if (use_hypno and is_valid_strict) else 0

        sig_window = signal[start:end]
        mask_window = vote_mask[start:end]

        if raw_unfiltered_array is not None:
            raw_window_segment = raw_unfiltered_array[start:end]
            all_raw_windows.append(raw_window_segment)

        if USE_INSTANCE_NORM:
            sig_window = normalization.normalize_data(sig_window)

        all_windows.append(sig_window)
        all_masks.append(mask_window)

    if use_hypno:
        n_pure = kept_midpoint - discarded_mixed
        n_transition = discarded_mixed

        log.info(f"Windows segmentation report:")
        log.info(f"Total windows included: {kept_midpoint}. Based on midpoint rule")
        log.info(f"     |__ N2 or N3 sleep stage windows: {n_pure}")
        log.info(f"     |__ Windows mixed with other stages: {n_transition}")

    return np.array(all_windows), np.array(all_masks), n_total, n_kept, np.array(all_raw_windows)


def main():
    log.info(f"Starting data preprocessing for: {SELECTED_DATASET}")
    log.info(f"Reading raw data from: {DATA_DIRECTORY}")
    log.info(f"Saving processed data to: {PROCESSED_DATA_DIR}")

    start_time = time.time()

    # Cleanup processed data and plots folder
    if PROCESSED_DATA_DIR.exists():
        log.info(f"Cleaning previous processed data from: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    if PLOTS_DIR.exists():
        log.info(f"Cleaning previous plots from: {PLOTS_DIR}")
        shutil.rmtree(PLOTS_DIR)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Plots will be saved to: {PLOTS_DIR}")

    # 1. Find Files
    patient_list = find_dataset_files(DATA_DIRECTORY)

    if not patient_list:
        log.error(f"No valid data files found in {DATA_DIRECTORY}. Check paths.py file.")
        return

    subject_stats = []

    for patient_file_group in patient_list:
        patient_id = patient_file_group['id']
        log.info(f"Processing patient: {patient_id}")

        # 2. Load Data
        raw = load_patient_data(patient_file_group)
        if not raw:
            continue

        fs = raw.info['sfreq']

        # Scorer annotations for visualization (DREAMS only)
        s1_events, s2_events = get_scorer_annotations(patient_file_group['annotation_files'], fs)

        original_raw_signal = raw.get_data()[0].copy()

        # Filter
        signal_data = raw.get_data()[0]
        filtered_signal = bandpassfilter.apply_bandpass_filter(
            signal_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER
        )

        # Plot trace
        try:
            plot_eeg_trace(filtered_signal, fs, s1_events, s2_events, patient_id, PLOTS_DIR)
        except Exception as e:
            log.warning(f"Signal visualization plotting failed: {e}")

        if not USE_INSTANCE_NORM:
            filtered_signal = normalization.normalize_data(filtered_signal)

        raw._data[0] = filtered_signal

        # 3. Load Hypnogram
        hypnogram = None
        hypnogram = load_hypnogram(patient_file_group)

        if hypnogram is None:
            log.warning(f"Skipping filtering for {patient_id} as no hypnogram found.")

        # 4. Segmentation
        x_windows, y_masks, n_union, n_kept, x_raw_windows = segment_data_with_filtering(
            raw, hypnogram,
            window_sec=WINDOW_SEC,
            overlap_sec=OVERLAP_SEC,
            raw_unfiltered_array=original_raw_signal
        )

        subject_stats.append({
            'id': patient_id,
            's1': len(s1_events),
            's2': len(s2_events),
            'union': n_union,
            'kept': n_kept
        })

        if len(x_windows) == 0:
            log.warning(f"No windows for {patient_id}. Check hypnogram/stages.")
            continue

        log.info(f"Final 1D data shape. X: {x_windows.shape}, Y: {y_masks.shape}")

        # 5. Save Model Input Examples
        channel_names = SIGNAL_VISUALIZATION_PARAMS['channel_names']
        try:
            save_model_input_examples(
                x_data=x_windows,
                y_data=y_masks,
                raw_windows=x_raw_windows,
                subject_id=patient_id,
                save_dir=PLOTS_DIR,
                fs=fs,
                n_examples=2,
                channel_names=channel_names
            )
        except Exception as e:
            log.warning(f"Could not save input examples for {patient_id}. Error: {e}")

        # 6. Save Data
        x_path = PROCESSED_DATA_DIR / f"{patient_id}_X_1D.npy"
        y_path = PROCESSED_DATA_DIR / f"{patient_id}_Y_1D.npy"

        np.save(x_path, x_windows)
        np.save(y_path, y_masks)
        log.info(f"Processed data saved to {x_path}")

    # Save Statistics
    if subject_stats:
        stats_file = PROCESSED_DATA_DIR / "subject_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(subject_stats, f, indent=4)
        log.info(f"Subject statistics saved to {stats_file}")

    end_time = time.time()
    log.info(f"Preprocessing complete. Total time: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()
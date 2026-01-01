# data_handler.py

import logging
from pathlib import Path
import numpy as np
import time
import shutil
import json
import matplotlib.pyplot as plt
from scipy.ndimage import label
from utils.logger import setup_logging
from configs.dreams_config import DATA_PARAMS
from preprocessing import bandpassfilter, normalization
from data_loaders import dreams_loader
import paths
from utils.signal_visualization import save_three_channel_examples

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

DATA_DIRECTORY = paths.RAW_DREAMS_DATA_DIR
PROCESSED_DATA_DIR = paths.PROCESSED_DATA_DIR
CURRENT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = paths.PLOTS_DIR

LOWCUT = DATA_PARAMS['lowcut']
HIGHCUT = DATA_PARAMS['highcut']
FILTER_ORDER = DATA_PARAMS['filter_order']
WINDOW_SEC = DATA_PARAMS['window_sec']
OVERLAP_SEC = DATA_PARAMS['overlap_sec']
USE_INSTANCE_NORM = DATA_PARAMS['use_instance_norm']
INCLUDED_STAGES = DATA_PARAMS['included_stages']
HYPNO_RES = DATA_PARAMS['hypnogram_resolution_sec']

def load_hypnogram(txt_dir: Path, subject_id: str):
    hypno_file = txt_dir / f"Hypnogram_{subject_id}.txt"
    if not hypno_file.exists():
        log.warning(f"Hypnogram file not found: {hypno_file}")
        return None

    try:
        hypno_data = np.loadtxt(hypno_file, dtype=int, skiprows=1)
    except Exception as e:
        try:
            hypno_data = np.loadtxt(hypno_file, dtype=int)
        except Exception as e:
            log.error(f"Failed to read hypnogram {hypno_file}: {e}")
            return None

    if hypno_data.ndim > 1:
        hypno_data = hypno_data.flatten()

    return hypno_data


def get_scorer_annotations(annotation_files, sfreq):
    scorer1_evs = []
    scorer2_evs = []

    for ann_file in annotation_files:
        mne_ann = dreams_loader._load_dreams_annotations_txt(ann_file, sfreq)
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


def plot_eeg_trace(signal, sfreq, s1_evs, s2_evs, subject_id, save_dir):
    center_time = 10.0
    found_interesting = False

    if len(s1_evs) > 0 and len(s2_evs) > 0:
        for s1_onset, s1_dur in s1_evs:
            s1_end = s1_onset + s1_dur
            for s2_onset, s2_dur in s2_evs:
                s2_end = s2_onset + s2_dur

                if max(s1_onset, s2_onset) < min(s1_end, s2_end):
                    if abs(s1_onset - s2_onset) > 0.1 or abs(s1_end - s2_end) > 0.1:
                        center_time = s1_onset + (s1_dur / 2)
                        found_interesting = True
                        break
            if found_interesting:
                break

    if not found_interesting:
        if len(s1_evs) > 0:
            center_time = s1_evs[0][0] + (s1_evs[0][1] / 2)
        elif len(s2_evs) > 0:
            center_time = s2_evs[0][0] + (s2_evs[0][1] / 2)

    win_len = 5.0
    start_time = max(0, center_time - (win_len / 2))
    end_time = start_time + win_len

    start_idx = int(start_time * sfreq)
    end_idx = int(end_time * sfreq)

    if end_idx > len(signal):
        end_idx = len(signal)
        start_idx = end_idx - int(win_len * sfreq)

    t_axis = np.linspace(start_time, end_time, end_idx - start_idx)
    segment = signal[start_idx:end_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, segment, color='black', linewidth=0.8, label='EEG', zorder=1)

    label_added = False
    for onset, dur in s1_evs:
        if (onset + dur) > start_time and onset < end_time:
            vis_start = max(onset, start_time)
            vis_end = min(onset + dur, end_time)
            plt.hlines(y=-10, xmin=vis_start, xmax=vis_end, linewidth=4, color='#EFB7B2',
                       label='Expert 1' if not label_added else "", zorder=3)
            label_added = True

    label_added = False
    for onset, dur in s2_evs:
        if (onset + dur) > start_time and onset < end_time:
            vis_start = max(onset, start_time)
            vis_end = min(onset + dur, end_time)
            plt.hlines(y=-15, xmin=vis_start, xmax=vis_end, linewidth=4, color='#6699CC',
                       label='Expert 2' if not label_added else "", zorder=3)
            label_added = True

    plot_len = len(t_axis)
    union_mask = np.zeros(plot_len, dtype=int)

    for evs in [s1_evs, s2_evs]:
        for onset, dur in evs:
            if (onset + dur) > start_time and onset < end_time:
                s_rel = int((max(onset, start_time) - start_time) * sfreq)
                e_rel = int((min(onset + dur, end_time) - start_time) * sfreq)
                s_rel = max(0, s_rel)
                e_rel = min(plot_len, e_rel)
                union_mask[s_rel:e_rel] = 1

    from scipy.ndimage import find_objects, label as nd_label
    labeled_union, num_features = nd_label(union_mask)
    slices = find_objects(labeled_union)

    label_added = False
    if num_features > 0:
        for sl in slices:
            start_idx_u = sl[0].start
            end_idx_u = sl[0].stop

            u_start_time = t_axis[start_idx_u]
            u_end_time = t_axis[end_idx_u] if end_idx_u < len(t_axis) else t_axis[-1]

            plt.axvspan(u_start_time, u_end_time, color='#9370DB', alpha=0.2, zorder=0,
                        label='UNION (Ground Truth)' if not label_added else "")

            label_added = True

    plt.title(f"Subject {subject_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    data_min = np.min(segment)
    data_max = np.max(segment)
    plt.ylim(min(data_min, -20) - 10, max(data_max, 10) + 10)

    plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
    plt.tight_layout()

    out_file = save_dir / f"{subject_id}_trace.png"
    plt.savefig(out_file, dpi=150)
    plt.close()


def segment_data_with_filtering(raw, hypnogram, window_sec, overlap_sec, raw_unfiltered_array=None):
    fs = raw.info['sfreq']
    signal = raw.get_data()[0]

    vote_mask = np.zeros_like(signal, dtype=np.float32)
    for annot in raw.annotations:
        if 'spindle' in annot['description']:
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

        filtered_mask = vote_mask * valid_stage_mask
        _, n_kept = label(filtered_mask)

        log.info(f"SPINDLE STATS (UNION): Total spindles found in raw data: {n_total}")
        log.info(f"SPINDLE STATS: Found {n_kept}/{n_total} spindles within included stages {INCLUDED_STAGES}.")
    else:
        n_kept = n_total
        log.info(f"SPINDLE STATS (UNION): Total spindles found in raw data: {n_total}")
        log.info(f"SPINDLE STATS: No hypnogram used. Keeping all {n_total} spindles.")

    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples

    all_windows = []
    all_masks = []
    all_raw_windows = []

    discarded_count = 0
    kept_count = 0
    use_hypno = hypnogram is not None

    for start in range(0, len(signal) - window_samples, step_samples):
        end = start + window_samples

        if use_hypno:
            midpoint_sec = (start + window_samples / 2) / fs
            hypno_idx = int(midpoint_sec / HYPNO_RES)

            if hypno_idx < len(hypnogram):
                stage = hypnogram[hypno_idx]
                if stage not in INCLUDED_STAGES:
                    discarded_count += 1
                    continue
            else:
                break

        kept_count += 1
        sig_window = signal[start:end]
        mask_window = vote_mask[start:end]

        if raw_unfiltered_array is not None:
            raw_window_segment = raw_unfiltered_array[start:end]
            all_raw_windows.append(raw_window_segment)

        if USE_INSTANCE_NORM:
            sig_window = normalization.normalize_data(sig_window)

        all_windows.append(sig_window)
        all_masks.append(mask_window)

    log.info(f"Hypnogram filtering: Kept {kept_count} windows, Discarded {discarded_count} (Wrong Stages).")
    return np.array(all_windows), np.array(all_masks), n_total, n_kept, np.array(all_raw_windows)

def main():
    log.info("Starting 1D Data Preprocessing")
    log.info(f"Reading Raw Data from: {DATA_DIRECTORY}")
    log.info(f"Saving Processed Data to: {PROCESSED_DATA_DIR}")

    start_time = time.time()

    if PROCESSED_DATA_DIR.exists():
        log.warning(f"Cleaning previous data from: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    examples_dir = PLOTS_DIR / "Three_channel_examples"
    if examples_dir.exists():
        log.warning(f"Cleaning previous plots from: {examples_dir}")
        shutil.rmtree(examples_dir)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Plots will be saved to: {PLOTS_DIR}")

    patient_list = dreams_loader.find_dreams_data_files(DATA_DIRECTORY)

    if not patient_list:
        log.error(f"No valid data files found in {DATA_DIRECTORY}. Please check paths.py file.")
        return

    subject_stats = []

    for patient_file_group in patient_list:
        patient_id = patient_file_group['id']
        log.info(f"\nProcessing patient: {patient_id}")

        raw = dreams_loader.load_dreams_patient_data(patient_file_group)
        if not raw:
            continue

        fs = raw.info['sfreq']
        s1_events, s2_events = get_scorer_annotations(patient_file_group['annotation_files'], fs)

        original_raw_signal = raw.get_data()[0].copy()

        signal_data = raw.get_data()[0]
        filtered_signal = bandpassfilter.apply_bandpass_filter(
            signal_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER
        )

        plot_eeg_trace(filtered_signal, fs, s1_events, s2_events, patient_id, PLOTS_DIR)

        if not USE_INSTANCE_NORM:
            filtered_signal = normalization.normalize_data(filtered_signal)

        raw._data[0] = filtered_signal

        hypnogram = load_hypnogram(DATA_DIRECTORY, patient_id)
        if hypnogram is None:
            log.warning(f"Skipping filtering for {patient_id} (No hypnogram found).")

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

        save_three_channel_examples(x_windows, y_masks, x_raw_windows, patient_id, PLOTS_DIR, fs=fs)

        x_path = PROCESSED_DATA_DIR / f"{patient_id}_X_1D.npy"
        y_path = PROCESSED_DATA_DIR / f"{patient_id}_Y_1D.npy"

        np.save(x_path, x_windows)
        np.save(y_path, y_masks)
        log.info(f"Saved to {x_path}")

    if subject_stats:
        stats_file = PROCESSED_DATA_DIR / "subject_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(subject_stats, f, indent=4)
        log.info(f"Subject statistics saved to {stats_file}")

    end_time = time.time()
    log.info(f"\nPreprocessing complete. Total time: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()
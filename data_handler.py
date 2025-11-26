# data_handler.py

import logging
from pathlib import Path
import numpy as np
import time
import shutil
from utils.logger import setup_logging
from config import DATA_PARAMS
from data_preprocess import handler, bandpassfilter, normalization

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).parent / 'Raw DREAMS'
LOWCUT = DATA_PARAMS['lowcut']
HIGHCUT = DATA_PARAMS['highcut']
FILTER_ORDER = DATA_PARAMS['filter_order']
WINDOW_SEC = DATA_PARAMS['window_sec']
OVERLAP_SEC = DATA_PARAMS['overlap_sec']


def segment_data(raw, window_sec: float, overlap_sec: float):
    fs = raw.info['sfreq']
    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples
    signal = raw.get_data()[0]

    vote_mask = np.zeros_like(signal, dtype=np.float32)

    for annot in raw.annotations:
        if 'spindle' in annot['description']:
            start_sample = int(annot['onset'] * fs)
            end_sample = int(start_sample + (annot['duration'] * fs))
            end_sample = min(end_sample, len(vote_mask))

            if start_sample < end_sample:
                vote_mask[start_sample:end_sample] += 1.0

        final_mask = np.clip(vote_mask, 0.0, 1.0).astype(np.float32)

        log.info(f"Created HARD UNION mask. Values: {np.unique(final_mask)}")
    all_windows, all_masks = [], []
    for start in range(0, len(signal) - window_samples, step_samples):
        end = start + window_samples
        if end <= len(signal):
            all_windows.append(signal[start:end])
            all_masks.append(final_mask[start:end])

    return np.array(all_windows), np.array(all_masks)


def main():
    log.info("---Starting Optimized 1D data preprocessing ---")
    start_time = time.time()

    output_dir = Path("./diagnostics_plots/processed_data")

    if output_dir.exists():
        log.warning(f"Cleaning previous data from: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    patient_list = handler.find_dreams_data_files(DATA_DIRECTORY)
    if not patient_list:
        log.error("No data files found. Stopping.")
        return

    for patient_file_group in patient_list:
        patient_id = patient_file_group['id']
        log.info(f"\n---Processing patient: {patient_id} ---")
        raw = handler.load_dreams_patient_data(patient_file_group)
        if not raw:
            log.warning(f"Failed to load patient {patient_id}. Skipping.")
            continue

        fs = raw.info['sfreq']
        log.info(f"Data loaded. Sample rate: {fs} Hz.")

        signal_data = raw.get_data()[0]

        filtered_signal = bandpassfilter.apply_bandpass_filter(
            signal_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER
        )
        log.info(f"Signal filtered ({LOWCUT}-{HIGHCUT} Hz).")

        normalized_signal = normalization.normalize_data(filtered_signal)
        log.info("Signal normalized (Z-score).")
        raw._data[0] = normalized_signal

        x_windows, y_masks = segment_data(raw, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC)
        log.info(f"Segmented to 1D data. X: {x_windows.shape}, Y: {y_masks.shape}")

        x_path = output_dir / f"{patient_id}_X_1D.npy"
        y_path = output_dir / f"{patient_id}_Y_1D.npy"

        np.save(x_path, x_windows)
        np.save(y_path, y_masks)

        log.info(f"Saved clean 1D signals to {output_dir}")

    end_time = time.time()
    log.info(f"\n---Preprocessing complete. Total time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
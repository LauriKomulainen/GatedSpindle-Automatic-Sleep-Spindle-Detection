# data_handler.py

import logging
from pathlib import Path
import numpy as np
import time
import shutil
from utils.logger import setup_logging
from training_parameters import DATA_PARAMS
from data_preprocess import handler, bandpassfilter, normalization, cwt_transform
from utils import diagnostics

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

    mask = np.zeros_like(signal, dtype=np.int32)
    for annot in raw.annotations:
        if 'spindle' in annot['description']:
            start_sample = int(annot['onset'] * fs)
            end_sample = int(start_sample + (annot['duration'] * fs))
            if end_sample < len(mask):
                mask[start_sample:end_sample] = 1
    log.info(f"Created mask with {np.sum(mask)} spindle samples from {len(raw.annotations)} annotations.")

    all_windows, all_masks = [], []
    for start in range(0, len(signal) - window_samples, step_samples):
        end = start + window_samples
        all_windows.append(signal[start:end])
        all_masks.append(mask[start:end])

    log.info(f"Segmented signal into {len(all_windows)} windows ({window_sec}s, {overlap_sec}s overlap).")
    return np.array(all_windows), np.array(all_masks)


def main():
    log.info("---Starting DREAMS data preprocessing ---")
    start_time = time.time()

    diag_dir = Path("./diagnostics_plots")
    examples_dir = Path("./diagnostics_plots/examples")
    output_dir = Path("./diagnostics_plots/processed_data")

    if diag_dir.exists():
        log.warning(f"Cleaning previous plots from: {diag_dir}")
        shutil.rmtree(diag_dir)
    if output_dir.exists():
        log.warning(f"Cleaning previous data from: {output_dir}")
        shutil.rmtree(output_dir)
    if examples_dir.exists():
        log.warning(f"Cleaning previous data from: {examples_dir}")
        shutil.rmtree(examples_dir)

    diag_dir.mkdir(exist_ok=True)
    examples_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

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
        log.info(f"Segmented to 1D data. X shape: {x_windows.shape}, Y shape: {y_masks.shape}")

        x_images, y_images = cwt_transform.transform_windows_to_images(
            x_windows, y_masks, fs
        )
        log.info(f"Converted to 2D images. X_images shape: {x_images.shape}, Y_images shape: {y_images.shape}")

        try:
            first_spindle_idx = np.where(np.sum(y_masks, axis=1) > 0)[0][0]
            if first_spindle_idx >= 0:
                log.info(f"Found first spindle in window {first_spindle_idx}. Saving diagnostic plot.")
                plot_save_path = examples_dir / f"{patient_id}_spindle_example_idx{first_spindle_idx}.png"
                diagnostics.save_diagnostic_plot(
                    signal_1d=x_windows[first_spindle_idx],
                    mask_1d=y_masks[first_spindle_idx],
                    image_2d=x_images[first_spindle_idx],
                    mask_2d=y_images[first_spindle_idx],
                    fs=fs,
                    save_path=plot_save_path
                )
        except IndexError:
            log.warning("No spindles found for this subject. No diagnostic plot will be saved.")
        except Exception as e:
            log.error(f"Failed to save diagnostic plot: {e}")

        x_path = output_dir / f"{patient_id}_X_images.npy"
        y_path = output_dir / f"{patient_id}_Y_images.npy"
        x_1d_path = output_dir / f"{patient_id}_X_1D.npy"
        np.save(x_path, x_images)
        np.save(y_path, y_images)
        np.save(x_1d_path, x_windows)
        log.info(f"Saved final 2D images and 1D signals to {output_dir}")

    end_time = time.time()
    log.info(f"\n---Full preprocessing complete. Total time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
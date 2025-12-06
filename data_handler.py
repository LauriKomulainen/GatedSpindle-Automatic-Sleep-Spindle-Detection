# data_handler.py

import logging
from pathlib import Path
import numpy as np
import time
import shutil
from scipy.ndimage import label  # --- UUSI IMPORT TILASTOJA VARTEN ---
from utils.logger import setup_logging
from config import DATA_PARAMS, PATHS
from data_preprocess import handler, bandpassfilter, normalization

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

# --- CONFIGURATION FROM CONFIG.PY ---
DATA_DIRECTORY = Path(PATHS['raw_data_dir'])
PROCESSED_DATA_DIR = Path(PATHS['processed_data_dir'])

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
    except Exception:
        try:
            hypno_data = np.loadtxt(hypno_file, dtype=int)
        except Exception as e:
            log.error(f"Failed to read hypnogram {hypno_file}: {e}")
            return None

    if hypno_data.ndim > 1:
        hypno_data = hypno_data.flatten()

    return hypno_data


def segment_data_with_filtering(raw, hypnogram, window_sec, overlap_sec):
    fs = raw.info['sfreq']
    signal = raw.get_data()[0]

    # 1. Luo maski annotaatioista (Ground Truth)
    vote_mask = np.zeros_like(signal, dtype=np.float32)
    for annot in raw.annotations:
        if 'spindle' in annot['description']:
            start_sample = int(annot['onset'] * fs)
            end_sample = int(start_sample + (annot['duration'] * fs))
            end_sample = min(end_sample, len(vote_mask))
            if start_sample < end_sample:
                vote_mask[start_sample:end_sample] = 1.0

    # --- UUSI LOGIIKKA: Laske spindlet tilastoja varten ---

    # A. Laske kaikki spindlet (ilman stage-rajoitusta)
    _, n_total = label(vote_mask)
    n_kept = 0

    # B. Laske spindlet, jotka osuvat valittuihin unitiloihin
    if hypnogram is not None:
        # Luodaan maski, joka on 1 vain sallituilla stage-alueilla
        valid_stage_mask = np.zeros_like(vote_mask)
        samples_per_epoch = int(HYPNO_RES * fs)

        for i, stage in enumerate(hypnogram):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            # Rajatarkistus
            if start >= len(valid_stage_mask): break
            end = min(end, len(valid_stage_mask))

            if stage in INCLUDED_STAGES:
                valid_stage_mask[start:end] = 1.0

        # Leikataan alkuperäinen maski unitila-maskilla
        filtered_mask = vote_mask * valid_stage_mask
        _, n_kept = label(filtered_mask)

        log.info(f"SPINDLE STATS: Found {n_kept}/{n_total} spindles within included stages {INCLUDED_STAGES}.")
    else:
        # Jos hypnogrammia ei ole, kaikki säilyvät
        n_kept = n_total
        log.info(f"SPINDLE STATS: No hypnogram used. Found {n_total} spindles total.")

    # -----------------------------------------------------

    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples

    all_windows, all_masks = [], []
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

        if USE_INSTANCE_NORM:
            sig_window = normalization.normalize_data(sig_window)

        all_windows.append(sig_window)
        all_masks.append(mask_window)

    log.info(f"Hypnogram filtering: Kept {kept_count} windows, Discarded {discarded_count} (Wrong Stages).")
    return np.array(all_windows), np.array(all_masks)


def main():
    log.info("--- Starting 1D Data Preprocessing ---")
    log.info(f"Reading Raw Data from: {DATA_DIRECTORY}")
    log.info(f"Saving Processed Data to: {PROCESSED_DATA_DIR}")

    start_time = time.time()

    if PROCESSED_DATA_DIR.exists():
        log.warning(f"Cleaning previous data from: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Etsi tiedostot (handler.py käyttää nyt config.py:n subjects_listiä)
    patient_list = handler.find_dreams_data_files(DATA_DIRECTORY)

    if not patient_list:
        log.error(f"No valid data files found in {DATA_DIRECTORY}. Check config and paths.")
        return

    txt_directory = DATA_DIRECTORY / 'TXT'
    if not txt_directory.exists():  # Fallback jos TXT on juuressa
        txt_directory = DATA_DIRECTORY

    for patient_file_group in patient_list:
        patient_id = patient_file_group['id']
        log.info(f"\n--- Processing patient: {patient_id} ---")

        raw = handler.load_dreams_patient_data(patient_file_group)
        if not raw:
            continue

        fs = raw.info['sfreq']
        hypnogram = load_hypnogram(txt_directory, patient_id)
        if hypnogram is None:
            log.warning(f"Skipping filtering for {patient_id} (No hypnogram).")

        # Filtteröinti
        signal_data = raw.get_data()[0]
        filtered_signal = bandpassfilter.apply_bandpass_filter(
            signal_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER
        )

        # HUOM: Jos käytämme Instance Normia, emme normalisoi koko signaalia tässä,
        # vaan vasta ikkunoinnin jälkeen (kuten yllä segment_data_with_filtering tekee).
        if not USE_INSTANCE_NORM:
            filtered_signal = normalization.normalize_data(filtered_signal)

        raw._data[0] = filtered_signal

        x_windows, y_masks = segment_data_with_filtering(
            raw, hypnogram, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC
        )

        if len(x_windows) == 0:
            log.warning(f"No windows for {patient_id}. Check hypnogram/stages.")
            continue

        log.info(f"Final 1D data shape. X: {x_windows.shape}, Y: {y_masks.shape}")

        x_path = PROCESSED_DATA_DIR / f"{patient_id}_X_1D.npy"
        y_path = PROCESSED_DATA_DIR / f"{patient_id}_Y_1D.npy"

        np.save(x_path, x_windows)
        np.save(y_path, y_masks)
        log.info(f"Saved to {x_path}")

    end_time = time.time()
    log.info(f"\n--- Preprocessing complete. Total time: {end_time - start_time:.2f} s ---")


if __name__ == "__main__":
    main()
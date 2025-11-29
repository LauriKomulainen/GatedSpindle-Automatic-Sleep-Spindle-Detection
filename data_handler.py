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

# Uudet parametrit
USE_INSTANCE_NORM = DATA_PARAMS['use_instance_norm']
INCLUDED_STAGES = DATA_PARAMS['included_stages']
HYPNO_RES = DATA_PARAMS['hypnogram_resolution_sec']


def load_hypnogram(txt_dir: Path, subject_id: str):
    """
    Loads the hypnogram text file.
    Format assumes one integer per line (or row), representing 5 seconds.
    Most DREAMS hypnogram files have a header line "[Hypnogram]", so we try skipping row 1.
    """
    hypno_file = txt_dir / f"Hypnogram_{subject_id}.txt"
    if not hypno_file.exists():
        log.warning(f"Hypnogram file not found: {hypno_file}")
        return None

    try:
        # Ensisijainen yritys: Oletetaan että otsikkorivi on olemassa -> skiprows=1
        hypno_data = np.loadtxt(hypno_file, dtype=int, skiprows=1)
    except Exception:
        # Jos epäonnistuu (esim. ValueError tai ei otsikkoa), yritetään lukea alusta
        try:
            hypno_data = np.loadtxt(hypno_file, dtype=int)
        except Exception as e:
            log.error(f"Failed to read hypnogram {hypno_file}: {e}")
            return None

    # Jos tiedostossa on outoa muotoilua
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
                vote_mask[start_sample:end_sample] = 1.0  # Binary mask

    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples

    all_windows, all_masks = [], []

    discarded_count = 0
    kept_count = 0

    # Jos hypnogrammia ei ole, oletetaan että kaikki käy (fallback)
    use_hypno = hypnogram is not None

    for start in range(0, len(signal) - window_samples, step_samples):
        end = start + window_samples

        # --- A. HYPNOGRAM FILTERING ---
        if use_hypno:
            # Laske ikkunan keskikohta sekunteina
            midpoint_sec = (start + window_samples / 2) / fs

            # Hae vastaava indeksi hypnogrammista
            hypno_idx = int(midpoint_sec / HYPNO_RES)

            # Varmistetaan ettei mennä yli hypnogrammin pituuden
            if hypno_idx < len(hypnogram):
                stage = hypnogram[hypno_idx]

                # Jos vaihe ei ole sallituissa, hylkää ikkuna
                if stage not in INCLUDED_STAGES:
                    discarded_count += 1
                    continue
            else:
                # Jos signaali on pidempi kuin hypnogrammi, hylätään varuiksi tai lopetetaan
                break

        kept_count += 1

        # Ota pätkät
        sig_window = signal[start:end]
        mask_window = vote_mask[start:end]

        # --- B. INSTANCE NORMALIZATION ---
        if USE_INSTANCE_NORM:
            # Normalisoi VAIN tämä ikkuna
            sig_window = normalization.normalize_data(sig_window)

        all_windows.append(sig_window)
        all_masks.append(mask_window)

    log.info(f"Hypnogram filtering: Kept {kept_count} windows, Discarded {discarded_count} (Wrong Stages).")

    return np.array(all_windows), np.array(all_masks)


def main():
    log.info("---Starting Optimized 1D data preprocessing with Hypnogram Filter & Norm Options ---")
    log.info(f"Config: Instance Norm={USE_INSTANCE_NORM}, Stages={INCLUDED_STAGES}")

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

    txt_directory = DATA_DIRECTORY / 'TXT'

    for patient_file_group in patient_list:
        patient_id = patient_file_group['id']
        log.info(f"\n---Processing patient: {patient_id} ---")

        # 1. Lataa Raw Data
        raw = handler.load_dreams_patient_data(patient_file_group)
        if not raw:
            continue

        fs = raw.info['sfreq']

        # 2. Lataa Hypnogrammi
        hypnogram = load_hypnogram(txt_directory, patient_id)
        if hypnogram is None:
            log.warning(f"Skipping filtering for {patient_id} (No hypnogram found). Using all data.")

        # 3. Bandpass Filter (0.3 - 30Hz) - Koko signaalille
        signal_data = raw.get_data()[0]
        filtered_signal = bandpassfilter.apply_bandpass_filter(
            signal_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER
        )
        log.info(f"Signal filtered ({LOWCUT}-{HIGHCUT} Hz).")

        # 4. Normalisointi (GLOBAL vs INSTANCE)
        if not USE_INSTANCE_NORM:
            filtered_signal = normalization.normalize_data(filtered_signal)
            log.info("Applied GLOBAL normalization to full signal.")
        else:
            log.info("Skipping global normalization (using Instance Normalization later).")

        raw._data[0] = filtered_signal

        # 5. Segmentointi & Suodatus & (Instance Norm)
        x_windows, y_masks = segment_data_with_filtering(
            raw, hypnogram, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC
        )

        if len(x_windows) == 0:
            log.warning(f"No windows left for {patient_id} after filtering! Check included_stages.")
            continue

        log.info(f"Final 1D data shape. X: {x_windows.shape}, Y: {y_masks.shape}")

        x_path = output_dir / f"{patient_id}_X_1D.npy"
        y_path = output_dir / f"{patient_id}_Y_1D.npy"

        np.save(x_path, x_windows)
        np.save(y_path, y_masks)

        log.info(f"Saved processed data to {output_dir}")

    end_time = time.time()
    log.info(f"\n---Preprocessing complete. Total time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
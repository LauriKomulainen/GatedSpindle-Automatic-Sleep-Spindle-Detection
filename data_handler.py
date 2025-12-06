# data_handler.py

import logging
from pathlib import Path
import numpy as np
from utils.logger import setup_logging
from data_preprocess import handler, bandpassfilter, normalization
from config import PATHS, DATA_PARAMS

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

DATA_DIRECTORY = Path(PATHS['raw_data_dir'])
PROCESSED_DIR = Path(PATHS['processed_data_dir'])

LOWCUT = DATA_PARAMS['lowcut']
HIGHCUT = DATA_PARAMS['highcut']
FILTER_ORDER = DATA_PARAMS['filter_order']
WINDOW_SEC = DATA_PARAMS['window_sec']
OVERLAP_SEC = DATA_PARAMS['overlap_sec']
USE_INSTANCE_NORM = DATA_PARAMS['use_instance_norm']
INCLUDED_STAGES = DATA_PARAMS['included_stages']
HYPNO_RES = DATA_PARAMS['hypnogram_resolution_sec']

SUBJECTS = DATA_PARAMS['subjects_list']
MERGE_MODE = DATA_PARAMS['annotation_merge_mode']


def load_hypnogram(txt_dir: Path, subject_id: str):
    hypno_file = txt_dir / f"Hypnogram_{subject_id}.txt"
    if not hypno_file.exists():
        idx = ''.join(filter(str.isdigit, subject_id))
        hypno_file_alt = txt_dir / f"Hypnogram_{idx}.txt"
        if hypno_file_alt.exists():
            hypno_file = hypno_file_alt
        else:
            log.warning(f"Hypnogram file not found: {hypno_file}")
            return None
    try:
        return np.loadtxt(hypno_file, dtype=int, skiprows=1)
    except:
        try:
            return np.loadtxt(hypno_file, dtype=int)
        except Exception as e2:
            log.error(f"Error loading hypnogram {hypno_file}: {e2}")
            return None


def preprocess_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fs = 200.0

    log.info("=" * 60)
    log.info("DATA PROCESSING CONFIGURATION")
    log.info(f"Subjects: {SUBJECTS}")
    log.info(f"Filter: {LOWCUT}-{HIGHCUT} Hz")
    log.info(f"Merge Mode: {MERGE_MODE}")
    log.info("=" * 60)

    for subject_id in SUBJECTS:
        log.info(f"\nProcessing {subject_id}...")

        # 1. Load with Merge Mode
        try:
            raw, mask = handler.load_data(DATA_DIRECTORY, subject_id, merge_mode=MERGE_MODE)
        except Exception as e:
            log.error(f"Failed to load data for {subject_id}: {e}")
            continue

        if raw is None:
            continue

        # 2. Hypnogram
        hypno = load_hypnogram(DATA_DIRECTORY, subject_id)
        if hypno is None:
            log.warning(f"Hypnogram missing for {subject_id}. Using all stages.")
            n_epochs = int(raw.n_times / fs / HYPNO_RES) + 1
            hypno = np.full(n_epochs, INCLUDED_STAGES[0])

        # 3. Filter
        signal_data = raw.get_data()[0]
        filtered_signal = bandpassfilter.apply_bandpass_filter(
            signal_data, fs, LOWCUT, HIGHCUT, FILTER_ORDER
        )
        log.info(f"Signal filtered ({LOWCUT}-{HIGHCUT} Hz).")

        if not USE_INSTANCE_NORM:
            filtered_signal = normalization.normalize_data(filtered_signal)

        raw._data[0] = filtered_signal

        # 4. Segment (with check)
        try:
            x_windows, y_masks = handler.segment_data(
                raw, mask, hypno,
                window_sec=WINDOW_SEC,
                overlap_sec=OVERLAP_SEC,
                included_stages=INCLUDED_STAGES
            )
        except Exception as e:
            log.error(f"Segmentation failed for {subject_id}: {e}")
            continue

        if len(x_windows) == 0:
            log.warning(f"No windows produced for {subject_id}.")
            continue

        log.info(f"Result: {len(x_windows)} windows (Shape: {x_windows.shape})")

        # 5. Save
        np.save(PROCESSED_DIR / f"{subject_id}_X_1D.npy", x_windows)
        np.save(PROCESSED_DIR / f"{subject_id}_Y_1D.npy", y_masks)
        log.info(f"Saved {subject_id}")


if __name__ == "__main__":
    preprocess_pipeline()
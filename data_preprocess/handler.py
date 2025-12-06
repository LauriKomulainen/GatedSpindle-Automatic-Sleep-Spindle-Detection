# data_preprocess/handler.py

import logging
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from config import DATA_PARAMS
from scipy.ndimage import label

log = logging.getLogger(__name__)

EXCLUDED_SUBJECTS = []
TARGET_SAMPLE_RATE = DATA_PARAMS['fs']

def _count_events(mask):
    if mask is None: return 0
    _, n = label(mask)
    return n


def _load_dreams_annotations_txt(file_path: Path, sfreq: float):
    annotations = []
    try:
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['start', 'duration'], skiprows=1)
        except:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['start', 'duration'])

        for _, row in df.iterrows():
            try:
                onset = float(row['start'])
                duration = float(row['duration'])
                annotations.append((onset, duration, 'spindle'))
            except ValueError:
                continue
    except Exception as e:
        log.warning(f"Failed to read annotations from {file_path}: {e}")

    if annotations:
        onsets = [x[0] for x in annotations]
        durations = [x[1] for x in annotations]
        descriptions = [x[2] for x in annotations]

        return mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    else:
        return mne.Annotations([], [], [])


def create_mask_from_annotations(raw, annotations):
    mask = np.zeros(len(raw.times), dtype=np.float32)
    fs = raw.info['sfreq']

    for ann in annotations:
        start_idx = int(ann['onset'] * fs)
        end_idx = start_idx + int(ann['duration'] * fs)
        start_idx = max(0, min(start_idx, len(mask)))
        end_idx = max(0, min(end_idx, len(mask)))
        mask[start_idx:end_idx] = 1.0
    return mask


# --- PÄÄFUNKTIOT ---

def load_data(data_dir: Path, subject_id: str, merge_mode='UNION'):
    try:
        idx = int(''.join(filter(str.isdigit, subject_id)))
    except:
        log.error(f"Invalid subject ID {subject_id}")
        return None, None

    edf_name = f"excerpt{idx}.edf"
    edf_path = data_dir / edf_name

    if not edf_path.exists():
        log.error(f"EDF not found: {edf_path}")
        return None, None

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        log.error(f"EDF load error: {e}")
        return None, None

    if 'C3-A1' in raw.ch_names:
        raw.pick(['C3-A1'])
    elif 'Cz-A1' in raw.ch_names:
        raw.pick(['Cz-A1'])
    else:
        raw.pick([raw.ch_names[0]])

    if raw.info['sfreq'] != TARGET_SAMPLE_RATE:
        raw.resample(TARGET_SAMPLE_RATE)

    txt1 = data_dir / f"Visual_scoring1_excerpt{idx}.txt"
    if not txt1.exists(): txt1 = data_dir / "Visual_scoring1.txt"

    txt2 = data_dir / f"Visual_scoring2_excerpt{idx}.txt"
    if not txt2.exists(): txt2 = data_dir / "Visual_scoring2.txt"

    mask1 = None
    mask2 = None

    if txt1.exists():
        ann1 = _load_dreams_annotations_txt(txt1, raw.info['sfreq'])
        mask1 = create_mask_from_annotations(raw, ann1)

    if txt2.exists():
        ann2 = _load_dreams_annotations_txt(txt2, raw.info['sfreq'])
        mask2 = create_mask_from_annotations(raw, ann2)

    c1 = _count_events(mask1)
    c2 = _count_events(mask2)
    log.info(f"--- Annotations for {subject_id} ---")
    log.info(f"  Scorer 1: {c1}")
    log.info(f"  Scorer 2: {c2}")

    # Merge Logic
    final_mask = np.zeros(len(raw.times), dtype=np.float32)

    if mask1 is not None and mask2 is not None:
        if merge_mode == 'INTERSECTION':
            final_mask = np.minimum(mask1, mask2)
            log.info(f"  --> INTERSECTION: {_count_events(final_mask)}")
        elif merge_mode == 'UNION':
            final_mask = np.maximum(mask1, mask2)
            log.info(f"  --> UNION: {_count_events(final_mask)}")
        else:
            final_mask = mask1
            log.info("  --> Scorer 1 Only")
    elif mask1 is not None:
        final_mask = mask1
        log.info("  --> Scorer 1 (Scorer 2 missing)")
    elif mask2 is not None:
        final_mask = mask2
        log.info("  --> Scorer 2 (Scorer 1 missing)")

    return raw, final_mask


def segment_data(raw, mask, hypnogram, window_sec, overlap_sec, included_stages):
    """
    Segmentoi signaalin. SISÄLTÄÄ TÄRKEÄN PITUUSTARKISTUKSEN.
    """
    fs = raw.info['sfreq']
    signal = raw.get_data()[0]

    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)
    hypno_res_sec = 5.0

    windows_x = []
    windows_y = []

    n_windows = (len(signal) - window_samples) // step_samples + 1

    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples

        if end_idx > len(signal):
            break

        mid_sec = (start_idx + end_idx) / 2.0 / fs
        hypno_idx = int(mid_sec / hypno_res_sec)

        if hypno_idx < len(hypnogram):
            stage = hypnogram[hypno_idx]
            if stage in included_stages:
                x_seg = signal[start_idx:end_idx]
                y_seg = mask[start_idx:end_idx]

                # VARMISTETAAN PITUUS
                if len(x_seg) == window_samples:
                    windows_x.append(x_seg)
                    windows_y.append(y_seg)

    if len(windows_x) == 0:
        return np.array([]), np.array([])

    return np.array(windows_x), np.array(windows_y)
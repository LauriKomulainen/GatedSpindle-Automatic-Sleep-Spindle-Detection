# data_preprocess/handler.py

import logging
from pathlib import Path
import mne
import numpy as np
from typing import List, Dict, Optional, Any
from config import DATA_PARAMS

log = logging.getLogger(__name__)

EXCLUDED_SUBJECTS = ['excerpt7', 'excerpt8']
TARGET_SAMPLE_RATE = DATA_PARAMS['fs']


def find_dreams_data_files(raw_data_path: Path) -> List[Dict[str, Any]]:
    log.info(f"Searching for DREAMS data files in: {raw_data_path}")
    edf_dir = raw_data_path / 'EDF'
    txt_dir = raw_data_path / 'TXT'

    if not edf_dir.is_dir() or not txt_dir.is_dir():
        log.error(f"Data directories not found! Ensure {edf_dir} and {txt_dir} exist.")
        return []

    subjects = []
    for edf_file in edf_dir.glob('*.edf'):
        basename = edf_file.stem
        if basename in EXCLUDED_SUBJECTS:
            log.warning(f"Skipping subject by request: {basename} (not ground truth)")
            continue

        txt_file_e1 = txt_dir / f"Visual_scoring1_{basename}.txt"
        txt_file_e2 = txt_dir / f"Visual_scoring2_{basename}.txt"
        annotation_files = []

        if txt_file_e1.exists():
            annotation_files.append(txt_file_e1)
        else:
            log.warning(f"Subject {basename} is missing Visual_scoring1 file.")
        if txt_file_e2.exists():
            annotation_files.append(txt_file_e2)
        else:
            log.warning(f"Subject {basename} is missing Visual_scoring2 file.")

        if annotation_files:
            subjects.append({
                'id': basename,
                'signal_file': edf_file,
                'annotation_files': annotation_files
            })
            log.info(f"Found subject {basename} with {len(annotation_files)} annotation file(s).")
        else:
            log.warning(f"Found {edf_file.name}, but no Visual_scoring files were found.")

    log.info(f"Found a total of {len(subjects)} processable DREAMS subjects.")
    return subjects


def _load_dreams_annotations_txt(txt_file_path: Path, sfreq: float) -> mne.Annotations:
    log.info(f"Loading annotations from: {txt_file_path.name}")
    try:
        annotations_data = np.loadtxt(txt_file_path, comments='#', skiprows=0)
    except ValueError:
        log.warning(f"Failed to read {txt_file_path.name} (maybe header?). Retrying, skipping row 1.")
        try:
            annotations_data = np.loadtxt(txt_file_path, comments='#', skiprows=1)
        except Exception as e:
            log.error(f"Failed to read text file {txt_file_path.name} permanently: {e}")
            return mne.Annotations([], [], [])
    except Exception as e:
        log.error(f"Failed to read text file {txt_file_path.name}: {e}")
        return mne.Annotations([], [], [])

    if annotations_data.ndim == 1:
        annotations_data = annotations_data.reshape(1, -1)
    if annotations_data.shape[1] < 2:
        log.error("Annotation file has incorrect format (less than 2 columns).")
        return mne.Annotations([], [], [])

    onsets, durations, descriptions = [], [], []

    for row in annotations_data:
        start_val, duration_val = row[0], row[1]

        start_sec = start_val
        duration_sec = duration_val

        if duration_sec <= 0:
            log.warning(f"Invalid (zero/negative) duration {duration_sec}s on row {row}. Skipping.")
            continue
        onsets.append(start_sec)
        durations.append(duration_sec)
        descriptions.append('spindle')

    log.info(f"Found and converted {len(onsets)} annotations.")
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def load_dreams_patient_data(patient_file_group: Dict[str, Any], eeg_channel: str = 'C3-A1') -> Optional[mne.io.Raw]:
    patient_id = patient_file_group['id']
    log.info(f"Loading data for patient: {patient_id}...")
    try:
        raw_signal = mne.io.read_raw_edf(patient_file_group['signal_file'], preload=True, verbose='WARNING')
        if len(raw_signal.ch_names) == 0:
            log.error(f"Patient {patient_id} signal file is empty. Skipping.")
            return None

        original_sfreq = raw_signal.info['sfreq']
        if original_sfreq != TARGET_SAMPLE_RATE:
            log.warning(
                f"Patient {patient_id} sample rate is {original_sfreq} Hz. Resampling -> {TARGET_SAMPLE_RATE} Hz.")
            raw_signal.resample(TARGET_SAMPLE_RATE, npad="auto")

        target_channel = eeg_channel
        if target_channel not in raw_signal.ch_names:
            log.warning(f"Channel '{target_channel}' (default) not found in patient {patient_id}.")
            log.info(f"AVAILABLE CHANNELS: {raw_signal.ch_names}")
            fallback_channels = [ch for ch in raw_signal.ch_names if 'C3' in ch.upper()]
            if not fallback_channels:
                log.warning("No C3 channel found. Searching for CZ channel...")
                fallback_channels = [ch for ch in raw_signal.ch_names if 'CZ' in ch.upper()]
            if fallback_channels:
                target_channel = fallback_channels[0]
                log.info(f"Using first available fallback channel: {target_channel}")
            else:
                log.error(f"No C3 or CZ channels found. Skipping patient.")
                return None
        raw_signal.pick([target_channel])

        sfreq = raw_signal.info['sfreq']
        combined_annotations = mne.Annotations([], [], [])
        total_count = 0

        for ann_file in patient_file_group['annotation_files']:
            annotations = _load_dreams_annotations_txt(ann_file, sfreq)
            total_count += len(annotations)
            combined_annotations = combined_annotations + annotations

        raw_signal.set_annotations(combined_annotations)
        log.info(f"Loaded {len(raw_signal.times) / sfreq:.0f} seconds of signal.")

        if len(combined_annotations) > 0:
            log.info(
                f"Combined {total_count} raw annotations from {len(patient_file_group['annotation_files'])} file(s).")
        else:
            log.warning(f"No annotations were found (found {len(combined_annotations)}).")
        return raw_signal
    except Exception as e:
        log.error(f"Error loading data for patient {patient_id}. Error: {e}")
        return None
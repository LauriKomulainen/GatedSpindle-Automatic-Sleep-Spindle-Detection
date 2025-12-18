# data_preprocess/handler.py

import logging
from pathlib import Path
import mne
import numpy as np
from typing import List, Dict, Optional, Any
from configs.dreams_config import DATA_PARAMS

log = logging.getLogger(__name__)


def find_dreams_data_files(raw_data_path: Path) -> List[Dict[str, Any]]:
    subjects_to_process = DATA_PARAMS['subjects_list']
    log.info(f"Searching for data in: {raw_data_path}")
    log.info(f"Subjects to process: {subjects_to_process}")

    found_subjects = []

    for subject_id in subjects_to_process:
        edf_name = f"{subject_id}.edf"
        edf_file = raw_data_path / edf_name

        if not edf_file.exists():
            log.warning(f"EDF file not found for {subject_id}: {edf_file}")
            continue

        annotation_files = []
        idx_str = ''.join(filter(str.isdigit, subject_id))  # "excerpt1" -> "1"

        txt_file_e1 = raw_data_path / f"Visual_scoring1_excerpt{idx_str}.txt"
        txt_file_e2 = raw_data_path / f"Visual_scoring2_excerpt{idx_str}.txt"

        # --- Logic for 7 & 8:
        if subject_id in ['excerpt7', 'excerpt8']:
            log.info(f"Subject {subject_id}: Single scorer mode (loading only Scorer 1).")
            if txt_file_e1.exists():
                annotation_files.append(txt_file_e1)
            else:
                log.warning(f"Subject {subject_id}: Scorer 1 annotation file missing!")

        # --- Logic for 1-6:
        else:
            if txt_file_e1.exists():
                annotation_files.append(txt_file_e1)
            else:
                log.warning(f"Subject {subject_id}: Scorer 1 missing.")

            if txt_file_e2.exists():
                annotation_files.append(txt_file_e2)
            else:
                log.warning(
                    f"Subject {subject_id}: Scorer 2 missing (normal if single scorer, but unexpected for 1-6).")

        if annotation_files:
            found_subjects.append({
                'id': subject_id,
                'signal_file': edf_file,
                'annotation_files': annotation_files
            })
            log.info(f"Found {subject_id} with {len(annotation_files)} annotation file(s).")
        else:
            log.warning(f"Skipping {subject_id} - No annotation files found.")

    log.info(f"Found a total of {len(found_subjects)} processable DREAMS subjects.")
    return found_subjects


def _load_dreams_annotations_txt(txt_file_path: Path, sfreq: float) -> mne.Annotations:
    try:
        annotations_data = np.loadtxt(txt_file_path, comments='#', skiprows=0)
    except ValueError:
        try:
            annotations_data = np.loadtxt(txt_file_path, comments='#', skiprows=1)
        except Exception as e:
            log.error(f"Failed to read {txt_file_path.name}: {e}")
            return mne.Annotations([], [], [])
    except Exception as e:
        log.error(f"Failed to read {txt_file_path.name}: {e}")
        return mne.Annotations([], [], [])

    if annotations_data.ndim == 1:
        annotations_data = annotations_data.reshape(1, -1)

    # DREAMS format: Start(sec) Duration(sec)
    if annotations_data.shape[1] < 2:
        return mne.Annotations([], [], [])

    onsets = []
    durations = []
    descriptions = []

    for row in annotations_data:
        start_sec = row[0]
        duration_sec = row[1]

        if duration_sec > 0:
            onsets.append(start_sec)
            durations.append(duration_sec)
            descriptions.append('spindle')

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)

def load_dreams_patient_data(patient_file_group: Dict[str, Any], eeg_channel: str = 'C3-A1') -> Optional[mne.io.Raw]:
    patient_id = patient_file_group['id']
    log.info(f"Loading data for patient: {patient_id}...")
    try:
        raw_signal = mne.io.read_raw_edf(patient_file_group['signal_file'], preload=True, verbose='ERROR')

        if raw_signal.info['sfreq'] != DATA_PARAMS['fs']:
            raw_signal.resample(DATA_PARAMS['fs'], npad="auto")

        target_channel = eeg_channel
        if target_channel not in raw_signal.ch_names:
            # Fallback
            fallback_channels = [ch for ch in raw_signal.ch_names if 'C3' in ch.upper()]
            if not fallback_channels:
                fallback_channels = [ch for ch in raw_signal.ch_names if 'CZ' in ch.upper()]

            if fallback_channels:
                target_channel = fallback_channels[0]
                log.info(f"Using fallback channel: {target_channel}")
            else:
                log.error(f"No suitable EEG channel found for {patient_id}.")
                return None

        raw_signal.pick([target_channel])

        # Ground Truth = Union
        sfreq = raw_signal.info['sfreq']
        combined_annotations = mne.Annotations([], [], [])

        for ann_file in patient_file_group['annotation_files']:
            annotations = _load_dreams_annotations_txt(ann_file, sfreq)
            combined_annotations = combined_annotations + annotations

        raw_signal.set_annotations(combined_annotations)
        return raw_signal

    except Exception as e:
        log.error(f"Error loading data for {patient_id}: {e}")
        return None
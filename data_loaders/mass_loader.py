# data_loaders/mass_loader.py

import logging
from pathlib import Path
import mne
import numpy as np
from typing import List, Dict, Optional, Any
from configs.mass_config import DATA_PARAMS

log = logging.getLogger(__name__)


def find_mass_data_files(raw_data_path: Path) -> List[Dict[str, Any]]:
    subjects_to_process = DATA_PARAMS['subjects_list']
    log.info(f"Searching for MASS data in: {raw_data_path}")

    found_subjects = []

    for subject_id in subjects_to_process:
        # File naming convention: "01-02-0001 PSG.edf"
        psg_file = raw_data_path / f"{subject_id} PSG.edf"
        base_file = raw_data_path / f"{subject_id} Base.edf"

        # Annotation files
        spindles_e1 = raw_data_path / f"{subject_id} Spindles_E1.edf"
        spindles_e2 = raw_data_path / f"{subject_id} Spindles_E2.edf"

        if not psg_file.exists():
            log.warning(f"PSG file not found for {subject_id}: {psg_file}")
            continue

        annotation_files = []
        if spindles_e1.exists():
            annotation_files.append(spindles_e1)
        else:
            log.warning(f"Expert 1 annotations for spindles missing for  {subject_id}")

        if spindles_e2.exists():
            annotation_files.append(spindles_e2)
        else:
            log.warning(f"Expert 2 annotations for spindles missing for {subject_id}")

        if annotation_files and base_file.exists():
            found_subjects.append({
                'id': subject_id,
                'signal_file': psg_file,
                'hypnogram_file': base_file,  # MASS keeps hypnogram in EDF
                'annotation_files': annotation_files
            })
            log.info(f"Found PSG, Base and {len(annotation_files)} scorers for {subject_id} )")
        else:
            log.warning(f"Skipping {subject_id}. Missing Base or Annotation files.")

    return found_subjects


def _read_mass_annotations_edf(edf_path: Path) -> mne.Annotations:
    """Reads annotations from MASS EDF+ file and standardizes labels to 'spindle'."""
    try:
        # read_annotations supports EDF+
        raw_annots = mne.read_annotations(edf_path)

        new_desc = []
        for desc in raw_annots.description:
            if "spindle" in desc.lower():
                new_desc.append("spindle")
            else:
                new_desc.append(desc)  # Keep others or filter them out

        # Replace descriptions
        raw_annots.description = np.array(new_desc)
        return raw_annots

    except Exception as e:
        log.error(f"Failed to read annotations from {edf_path.name}: {e}")
        return mne.Annotations([], [], [])


def load_mass_patient_data(patient_file_group: Dict[str, Any], eeg_channel: str = 'EEG C3-CLE') -> Optional[mne.io.Raw]:
    patient_id = patient_file_group['id']
    log.info(f"Loading MASS data for: {patient_id}...")

    try:
        # 1. Load Raw Signal
        # MASS files often have many channels, select EEG C3-CLE
        raw = mne.io.read_raw_edf(patient_file_group['signal_file'], preload=True, verbose='ERROR')

        # Resample if needed (MASS is 256Hz usually)
        if raw.info['sfreq'] != DATA_PARAMS['fs']:
            log.info(f"Resampling from {raw.info['sfreq']} to {DATA_PARAMS['fs']} Hz")
            raw.resample(DATA_PARAMS['fs'], npad="auto")

        # Select Channel
        if eeg_channel in raw.ch_names:
            raw.pick([eeg_channel])
        else:
            # Fallback search
            available = [ch for ch in raw.ch_names if "C3" in ch and "EEG" in ch]
            if available:
                log.info(f"Requested {eeg_channel} not found. Using {available[0]}")
                raw.pick([available[0]])
            else:
                log.error(f"Channel {eeg_channel} not found in {raw.ch_names}")
                return None

        # 2. Load and Combine Spindle Annotations
        combined_annots = mne.Annotations([], [], [])

        for ann_file in patient_file_group['annotation_files']:
            annots = _read_mass_annotations_edf(ann_file)
            # Adjust onset if resampling caused shift? Usually MNE handles this if attached to Raw.
            # But here we attach after. MNE Annotations use time (seconds), so it is sampling-rate independent.
            combined_annots = combined_annots + annots

        # Set annotations to Raw
        raw.set_annotations(combined_annots)

        return raw

    except Exception as e:
        log.error(f"Error loading MASS data for {patient_id}: {e}")
        return None


def load_mass_hypnogram(patient_file_group: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extracts hypnogram from Base.edf.
    MASS SS2 Base.edf contains annotations 'Sleep stage ?'.
    """
    base_path = patient_file_group.get('hypnogram_file')
    if not base_path or not base_path.exists():
        return None

    try:
        annots = mne.read_annotations(base_path)

        # MNE sorts annotations by time.
        # We need to map descriptions to integers:
        # W=5, R=4, 1=1, 2=2, 3=3, 4=3 (if combining N3/N4)
        # Or standard: W=0, N1=1, N2=2, N3=3, REM=4 (Modify according to your project standard)

        # Example mapping based on typical MASS strings:
        # "Sleep stage W", "Sleep stage 1", etc.

        # This implementation depends on how you want the output array (sample-wise or epoch-wise).
        # Typically data_handler expects an epoch-wise array (e.g. one value per 30s).

        # Simplified parser:
        stages = []
        # Check the logic for your specific usage in data_handler
        # For now, let's just return the raw events or similar.

        # TODO: Implement specific mapping logic here based on your stage encoding
        # For now, returning None to indicate implementation needed based on specific mapping
        log.info(f"Hypnogram found in {base_path.name}, but parsing logic needs to be defined based on stage mapping.")
        return None

    except Exception as e:
        log.error(f"Error reading hypnogram {base_path}: {e}")
        return None
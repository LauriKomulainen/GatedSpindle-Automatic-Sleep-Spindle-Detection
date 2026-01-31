"""
MASS Dataset Loader
====================
This module handles loading and preprocessing of polysomnography (PSG) data
from the MASS (Montreal Archive of Sleep Studies) dataset.

Key functionality:
- Load EDF files containing EEG signals and sleep staging
- Align all files to common t=0 reference using subsecond timestamps
- Resample signals to target frequency
- Extract and merge sleep spindle annotations from multiple experts

Time alignment strategy:
------------------------
Each EDF file has a subsecond timestamp (starttime_subsecond * 1e-7 seconds).
We normalize everything relative to the PSG file:

    PSG signal      -> starts at t=0 (reference)
    Base.edf        -> onset += (base_subsecond - psg_subsecond)
    Spindles_E1.edf -> onset += (spindle_subsecond - psg_subsecond)
    Spindles_E2.edf -> onset += (spindle_subsecond - psg_subsecond)

This ensures all timestamps are in the same coordinate system.

Scorer modes:
-------------
    'E1'    -> Use only Expert 1 annotations (Spindles_E1.edf) - 19 patients
    'E2'    -> Use only Expert 2 annotations (Spindles_E2.edf) - 15 patients
    'UNION' -> Merge annotations from both experts (overlapping spindles merged)
"""

import os
import logging
from pathlib import Path

import numpy as np
import pyedflib
from scipy.interpolate import interp1d
from scipy.signal import resample_poly

from configs.mass_config import DATA_PARAMS
from signal_processing import bandpassfilter

log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

CHANNEL_NAME = DATA_PARAMS['channels'][0]
TARGET_FS = float(DATA_PARAMS['fs'])
PAGE_DURATION = DATA_PARAMS['page_duration']
SCORER_MODE = DATA_PARAMS.get('scorer_mode', 'UNION')  # Default to UNION

# Sleep spindle duration constraints (in seconds)
MIN_SPINDLE_DURATION = 0.3
MAX_SPINDLE_DURATION = 3.0


# =============================================================================
# DATA CONTAINER CLASS
# =============================================================================

class MassRaw:
    """
    Container for processed MASS recording data.

    Mimics MNE-Python's Raw object interface for compatibility
    with downstream processing pipelines.
    """

    def __init__(self, signal: np.ndarray, fs: float, annotations: list):
        self._data = [signal]
        self.info = {'sfreq': fs}
        self.annotations = annotations

    def get_data(self) -> np.ndarray:
        """Return signal data as 2D array (channels x samples)."""
        return np.array(self._data)


# =============================================================================
# SIGNAL PROCESSING UTILITIES
# =============================================================================

def resample_signal_linear(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """Resample signal using linear interpolation."""
    if np.isclose(fs_old, fs_new, atol=1e-5):
        return signal

    t_old = np.arange(len(signal)) / fs_old
    n_new_samples = int(t_old[-1] * fs_new) + 1
    t_new = np.arange(n_new_samples) / fs_new

    interpolator = interp1d(t_old, signal, kind='linear', fill_value="extrapolate")
    return interpolator(t_new).astype(np.float32)


# =============================================================================
# HYPNOGRAM LOADING
# =============================================================================

def _read_hypnogram_data(path_states_file: str, time_offset: float):
    """
    Read sleep staging (hypnogram) from Base.edf file.

    Args:
        path_states_file: Path to Base.edf
        time_offset: Offset to add to onsets (base_subsecond - psg_subsecond)

    Returns:
        tuple: (hypnogram_array, first_epoch_onset)
    """
    if not os.path.isfile(path_states_file):
        return None, 0

    with pyedflib.EdfReader(str(path_states_file)) as f:
        annotations = f.readAnnotations()

    onsets = np.array(annotations[0])
    durations = np.round(np.array(annotations[1]))
    stages_str = annotations[2]

    # Filter valid epochs
    valid_mask = durations == PAGE_DURATION
    onsets = onsets[valid_mask]
    stages_str = stages_str[valid_mask]

    if len(onsets) == 0:
        return None, 0

    # Extract stage labels and sort
    stages_char = np.asarray([s[-1] for s in stages_str])
    sorted_idx = np.argsort(onsets)
    onsets = onsets[sorted_idx]
    stages_char = stages_char[sorted_idx]

    # Apply time offset to align with PSG t=0
    onsets = onsets + time_offset

    # Build hypnogram array
    start_time = onsets[0]
    page_indices = np.round((onsets - start_time) / PAGE_DURATION).astype(np.int32)
    n_pages = 1 + page_indices[-1]
    hypnogram = np.full(n_pages + 1, '?', dtype='<U1')

    for idx, stage in zip(page_indices, stages_char):
        if idx < len(hypnogram):
            hypnogram[idx] = stage

    return hypnogram, start_time


# =============================================================================
# SPINDLE ANNOTATION PROCESSING
# =============================================================================

def _load_spindle_annotations(
        file_group: dict,
        psg_subsecond: float,
        crop_start_sec: float,
        n_samples_valid: int,
        scorer_mode: str = 'UNION'
) -> np.ndarray:
    """
    Load sleep spindle annotations from expert scorers.

    Args:
        file_group: Dictionary with file paths
        psg_subsecond: PSG subsecond offset (reference)
        crop_start_sec: Where the cropped signal starts (seconds from PSG t=0)
        n_samples_valid: Total valid samples after cropping
        scorer_mode: 'E1' (Expert 1), 'E2' (Expert 2), or 'UNION' (merge both)

    Returns:
        2D array of shape (n_spindles, 2) with [onset_sample, offset_sample]
    """
    # Determine which files to load based on scorer_mode
    if scorer_mode == 'E1':
        keys_to_load = ['file_marks_1']
    elif scorer_mode == 'E2':
        keys_to_load = ['file_marks_2']
    else:  # UNION
        keys_to_load = ['file_marks_1', 'file_marks_2']

    raw_marks = []

    for key in keys_to_load:
        marks_path = str(file_group[key])

        if not os.path.isfile(marks_path):
            continue

        with pyedflib.EdfReader(marks_path) as f:
            annotations = f.readAnnotations()
            onsets = np.array(annotations[0])
            durations = np.array(annotations[1])
            spindle_subsecond = f.starttime_subsecond * 1e-7

        if len(onsets) == 0:
            continue

        # Align to PSG t=0: add offset difference
        time_offset = spindle_subsecond - psg_subsecond
        onsets = onsets + time_offset

        # Shift relative to cropped signal start
        onsets = onsets - crop_start_sec

        # Convert to samples
        onset_samples = np.round(onsets * TARGET_FS).astype(int)
        offset_samples = onset_samples + np.round(durations * TARGET_FS).astype(int)

        # Filter to valid range
        marks = np.stack((onset_samples, offset_samples), axis=1)
        valid_mask = (marks[:, 0] >= 0) & (marks[:, 1] < n_samples_valid)
        raw_marks.append(marks[valid_mask])

    if len(raw_marks) == 0:
        return np.empty((0, 2), dtype=int)

    # Combine all marks
    combined = np.vstack(raw_marks)
    combined = combined[combined[:, 0].argsort()]

    # Merge overlapping spindles (for UNION mode, or in case of duplicates)
    merged = _merge_overlapping_intervals(combined)

    # Filter by duration
    durations_sec = (merged[:, 1] - merged[:, 0]) / TARGET_FS
    valid_mask = (durations_sec >= MIN_SPINDLE_DURATION) & (durations_sec <= MAX_SPINDLE_DURATION)

    return merged[valid_mask]


def _merge_overlapping_intervals(intervals: np.ndarray) -> np.ndarray:
    """Merge overlapping intervals into non-overlapping segments."""
    if len(intervals) == 0:
        return intervals

    merged = [intervals[0].copy()]

    for current in intervals[1:]:
        if current[0] < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current.copy())

    return np.array(merged)


# =============================================================================
# MAIN DATA LOADING FUNCTION
# =============================================================================

def load_mass_patient_data(file_group: dict):
    """
    Load and preprocess a complete MASS patient recording.

    Time alignment:
    - PSG signal is the reference (t=0)
    - All other files are aligned using: onset += (file_subsecond - psg_subsecond)

    Scorer mode (from config):
    - 'E1': Use only Expert 1 annotations
    - 'E2': Use only Expert 2 annotations
    - 'UNION': Merge annotations from both experts

    Args:
        file_group: Dictionary containing file paths

    Returns:
        MassRaw object or None if loading fails
    """
    eeg_path = str(file_group['file_eeg'])
    states_path = str(file_group['file_states'])
    patient_id = file_group['id']

    log.info(f"Loading patient: {patient_id} (scorer_mode: {SCORER_MODE})")

    # -------------------------------------------------------------------------
    # STEP 1: Load PSG signal (this is our t=0 reference)
    # -------------------------------------------------------------------------

    with pyedflib.EdfReader(eeg_path) as f:
        psg_subsecond = f.starttime_subsecond * 1e-7
        channel_labels = f.getSignalLabels()
        target_channel = CHANNEL_NAME if CHANNEL_NAME in channel_labels else channel_labels[0]
        channel_idx = channel_labels.index(target_channel)
        fs_header = f.samplefrequency(channel_idx)
        signal = f.readSignal(channel_idx)

    log.info(f"  PSG subsecond: {psg_subsecond:.6f}s (this is t=0 reference)")

    # -------------------------------------------------------------------------
    # STEP 2: Get Base.edf offset and calculate time adjustment
    # -------------------------------------------------------------------------

    with pyedflib.EdfReader(states_path) as f:
        base_subsecond = f.starttime_subsecond * 1e-7

    base_time_offset = base_subsecond - psg_subsecond
    log.info(f"  Base subsecond: {base_subsecond:.6f}s (offset: {base_time_offset:+.6f}s)")

    # -------------------------------------------------------------------------
    # STEP 3: Resample signal
    # -------------------------------------------------------------------------

    fs_rounded = int(np.round(fs_header))
    signal = resample_signal_linear(signal, fs_header, fs_rounded)

    if fs_rounded != TARGET_FS:
        signal = resample_poly(signal, int(TARGET_FS), fs_rounded)

    # -------------------------------------------------------------------------
    # STEP 4: Load hypnogram (with time offset applied)
    # -------------------------------------------------------------------------

    hypnogram, hypnogram_start = _read_hypnogram_data(states_path, base_time_offset)

    if hypnogram is None:
        log.warning(f"  No valid hypnogram found for {patient_id}")
        return None

    log.info(f"  Hypnogram starts at: {hypnogram_start:.2f}s (in PSG time)")

    # -------------------------------------------------------------------------
    # STEP 5: Crop signal to hypnogram region
    # -------------------------------------------------------------------------

    crop_start_sec = max(0, hypnogram_start)
    start_sample = int(crop_start_sec * TARGET_FS)
    signal = signal[start_sample:]

    # -------------------------------------------------------------------------
    # STEP 6: Apply bandpass filter
    # -------------------------------------------------------------------------

    signal = bandpassfilter.apply_bandpass_filter(
        signal, TARGET_FS,
        DATA_PARAMS['lowcut'], DATA_PARAMS['highcut'], DATA_PARAMS['filter_order']
    )

    # -------------------------------------------------------------------------
    # STEP 7: Match signal length to hypnogram
    # -------------------------------------------------------------------------

    page_samples = int(PAGE_DURATION * TARGET_FS)
    n_pages = min(len(signal) // page_samples, len(hypnogram))
    n_samples_valid = n_pages * page_samples

    signal = signal[:n_samples_valid]
    hypnogram = hypnogram[:n_pages + 1]

    # -------------------------------------------------------------------------
    # STEP 8: Load spindle annotations (with time offsets applied)
    # -------------------------------------------------------------------------

    spindle_samples = _load_spindle_annotations(
        file_group, psg_subsecond, crop_start_sec, n_samples_valid,
        scorer_mode=SCORER_MODE
    )

    annotations = [
        {
            'description': 'spindle',
            'onset': onset / TARGET_FS,
            'duration': (offset - onset) / TARGET_FS
        }
        for onset, offset in spindle_samples
    ]

    # -------------------------------------------------------------------------
    # STEP 9: Create and return data container
    # -------------------------------------------------------------------------

    raw = MassRaw(signal, TARGET_FS, annotations)
    raw.info['crop_start_sec'] = crop_start_sec
    raw.info['psg_subsecond'] = psg_subsecond
    raw.info['base_subsecond'] = base_subsecond
    raw.info['scorer_mode'] = SCORER_MODE

    log.info(f"  Loaded {len(annotations)} spindles, {n_pages} pages")

    return raw


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_mass_hypnogram(file_group: dict) -> np.ndarray:
    """Load only the hypnogram for a patient."""
    states_path = str(file_group['file_states'])

    # For standalone hypnogram loading, offset doesn't matter for sleep stages
    hypnogram, _ = _read_hypnogram_data(states_path, 0)
    return hypnogram


def find_mass_data_files(data_dir: str) -> list:
    """
    Discover all valid MASS dataset files in a directory.

    Patient inclusion depends on scorer_mode:
    - 'E1': Include patients that have Spindles_E1.edf (all 19)
    - 'E2': Include patients that have Spindles_E2.edf (15 patients)
    - 'UNION': Include patients that have at least one spindle file

    Args:
        data_dir: Path to directory containing MASS EDF files

    Returns:
        List of file group dictionaries
    """
    data_dir = Path(data_dir)
    file_groups = []

    # Check all 19 patients (01-02-0001 to 01-02-0019)
    for pid in range(1, 20):
        patient_str = f"01-02-{pid:04d}"

        psg_path = data_dir / f"{patient_str} PSG.edf"
        base_path = data_dir / f"{patient_str} Base.edf"
        e1_path = data_dir / f"{patient_str} Spindles_E1.edf"
        e2_path = data_dir / f"{patient_str} Spindles_E2.edf"

        # Must have PSG and Base files
        if not psg_path.exists() or not base_path.exists():
            continue

        # Check spindle files based on scorer_mode
        has_e1 = e1_path.exists()
        has_e2 = e2_path.exists()

        include_patient = False
        if SCORER_MODE == 'E1':
            include_patient = has_e1
        elif SCORER_MODE == 'E2':
            include_patient = has_e2
        else:  # UNION
            include_patient = has_e1 and has_e2

        if include_patient:
            file_groups.append({
                'id': patient_str,
                'file_eeg': psg_path,
                'file_states': base_path,
                'file_marks_1': e1_path,
                'file_marks_2': e2_path
            })

    log.info(f"Found {len(file_groups)} valid MASS recordings for scorer_mode='{SCORER_MODE}'")
    return file_groups
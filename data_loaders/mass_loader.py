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

Scorer modes (applied at dataset/evaluation time, NOT at loading time):
-----------------------------------------------------------------------
    'E1'    -> Use only Expert 1 annotations (Spindles_E1.edf) - 19 patients
    'E2'    -> Use only Expert 2 annotations (Spindles_E2.edf) - 15 patients
    'UNION' -> Merge annotations from both experts (overlapping spindles merged)

IMPORTANT: This loader always processes ALL 19 patients and returns annotations
for ALL available scorers. The scorer_mode filtering happens downstream in
build_dataset.py (which saves separate Y files per scorer) and dataset.py
(which selects the correct Y file at runtime).
"""

import os
import logging
from pathlib import Path
import numpy as np
import pyedflib
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from configs.mass_config import DATA_PARAMS

log = logging.getLogger(__name__)

# CONFIGURATION CONSTANTS
CHANNEL_NAME = DATA_PARAMS['channels'][0]
TARGET_FS = float(DATA_PARAMS['fs'])
PAGE_DURATION = DATA_PARAMS['page_duration']

# DATA CONTAINER CLASS
class MassRaw:
    """
    Container for processed MASS recording data.
    """
    def __init__(self, signal: np.ndarray, fs: float, annotations: list):
        self._data = [signal]
        self.info = {'sfreq': fs}
        self.annotations = annotations

        # Per-scorer annotations (populated by load_mass_patient_data)
        self.annotations_by_scorer = {}

    def get_data(self) -> np.ndarray:
        """Return signal data as 2D array (channels x samples)."""
        return np.array(self._data)


# SIGNAL PROCESSING UTILITIES
# Resample a signal from one sample rate to another using linear interpolation.
def resample_signal_linear(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    if np.isclose(fs_old, fs_new, atol=1e-5):
        return signal

    t_old = np.arange(len(signal)) / fs_old
    n_new_samples = int(t_old[-1] * fs_new) + 1
    t_new = np.arange(n_new_samples) / fs_new

    interpolator = interp1d(t_old, signal, kind='linear', fill_value="extrapolate")
    return interpolator(t_new).astype(np.float32)


# HYPNOGRAM LOADING
# Read sleep stage labels from Base.edf annotations.
# Returns (hypnogram_array, first_epoch_onset_sec) or (None, 0) if no valid epochs found.
# The time_offset shifts all onsets so they align with PSG t=0.
def _read_hypnogram_data(path_states_file: str, time_offset: float):
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

    # Build hypnogram array — one entry per page, indexed 0..n_pages-1
    start_time = onsets[0]
    page_indices = np.round((onsets - start_time) / PAGE_DURATION).astype(np.int32)
    n_pages = 1 + page_indices[-1]
    hypnogram = np.full(n_pages, '?', dtype='<U1')

    for idx, stage in zip(page_indices, stages_char):
        if idx < len(hypnogram):
            hypnogram[idx] = stage

    return hypnogram, start_time


# SPINDLE ANNOTATION PROCESSING
# Load spindle annotations from a single scorer's EDF file (E1 or E2).
# Aligns timestamps to PSG t=0, shifts them relative to the cropped signal start,
# converts to sample indices, and filters out any events outside the valid range.
# Returns shape (n_spindles, 2) with columns [onset_sample, offset_sample].
def _load_spindle_annotations_single(
        marks_path: str,
        psg_subsecond: float,
        crop_start_sec: float,
        n_samples_valid: int,
) -> np.ndarray:
    if not os.path.isfile(marks_path):
        return np.empty((0, 2), dtype=int)

    with pyedflib.EdfReader(marks_path) as f:
        annotations = f.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        spindle_subsecond = f.starttime_subsecond * 1e-7

    if len(onsets) == 0:
        return np.empty((0, 2), dtype=int)

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
    return marks[valid_mask]


# Merge spindle annotations from two experts into a single set (UNION mode).
# Overlapping or nearly-touching spindles (within 0.3s gap) are merged together.
def _merge_annotations_union(marks_e1: np.ndarray, marks_e2: np.ndarray) -> np.ndarray:
    if len(marks_e1) == 0 and len(marks_e2) == 0:
        return np.empty((0, 2), dtype=int)
    if len(marks_e1) == 0:
        return marks_e2
    if len(marks_e2) == 0:
        return marks_e1

    combined = np.vstack([marks_e1, marks_e2])
    combined = combined[combined[:, 0].argsort()]

    merge_gap_sec = 0.3
    gap_samples = int(merge_gap_sec * TARGET_FS)
    return _merge_overlapping_intervals(combined, gap_samples=gap_samples)


# Merge overlapping or nearly-touching intervals into non-overlapping segments.
# Intervals closer than gap_samples are treated as one continuous event.
def _merge_overlapping_intervals(intervals: np.ndarray, gap_samples: int = 0) -> np.ndarray:
    if len(intervals) == 0:
        return intervals

    merged = [intervals[0].copy()]

    for current in intervals[1:]:
        if current[0] <= (merged[-1][1] + gap_samples):
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current.copy())

    return np.array(merged)


# Convert sample-based [onset, offset] pairs into annotation dicts with onset/duration in seconds.
def _samples_to_annotations(spindle_samples: np.ndarray) -> list:
    return [
        {
            'description': 'spindle',
            'onset': onset / TARGET_FS,
            'duration': (offset - onset) / TARGET_FS
        }
        for onset, offset in spindle_samples
    ]


# MAIN DATA LOADING FUNCTION
# Load and preprocess a single MASS patient recording.
# The PSG file defines t=0; all other files (Base.edf, Spindles_E1/E2.edf)
# are time-aligned using their subsecond timestamps relative to PSG.
# Returns a MassRaw object with signal, hypnogram, and per-scorer annotations,
# or None if the recording can't be loaded.
def load_mass_patient_data(file_group: dict):
    eeg_path = str(file_group['file_eeg'])
    states_path = str(file_group['file_states'])
    patient_id = file_group['id']

    log.info(f"Loading patient: {patient_id}")

    # 1: Load PSG signal (this is our t=0 reference)
    with pyedflib.EdfReader(eeg_path) as f:
        psg_subsecond = f.starttime_subsecond * 1e-7
        channel_labels = f.getSignalLabels()
        target_channel = CHANNEL_NAME if CHANNEL_NAME in channel_labels else channel_labels[0]
        channel_idx = channel_labels.index(target_channel)
        fs_header = f.samplefrequency(channel_idx)
        signal = f.readSignal(channel_idx)

    log.info(f"  PSG subsecond: {psg_subsecond:.6f}s (this is t=0 reference)")

    # 2: Get Base.edf offset and calculate time adjustment
    with pyedflib.EdfReader(states_path) as f:
        base_subsecond = f.starttime_subsecond * 1e-7

    base_time_offset = base_subsecond - psg_subsecond
    log.info(f"  Base subsecond: {base_subsecond:.6f}s (offset: {base_time_offset:+.6f}s)")

    # 3: Resample signal
    fs_rounded = int(np.round(fs_header))
    signal = resample_signal_linear(signal, fs_header, fs_rounded)

    if fs_rounded != TARGET_FS:
        signal = resample_poly(signal, int(TARGET_FS), fs_rounded)

    # 4: Load hypnogram (with time offset applied)
    hypnogram, hypnogram_start = _read_hypnogram_data(states_path, base_time_offset)

    if hypnogram is None:
        log.warning(f"  No valid hypnogram found for {patient_id}")
        return None

    log.info(f"  Hypnogram starts at: {hypnogram_start:.2f}s (in PSG time)")

    # 5: Crop signal to hypnogram region
    crop_start_sec = max(0, hypnogram_start)
    start_sample = int(crop_start_sec * TARGET_FS)
    signal = signal[start_sample:]

    # 6: Match signal length to hypnogram — both must cover exactly n_pages
    page_samples = int(PAGE_DURATION * TARGET_FS)
    n_pages = min(len(signal) // page_samples, len(hypnogram))
    n_samples_valid = n_pages * page_samples

    signal = signal[:n_samples_valid]
    hypnogram = hypnogram[:n_pages]

    # 7: Load spindle annotations from ALL available scorers
    e1_path = str(file_group['file_marks_1'])
    e2_path = str(file_group['file_marks_2'])

    marks_e1 = _load_spindle_annotations_single(
        e1_path, psg_subsecond, crop_start_sec, n_samples_valid
    )
    marks_e2 = _load_spindle_annotations_single(
        e2_path, psg_subsecond, crop_start_sec, n_samples_valid
    )
    marks_union = _merge_annotations_union(marks_e1, marks_e2)

    # 8: Build annotation dicts for each scorer mode
    annotations_e1 = _samples_to_annotations(marks_e1)
    annotations_e2 = _samples_to_annotations(marks_e2)
    annotations_union = _samples_to_annotations(marks_union)

    # 9: Create and return data container
    # Default annotations = UNION for backward compatibility
    raw = MassRaw(signal, TARGET_FS, annotations_union)
    raw.info['crop_start_sec'] = crop_start_sec
    raw.info['psg_subsecond'] = psg_subsecond
    raw.info['base_subsecond'] = base_subsecond

    # Store per-scorer annotations
    raw.annotations_by_scorer = {
        'E1': annotations_e1,
        'E2': annotations_e2,
        'UNION': annotations_union,
    }

    # Store per-scorer sample arrays (for build_dataset to create masks)
    raw.info['marks_by_scorer'] = {
        'E1': marks_e1,
        'E2': marks_e2,
        'UNION': marks_union,
    }

    log.info(
        f"  Spindles loaded - E1: {len(annotations_e1)}, "
        f"E2: {len(annotations_e2)}, UNION: {len(annotations_union)}, "
        f"{n_pages} pages"
    )

    return raw


# UTILITY FUNCTIONS
# Load only the hypnogram (sleep stage labels) for a patient, without the EEG signal.
def load_mass_hypnogram(file_group: dict) -> np.ndarray:
    states_path = str(file_group['file_states'])
    hypnogram, _ = _read_hypnogram_data(states_path, 0)
    return hypnogram


# Scan the data directory and return a list of file groups for all valid MASS patients.
# A patient is valid if PSG, Base, and at least E1 annotation files exist.
# All 19 patients are included; E2 availability is tracked but never excludes a patient.
def find_mass_data_files(data_dir: str) -> list:
    data_dir = Path(data_dir)
    file_groups = []

    for pid in range(1, 20):
        patient_str = f"01-02-{pid:04d}"

        psg_path = data_dir / f"{patient_str} PSG.edf"
        base_path = data_dir / f"{patient_str} Base.edf"
        e1_path = data_dir / f"{patient_str} Spindles_E1.edf"
        e2_path = data_dir / f"{patient_str} Spindles_E2.edf"

        # Must have PSG, Base, and at least E1 file
        if not psg_path.exists() or not base_path.exists():
            continue
        if not e1_path.exists():
            log.warning(f"  Skipping {patient_str}: no Spindles_E1.edf")
            continue

        file_groups.append({
            'id': patient_str,
            'file_eeg': psg_path,
            'file_states': base_path,
            'file_marks_1': e1_path,
            'file_marks_2': e2_path,
            'has_e2': e2_path.exists(),
        })

    n_with_e2 = sum(1 for fg in file_groups if fg['has_e2'])
    log.info(f"Found {len(file_groups)} MASS recordings ({n_with_e2} with E2 annotations)")
    return file_groups
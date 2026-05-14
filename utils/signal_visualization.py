# utils/signal_visualization.py

"""
Visualization utilities for sleep spindle detection pipeline.

This module can be used in two ways:
    1) As a library: import save_model_input_examples and plot_eeg_trace.
    2) As a standalone script: run after the dataset has been built.

Standalone usage:
    python -m utils.signal_visualization --dataset dreams
    python -m utils.signal_visualization --dataset mass

The standalone mode:
    - Reads subject_stats.json from PROCESSED_DATA_DIR (raises if missing)
    - Re-loads raw data via the dataset loader for each subject listed there
    - Applies bandpass filtering, segments the signal, and generates plots
    - Saves plots under PROCESSED_DATA_DIR/plots/

Generates:
    - Model input example plots (multi-channel view with spindle annotations)
    - EEG trace plots with expert scorer annotations
"""

from pathlib import Path
import argparse
import json
import logging
import sys
from paths import PLOTS_DIR
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

from core.features import compute_input_channels

log = logging.getLogger(__name__)

# Consistent color scheme
COLORS_CHANNELS = ['#2c3e50', '#2980b9', '#8e44ad']
COLOR_RAW = '#7f8c8d'
COLOR_SPINDLE = '#e74c3c'


def _select_example_indices(has_spindle_indices: np.ndarray, n_examples: int) -> np.ndarray:
    """
    Select window indices distributed across start, middle, and end of recording.
    """
    n_available = len(has_spindle_indices)

    if n_available <= n_examples:
        return has_spindle_indices

    part = n_examples // 3
    remainder = n_examples % 3

    start_sel = has_spindle_indices[:part + remainder]

    mid_center = n_available // 2
    mid_start = max(0, mid_center - part // 2)
    mid_sel = has_spindle_indices[mid_start:mid_start + part] if part > 0 else np.array([], dtype=int)

    end_sel = has_spindle_indices[-part:] if part > 0 else np.array([], dtype=int)

    chosen = np.unique(np.concatenate([start_sel, mid_sel, end_sel]))
    if len(chosen) > n_examples:
        chosen = np.sort(np.random.choice(chosen, n_examples, replace=False))

    return chosen


def save_model_input_examples(x_data, y_data, raw_windows, subject_id, save_dir,
                              n_examples=10, fs=256, channel_names=None):
    """
    Save example plots showing model input channels with spindle annotations.
    """
    if not n_examples or n_examples <= 0:
        return

    if channel_names is None:
        channel_names = ['EEG', 'Sigma', 'Hilbert Envelope']

    subject_dir = Path(save_dir) / "Model_input_examples" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    has_spindle_indices = np.where(np.any(y_data > 0, axis=1))[0]
    if len(has_spindle_indices) == 0:
        log.warning(f"No spindles found for {subject_id}, skipping.")
        return

    chosen_indices = _select_example_indices(has_spindle_indices, n_examples)
    window_sec = x_data.shape[1] / fs
    t_axis = np.linspace(0, window_sec, x_data.shape[1])

    for i, idx in enumerate(chosen_indices):
        raw_signal = x_data[idx]
        raw_unfiltered = raw_windows[idx]
        mask = y_data[idx]

        channels = compute_input_channels(raw_signal, fs)
        n_channels = channels.shape[0]

        fig, axs = plt.subplots(
            n_channels + 1, 1,
            figsize=(10, 2.5 * (n_channels + 1)),
            sharex=True
        )

        # Row 0: Raw unfiltered signal (reference)
        axs[0].plot(t_axis, raw_unfiltered, color=COLOR_RAW, linewidth=0.8)
        axs[0].fill_between(
            t_axis, raw_unfiltered.min(), raw_unfiltered.max(),
            where=(mask > 0),
            color=COLOR_SPINDLE, alpha=0.25
        )
        axs[0].set_ylabel("Raw EEG (µV)")
        axs[0].grid(True, alpha=0.3)

        # Rows 1..n: Model input channels
        for ch_idx in range(n_channels):
            ax = axs[ch_idx + 1]
            signal = channels[ch_idx]
            color = COLORS_CHANNELS[ch_idx % len(COLORS_CHANNELS)]
            ch_label = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH {ch_idx + 1}"

            ax.plot(t_axis, signal, color=color, linewidth=1)
            ax.fill_between(
                t_axis, signal.min(), signal.max(),
                where=(mask > 0),
                color=COLOR_SPINDLE, alpha=0.25, label='Spindle'
            )
            ax.set_ylabel(ch_label)
            ax.grid(True, alpha=0.3)

        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()

        save_path = subject_dir / f"excerpt_{i + 1}_win_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    log.info(f"Saved {len(chosen_indices)} input examples to {subject_dir}")


def plot_eeg_trace(signal, sfreq, s1_evs, s2_evs, subject_id, save_dir):
    """
    Plot a 5-second EEG segment with expert scorer annotations.
    Tries to find an interesting region where scorers partially overlap.
    """
    center_time = _find_interesting_center(s1_evs, s2_evs)

    win_len = 5.0
    start_time = max(0, center_time - win_len / 2)
    end_time = start_time + win_len
    start_idx = int(start_time * sfreq)
    end_idx = int(end_time * sfreq)

    if end_idx > len(signal):
        end_idx = len(signal)
        start_idx = end_idx - int(win_len * sfreq)

    t_axis = np.linspace(start_time, end_time, end_idx - start_idx)
    segment = signal[start_idx:end_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, segment, color='black', linewidth=0.8, label='EEG', zorder=1)

    # Expert 1 annotations
    _plot_scorer_events(s1_evs, start_time, end_time, y=-10,
                        color='#EFB7B2', label_text='Expert 1')

    # Expert 2 annotations
    _plot_scorer_events(s2_evs, start_time, end_time, y=-15,
                        color='#6699CC', label_text='Expert 2')

    # Union shading
    _plot_union_shading(s1_evs, s2_evs, start_time, end_time, t_axis, sfreq)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    data_min, data_max = np.min(segment), np.max(segment)
    plt.ylim(min(data_min, -20) - 10, max(data_max, 10) + 10)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{subject_id}_trace.png", dpi=150)
    plt.close()


def _find_interesting_center(s1_evs, s2_evs, default=10.0):
    """Find a timepoint where scorer annotations partially overlap."""
    if s1_evs and s2_evs:
        for s1_onset, s1_dur in s1_evs:
            s1_end = s1_onset + s1_dur
            for s2_onset, s2_dur in s2_evs:
                s2_end = s2_onset + s2_dur
                if max(s1_onset, s2_onset) < min(s1_end, s2_end):
                    if abs(s1_onset - s2_onset) > 0.1 or abs(s1_end - s2_end) > 0.1:
                        return s1_onset + s1_dur / 2

    if s1_evs:
        return s1_evs[0][0] + s1_evs[0][1] / 2
    if s2_evs:
        return s2_evs[0][0] + s2_evs[0][1] / 2

    return default


def _plot_scorer_events(events, start_time, end_time, y, color, label_text):
    """Plot horizontal bars for a single scorer's annotations."""
    label_added = False
    for onset, dur in events:
        if (onset + dur) > start_time and onset < end_time:
            vis_start = max(onset, start_time)
            vis_end = min(onset + dur, end_time)
            plt.hlines(y=y, xmin=vis_start, xmax=vis_end, linewidth=4, color=color,
                       label=label_text if not label_added else "", zorder=3)
            label_added = True


def _plot_union_shading(s1_evs, s2_evs, start_time, end_time, t_axis, sfreq):
    """Shade the union of both scorers' annotations."""
    plot_len = len(t_axis)
    union_mask = np.zeros(plot_len, dtype=int)

    for events in [s1_evs, s2_evs]:
        for onset, dur in events:
            if (onset + dur) > start_time and onset < end_time:
                s_rel = max(0, int((max(onset, start_time) - start_time) * sfreq))
                e_rel = min(plot_len, int((min(onset + dur, end_time) - start_time) * sfreq))
                union_mask[s_rel:e_rel] = 1

    labeled_union, n_regions = label(union_mask)
    if n_regions == 0:
        return

    label_added = False
    for sl in find_objects(labeled_union):
        u_start = t_axis[sl[0].start]
        u_end = t_axis[min(sl[0].stop, plot_len - 1)]
        plt.axvspan(u_start, u_end, color='#9370DB', alpha=0.2, zorder=0,
                    label='Union' if not label_added else "")
        label_added = True


# =============================================================================
# STANDALONE SCRIPT MODE
# =============================================================================
# When run via `python -m utils.signal_visualization`, the module reads
# subject_stats.json from PROCESSED_DATA_DIR and generates plots for each
# subject listed there. This decouples plot generation from dataset building.


def _segment_raw_for_visualization(
    filtered_signal: np.ndarray,
    raw_unfiltered: np.ndarray,
    hypnogram: np.ndarray,
    window_sec: float,
    overlap_sec: float,
    included_stages: list,
    hypnogram_resolution_sec: float,
    fs: float,
):
    """
    Re-segment the signal to produce raw_windows for visualization.

    This mirrors the midpoint-rule segmentation used by build_*_dataset.py
    but only returns the raw (unfiltered) windows. The filtered windows
    and masks are loaded from disk in the caller.

    Returns:
        np.ndarray of raw window slices, shape (n_windows, window_samples)
    """
    signal_length = len(filtered_signal)
    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)

    use_hypno = hypnogram is not None
    raw_windows = []

    for start in range(0, signal_length - window_samples, step_samples):
        end = start + window_samples

        if use_hypno:
            midpoint_sec = (start + window_samples / 2) / fs
            mid_idx = int(midpoint_sec / hypnogram_resolution_sec)
            if mid_idx >= len(hypnogram):
                break
            if hypnogram[mid_idx] not in included_stages:
                continue

        raw_windows.append(raw_unfiltered[start:end])

    return np.array(raw_windows) if raw_windows else np.array([])


def _generate_plots_dreams(processed_dir: Path, subject_stats: dict):
    """Generate visualization plots for all DREAMS subjects in subject_stats."""
    from configs.dreams_config import DATA_PARAMS
    from configs.dreams_model_config import SIGNAL_VISUALIZATION_PARAMS
    from data_loaders import dreams_loader
    from signal_processing import bandpassfilter
    import paths

    plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_viz_examples = SIGNAL_VISUALIZATION_PARAMS.get("input_examples", 0)
    if not n_viz_examples or n_viz_examples <= 0:
        log.warning("SIGNAL_VISUALIZATION_PARAMS['input_examples'] is 0 or unset — nothing to plot.")
        return

    channel_names = SIGNAL_VISUALIZATION_PARAMS.get(
        "channel_names", ["EEG", "Sigma", "Hilbert Envelope"]
    )

    # Re-discover the raw data files (loader uses configured subject list)
    patient_list = dreams_loader.find_dreams_data_files(paths.RAW_DREAMS_DATA_DIR)
    patient_by_id = {p["id"]: p for p in patient_list}

    # Map subject_stats subjects to discovered patient groups
    stats_ids = {s["id"] for s in subject_stats["subjects"]}

    for sid in sorted(stats_ids):
        patient_group = patient_by_id.get(sid)
        if patient_group is None:
            log.warning(f"Subject {sid} in subject_stats.json but no raw files found — skipping.")
            continue

        log.info(f"Generating plots for: {sid}")

        # Load raw data + filter (same pipeline as build_dreams_dataset.py)
        raw = dreams_loader.load_dreams_patient_data(patient_group)
        if raw is None:
            log.warning(f"  Failed to load {sid} — skipping.")
            continue

        fs = raw.info["sfreq"]
        raw_unfiltered = raw.get_data()[0].copy()

        filtered = bandpassfilter.apply_bandpass_filter(
            raw.get_data()[0], fs,
            DATA_PARAMS["lowcut"], DATA_PARAMS["highcut"], DATA_PARAMS["filter_order"]
        )

        # EEG trace plot (uses raw scorer events)
        from build_dreams_dataset import get_scorer_annotations as get_dreams_scorers
        try:
            s1_evs, s2_evs = get_dreams_scorers(patient_group, fs)
            plot_eeg_trace(filtered, fs, s1_evs, s2_evs, sid, plots_dir)
            log.info(f"  Saved EEG trace plot: {sid}_trace.png")
        except Exception as e:
            log.warning(f"  EEG trace plotting failed for {sid}: {e}")

        # Model input examples: load X and Y from disk; reconstruct raw_windows
        x_path = processed_dir / f"{sid}_X_1D.npy"
        y_path = processed_dir / f"{sid}_Y_1D.npy"
        if not x_path.exists() or not y_path.exists():
            log.warning(f"  Missing .npy files for {sid} — skipping input examples.")
            continue

        x_data = np.load(x_path)
        y_data = np.load(y_path)

        hypnogram = dreams_loader.load_dreams_hypnogram(patient_group)
        raw_windows = _segment_raw_for_visualization(
            filtered_signal=filtered,
            raw_unfiltered=raw_unfiltered,
            hypnogram=hypnogram,
            window_sec=DATA_PARAMS["window_sec"],
            overlap_sec=DATA_PARAMS["overlap_sec"],
            included_stages=DATA_PARAMS["included_stages"],
            hypnogram_resolution_sec=DATA_PARAMS["hypnogram_resolution_sec"],
            fs=fs,
        )

        if len(raw_windows) != len(x_data):
            log.warning(
                f"  raw_windows count ({len(raw_windows)}) does not match X_1D count "
                f"({len(x_data)}) for {sid}. Skipping input examples — rebuild dataset "
                "or check that config matches."
            )
            continue

        try:
            save_model_input_examples(
                x_data=x_data,
                y_data=y_data,
                raw_windows=raw_windows,
                subject_id=sid,
                save_dir=plots_dir,
                fs=fs,
                n_examples=n_viz_examples,
                channel_names=channel_names,
            )
        except Exception as e:
            log.warning(f"  Could not save input examples for {sid}: {e}")


def _generate_plots_mass(processed_dir: Path, subject_stats: dict):
    """Generate visualization plots for all MASS subjects in subject_stats."""
    from configs.mass_config import DATA_PARAMS
    from configs.mass_model_config import SIGNAL_VISUALIZATION_PARAMS
    from data_loaders import mass_loader
    from signal_processing import bandpassfilter
    import paths

    plots_dir = processed_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_viz_examples = SIGNAL_VISUALIZATION_PARAMS.get("input_examples", 0)
    if not n_viz_examples or n_viz_examples <= 0:
        log.warning("SIGNAL_VISUALIZATION_PARAMS['input_examples'] is 0 or unset — nothing to plot.")
        return

    channel_names = SIGNAL_VISUALIZATION_PARAMS.get(
        "channel_names", ["EEG", "Sigma", "Hilbert Envelope"]
    )

    patient_list = mass_loader.find_mass_data_files(paths.RAW_MASS_DATA_DIR)
    patient_by_id = {p["id"]: p for p in patient_list}

    stats_ids = {s["id"] for s in subject_stats["subjects"]}

    for sid in sorted(stats_ids):
        patient_group = patient_by_id.get(sid)
        if patient_group is None:
            log.warning(f"Subject {sid} in subject_stats.json but no raw files found — skipping.")
            continue

        log.info(f"Generating plots for: {sid}")

        raw = mass_loader.load_mass_patient_data(patient_group)
        if raw is None:
            log.warning(f"  Failed to load {sid} — skipping.")
            continue

        fs = raw.info["sfreq"]
        raw_unfiltered = raw.get_data()[0].copy()

        filtered = bandpassfilter.apply_bandpass_filter(
            raw.get_data()[0], fs,
            DATA_PARAMS["lowcut"], DATA_PARAMS["highcut"], DATA_PARAMS["filter_order"]
        )

        # EEG trace plot
        from build_mass_dataset import get_scorer_annotations as get_mass_scorers
        try:
            s1_evs, s2_evs = get_mass_scorers(patient_group, fs)
            plot_eeg_trace(filtered, fs, s1_evs, s2_evs, sid, plots_dir)
            log.info(f"  Saved EEG trace plot: {sid}_trace.png")
        except Exception as e:
            log.warning(f"  EEG trace plotting failed for {sid}: {e}")

        # Model input examples: load X and UNION Y from disk; reconstruct raw_windows
        x_path = processed_dir / f"{sid}_X_1D.npy"
        y_path = processed_dir / f"{sid}_Y_UNION.npy"
        if not x_path.exists() or not y_path.exists():
            log.warning(f"  Missing .npy files for {sid} — skipping input examples.")
            continue

        x_data = np.load(x_path)
        y_data = np.load(y_path)

        hypnogram = mass_loader.load_mass_hypnogram(patient_group)
        raw_windows = _segment_raw_for_visualization(
            filtered_signal=filtered,
            raw_unfiltered=raw_unfiltered,
            hypnogram=hypnogram,
            window_sec=DATA_PARAMS["window_sec"],
            overlap_sec=DATA_PARAMS["overlap_sec"],
            included_stages=DATA_PARAMS["included_stages"],
            hypnogram_resolution_sec=DATA_PARAMS["hypnogram_resolution_sec"],
            fs=fs,
        )

        if len(raw_windows) != len(x_data):
            log.warning(
                f"  raw_windows count ({len(raw_windows)}) does not match X_1D count "
                f"({len(x_data)}) for {sid}. Skipping input examples — rebuild dataset "
                "or check that config matches."
            )
            continue

        try:
            save_model_input_examples(
                x_data=x_data,
                y_data=y_data,
                raw_windows=raw_windows,
                subject_id=sid,
                save_dir=plots_dir,
                fs=fs,
                n_examples=n_viz_examples,
                channel_names=channel_names,
            )
        except Exception as e:
            log.warning(f"  Could not save input examples for {sid}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots for a processed dataset. "
                    "Run AFTER build_*_dataset.py has produced subject_stats.json."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dreams", "mass"],
        help="Which dataset to visualize.",
    )
    args = parser.parse_args()

    from utils.logger import setup_logging
    setup_logging(f"visualize_{args.dataset}.log")

    import paths
    processed_dir = Path(paths.PROCESSED_DATA_DIR)
    stats_path = processed_dir / "subject_stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"subject_stats.json not found at {stats_path}. "
            f"Run build_{args.dataset}_dataset.py first to generate the processed dataset."
        )

    with open(stats_path) as f:
        subject_stats = json.load(f)

    expected_dataset = args.dataset.upper()
    actual_dataset = subject_stats.get("dataset", "").upper()
    if actual_dataset != expected_dataset:
        raise ValueError(
            f"subject_stats.json reports dataset='{actual_dataset}' "
            f"but --dataset={args.dataset} was requested. "
            f"Rebuild the dataset or pass the correct --dataset flag."
        )

    log.info(f"Generating plots for {expected_dataset} dataset")
    log.info(f"Processed dir: {processed_dir}")
    log.info(f"Subjects in stats: {len(subject_stats['subjects'])}")

    if args.dataset == "dreams":
        _generate_plots_dreams(processed_dir, subject_stats)
    else:
        _generate_plots_mass(processed_dir, subject_stats)

    log.info("Plot generation complete.")


if __name__ == "__main__":
    main()
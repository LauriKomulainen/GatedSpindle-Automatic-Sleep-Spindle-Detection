# utils/signal_visualization.py

"""
Visualization utilities for sleep spindle detection pipeline.

Generates:
- Model input example plots (multi-channel view with spindle annotations)
- EEG trace plots with expert scorer annotations
"""

from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from core.dataset import compute_input_channels
from configs.config_loader import DATA_PARAMS
from configs.model_config import SIGNAL_VISUALIZATION_PARAMS

log = logging.getLogger(__name__)

WINDOW_SEC = DATA_PARAMS['window_sec']
CHANNEL_NAMES = SIGNAL_VISUALIZATION_PARAMS.get(
    'channel_names', ['EEG', 'Sigma', 'Hilbert Envelope']
)

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
        channel_names = CHANNEL_NAMES

    subject_dir = Path(save_dir) / "Model_input_examples" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    has_spindle_indices = np.where(np.any(y_data > 0, axis=1))[0]
    if len(has_spindle_indices) == 0:
        log.warning(f"No spindles found for {subject_id}, skipping.")
        return

    chosen_indices = _select_example_indices(has_spindle_indices, n_examples)
    t_axis = np.linspace(0, WINDOW_SEC, x_data.shape[1])

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
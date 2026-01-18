# utils/signal_visualization.py

from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
from configs.dreams_config import DATA_PARAMS
from utils.logger import setup_logging
from core.dataset import compute_input_channels
from scipy.ndimage import label, find_objects

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

WINDOW_SEC = DATA_PARAMS['window_sec']

def save_model_input_examples(x_data, y_data, raw_windows, subject_id, save_dir,
                              n_examples=2, fs=100, channel_names=None):
    save_dir = Path(save_dir) / "Model_input_examples"
    save_dir.mkdir(parents=True, exist_ok=True)

    has_spindle_indices = np.where(np.any(y_data > 0, axis=1))[0]

    if len(has_spindle_indices) >= n_examples:
        chosen_indices = np.random.choice(has_spindle_indices, n_examples, replace=False)
    elif len(x_data) > 0:
        chosen_indices = np.random.choice(len(x_data), min(len(x_data), n_examples), replace=False)
    else:
        return

    t_axis = np.linspace(0, WINDOW_SEC, x_data.shape[1])

    for idx in chosen_indices:
        # 1. load data
        base_signal = x_data[idx]
        sig_true_raw = raw_windows[idx]
        mask = y_data[idx]

        # 2. channels from dataset.py
        generated_channels = compute_input_channels(base_signal, fs)

        n_channels = generated_channels.shape[0]

        # 3. Plot channels
        fig, axs = plt.subplots(n_channels + 1, 1, figsize=(10, 3 * (n_channels + 1)), sharex=True)

        fig.suptitle(f"Subject {subject_id} - Window {idx}", fontsize=16)

        # Row 0: Raw Signal (context)
        axs[0].plot(t_axis, sig_true_raw, color='#7f8c8d', linewidth=0.8, label='Original Raw')
        axs[0].set_ylabel("uV")
        axs[0].legend(loc='upper right', fontsize='x-small')
        axs[0].set_title("Reference: Raw Signal")
        axs[0].grid(True, alpha=0.3)

        colors = ['#2c3e50', '#2980b9', '#8e44ad', '#d35400', '#27ae60', '#c0392b']

        for i in range(n_channels):
            ax = axs[i + 1]
            signal = generated_channels[i]
            color = colors[i % len(colors)]

            if channel_names is not None and i < len(channel_names):
                ch_label = channel_names[i]
            else:
                ch_label = f"CH {i + 1}"

            ax.plot(t_axis, signal, color=color, linewidth=1, label=ch_label)

            # Spindle labels
            ax.fill_between(t_axis, min(signal), max(signal), where=(mask > 0),
                            color='#e74c3c', alpha=0.3, label='Spindle Label' if i == 0 else "")

            ax.set_title(ch_label)

            if "Raw" in ch_label and not "EEG" in ch_label:
                ylabel_text = "uV"
            else:
                ylabel_text = "Norm"

            ax.set_ylabel(ylabel_text)

            ax.legend(loc='upper right', fontsize='x-small')
            ax.grid(True, alpha=0.3)

        axs[-1].set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(save_dir / f"{subject_id}_win_{idx}_dynamic_inputs.png")
        plt.close()

    log.info(f"Saved input example plots to {save_dir}")


def plot_eeg_trace(signal, sfreq, s1_evs, s2_evs, subject_id, save_dir):
    """Visualizes a segment of EEG with annotations."""
    center_time = 10.0
    found_interesting = False

    if len(s1_evs) > 0 and len(s2_evs) > 0:
        for s1_onset, s1_dur in s1_evs:
            s1_end = s1_onset + s1_dur
            for s2_onset, s2_dur in s2_evs:
                s2_end = s2_onset + s2_dur
                if max(s1_onset, s2_onset) < min(s1_end, s2_end):
                    if abs(s1_onset - s2_onset) > 0.1 or abs(s1_end - s2_end) > 0.1:
                        center_time = s1_onset + (s1_dur / 2)
                        found_interesting = True
                        break
            if found_interesting: break

    if not found_interesting:
        if len(s1_evs) > 0:
            center_time = s1_evs[0][0] + (s1_evs[0][1] / 2)
        elif len(s2_evs) > 0:
            center_time = s2_evs[0][0] + (s2_evs[0][1] / 2)

    win_len = 5.0
    start_time = max(0, center_time - (win_len / 2))
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

    # Plot Scorer 1
    label_added = False
    for onset, dur in s1_evs:
        if (onset + dur) > start_time and onset < end_time:
            vis_start = max(onset, start_time)
            vis_end = min(onset + dur, end_time)
            plt.hlines(y=-10, xmin=vis_start, xmax=vis_end, linewidth=4, color='#EFB7B2',
                       label='Expert 1' if not label_added else "", zorder=3)
            label_added = True

    # Plot Scorer 2
    label_added = False
    for onset, dur in s2_evs:
        if (onset + dur) > start_time and onset < end_time:
            vis_start = max(onset, start_time)
            vis_end = min(onset + dur, end_time)
            plt.hlines(y=-15, xmin=vis_start, xmax=vis_end, linewidth=4, color='#6699CC',
                       label='Expert 2' if not label_added else "", zorder=3)
            label_added = True

    # Plot Union (Ground Truth)
    plot_len = len(t_axis)
    union_mask = np.zeros(plot_len, dtype=int)
    for evs in [s1_evs, s2_evs]:
        for onset, dur in evs:
            if (onset + dur) > start_time and onset < end_time:
                s_rel = int((max(onset, start_time) - start_time) * sfreq)
                e_rel = int((min(onset + dur, end_time) - start_time) * sfreq)
                s_rel = max(0, s_rel)
                e_rel = min(plot_len, e_rel)
                union_mask[s_rel:e_rel] = 1

    labeled_union, num_features = label(union_mask)
    slices = find_objects(labeled_union)

    label_added = False
    if num_features > 0:
        for sl in slices:
            start_idx_u = sl[0].start
            end_idx_u = sl[0].stop
            u_start_time = t_axis[start_idx_u]
            u_end_time = t_axis[end_idx_u] if end_idx_u < len(t_axis) else t_axis[-1]
            plt.axvspan(u_start_time, u_end_time, color='#9370DB', alpha=0.2, zorder=0,
                        label='UNION (Ground Truth)' if not label_added else "")
            label_added = True

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    data_min = np.min(segment)
    data_max = np.max(segment)
    plt.ylim(min(data_min, -20) - 10, max(data_max, 10) + 10)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_dir / f"{subject_id}_trace.png", dpi=150)
    plt.close()
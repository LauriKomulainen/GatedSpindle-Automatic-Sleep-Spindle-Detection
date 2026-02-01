from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
from utils.logger import setup_logging
from core.dataset import compute_input_channels
from scipy.ndimage import label, find_objects
from configs.config_loader import DATA_PARAMS

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

WINDOW_SEC = DATA_PARAMS['window_sec']
OVERLAP_SEC = DATA_PARAMS.get('overlap_sec', 0)


def save_model_input_examples(x_data, y_data, raw_windows, subject_id, save_dir,
                              n_examples=10, fs=256, channel_names=None,
                              y_data_corrected=None,
                              correction_val=0.0):
    """
    Saves example plots of model inputs.
    Dynamically distributes n_examples across start, middle, and end of the recording.
    """
    # Jos n_examples on None tai 0, lopetetaan heti
    if not n_examples or n_examples <= 0:
        return

    subject_dir = Path(save_dir) / "Model_input_examples" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify windows containing spindles
    has_spindle_indices = np.where(np.any(y_data > 0, axis=1))[0]
    n_spindles = len(has_spindle_indices)

    if n_spindles == 0:
        log.warning(f"No spindles found for subject {subject_id}. Skipping plotting.")
        return

    # 2. Select Indices (Dynamic Start/Middle/End split)
    if n_spindles <= n_examples:
        chosen_indices = has_spindle_indices
    else:
        part = n_examples // 3
        remainder = n_examples % 3

        n_start = part + remainder
        first_selection = has_spindle_indices[:n_start]

        n_end = part
        last_selection = has_spindle_indices[-n_end:] if n_end > 0 else np.array([], dtype=int)

        n_mid = part
        middle_selection = np.array([], dtype=int)

        if n_mid > 0:
            mid_idx = n_spindles // 2
            start_mid = max(0, mid_idx - (n_mid // 2))
            end_mid = start_mid + n_mid
            if end_mid > n_spindles:
                end_mid = n_spindles
                start_mid = max(0, end_mid - n_mid)

            middle_selection = has_spindle_indices[start_mid:end_mid]

        chosen_indices = np.unique(np.concatenate([first_selection, middle_selection, last_selection]))
        if len(chosen_indices) > n_examples:
            chosen_indices = np.sort(np.random.choice(chosen_indices, n_examples, replace=False))

    t_axis = np.linspace(0, WINDOW_SEC, x_data.shape[1])
    is_comparison = y_data_corrected is not None

    for i, idx in enumerate(chosen_indices):
        excerpt_num = i + 1
        base_signal = x_data[idx]
        sig_true_raw = raw_windows[idx]
        mask_original = y_data[idx]
        mask_corrected = y_data_corrected[idx] if is_comparison else None

        generated_channels = compute_input_channels(base_signal, fs)
        n_channels = generated_channels.shape[0]
        colors = ['#2c3e50', '#2980b9', '#8e44ad', '#d35400', '#27ae60', '#c0392b']

        # PLOTTING
        filename = f"excerpt_{excerpt_num}_win_{idx}_comparison.png" if is_comparison else f"excerpt_{excerpt_num}_win_{idx}.png"
        save_path = subject_dir / filename

        if is_comparison:
            # Comparison mode: two columns side by side
            fig, axs = plt.subplots(n_channels + 1, 2, figsize=(20, 3 * (n_channels + 1)),
                                    sharex=True, sharey='row')

            # Main title
            main_title = f"Subject {subject_id} - Window {idx}"
            fig.suptitle(main_title, fontsize=12, y=0.98)

            for row in range(n_channels + 1):
                if row == 0:
                    signal_data = sig_true_raw
                    ch_label = "Raw Signal"
                    color = '#7f8c8d'
                    ylabel = "µV"
                else:
                    signal_data = generated_channels[row - 1]
                    color = colors[(row - 1) % len(colors)]
                    ch_label = channel_names[row - 1] if channel_names and (row - 1) < len(
                        channel_names) else f"CH {row}"
                    ylabel = "µV" if ("Raw" in ch_label and "EEG" not in ch_label) else "Norm"

                # Left column (Original)
                ax_orig = axs[row, 0]
                ax_orig.plot(t_axis, signal_data, color=color, linewidth=0.8 if row == 0 else 1)
                ax_orig.set_ylabel(ylabel, fontsize=9)
                if row > 0:
                    ax_orig.fill_between(t_axis, min(signal_data), max(signal_data),
                                         where=(mask_original > 0),
                                         color='#e74c3c', alpha=0.3, label='Original')
                    ax_orig.set_title(ch_label, fontsize=10)
                ax_orig.grid(True, alpha=0.3)

                # Right column (Corrected)
                ax_corr = axs[row, 1]
                ax_corr.plot(t_axis, signal_data, color=color, linewidth=0.8 if row == 0 else 1)
                if row > 0:
                    ax_corr.fill_between(t_axis, min(signal_data), max(signal_data),
                                         where=(mask_corrected > 0),
                                         color='#e74c3c', alpha=0.3, label='Corrected')
                    ax_corr.set_title(ch_label, fontsize=10)
                ax_corr.grid(True, alpha=0.3)

            axs[-1, 0].set_xlabel("Time (s)")
            axs[-1, 1].set_xlabel("Time (s)")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        else:
            # Standard mode: single column
            fig, axs = plt.subplots(n_channels + 1, 1, figsize=(10, 3 * (n_channels + 1)), sharex=True)

            # Main title
            main_title = f"Subject {subject_id} - Window {idx}"
            fig.suptitle(main_title, fontsize=12, y=0.98)

            axs[0].plot(t_axis, sig_true_raw, color='#7f8c8d', linewidth=0.8, label='Original Raw')
            axs[0].set_ylabel("µV")
            axs[0].set_title("Reference: Raw Signal", fontsize=10)
            axs[0].grid(True, alpha=0.3)

            for ch_idx in range(n_channels):
                ax = axs[ch_idx + 1]
                signal = generated_channels[ch_idx]
                color = colors[ch_idx % len(colors)]
                ch_label = channel_names[ch_idx] if channel_names and ch_idx < len(
                    channel_names) else f"CH {ch_idx + 1}"

                ax.plot(t_axis, signal, color=color, linewidth=1, label=ch_label)
                ax.fill_between(t_axis, min(signal), max(signal),
                                where=(mask_original > 0),
                                color='#e74c3c', alpha=0.3, label='Spindle')

                ax.set_title(ch_label, fontsize=10)
                ylabel_text = "µV" if ("Raw" in ch_label and "EEG" not in ch_label) else "Norm"
                ax.set_ylabel(ylabel_text)
                ax.grid(True, alpha=0.3)

            axs[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()

    log.info(f"Saved {len(chosen_indices)} input example images to {subject_dir}")


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
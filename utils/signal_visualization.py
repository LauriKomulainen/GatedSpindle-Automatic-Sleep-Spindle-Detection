# utils/signal_visualization.py

from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
from preprocessing import bandpassfilter, normalization
from configs.dreams_config import DATA_PARAMS
from utils.logger import setup_logging

setup_logging("data_handler.log")
log = logging.getLogger(__name__)

WINDOW_SEC = DATA_PARAMS['window_sec']

def save_three_channel_examples(x_data, y_data, raw_windows, subject_id, save_dir, n_examples=5, fs=100):
    """
    Visualize 3 channels:
    1. RAW
    2. Model Input (0.3-30Hz + normalization)
    3. Sigma (11-16Hz + normalization)
    """
    save_dir = Path(save_dir) / "Three_channel_examples"
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
        # 1. RAW
        sig_true_raw = raw_windows[idx]

        # 2. Model Input (0.3-30Hz)
        sig_input = x_data[idx]

        # 3. Sigma (11-16Hz)
        sig_sigma = bandpassfilter.apply_bandpass_filter(
            sig_input, fs, lowcut=11.0, highcut=16.0, order=4
        )
        sig_sigma = normalization.normalize_data(sig_sigma)

        mask = y_data[idx]
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot 1: RAW
        axs[0].plot(t_axis, sig_true_raw, color='#555555', linewidth=0.8, label='1. Raw (Clean)')
        axs[0].set_ylabel("Amplitude (uV)")
        axs[0].set_title(f"Subject {subject_id} - Window {idx}")
        axs[0].legend(loc='upper right', fontsize='x-small')
        axs[0].grid(True, alpha=0.3)

        # Plot 2: Model Input (0.3-30 Hz)
        axs[1].plot(t_axis, sig_input, color='#333333', linewidth=1, label='2. Model Input (0.3-30Hz)')
        axs[1].fill_between(t_axis, min(sig_input), max(sig_input), where=(mask > 0),
                            color='#e74c3c', alpha=0.3)
        axs[1].set_ylabel("Norm. Amplitude")
        axs[1].legend(loc='upper right', fontsize='x-small')
        axs[1].grid(True, alpha=0.3)

        # Plot 3: Sigma (11-16Hz)
        axs[2].plot(t_axis, sig_sigma, color='#2980b9', linewidth=1, label='3. Sigma (11-16Hz)')
        axs[2].fill_between(t_axis, min(sig_sigma), max(sig_sigma), where=(mask > 0),
                            color='#e74c3c', alpha=0.3)
        axs[2].set_ylabel("Norm. Amplitude")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend(loc='upper right', fontsize='x-small')
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f"{subject_id}_win_{idx}_raw_comparison.png")
        plt.close()

    log.info(f"Saved 3-channel comparison plots to {save_dir}")
# utils/signal_visualization.py

from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
from configs.dreams_config import DATA_PARAMS
from utils.logger import setup_logging
from core.dataset import compute_input_channels

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
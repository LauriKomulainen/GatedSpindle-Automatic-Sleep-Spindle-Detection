# diagnostics.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import os
import logging

log = logging.getLogger(__name__)

from data_preprocess.cwt_transform import (
    CWT_FREQ_LOW, CWT_FREQ_HIGH, CWT_FREQ_BINS,
    SPINDLE_FREQ_LOW, SPINDLE_FREQ_HIGH
)


# (save_diagnostic_plot pysyy samana, se on jo oikein 3-kanavaiselle kuvalle)
def save_diagnostic_plot(
        signal_1d: np.ndarray,
        mask_1d: np.ndarray,
        image_3c: np.ndarray,
        mask_2d: np.ndarray,
        fs: float,
        save_path: Path
):
    """
    Tallentaa 5-osaisen diagnostiikkakuvan yhdestä ikkunasta.
    Näyttää 1D-signaalin, 3 CWT-kanavaa ja 2D-maskin.
    """
    try:
        image_ch0 = np.squeeze(image_3c[0])
        image_ch1 = np.squeeze(image_3c[1])
        image_ch2 = np.squeeze(image_3c[2])
        mask_2d = np.squeeze(mask_2d)

        window_sec = len(signal_1d) / fs
        time_axis = np.linspace(0, window_sec, len(signal_1d))

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1,
            figsize=(10, 18),
            gridspec_kw={'height_ratios': [1, 2, 2, 2, 2]},
            sharex=True
        )
        fig.suptitle(f"Diagnostic Plot: {save_path.stem}", fontsize=16)

        cwt_extent = [0, window_sec, CWT_FREQ_LOW, CWT_FREQ_HIGH]

        ax1.plot(time_axis, signal_1d, color='black', label='Signal (Z-score)')
        ax1.fill_between(time_axis,
                         ax1.get_ylim()[0],
                         ax1.get_ylim()[1],
                         where=(mask_1d == 1),
                         color='lime',
                         alpha=0.4,
                         label='Ground Truth Spindle')
        ax1.set_title("1. Original 1D Signal & Target Mask")
        ax1.set_ylabel("Amplitude (Z-score)")
        ax1.legend(loc='upper right')
        ax1.grid(linestyle='--', alpha=0.6)

        ax2.imshow(image_ch0, aspect='auto', origin='lower', cmap='jet', extent=cwt_extent)
        ax2.set_title("2. Model Input: Channel 0 (Main 1-35 Hz)")
        ax2.set_ylabel("Frequency (Hz)")

        ax3.imshow(image_ch1, aspect='auto', origin='lower', cmap='viridis', extent=cwt_extent)
        ax3.set_title("3. Model Input: Channel 1 (Delta Context 0.5-4 Hz)")
        ax3.set_ylabel("Frequency (Hz)")

        # KORJAUS: Päivitä otsikko näyttämään ALPHA (8-12 Hz)
        ax4.imshow(image_ch2, aspect='auto', origin='lower', cmap='hot', extent=cwt_extent)
        ax4.set_title("4. Model Input: Channel 2 (Alpha Context 8.0-12.0 Hz)")
        ax4.set_ylabel("Frequency (Hz)")

        ax5.imshow(mask_2d, aspect='auto', origin='lower', cmap='gray', extent=cwt_extent)
        ax5.set_title("5. Model Target: 2D Mask (Y_image)")
        ax5.set_ylabel("Frequency (Hz)")
        ax5.set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        log.info(f"Diagnostic plot saved: {save_path}")
    except ImportError:
        log.error("matplotlib not found. 'pip install matplotlib' to create plots.")
    except Exception as e:
        log.error(f"Failed to create diagnostic plot: {e}")
        plt.close('all')


def _convert_2d_mask_to_1d(mask_2d: np.ndarray) -> np.ndarray:
    """ Muuntaa 2D-ennustusmaskin 1D-aikasarjaksi. """
    frequencies = np.linspace(CWT_FREQ_LOW, CWT_FREQ_HIGH, CWT_FREQ_BINS)
    y_indices = np.where(
        (frequencies >= SPINDLE_FREQ_LOW) &
        (frequencies <= SPINDLE_FREQ_HIGH)
    )[0]
    if len(y_indices) == 0:
        return np.zeros(mask_2d.shape[1], dtype=bool)
    spindle_band = mask_2d[y_indices, :]
    mask_1d = np.any(spindle_band, axis=0)
    return mask_1d


# --- TÄMÄ FUNKTIO KORJATTU ---
def save_prediction_plot(
        model,
        loader,
        output_dir: str,
        fs: float,
        num_to_save: int,
        prefix: str
):
    """
    Tallentaa 'num_to_save' määrän ennustuskuvia 2-paneelin muodossa.
    """
    log.info(f"Saving {num_to_save} prediction examples ({prefix})...")
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    model.eval()

    saved_count = 0
    with torch.no_grad():
        try:
            random_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=1, shuffle=True, num_workers=0)
        except Exception:
            random_loader = loader

        # --- KORJAUS 1: Datalataaja palauttaa 3 arvoa ---
        for i, (images_seq, masks_2d, signals_1d) in enumerate(random_loader):
            if saved_count >= num_to_save:
                break

            # --- KORJAUS 2: Syötä sekvenssi mallille ---
            images_seq, masks_2d = images_seq.to(device), masks_2d.to(device)
            outputs_2d = model(images_seq)  # Malli ottaa sekvenssin (B,S,C,H,W)

            outputs_2d = torch.sigmoid(outputs_2d)
            preds_2d = (outputs_2d > 0.5).float()

            signal_1d_np = signals_1d[0].cpu().numpy().squeeze()
            mask_2d_true_np = masks_2d[0].cpu().numpy().squeeze()
            pred_2d_np = preds_2d[0].cpu().numpy().squeeze()

            mask_1d_true = _convert_2d_mask_to_1d(mask_2d_true_np)
            mask_1d_pred = _convert_2d_mask_to_1d(pred_2d_np)

            if len(mask_1d_pred) < len(signal_1d_np):
                pad_width = len(signal_1d_np) - len(mask_1d_pred)
                mask_1d_pred = np.pad(mask_1d_pred, (0, pad_width), 'constant', constant_values=False)

            window_sec = len(signal_1d_np) / fs
            time_axis = np.linspace(0, window_sec, len(signal_1d_np))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.suptitle(f"Prediction: {prefix}_example_{i}", fontsize=16)

            ax1.set_title("1. Ground Truth (Signal + Target Label)")
            ax1.plot(time_axis, signal_1d_np, color='black', label='Signal')
            ax1.fill_between(time_axis,
                             ax1.get_ylim()[0],
                             ax1.get_ylim()[1],
                             where=mask_1d_true,
                             color='lime',
                             alpha=0.5,
                             label='Ground Truth')
            ax1.set_ylabel("Amplitude (Z-score)")
            ax1.legend(loc='upper right')
            ax1.grid(linestyle='--', alpha=0.6)

            ax2.set_title("2. Model Prediction (Signal + Predicted Label)")
            ax2.plot(time_axis, signal_1d_np, color='black', label='Signal')
            ax2.fill_between(time_axis,
                             ax2.get_ylim()[0],
                             ax2.get_ylim()[1],
                             where=mask_1d_pred,
                             color='red',
                             alpha=0.5,
                             label='Model Prediction')
            ax2.set_ylabel("Amplitude (Z-score)")
            ax2.set_xlabel("Time (s)")
            ax2.legend(loc='upper right')
            ax2.grid(linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(output_dir, f"{prefix}_prediction_{i}.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            saved_count += 1
# signal_processing/transforms.py

import torch
import numpy as np
import random
from scipy.signal import hilbert, stft
from signal_processing.normalization import normalize_data
from signal_processing.bandpassfilter import apply_bandpass_filter


def compute_stft_sigma_channel(raw_signal_np, fs, nperseg=128, target_len=None):
    """
    Compute a sigma-band (11-16 Hz) power envelope via STFT.

    Returns a 1D array (same length as raw_signal_np) representing
    the interpolated sigma-band power over time.
    """
    noverlap = nperseg // 2
    f, t, Zxx = stft(raw_signal_np, fs=fs, nperseg=nperseg,
                     noverlap=noverlap, boundary=None)

    # Select sigma band (11-16 Hz)
    sigma_mask = (f >= 11.0) & (f <= 16.0)
    sigma_power = np.mean(np.abs(Zxx[sigma_mask, :]) ** 2, axis=0)

    # Interpolate back to original signal length
    if target_len is None:
        target_len = len(raw_signal_np)

    sigma_interp = np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(sigma_power)),
        sigma_power
    )

    return sigma_interp


def compute_input_channels(raw_signal_np, fs):
    """
    Compute multi-channel input: Raw EEG, Sigma band, Hilbert Envelope.

    FIX: Hilbert envelope is now computed from the UN-normalized sigma signal
    to preserve amplitude contrast. Each channel is normalized exactly once.
    Previously, the sigma signal was normalized before Hilbert computation,
    and the envelope was normalized again — double normalization compressed
    spindle amplitude peaks and hurt recall.
    """
    channels = []

    # CH1: Raw EEG (already filtered + normalized during preprocessing)
    ch1 = torch.tensor(raw_signal_np, dtype=torch.float32).unsqueeze(0)
    channels.append(ch1)

    # Sigma bandpass filter
    sigma_raw = apply_bandpass_filter(raw_signal_np, fs, 11, 16, order=4)

    # CH2: Sigma band — normalize once
    sigma_norm = normalize_data(sigma_raw)
    ch2 = torch.tensor(sigma_norm.copy(), dtype=torch.float32).unsqueeze(0)
    channels.append(ch2)

    # CH3: Hilbert envelope — computed from UN-normalized sigma, then normalized once
    analytic_signal = hilbert(sigma_raw)
    amplitude_envelope = np.abs(analytic_signal)
    env_norm = normalize_data(amplitude_envelope)
    ch3 = torch.tensor(env_norm.copy(), dtype=torch.float32).unsqueeze(0)
    channels.append(ch3)

    return torch.cat(channels, dim=0)


class RandomAugment1D:
    """
    Applies random augmentations to 1D EEG signal and mask.
    Previously only the mask was zeroed, which created corrupted negative examples
    where spindle-like waveforms were paired with zero masks at boundaries.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal, mask=None):
        # Gain jitter (signal only - mask unaffected)
        if random.random() < 0.5:
            signal = signal * random.uniform(0.9, 1.1)

        # Additive Gaussian noise (signal only)
        if random.random() < 0.5:
            noise = torch.randn_like(signal) * 0.03
            signal = signal + noise

        # Invert the signal vertically with 50% probability
        if random.random() < 0.5:
            signal = signal * -1.0
        # --------

        # Time shift - apply SAME shift to both signal and mask
        if random.random() < 0.5:
            shift = random.randint(-50, 50)
            signal = torch.roll(signal, shift, dims=-1)
            if mask is not None:
                mask = torch.roll(mask, shift, dims=-1)

            # Zero out wrapped region in BOTH signal and mask
            if shift > 0:
                signal[:, :shift] = 0.0
                if mask is not None:
                    mask[:shift] = 0.0
            elif shift < 0:
                signal[:, shift:] = 0.0
                if mask is not None:
                    mask[shift:] = 0.0

        # Channel dropout - randomly zero one input channel
        if random.random() < 0.3:
            ch_idx = random.randint(0, signal.shape[0] - 1)
            signal[ch_idx, :] = 0.0

        return signal, mask
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
    Computes input channels: Raw, Sigma, Hilbert Envelope, STFT Sigma Power.
    """
    channels = []

    # CH1: Raw EEG
    ch1 = torch.tensor(raw_signal_np, dtype=torch.float32).unsqueeze(0)
    channels.append(ch1)

    # CH2: Sigma (11-16 Hz)
    sigma_signal = apply_bandpass_filter(raw_signal_np, fs, 11, 16, order=4)
    sigma_signal = normalize_data(sigma_signal)
    ch2 = torch.tensor(sigma_signal.copy(), dtype=torch.float32).unsqueeze(0)
    channels.append(ch2)

    # CH3: Hilbert Envelope
    analytic_signal = hilbert(sigma_signal)
    amplitude_envelope = np.abs(analytic_signal)
    env_norm = normalize_data(amplitude_envelope)
    ch3 = torch.tensor(env_norm.copy(), dtype=torch.float32).unsqueeze(0)
    channels.append(ch3)

    return torch.cat(channels, dim=0)


class RandomAugment1D:
    """
    Augmentation that operates on BOTH signal and mask jointly
    to maintain temporal alignment.
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

        # Time shift - apply SAME shift to both signal and mask
        if random.random() < 0.5:
            shift = random.randint(-50, 50)
            signal = torch.roll(signal, shift, dims=-1)
            if mask is not None:
                mask = torch.roll(mask, shift, dims=-1)
                # Zero out the wrapped-around region so we don't
                # create false positives at boundaries
                if shift > 0:
                    mask[:shift] = 0.0
                elif shift < 0:
                    mask[shift:] = 0.0

        # Channel dropout - randomly zero one input channel
        if random.random() < 0.3:
            ch_idx = random.randint(0, signal.shape[0] - 1)
            signal[ch_idx] = 0.0

        if mask is not None:
            return signal, mask
        return signal
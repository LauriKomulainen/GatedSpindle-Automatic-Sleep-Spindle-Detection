# core/features.py

from scipy.signal import hilbert
import numpy as np
import torch
from signal_processing.normalization import normalize_data
from signal_processing.bandpassfilter import apply_bandpass_filter

def compute_input_channels(raw_signal_np, fs):
    """
    Compute multi-channel input: Raw EEG, Sigma band, Hilbert Envelope.

    NOTE: This exact pipeline is documented in Master Thesis Section 7.3.
    Do not edit, so the text and code stay in sync!
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

    return torch.cat(channels, dim=0)
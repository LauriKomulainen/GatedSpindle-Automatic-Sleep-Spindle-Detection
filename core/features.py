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
# core/features.py

from scipy.signal import hilbert
import numpy as np
import torch
from signal_processing.normalization import normalize_data
from signal_processing.bandpassfilter import apply_bandpass_filter

def compute_input_channels(raw_signal_np, fs):
    """
    Compute multi-channel input tensor for the model.

    Active channels:
        CH1: Preprocessed EEG (raw input from preprocessing stage)
        CH2: Sigma band (11-16 Hz)

    To add a new channel, compute it below and append to `channels`.
    Each channel must have shape (1, signal_length) and dtype float32.
    """
    channels = []

    # CH1: "Raw" EEG
    ch1 = torch.tensor(raw_signal_np, dtype=torch.float32).unsqueeze(0)
    channels.append(ch1)

    # CH2: Sigma (11-16 Hz)
    sigma_signal = apply_bandpass_filter(raw_signal_np, fs, 11, 16, order=4)
    sigma_signal = normalize_data(sigma_signal)
    ch2 = torch.tensor(sigma_signal.copy(), dtype=torch.float32).unsqueeze(0)
    channels.append(ch2)

    # Optional channels for future usage
    # Uncomment the block below to include the Hilbert amplitude envelope
    # of the sigma band as an additional channel.
    #
    # analytic_signal = hilbert(sigma_signal)
    # amplitude_envelope = np.abs(analytic_signal)
    # env_norm = normalize_data(amplitude_envelope)
    # ch_hilbert = torch.tensor(env_norm.copy(), dtype=torch.float32).unsqueeze(0)
    # channels.append(ch_hilbert)

    return torch.cat(channels, dim=0)
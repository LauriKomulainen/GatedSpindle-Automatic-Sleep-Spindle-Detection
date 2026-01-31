# signal_processing/transforms.py

import torch
import numpy as np
import random
from scipy.signal import hilbert
from signal_processing.normalization import normalize_data
from signal_processing.bandpassfilter import apply_bandpass_filter


def compute_input_channels(raw_signal_np, fs):
    """
    Computes input channels: Raw, Sigma, Hilbert Envelope.
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

    # CH4: Delta (0.4-4 Hz)
    delta_signal = apply_bandpass_filter(raw_signal_np, fs, 0.4, 4, order=4)
    delta_signal = normalize_data(delta_signal)
    ch4 = torch.tensor(delta_signal.copy(), dtype=torch.float32).unsqueeze(0)
    #channels.append(ch4)

    return torch.cat(channels, dim=0)


class RandomAugment1D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        if random.random() < self.p:
            gain = random.uniform(0.9, 1.1)
            signal = signal * gain
        return signal
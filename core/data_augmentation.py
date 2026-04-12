# core/data_augmentation.py

import random
import torch

class RandomAugment1D:
    """
    Applies random augmentations to 1D EEG signal and mask.
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

        # Invert the signal vertically
        if random.random() < 0.5:
            signal = signal * -1.0

        # Time shift, same shift is applied to both signal and mask
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

        # Channel dropout. Randomly zero one input channel
        if random.random() < 0.3:
            ch_idx = random.randint(0, signal.shape[0] - 1)
            signal[ch_idx, :] = 0.0

        return signal, mask
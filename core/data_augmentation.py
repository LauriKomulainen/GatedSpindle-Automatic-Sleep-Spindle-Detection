# core/data_augmentation.py

import random
import torch

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
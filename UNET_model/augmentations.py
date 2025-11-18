# UNET_model/augmentations.py

import torch.nn as nn
import random


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_prob=0.3, time_mask_prob=0.3, freq_mask_param=10, time_mask_param=20,
                 protected_channels=None):
        super(SpecAugment, self).__init__()
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        # Kanavat, joihin EI kosketa (esim. [2] lihaskanavalle)
        self.protected_channels = protected_channels if protected_channels else []

    def forward(self, x):
        """
        x: (B, C, F, T)
        """
        batch_size, num_channels, num_freq_bins, num_frames = x.shape
        x_aug = x.clone()

        # FREQUENCY MASKING
        if random.uniform(0, 1) < self.freq_mask_prob:
            for i in range(batch_size):
                f = random.randint(0, self.freq_mask_param)
                if num_freq_bins > f:
                    f0 = random.randint(0, num_freq_bins - f)

                    # Maskataan kaikki kanavat PAITSI suojatut
                    for c in range(num_channels):
                        if c not in self.protected_channels:
                            x_aug[i, c, f0:f0 + f, :] = 0

        # TIME MASKING (voidaan soveltaa kaikkiin, koska aikasiirtymä ei poista artefaktia kokonaan)
        if random.uniform(0, 1) < self.time_mask_prob:
            for i in range(batch_size):
                t = random.randint(0, self.time_mask_param)
                if num_frames > t:
                    t0 = random.randint(0, num_frames - t)

                    for c in range(num_channels):
                        if c not in self.protected_channels:
                            x_aug[i, c, :, t0:t0 + t] = 0

        return x_aug
# UNET_model/augmentations.py

import torch.nn as nn
import random


class SpecAugment(nn.Module):
    """
    SpecAugment-tyyppinen augmentaatio CWT-kuville (torchaudio.transforms-kirjaston
    sijaan, jotta riippuvuuksia ei tarvita).

    Tämä ajetaan GPU:lla batchin sisällä.
    """

    def __init__(self, freq_mask_prob=0.3, time_mask_prob=0.3, freq_mask_param=10, time_mask_param=20):
        super(SpecAugment, self).__init__()
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, x):
        """
        x: Sisääntuleva CWT-kuvabatch (B, C, F, T)
        """
        batch_size, _, num_freq_bins, num_frames = x.shape

        x_aug = x.clone()

        if random.uniform(0, 1) < self.freq_mask_prob:
            for i in range(batch_size):
                f = random.randint(0, self.freq_mask_param)
                if num_freq_bins > f:
                    f0 = random.randint(0, num_freq_bins - f)
                    x_aug[i, :, f0:f0 + f, :] = 0

        if random.uniform(0, 1) < self.time_mask_prob:
            for i in range(batch_size):
                t = random.randint(0, self.time_mask_param)
                if num_frames > t:
                    t0 = random.randint(0, num_frames - t)
                    x_aug[i, :, :, t0:t0 + t] = 0

        return x_aug
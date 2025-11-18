# UNET_model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Tversky Loss F1-scoren optimointiin ja FP-arvojen vähentämiseen.
        alpha: Paino False Negativeille (Recall). Esim 0.3.
        beta:  Paino False Positiveille (Precision). Esim 0.7 (Korkea arvo vähentää vääriä hälytyksiä).
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs_logits, targets):
        # Varmistetaan dimension yhteensopivuus
        h_out, w_out = outputs_logits.shape[2], outputs_logits.shape[3]
        h_targ, w_targ = targets.shape[2], targets.shape[3]
        if h_out != h_targ or w_out != w_targ:
            targets = F.interpolate(targets, size=(h_out, w_out), mode='nearest')

        # Sigmoid muunnos logiteille
        inputs = torch.sigmoid(outputs_logits)

        # Flattens
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Lasketaan Tversky indeksi
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        return 1 - tversky
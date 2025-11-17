# UNET_model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Yhdistetty Dice Loss ja Binary Cross-Entropy Loss.
    Tämä ottaa sisään logiitit (raaka mallin ulostulo) ja on
    numeerisesti vakaampi.
    """

    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        # BCEWithLogitsLoss on vakaampi ja odottaa logiitteja
        self.bce = nn.BCEWithLogitsLoss()

    def _dice_loss(self, outputs_sigmoid, targets):
        """ Sisäinen Dice-lasku, joka odottaa sigmoid-aktivoitua syötettä """
        outputs = outputs_sigmoid.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

    def forward(self, outputs_logits, targets):
        """
        outputs_logits: Mallin raaka ulostulo (B, C, H, W)
        targets: Maali-maski (B, C, H, W)
        """
        h_out, w_out = outputs_logits.shape[2], outputs_logits.shape[3]
        h_targ, w_targ = targets.shape[2], targets.shape[3]
        if h_out != h_targ or w_out != w_targ:
            targets = F.interpolate(targets, size=(h_out, w_out), mode='nearest')

        bce_loss = self.bce(outputs_logits, targets)

        outputs_sigmoid = torch.sigmoid(outputs_logits)
        dice_loss = self._dice_loss(outputs_sigmoid, targets)

        combined_loss = (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)

        return combined_loss
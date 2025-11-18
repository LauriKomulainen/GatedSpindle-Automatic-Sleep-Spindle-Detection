# UNET_model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Yhdistetty Dice Loss ja Binary Cross-Entropy Loss.
    Tämä ottaa sisään logiitit (raaka mallin ulostulo) ja on
    numeerisesti vakaampi.

    LISÄTTY: 'fp_weight' -parametri, jolla rangaistaan vääriä positiivisia
    (taustan ennustamista positiiviseksi) kovemmin.
    """

    def __init__(self, smooth=1.0, bce_weight=0.5, fp_weight=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.fp_weight = fp_weight  # Tällä rangaistaan FP-virheitä

        # Emme voi käyttää sisäänrakennettua painotusta, koska haluamme
        # painottaa negatiivisia (tausta) näytteitä, ei positiivisia.
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

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

        # --- BCE-häviön laskenta painotuksella ---

        # 1. Laske raaka BCE-häviö jokaiselle pikselille
        bce_loss_raw = self.bce(outputs_logits, targets)

        # 2. Luo painokartta
        # Oletusarvo on 1.0 kaikille
        weights = torch.ones_like(targets)
        # Aseta korkeampi painoarvo niille pikseleille,
        # jotka ovat taustaa (target == 0)
        weights[targets == 0] = self.fp_weight

        # 3. Laske painotettu keskiarvo
        bce_loss = (bce_loss_raw * weights).mean()

        # --- Dice-häviön laskenta (pysyy samana) ---
        outputs_sigmoid = torch.sigmoid(outputs_logits)
        dice_loss = self._dice_loss(outputs_sigmoid, targets)

        # --- Yhdistetty häviö ---
        combined_loss = (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)

        return combined_loss
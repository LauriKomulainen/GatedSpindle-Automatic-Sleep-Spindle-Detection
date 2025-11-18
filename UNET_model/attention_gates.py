# UNET_model/attention_gates.py

import torch.nn as nn
import torch.nn.functional as F  # <-- LISÄTTY IMPORT


class AttentionGate(nn.Module):
    """
    Attention Gate -moduuli U-Netin skip connection -vaiheisiin.
    """

    def __init__(self, f_g, f_l, f_int):
        """
        Parametrit:
        F_g: Kanavat gating signalista (ylössamplattu polku, esim. 512)
        F_l: Kanavat skip connectionista (encoder-polku, esim. 512)
        F_int: Kanavat välitasossa (esim. 256)
        """
        super(AttentionGate, self).__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: Gating signal (dekooderista, esim. (B, C, 62, 124))
        x: Skip connection (enkooderista, esim. (B, C, 62, 125))
        """
        g1 = self.Wg(g)
        x1 = self.Wx(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)

        psi = self.psi(psi)
        return x * psi
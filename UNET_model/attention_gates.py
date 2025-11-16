# UNET_model/attention_gates.py

import torch.nn as nn

class AttentionGate(nn.Module):
    """
    Attention Gate -moduuli U-Netin skip connection -vaiheisiin.
    Tämä on Additive Attention Gate (AG) PyTorch-toteutus.
    """

    def __init__(self, f_g, f_l, f_int):
        """
        Parametrit:
        F_g: Kanavat gating signalista (ylössamplattu polku, esim. 512)
        F_l: Kanavat skip connectionista (encoder-polku, esim. 512)
        F_int: Kanavat välitasossa (esim. 256)
        """
        super(AttentionGate, self).__init__()

        # Wg: 1x1 Conv gating signalille (g)
        self.Wg = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        # Wx: 1x1 Conv skip connectionille (x)
        self.Wx = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        # Huomio-painot (a): Yhdistetty ulostulo -> 1 kanava -> Sigmoid [0, 1]
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # ReLU-aktivaatio käytetään yhdistämisen jälkeen
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # 1. Gating
        g1 = self.Wg(g)

        # 2. Skip Connection
        x1 = self.Wx(x)

        # 3. Yhdistetään ja aktivoidaan (g1 + x1)
        psi = self.relu(g1 + x1)

        # 4. Lasketaan huomio-painot (Attention Coefficients, a)
        psi = self.psi(psi)

        # 5. Huomion sulauttaminen: Kerrotaan alkuperäinen skip connection -signaali huomio-painoilla.
        return x * psi
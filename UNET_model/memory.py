# UNET_model/memory.py

import torch
import torch.nn as nn


class Bottleneck3D(nn.Module):
    """
    Korvaa hitaan BiConvLSTM:n nopealla 3D-konvoluutiolla.
    Käsittelee aikasarjan (S=3) yhtenä blokkina.
    """

    def __init__(self, in_channels, hidden_channels):
        super(Bottleneck3D, self).__init__()

        # Conv3d odottaa inputtia: (N, C, D, H, W), missä D on aika (sequence length)
        # Käytämme kernel_size=(3, 3, 3), jotta malli näkee menneisyyden, nykyhetken ja tulevaisuuden.
        self.conv1 = nn.Conv3d(
            in_channels,
            hidden_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),  # Säilyttää dimension (S=3 pysyy 3:na)
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            hidden_channels,
            hidden_channels,  # Pidetään kanavat samana
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(hidden_channels)

        # 1x1x1 konvoluutio sovittamaan kanavamäärä takaisin alkuperäiseen,
        # jos hidden_channels != in_channels tai haluamme sekoittaa piirteitä.
        self.final_conv = nn.Conv3d(
            hidden_channels,
            in_channels * 2,  # Tuplataan kanavat kuten BiLSTM teki (forward+backward)
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        """
        Input: (Batch, Channels, Sequence, Height, Width)
        """
        # Residual connection
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Palautetaan muoto vastaamaan BiConvLSTM:n outputtia (tuplatut kanavat)
        out = self.final_conv(out)

        return out
# UNET_model/memory.py

import torch.nn as nn


class Bottleneck3D(nn.Module):
    """
    Lightweight Bottleneck (Kevytversio).

    Alkuperäinen Conv3d(3x3x3) on korvattu kevyemmällä Conv3d(3x1x1) -operaatiolla.
    Tämä keskittyy sekoittamaan tietoa vain AIKADIMENSION (Sequence) yli,
    jättäen spatiaaliset piirteet (H, W) rauhaan, koska Encoder hoiti ne jo.

    Tämä vähentää parametreja ja estää ylisovitusta (Overfitting).
    """

    def __init__(self, in_channels, hidden_channels):
        super(Bottleneck3D, self).__init__()

        reduced_channels = hidden_channels // 2

        self.reduce_conv = nn.Conv3d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm3d(reduced_channels)

        self.conv_time = nn.Conv3d(
            reduced_channels,
            reduced_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.bn_time = nn.BatchNorm3d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)

        self.expand_conv = nn.Conv3d(
            reduced_channels,
            in_channels * 2,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        """
        Input: (Batch, Channels, Sequence, Height, Width)
        """

        out = self.reduce_conv(x)
        out = self.bn_reduce(out)
        out = self.relu(out)

        out = self.conv_time(out)
        out = self.bn_time(out)
        out = self.relu(out)

        out = self.expand_conv(out)

        return out
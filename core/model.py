# UNET_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


log = logging.getLogger(__name__)

class DiceBCELoss(nn.Module):
    """
    Combined Dice loss and Binary Cross-Entropy (BCE) loss.

    NOTE (self): This exact pipeline is documented in Master Thesis Section 7.4.
    Do not edit, so the text and code stay in sync!
    """
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (inputs * targets).sum(dim=1)
        dice_loss = (1 - (2. * intersection + self.smooth) /
                     (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)).mean()
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return 0.5 * bce + 0.5 * dice_loss


class ConvBlock(nn.Module):
    """
    Residual convolutional block with two 1D convolutions.

    Consists of:
    - Conv1D → InstanceNorm → ReLU
    - Conv1D → InstanceNorm
    - Residual (skip) connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=11, padding=5):
        super(ConvBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)

        # Residual shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm1d(out_channels, affine=True)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual addition
        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    """
    Decoder block consisting of upsampling followed by a convolutional block.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DecoderBlock, self).__init__()

        # Linear upsampling along the temporal dimension
        self.up = nn.Upsample(
            scale_factor=scale_factor,
            mode="linear",
            align_corners=False
        )

        # Convolutional refinement after concatenation
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Temporal dimensions match before concatenation
        if x.size(2) != skip.size(2):
            x = F.interpolate(
                x,
                size=skip.size(2),
                mode='linear',
                align_corners=False
            )

        # Concatenate decoder and encoder features
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class GatedUNet(nn.Module):
    """
    1D Gated U-Net architecture for time-series segmentation.

    The model consists of:
    - An encoder–decoder structure composed of residual convolutional blocks
      (22 Conv1D layers in total, including projection shortcuts)
    - Symmetric skip connections between encoder and decoder stages
    - Instance normalization and ReLU non-linearities throughout the network
    - A two-layer MLP gating head (256→64→1) operating on globally pooled bottleneck features
    """
    def __init__(self, features, dropout_rate=0.2, use_gating_branch=True):
        super(GatedUNet, self).__init__()
        self.use_gating_branch = use_gating_branch

        # Input normalization
        self.input_norm = nn.InstanceNorm1d(features, affine=True)

        # Encoder
        self.enc1 = ConvBlock(features, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_rate)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout_rate)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout_rate)


        """
        Bottleneck
        
        NOTE (self): This exact pipeline is documented in Master Thesis Section 7.4.
        Do not edit, so the text and code stay in sync!
        """
        self.bottleneck = ConvBlock(128, 256)

        """
        Global Gating Branch

        NOTE: This exact pipeline is documented in Master Thesis Section 7.4.
        Do not edit, so the text and code stay in sync!
        """
        if self.use_gating_branch:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.gate_fc = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

        # Decoder
        self.dec1 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec3 = DecoderBlock(64 + 32, 32)

        # Final segmentation output
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)

        # Encoder forward pass
        e1 = self.enc1(x)
        p1 = self.drop1(self.pool1(e1))

        e2 = self.enc2(p1)
        p2 = self.drop2(self.pool2(e2))

        e3 = self.enc3(p2)
        p3 = self.drop3(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(p3)

        # Global gating output
        if self.use_gating_branch:
            gate_logits = self.gate_fc(
                self.global_pool(b).view(b.size(0), -1)
            )
        else:
            gate_logits = torch.zeros(x.size(0), 1, device=x.device)

        # Decoder forward pass
        d1 = self.dec1(b, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)

        # Segmentation mask output (logits)
        mask_logits = self.final_conv(d3)

        return mask_logits, gate_logits
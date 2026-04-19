# core/DiceBCELoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    Combined Dice loss and Binary Cross-Entropy (BCE) loss.
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
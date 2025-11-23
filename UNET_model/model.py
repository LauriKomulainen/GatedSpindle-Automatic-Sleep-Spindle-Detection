# UNET_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
import os
from torch.optim.swa_utils import AveragedModel, SWALR

log = logging.getLogger(__name__)


# --- MALLILUOKAT (DiceBCE, CBAM, ConvBlock, DecoderBlock, PositionalEncoding, UNet) TÄSSÄ ---

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return 0.5 * bce + 0.5 * dice_loss


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ConvBlock, self).__init__()
        padding = "same"
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.cbam = CBAMBlock(out_channels)
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
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(2) != skip.size(2):
            x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class UNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(UNet, self).__init__()
        self.input_norm = nn.InstanceNorm1d(3, affine=True)
        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_rate)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout_rate)
        self.d_model = 128
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=256,
                                                   dropout=dropout_rate, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dec1 = DecoderBlock(256, 64)
        self.dec2 = DecoderBlock(128, 32)
        self.dec3 = DecoderBlock(64, 32)
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_norm(x)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        p1 = self.drop1(p1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        p2 = self.drop2(p2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        p3 = self.drop3(p3)
        bn = p3.permute(2, 0, 1)
        bn = self.pos_encoder(bn)
        bn = self.transformer(bn)
        bn = bn.permute(1, 2, 0)
        d1 = self.dec1(bn, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)
        return self.final_conv(d3)


# --- SWA TRAINING LOOP WITH COSINE ANNEALING (Korjattu tallennus) ---

def train_model(model, train_loader, val_loader, optimizer_type, learning_rate, num_epochs, early_stopping_patience,
                output_dir, fs):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    swa_model = AveragedModel(model)
    swa_start_epoch = int(num_epochs * 0.65)
    swa_scheduler = SWALR(optimizer, swa_lr=learning_rate * 0.2)

    criterion = DiceBCELoss().to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    log.info(f"SWA + CosineAnnealing activated. SWA starts at epoch {swa_start_epoch}.")

    for epoch in range(num_epochs):
        model.train()
        ep_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if np.random.random() < 0.3:
                lam = np.random.beta(0.4, 0.4)
                index = torch.randperm(x.size(0)).to(device)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                out = model(mixed_x)
                y_a, y_b = y, y[index]
                loss = lam * criterion(out.squeeze(1), y_a.float()) + (1 - lam) * criterion(out.squeeze(1), y_b.float())
            else:
                out = model(x)
                loss = criterion(out.squeeze(1), y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

            if epoch < swa_start_epoch:
                scheduler.step(epoch + x.size(0) / len(train_loader))

        avg_train = ep_loss / len(train_loader)
        train_losses.append(avg_train)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr_now = swa_scheduler.get_last_lr()[0]
        else:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out.squeeze(1), y.float())
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            lr_now = optimizer.param_groups[0]['lr']

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

        log.info(f"Epoch {epoch + 1}: Train {avg_train:.4f} | LR: {lr_now:.6f}")

    log.info("Updating SWA Batch Norm statistics...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # KORJAUS: Tallenna SWA-kääreen alla oleva malli oikeilla avaimilla
    swa_path = os.path.join(output_dir, 'unet_model_best.pth')
    torch.save(swa_model.module.state_dict(), swa_path) # <-- TÄMÄ ON KORJAUS
    log.info(f"SWA Model saved to {swa_path}")

    return train_losses, val_losses
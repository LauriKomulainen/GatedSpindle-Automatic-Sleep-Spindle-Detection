# UNET_model/model.py

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return bce + dice_loss

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size(2) != x1.size(2):
            g1 = F.interpolate(g1, size=x1.size(2), mode='linear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x): return self.single_conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            SingleConv(in_channels, out_channels, kernel_size, "same", dilation),
            SingleConv(out_channels, out_channels, kernel_size, "same", dilation),
        )

    def forward(self, x): return self.conv_block(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, dilation):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False)
        self.single_conv = SingleConv(in_channels, out_channels, 1, "same", dilation)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, dilation)
        self.attention_block = AttentionBlock(out_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.single_conv(x1)
        if x1.size(2) != x2.size(2):
            x1 = F.interpolate(x1, size=x2.size(2), mode='linear', align_corners=False)
        psi = self.attention_block(x1, x2)
        x = torch.cat([x2 * psi, x1], dim=1)
        return self.conv_block(x)


# --- NOVELTY: TRANSFORMER BOTTLENECK ---

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
        # x: (Length, Batch, Channels)
        return x + self.pe[:x.size(0), :]


class UNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(UNet, self).__init__()
        self.bn = nn.BatchNorm1d(3)  # Paluu 3 kanavaan (Raw, Sigma, TEO) - Puhtaampi signaali
        kernel_size = 7

        # Encoder (CNN)
        self.encoders = nn.ModuleList([
            ConvBlock(3, 32, kernel_size, 1),
            ConvBlock(32, 64, kernel_size, 1),
            ConvBlock(64, 128, kernel_size, 1)
        ])

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AvgPool1d(2)

        # --- TRANSFORMER BOTTLENECK ---
        # Tämä korvaa LSTM:n ja tuo globaalin kontekstin
        self.d_model = 256  # Projektio 128 -> 256

        self.project_in = nn.Conv1d(128, self.d_model, kernel_size=1)  # 128 (Enc out) -> 256 (Trans in)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,  # 8 päätä antaa tarkemman huomion
            dim_feedforward=512,
            dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)  # 3 kerrosta syvyyttä

        # Decoder (CNN)
        self.decoders = nn.ModuleList([
            Decoder(256, 128, 2, kernel_size, 1),  # Input 256 (Transformerista)
            Decoder(128, 64, 2, kernel_size, 1),
            Decoder(64, 32, 2, kernel_size, 1)
        ])

        self.dense = nn.Conv1d(32, 1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.bn(x)
        features_enc = []

        # 1. CNN Encoder Path
        for enc in self.encoders:
            x = enc(x)
            features_enc.append(x)
            x = self.dropout(x)
            x = self.pool(x)

        # 2. Transformer Bottleneck Path
        # x shape: (Batch, 128, Length)
        x = self.project_in(x)  # -> (Batch, 256, Length)
        x = x.permute(2, 0, 1)  # -> (Length, Batch, 256) [Transformer format]
        x = self.pos_encoder(x)  # Add position info
        x = self.transformer_encoder(x)  # Global Attention Magic
        x = x.permute(1, 2, 0)  # -> (Batch, 256, Length) [CNN format]

        # 3. CNN Decoder Path
        features_enc.reverse()
        for dec, x_enc in zip(self.decoders, features_enc):
            x = self.dropout(x)
            x = dec(x, x_enc)

        logits = self.dense(x)
        return logits


def train_model(model, train_loader, val_loader, optimizer_type, learning_rate, num_epochs, early_stopping_patience,
                output_dir, fs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Vaihdetaan DiceBCE-lossiin, joka on stabiilimpi segmentaatiossa
    criterion = DiceBCELoss().to(device)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    mixup_alpha = 0.2

    for epoch in range(num_epochs):
        model.train()
        ep_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Mixup regularization
            if np.random.random() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
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

        avg_train = ep_loss / len(train_loader)
        train_losses.append(avg_train)

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

        log.info(f"Epoch {epoch + 1}/{num_epochs}: Train {avg_train:.4f}, Val {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
            patience_counter = 0
            log.info(f"Validation loss improved. Saved model. (Patience: 0/{early_stopping_patience})")
        else:
            patience_counter += 1
            log.info(f"Validation loss did not improve. Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                log.info("Early stopping triggered.")
                break

    return train_losses, val_losses
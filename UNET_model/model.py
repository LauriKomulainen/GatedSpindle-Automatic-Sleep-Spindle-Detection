# UNET_model/model.py

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np  # Mixupia varten
from tqdm import tqdm

log = logging.getLogger(__name__)


# ... (Kopioi tähän kaikki luokat: AttentionBlock, SingleConv, ConvBlock, Decoder, PositionalEncoding, UNet) ...
# ... (Ne pysyvät täysin samoina kuin edellisessä vastauksessa) ...
# ... (Tilan säästämiseksi en toista luokkamäärityksiä, käytä edellistä versiota niille) ...

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
        self.bn = nn.BatchNorm1d(3)
        kernel_size = 7

        self.encoders = nn.ModuleList([
            ConvBlock(3, 32, kernel_size, 1),
            ConvBlock(32, 64, kernel_size, 1),
            ConvBlock(64, 128, kernel_size, 1)
        ])

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.AvgPool1d(2)

        self.d_model = 256
        self.project_in = nn.Conv1d(128, self.d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.decoders = nn.ModuleList([
            Decoder(256, 128, 2, kernel_size, 1),
            Decoder(128, 64, 2, kernel_size, 1),
            Decoder(64, 32, 2, kernel_size, 1)
        ])

        self.dense = nn.Conv1d(32, 1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.bn(x)
        features_enc = []
        for enc in self.encoders:
            x = enc(x)
            features_enc.append(x)
            x = self.dropout(x)
            x = self.pool(x)

        x = self.project_in(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)

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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 0

    # --- MIXUP PARAMETRIT ---
    mixup_alpha = 0.2

    for epoch in range(num_epochs):
        model.train()
        ep_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # --- NOVEL: MIXUP IMPLEMENTATION ---
            if np.random.random() < 0.5:  # 50% todennäköisyys Mixupille
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(x.size(0)).to(device)

                mixed_x = lam * x + (1 - lam) * x[index, :]

                # Lasketaan ennuste sekoitetulle datalle
                out = model(mixed_x)

                # Loss on sekoitus kahdesta targetista
                # BCEWithLogitsLoss hyväksyy float-targetit (Soft Labels)
                y_a, y_b = y, y[index]
                loss = lam * criterion(out.squeeze(1), y_a.float()) + (1 - lam) * criterion(out.squeeze(1), y_b.float())
            else:
                # Normaali koulutus
                out = model(x)
                loss = criterion(out.squeeze(1), y.float())
            # -----------------------------------

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

        log.info(f"Epoch {epoch + 1}: Train {avg_train:.4f}, Val {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
            patience = 0
            log.info("Validation loss improved. Saved model.")
        else:
            patience += 1
            if patience >= early_stopping_patience:
                log.info("Early stopping.")
                break

    return train_losses, val_losses
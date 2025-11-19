# UNET_model/model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import torch.nn.functional as F

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from .attention_gates import AttentionGate
from .augmentations import SpecAugment
from .losses import TverskyLoss
from .memory import Bottleneck3D
log = logging.getLogger(__name__)

class UNet(nn.Module):
    def __init__(self, dropout_rate):
        super(UNet, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        self.spec_aug = SpecAugment(
            freq_mask_prob=0.3,
            time_mask_prob=0.3,
            freq_mask_param=15,
            time_mask_param=30,
            protected_channels=[2]
        )

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                self.dropout,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # --- Encoder ---
        self.e1 = conv_block(3, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e5_bottleneck_conv = conv_block(512, 1024)

        # --- 3D BOTTLENECK ---
        self.bottleneck_3d = Bottleneck3D(in_channels=1024, hidden_channels=512)
        self.bottleneck_reducer = nn.Conv2d(2048, 1024, kernel_size=1)

        # --- Decoder ---
        self.up6 = up_conv(1024, 512)
        self.ag6 = AttentionGate(f_g=512, f_l=512, f_int=256)
        self.d6 = conv_block(1024, 512)

        self.up7 = up_conv(512, 256)
        self.ag7 = AttentionGate(f_g=256, f_l=256, f_int=128)
        self.d7 = conv_block(512, 256)

        self.up8 = up_conv(256, 128)
        self.ag8 = AttentionGate(f_g=128, f_l=128, f_int=64)
        self.d8 = conv_block(256, 128)

        self.up9 = up_conv(128, 64)
        self.ag9 = AttentionGate(f_g=64, f_l=64, f_int=32)
        self.d9 = conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x_cnn = x.view(b * s, c, h, w)

        if self.training:
            x_cnn = self.spec_aug(x_cnn)

        e1 = self.e1(x_cnn)
        p1 = self.pool(e1)
        e2 = self.e2(p1)
        p2 = self.pool(e2)
        e3 = self.e3(p2)
        p3 = self.pool(e3)
        e4 = self.e4(p3)
        p4 = self.pool(e4)
        e5 = self.e5_bottleneck_conv(p4)

        _, c_enc, h_enc, w_enc = e5.shape
        e5_seq = e5.view(b, s, c_enc, h_enc, w_enc).permute(0, 2, 1, 3, 4)

        temporal_out = self.bottleneck_3d(e5_seq)
        temporal_out = temporal_out.permute(0, 2, 1, 3, 4).contiguous()
        temporal_out = temporal_out.view(b * s, -1, h_enc, w_enc)

        lstm_out = self.bottleneck_reducer(temporal_out)

        e1_center = e1.view(b, s, 64, h, w)[:, s // 2]
        e2_center = e2.view(b, s, 128, h // 2, w // 2)[:, s // 2]
        e3_center = e3.view(b, s, 256, h // 4, w // 4)[:, s // 2]
        e4_center = e4.view(b, s, 512, h // 8, w // 8)[:, s // 2]

        lstm_out_center = lstm_out.view(b, s, 1024, h_enc, w_enc)[:, s // 2]

        d6 = self.up6(lstm_out_center)
        ag6 = self.ag6(g=d6, x=e4_center)
        if d6.shape[2:] != ag6.shape[2:]: d6 = F.interpolate(d6, size=ag6.shape[2:], mode='bilinear',
                                                             align_corners=False)
        d6 = self.d6(torch.cat((ag6, d6), dim=1))

        d7 = self.up7(d6)
        ag7 = self.ag7(g=d7, x=e3_center)
        if d7.shape[2:] != ag7.shape[2:]: d7 = F.interpolate(d7, size=ag7.shape[2:], mode='bilinear',
                                                             align_corners=False)
        d7 = self.d7(torch.cat((ag7, d7), dim=1))

        d8 = self.up8(d7)
        ag8 = self.ag8(g=d8, x=e2_center)
        if d8.shape[2:] != ag8.shape[2:]: d8 = F.interpolate(d8, size=ag8.shape[2:], mode='bilinear',
                                                             align_corners=False)
        d8 = self.d8(torch.cat((ag8, d8), dim=1))

        d9 = self.up9(d8)
        ag9 = self.ag9(g=d9, x=e1_center)
        if d9.shape[2:] != ag9.shape[2:]: d9 = F.interpolate(d9, size=ag9.shape[2:], mode='bilinear',
                                                             align_corners=False)
        d9 = self.d9(torch.cat((ag9, d9), dim=1))

        out = self.out_conv(d9)
        return out


def train_model(model, train_loader, val_loader, optimizer_type, learning_rate, num_epochs, early_stopping_patience,
                output_dir, fs):
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'

    device = torch.device(device_type)
    log.info(f"Using device: {device} (AMP enabled: {device_type != 'cpu'})")
    model.to(device)

    # MUUTOS: Alpha 0.7 painottaa Recallia (vähemmän false negativeja)
    criterion = TverskyLoss(alpha=0.3, beta=0.7).to(device)

    # MUUTOS: weight_decay=1e-4 lisätty estämään ylisovitusta
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    scaler = None
    if device_type != 'cpu':
        try:
            scaler = GradScaler(device=device_type)
        except Exception:
            if device_type == 'cuda':
                scaler = GradScaler()
            else:
                log.warning("GradScaler init failed for MPS, trying without scaler (float16 only).")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    log.info(f"Starting UNet (3D-CNN Bottleneck) training...")

    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss = 0

        train_iterator = tqdm(train_loader, desc=f"Training")
        for images_seq, masks_2d, _ in train_iterator:
            images_seq = images_seq.to(device)
            masks_2d = masks_2d.to(device)

            optimizer.zero_grad()

            if device_type != 'cpu':
                with autocast(device_type=device_type):
                    seg_logits = model(images_seq)
                    loss = criterion(seg_logits, masks_2d)
            else:
                seg_logits = model(images_seq)
                loss = criterion(seg_logits, masks_2d)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Avg Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0

        val_iterator = tqdm(val_loader, desc=f"Validation")
        with torch.no_grad():
            for images_seq, masks_2d, _ in val_iterator:
                images_seq = images_seq.to(device)
                masks_2d = masks_2d.to(device)

                if device_type != 'cpu':
                    with autocast(device_type=device_type):
                        seg_logits = model(images_seq)
                        loss = criterion(seg_logits, masks_2d)
                else:
                    seg_logits = model(images_seq)
                    loss = criterion(seg_logits, masks_2d)

                total_val_loss += loss.item()
                val_iterator.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
            log.info(f"Validation loss improved, saving model.")
        else:
            patience_counter += 1
            log.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            log.info(f"Early stopping triggered.")
            break

    return train_losses, val_losses
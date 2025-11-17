# UNET_model/model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import torch.nn.functional as F

from utils.diagnostics import save_prediction_plot
from .attention_gates import AttentionGate
from .augmentations import SpecAugment
from .losses import DiceBCELoss
from .memory import BiConvLSTM  # --- LISÄYS: Tuo uusi muistikerros ---

log = logging.getLogger(__name__)


class UNet(nn.Module):
    def __init__(self, dropout_rate):
        super(UNet, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

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

        # Encoder (pysyy samana, 3 kanavaa sisään)
        self.conv1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = conv_block(512, 1024)

        # --- LISÄYS: Ajallinen muistikerros (Bi-ConvLSTM) ---
        # Tämä tulee pullonkaulaan (conv5:n jälkeen)
        # hidden_dim 512 -> ulostulo on 2 * 512 = 1024, joka vastaa conv5:n kanavia
        self.temporal_memory = BiConvLSTM(input_dim=1024,
                                          hidden_dim=512,
                                          kernel_size=(3, 3),
                                          num_layers=1)
        # --------------------------------------------------

        # Decoder
        # Huom: up6 ottaa nyt 1024 kanavaa sisään BiConvLSTM:ltä
        self.ag6 = AttentionGate(f_g=512, f_l=512, f_int=256)
        self.up6 = up_conv(1024, 512)
        self.conv6 = conv_block(1024, 512)

        self.ag7 = AttentionGate(f_g=256, f_l=256, f_int=128)
        self.up7 = up_conv(512, 256)
        self.conv7 = conv_block(512, 256)

        self.ag8 = AttentionGate(f_g=128, f_l=128, f_int=64)
        self.up8 = up_conv(256, 128)
        self.conv8 = conv_block(256, 128)

        self.ag9 = AttentionGate(f_g=64, f_l=64, f_int=32)
        self.up9 = up_conv(128, 64)
        self.conv9 = conv_block(128, 64)

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)

        self.augmenter = SpecAugment()

    def forward(self, x):
        """
        MUUTOS: x on nyt sekvenssi (B, S, C, H, W)
        esim. (16, 3, 3, 64, 500)
        """
        b, s, c, h, w = x.shape

        # Yhdistä Batch ja Sequence -ulottuvuudet
        x = x.view(b * s, c, h, w)  # -> (48, 3, 64, 500)

        # Augmentaatio (sovelletaan jokaiseen kuvaan erikseen)
        if self.training:
            x = self.augmenter(x)

        # Encoder-polku
        c1 = self.conv1(x);
        p1 = self.pool1(c1)
        c2 = self.conv2(p1);
        p2 = self.pool2(c2)
        c3 = self.conv3(p2);
        p3 = self.pool3(c3)
        c4 = self.conv4(p3);
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)  # Muoto: (B*S, 1024, H_bottle, W_bottle)

        # --- MUUTOS: Käsittele sekvenssi BiConvLSTM:llä ---

        # 1. Pura Batch ja Sequence -ulottuvuudet
        _, c_bottle, h_bottle, w_bottle = c5.shape
        c5_seq = c5.view(b, s, c_bottle, h_bottle, w_bottle)  # -> (B, S, 1024, H, W)

        # 2. Syötä ajalliselle muistikerrokselle
        # Ulostulo on (B, 2*hidden_dim, H, W)
        context_map = self.temporal_memory(c5_seq)  # -> (B, 1024, H, W)

        # 3. 'context_map' korvaa nyt 'c5'
        # ------------------------------------------------

        # Decoder-polku (käyttää 'context_map' syötteenä)
        up_6 = self.up6(context_map)  # Huom: EI c5
        c4_squashed = c4.view(b, s, 512, c4.shape[2], c4.shape[3])
        c4_middle = c4_squashed[:, s // 2, :, :, :]  # Ota vain keskimmäinen skip-connection

        c4_cropped = F.interpolate(c4_middle, size=up_6.shape[2:], mode='bilinear', align_corners=False)
        attn_c4 = self.ag6(c4_cropped, up_6)
        merge6 = torch.cat([up_6, attn_c4], dim=1);
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c3_squashed = c3.view(b, s, 256, c3.shape[2], c3.shape[3])
        c3_middle = c3_squashed[:, s // 2, :, :, :]

        c3_cropped = F.interpolate(c3_middle, size=up_7.shape[2:], mode='bilinear', align_corners=False)
        attn_c3 = self.ag7(c3_cropped, up_7)
        merge7 = torch.cat([up_7, attn_c3], dim=1);
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c2_squashed = c2.view(b, s, 128, c2.shape[2], c2.shape[3])
        c2_middle = c2_squashed[:, s // 2, :, :, :]

        c2_cropped = F.interpolate(c2_middle, size=up_8.shape[2:], mode='bilinear', align_corners=False)
        attn_c2 = self.ag8(c2_cropped, up_8)
        merge8 = torch.cat([up_8, attn_c2], dim=1);
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1_squashed = c1.view(b, s, 64, c1.shape[2], c1.shape[3])
        c1_middle = c1_squashed[:, s // 2, :, :, :]

        c1_cropped = F.interpolate(c1_middle, size=up_9.shape[2:], mode='bilinear', align_corners=False)
        attn_c1 = self.ag9(c1_cropped, up_9)
        merge9 = torch.cat([up_9, attn_c1], dim=1);
        c9 = self.conv9(merge9)

        seg_logits = self.conv10(c9)

        return seg_logits  # Palauta vain yksi maski


# --- Koulutusfunktio ---
# Tämä yksinkertaistuu takaisin, koska meillä on taas vain yksi häviö
def train_model(model, train_loader, val_loader,
                optimizer_type, learning_rate, num_epochs,
                early_stopping_patience,
                output_dir: str,
                fs: float):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    log.info(f"Using device: {device}")
    model.to(device)

    criterion = DiceBCELoss(bce_weight=0.5).to(device)

    optimizer = {
        'Adam': optim.Adam(model.parameters(), lr=learning_rate),
        'SGD': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    }.get(optimizer_type)

    if optimizer is None:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    log.info(f"Starting Conv-BiLSTM-UNet training (Sequential Input)...")

    for epoch in range(num_epochs):
        model.train()
        running_loss_total = 0.0
        log.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # --- MUUTOS: Dataloader palauttaa nyt 3 elementtiä ---
        for batch_idx, (images_seq, masks, _) in enumerate(tqdm(train_loader, desc="Training")):
            # images_seq on (B, S, C, H, W)
            # masks on (B, C, H, W)
            images_seq, masks = images_seq.to(device), masks.to(device)

            optimizer.zero_grad()

            seg_logits = model(images_seq)  # Syötä sekvenssi

            total_loss = criterion(seg_logits, masks)  # Vertaa keskimmäiseen maskiin

            total_loss.backward()
            optimizer.step()

            running_loss_total += total_loss.item()

        avg_train_loss = running_loss_total / len(train_loader)
        train_losses.append(avg_train_loss)
        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Avg Training Loss: {avg_train_loss:.4f}")

        # --- VALIDOINTI ---
        model.eval()
        running_val_loss_total = 0.0

        with torch.no_grad():
            for images_seq, masks, _ in tqdm(val_loader, desc="Validation"):
                images_seq, masks = images_seq.to(device), masks.to(device)

                seg_logits = model(images_seq)
                total_loss = criterion(seg_logits, masks)
                running_val_loss_total += total_loss.item()

        avg_val_loss = running_val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)

        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")

        try:
            log.info(f"Saving 5 example images from epoch {epoch + 1}...")
            epoch_plot_dir = os.path.join(output_dir, f"epoch_{epoch + 1:02d}_predictions")
            os.makedirs(epoch_plot_dir, exist_ok=True)
            # Huom: save_prediction_plot tarvitsee päivityksen
            save_prediction_plot(
                model=model, loader=val_loader, output_dir=epoch_plot_dir,
                fs=fs, num_to_save=5, prefix=f"epoch_{epoch + 1:02d}"
            )
        except Exception as e:
            log.error(f"Failed to save example images on epoch {epoch + 1}: {e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
            log.info(f"Validation loss improved, saving model to: {output_dir}")
        else:
            patience_counter += 1
            log.info(f"Validation loss did not improve... {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                log.info(f"Stopping training early after {patience_counter} epochs without improvement.")
                break

    log.info("Training complete.")
    return train_losses, val_losses
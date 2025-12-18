# UNET_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from torch.optim.swa_utils import AveragedModel, SWALR
from configs.dreams_config import TRAINING_PARAMS

log = logging.getLogger(__name__)

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ConvBlock, self).__init__()
        padding = "same"
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)
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


class GatedUNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(GatedUNet, self).__init__()
        self.input_norm = nn.InstanceNorm1d(2, affine=True)
        self.enc1 = ConvBlock(2, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_rate)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout_rate)
        self.bottleneck = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.gate_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.dec1 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec3 = DecoderBlock(64 + 32, 32)
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
        b = self.bottleneck(p3)
        gate_logits = self.gate_fc(self.global_pool(b).view(b.size(0), -1))
        d1 = self.dec1(b, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)
        mask_logits = self.final_conv(d3)
        return mask_logits, gate_logits


def train_model(model, train_loader, val_loader, optimizer_type, learning_rate, num_epochs, early_stopping_patience,
                output_dir, fs, use_swa=True):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model.to(device)
    # Haetaan parametrit configista (fallback oletusarvoihin jos puuttuu)
    weight_decay = TRAINING_PARAMS.get('weight_decay', 1e-2)

    # --- OPTIMIZER SELECTION ---
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    swa_model = None
    swa_scheduler = None

    swa_start_epoch = int(num_epochs * 0.60)
    planned_swa_len = num_epochs - swa_start_epoch
    actual_stop_epoch = num_epochs

    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=learning_rate * 0.1)
        log.info(f"SWA Enabled. Planned start: {swa_start_epoch}, Planned length: {planned_swa_len} epochs.")
    else:
        log.info("SWA Disabled. Standard Early Stopping will apply strictly.")

    criterion_seg = DiceBCELoss().to(device)
    criterion_cls = nn.BCEWithLogitsLoss().to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # --- TURVAKYTKIN: Lopetetaan, jos uusi takaraja on saavutettu ---
        if epoch >= actual_stop_epoch:
            log.info(f"Reached adjusted training limit ({actual_stop_epoch} epochs). Stopping.")
            break

        model.train()
        ep_loss = 0

        for x, y_mask, y_label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            x, y_mask, y_label = x.to(device), y_mask.to(device), y_label.to(device)
            optimizer.zero_grad()

            mask_logits, gate_logits = model(x)
            loss_seg = criterion_seg(mask_logits.squeeze(1), y_mask.float())
            loss_cls = criterion_cls(gate_logits, y_label)
            loss = 0.6 * loss_seg + 0.4 * loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

            if use_swa and epoch < swa_start_epoch:
                scheduler.step(epoch + x.size(0) / len(train_loader))
            elif not use_swa:
                scheduler.step(epoch + x.size(0) / len(train_loader))

        avg_train = ep_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- SWA Update Logic ---
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr_now = swa_scheduler.get_last_lr()[0]
        else:
            lr_now = optimizer.param_groups[0]['lr']

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_mask, y_label in val_loader:
                x, y_mask, y_label = x.to(device), y_mask.to(device), y_label.to(device)
                mask_logits, gate_logits = model(x)
                l_seg = criterion_seg(mask_logits.squeeze(1), y_mask.float())
                l_cls = criterion_cls(gate_logits, y_label)
                val_loss += (0.6 * l_seg + 0.4 * l_cls).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        # --- Checkpoints & Logging Logic (UPDATED) ---
        status_msg = ""

        # Tallennetaan aina paras malli, vaikka oltaisiin SWA-vaiheessa (harvinaista mutta mahdollista)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Nollataan patience vain jos EI olla vielä SWA-vaiheessa
            if not (use_swa and epoch >= swa_start_epoch):
                patience_counter = 0

            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
            best_tag = "(New Best!) "
        else:
            if not (use_swa and epoch >= swa_start_epoch):
                patience_counter += 1
            best_tag = ""

        # Määritetään lokiviesti tilanteen mukaan
        if use_swa and epoch >= swa_start_epoch:
            # Olemme SWA-vaiheessa -> Tulostetaan SWA-edistyminen
            swa_progress = epoch - swa_start_epoch + 1
            swa_total = actual_stop_epoch - swa_start_epoch
            status_msg = f"{best_tag}(SWA Phase: {swa_progress}/{swa_total})"
        else:
            # Olemme normaalissa vaiheessa -> Tulostetaan Patience
            status_msg = f"{best_tag}(Patience: {patience_counter}/{early_stopping_patience})"

        log.info(f"Epoch {epoch + 1}: Train {avg_train:.4f} | Val {avg_val:.4f} | LR: {lr_now:.6f} | {status_msg}")

        # 1. Jos SWA on POIS päältä -> normaali stoppi
        if not use_swa and patience_counter >= early_stopping_patience:
            log.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # 2. Jos SWA on PÄÄLLÄ, mutta patience loppuu ENNEN kuin SWA on alkanut:
        if use_swa and epoch < swa_start_epoch and patience_counter >= early_stopping_patience:
            log.info(f"Early stopping triggered (Epoch {epoch + 1}). Forcing SWA start NOW.")

            swa_start_epoch = epoch + 1
            patience_counter = 0

            new_limit = swa_start_epoch + planned_swa_len
            actual_stop_epoch = min(num_epochs, new_limit)

            log.info(
                f"Adjusting training limit: Will run SWA until epoch {actual_stop_epoch} (Duration: {planned_swa_len})")

    if use_swa and swa_model is not None:
        if swa_start_epoch < actual_stop_epoch:
            log.info("Updating SWA Batch Norm statistics...")
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            swa_path = os.path.join(output_dir, 'unet_model_swa.pth')
            torch.save(swa_model.module.state_dict(), swa_path)
            log.info("SWA model saved.")
        else:
            log.warning("SWA was enabled but loop finished before SWA updates. Check parameters.")

    return train_losses, val_losses
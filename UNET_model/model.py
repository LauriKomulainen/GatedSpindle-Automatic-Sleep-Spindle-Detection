# UNET_model/model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

from utils.diagnostics import save_prediction_plot
from .attention_gates import AttentionGate

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

        self.conv1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = conv_block(512, 1024)

        # Korjatut kanavamäärät
        self.ag6 = AttentionGate(f_g=512, f_l=512, f_int=256)
        self.up6 = up_conv(1024, 512)
        self.conv6 = conv_block(1024, 512)  # 512 (up) + 512 (skip)

        self.ag7 = AttentionGate(f_g=256, f_l=256, f_int=128)
        self.up7 = up_conv(512, 256)
        self.conv7 = conv_block(512, 256)  # 256 (up) + 256 (skip)

        self.ag8 = AttentionGate(f_g=128, f_l=128, f_int=64)
        self.up8 = up_conv(256, 128)
        self.conv8 = conv_block(256, 128)  # 128 (up) + 128 (skip)

        self.ag9 = AttentionGate(f_g=64, f_l=64, f_int=32)
        self.up9 = up_conv(128, 64)
        self.conv9 = conv_block(128, 64)  # 64 (up) + 64 (skip)

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x);
        p1 = self.pool1(c1)
        c2 = self.conv2(p1);
        p2 = self.pool2(c2)
        c3 = self.conv3(p2);
        p3 = self.pool3(c3)
        c4 = self.conv4(p3);
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        c4_cropped = c4[:, :, :up_6.shape[2], :up_6.shape[3]]
        attn_c4 = self.ag6(c4_cropped, up_6)
        merge6 = torch.cat([up_6, attn_c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c3_cropped = c3[:, :, :up_7.shape[2], :up_7.shape[3]]
        attn_c3 = self.ag7(c3_cropped, up_7)
        merge7 = torch.cat([up_7, attn_c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c2_cropped = c2[:, :, :up_8.shape[2], :up_8.shape[3]]
        attn_c2 = self.ag8(c2_cropped, up_8)
        merge8 = torch.cat([up_8, attn_c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1_cropped = c1[:, :, :up_9.shape[2], :up_9.shape[3]]
        attn_c1 = self.ag9(c1_cropped, up_9)
        merge9 = torch.cat([up_9, attn_c1], dim=1)
        c9 = self.conv9(merge9)

        return self.conv10(c9)


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

    # Pikselipohjainen DiceLoss koulutusta varten (OK)
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.):
            super(DiceLoss, self).__init__()
            self.smooth = smooth

        def forward(self, outputs, targets):
            outputs = torch.sigmoid(outputs)
            h_out, w_out = outputs.shape[2], outputs.shape[3]
            h_targ, w_targ = targets.shape[2], targets.shape[3]
            if h_out != h_targ or w_out != w_targ:
                targets = targets[:, :, :h_out, :w_out]
            outputs = outputs.reshape(-1)
            targets = targets.reshape(-1)
            intersection = (outputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
            return 1 - dice

    criterion = DiceLoss()
    optimizer = {
        'Adam': optim.Adam(model.parameters(), lr=learning_rate),
        'SGD': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    }.get(optimizer_type)

    if optimizer is None:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    log.info("Starting model training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        log.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for batch_idx, (images, masks, _) in enumerate(tqdm(train_loader, desc="Training")):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")

        try:
            log.info(f"Saving 5 example images from epoch {epoch + 1}...")
            epoch_plot_dir = os.path.join(output_dir, f"epoch_{epoch + 1:02d}_predictions")
            os.makedirs(epoch_plot_dir, exist_ok=True)
            save_prediction_plot(
                model=model, loader=val_loader, output_dir=epoch_plot_dir,
                fs=fs, num_to_save=5, prefix=f"epoch_{epoch + 1:02d}"
            )
        except Exception as e:
            log.error(f"Failed to save example images on epoch {epoch + 1}: {e}")

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
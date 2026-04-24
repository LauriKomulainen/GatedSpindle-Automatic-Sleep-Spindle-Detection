import torch
import torch.nn as nn
import logging
import os
from tqdm import tqdm
from core.DiceBCELoss import DiceBCELoss
from torch.optim.swa_utils import AveragedModel, SWALR
from core.config_loader import TRAINING_PARAMS
use_gating = TRAINING_PARAMS.get('use_gating_branch')
seg_weight = TRAINING_PARAMS.get('seg_loss_weight')
cls_weight = 1.0 - seg_weight

log = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, learning_rate, num_epochs, early_stopping_patience,
                output_dir, use_swa):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model.to(device)
    weight_decay = TRAINING_PARAMS.get('weight_decay', 1e-2)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    swa_model = None
    swa_scheduler = None

    # SWA start
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
            if use_gating:
                loss_cls = criterion_cls(gate_logits, y_label)
                loss = seg_weight * loss_seg + cls_weight * loss_cls
            else:
                loss = loss_seg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

        avg_train = ep_loss / len(train_loader)
        train_losses.append(avg_train)

        # SWA Update Logic
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
                if use_gating:
                    l_cls = criterion_cls(gate_logits, y_label)
                    val_loss += (seg_weight * l_seg + cls_weight * l_cls).item()
                else:
                    val_loss += l_seg.item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        if not use_swa or epoch < swa_start_epoch:
            scheduler.step(avg_val)

        status_msg = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            if not (use_swa and epoch >= swa_start_epoch):
                patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_model_best.pth'))
            best_tag = "(New Best!) "
        else:
            if not (use_swa and epoch >= swa_start_epoch):
                patience_counter += 1
            best_tag = ""

        if use_swa and epoch >= swa_start_epoch:
            swa_progress = epoch - swa_start_epoch + 1
            swa_total = actual_stop_epoch - swa_start_epoch
            status_msg = f"{best_tag}(SWA Phase: {swa_progress}/{swa_total})"
        else:
            status_msg = f"{best_tag}(Patience: {patience_counter}/{early_stopping_patience})"

        log.info(f"Epoch {epoch + 1}: Train {avg_train:.4f} | Val {avg_val:.4f} | LR: {lr_now:.6f} | {status_msg}")

        if not use_swa and patience_counter >= early_stopping_patience:
            log.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if use_swa and epoch < swa_start_epoch and patience_counter >= early_stopping_patience:
            log.info(f"Early stopping triggered (Epoch {epoch + 1}). Forcing SWA start NOW.")
            swa_start_epoch = epoch + 1
            patience_counter = 0
            new_limit = swa_start_epoch + planned_swa_len
            actual_stop_epoch = min(num_epochs, new_limit)
            log.info(f"Adjusting training limit: Will run SWA until epoch {actual_stop_epoch} (Duration: {planned_swa_len})")

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
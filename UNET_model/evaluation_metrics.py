# UNET_model/evaluation_metrics.py

import logging
import torch
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
from typing import List, Dict, Tuple
from training_parameters import CWT_PARAMS, METRIC_PARAMS, DATA_PARAMS
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _internal_convert_2d_mask_to_1d(mask_2d: np.ndarray) -> np.ndarray:
    """Muuntaa 2D CWT-maskin 1D-aikasarjaksi (boolean)."""
    frequencies = np.linspace(CWT_PARAMS['freq_low'], CWT_PARAMS['freq_high'], CWT_PARAMS['freq_bins'])
    y_indices = np.where(
        (frequencies >= METRIC_PARAMS['spindle_freq_low']) &
        (frequencies <= METRIC_PARAMS['spindle_freq_high'])
    )[0]
    if len(y_indices) == 0:
        return np.zeros(mask_2d.shape[1], dtype=bool)
    spindle_band = mask_2d[y_indices, :]
    return np.any(spindle_band, axis=0)


def _get_events_from_mask(mask_1d: np.ndarray, fs: float) -> List[Tuple[int, int]]:
    """
    Etsii 1D-binäärimaskista tapahtumat (alku, loppu) ja suodattaa ne keston mukaan.
    """
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs

    labeled_array, num_features = label(mask_1d)
    events = []

    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        if len(indices) > 0:
            duration_samples = len(indices)
            if min_samples <= duration_samples <= max_samples:
                events.append((indices[0], indices[-1]))  # (start_index, end_index)
            else:
                log.debug(f"Discarding event: duration {duration_samples / fs:.2f}s is outside range.")

    return events


def _calculate_iou(event1: Tuple[int, int], event2: Tuple[int, int]) -> float:
    """Laskee Intersection over Union (IoU) kahdelle 1D-tapahtumalle."""
    start1, end1 = event1
    start2, end2 = event2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _stitch_predictions(all_preds: torch.Tensor, step_samples: int) -> torch.Tensor:
    """
    Ompelee (stitches) limittäiset ikkuna-ennusteet yhteen käyttämällä
    painotettua keskiarvoa (Hann-ikkuna) saumojen poistamiseksi.
    """
    num_windows, _, height, window_len = all_preds.shape

    all_preds = all_preds.cpu().float()

    final_len = (num_windows - 1) * step_samples + window_len

    stitched_sum = torch.zeros((1, 1, height, final_len), dtype=torch.float32)
    stitched_weights = torch.zeros((1, 1, height, final_len), dtype=torch.float32)

    window_weights_1d = torch.hann_window(window_len, periodic=False)

    window_weights = window_weights_1d.view(1, 1, 1, window_len)

    log.debug(f"Stitching {num_windows} windows. Final length: {final_len} samples.")

    for i in range(num_windows):
        start = i * step_samples
        end = start + window_len

        stitched_sum[:, :, :, start:end] += all_preds[i:i + 1] * window_weights

        stitched_weights[:, :, :, start:end] += window_weights

    stitched_weights[stitched_weights == 0] = 1e-6

    stitched_pred = stitched_sum / stitched_weights

    return stitched_pred


# --- PÄÄFUNKTIO (KORJATTU) ---

def compute_event_based_metrics(model, data_loader, threshold: float = 0.5) -> Dict[str, float]:
    """
    Laskee segmentoinnin metriikat TAPAHTUMAPOHJAISESTI.
    Ottaa nyt vastaan kynnysarvon (threshold).
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model.to(device)
    model.eval()  # Asettaa mallin eval-tilaan (tärkeää augmentaation poistamiseksi)

    total_tp, total_fp, total_fn = 0, 0, 0
    iou_scores_of_tps = []

    iou_threshold = METRIC_PARAMS['iou_threshold']
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    log.info(
        f"Stitching predictions and calculating EVENT-BASED metrics (IoU threshold: {iou_threshold}, Decision threshold: {threshold:.2f})...")

    all_preds_list = []
    all_masks_list = []
    with torch.no_grad():
        # --- KORJAUS 1: Purkaa 3 arvoa (ignoraa 1D-signaalin) ---
        for images_seq, masks_2d, _ in tqdm(data_loader, desc="1/3: Running Inference"):
            images_seq = images_seq.to(device)

            seg_logits = model(images_seq)
            outputs_2d = torch.sigmoid(seg_logits)

            preds_2d = (outputs_2d > threshold).float()

            all_preds_list.append(preds_2d.cpu())
            all_masks_list.append(masks_2d.cpu())

    all_preds_tensor = torch.cat(all_preds_list, dim=0)
    all_masks_tensor = torch.cat(all_masks_list, dim=0)

    log.info("2/3: Stitching predictions into continuous signal...")

    h_out, w_out = all_preds_tensor.shape[2], all_preds_tensor.shape[3]
    all_masks_tensor_resized = F.interpolate(all_masks_tensor, size=(h_out, w_out), mode='nearest')

    stitched_pred_2d = _stitch_predictions(all_preds_tensor, step_samples)
    stitched_mask_2d = _stitch_predictions(all_masks_tensor_resized, step_samples)

    log.info("Converting 2D stitched images to 1D time series...")
    mask_1d_true = _internal_convert_2d_mask_to_1d(stitched_mask_2d.squeeze().numpy())
    mask_1d_pred = _internal_convert_2d_mask_to_1d(stitched_pred_2d.squeeze().numpy())

    true_events = _get_events_from_mask(mask_1d_true, fs)
    pred_events = _get_events_from_mask(mask_1d_pred, fs)

    log.info(f"3/3: Comparing events... Found {len(true_events)} true events and {len(pred_events)} predicted events.")

    # (TP/FP/FN-laskenta pysyy samana)
    matched_true_events = []
    for pred in pred_events:
        found_match = False
        best_iou = 0
        best_true_idx = -1

        for j, true in enumerate(true_events):
            if j in matched_true_events:
                continue
            iou = _calculate_iou(pred, true)
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_true_idx = j
                found_match = True

        if found_match:
            total_tp += 1
            matched_true_events.append(best_true_idx)
            iou_scores_of_tps.append(best_iou)
        else:
            total_fp += 1

    total_fn = len(true_events) - len(matched_true_events)

    epsilon = 1e-6
    precision = (total_tp + epsilon) / (total_tp + total_fp + epsilon)
    recall = (total_tp + epsilon) / (total_tp + total_fn + epsilon)
    dice_f1 = (2 * total_tp + epsilon) / (2 * total_tp + total_fp + total_fn + epsilon)
    iou_jaccard = (total_tp + epsilon) / (total_tp + total_fp + total_fn + epsilon)
    miou = np.mean(iou_scores_of_tps) if iou_scores_of_tps else 0.0

    return {
        "F1-score": dice_f1,
        "mIoU (TPs only)": miou,
        "Jaccard (Event IoU)": iou_jaccard,
        "Precision": precision,
        "Recall": recall,
        "TP (events)": total_tp,
        "FP (events)": total_fp,
        "FN (events)": total_fn
    }


# --- TÄMÄ FUNKTIO KORJATTU ---
def find_optimal_threshold(model, val_loader) -> float:
    """
    Ajaa mallin validointidatan läpi ja etsii parhaan
    kynnysarvon, joka maksimoi F1-pisteet.
    """
    log.info("Finding optimal decision threshold using validation data...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model.to(device)
    model.eval()

    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    all_probs_list = []
    all_masks_list = []
    with torch.no_grad():
        # --- KORJAUS 2: Purkaa 3 arvoa ---
        for images_seq, masks_2d, _ in tqdm(val_loader, desc="Optimizing Threshold"):
            images_seq = images_seq.to(device)

            seg_logits = model(images_seq)
            outputs_2d_probs = torch.sigmoid(seg_logits)

            all_probs_list.append(outputs_2d_probs.cpu())
            all_masks_list.append(masks_2d.cpu())

    all_probs_tensor = torch.cat(all_probs_list, dim=0)
    all_masks_tensor = torch.cat(all_masks_list, dim=0)

    h_out, w_out = all_probs_tensor.shape[2], all_probs_tensor.shape[3]
    all_masks_tensor_resized = F.interpolate(all_masks_tensor, size=(h_out, w_out), mode='nearest')

    stitched_prob_2d = _stitch_predictions(all_probs_tensor, step_samples)
    stitched_mask_2d = _stitch_predictions(all_masks_tensor_resized, step_samples)

    mask_1d_true = _internal_convert_2d_mask_to_1d(stitched_mask_2d.squeeze().numpy())
    true_events = _get_events_from_mask(mask_1d_true, fs)

    best_f1 = -1.0
    best_threshold = 0.5

    search_space = np.arange(0.2, 0.95, 0.05)
    log.info(f"Testing {len(search_space)} thresholds: {np.round(search_space, 2)}")

    # (Loppuosa F1-laskennasta pysyy samana)
    for threshold in search_space:
        mask_1d_pred = _internal_convert_2d_mask_to_1d((stitched_prob_2d.squeeze().numpy() > threshold))
        pred_events = _get_events_from_mask(mask_1d_pred, fs)
        total_tp, total_fp = 0, 0
        matched_true_events = []

        for pred in pred_events:
            found_match = False
            for j, true in enumerate(true_events):
                if j in matched_true_events:
                    continue
                iou = _calculate_iou(pred, true)
                if iou > METRIC_PARAMS['iou_threshold']:
                    matched_true_events.append(j)
                    found_match = True
                    break
            if found_match:
                total_tp += 1
            else:
                total_fp += 1

        total_fn = len(true_events) - len(matched_true_events)
        epsilon = 1e-6
        f1_score = (2 * total_tp + epsilon) / (2 * total_tp + total_fp + total_fn + epsilon)
        log.debug(f"Threshold {threshold:.2f} -> F1: {f1_score:.4f} (TP:{total_tp}, FP:{total_fp}, FN:{total_fn})")

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

    log.info(f"Optimal threshold found: {best_threshold:.2f} (F1-score: {best_f1:.4f})")
    return best_threshold
# UNET_model/evaluation_metrics.py

import logging
import torch
import numpy as np
from scipy.ndimage import label, find_objects
from tqdm import tqdm
from typing import List, Dict, Tuple
from training_parameters import CWT_PARAMS, METRIC_PARAMS, DATA_PARAMS
import torch.nn.functional as F

log = logging.getLogger(__name__)

FIXED_BORDER_THRESH = 0.2

def _find_events_dual_thresh(prob_1d: np.ndarray,
                             peak_thresh: float,
                             border_thresh: float,
                             fs: float) -> List[Tuple[int, int]]:
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs

    border_mask = prob_1d > border_thresh
    labeled_borders, num_border_regions = label(border_mask)

    if num_border_regions == 0:
        return []

    peak_mask = prob_1d > peak_thresh
    labels_of_peaks = labeled_borders[peak_mask]
    valid_region_labels = np.unique(labels_of_peaks[labels_of_peaks > 0])

    if len(valid_region_labels) == 0:
        return []

    valid_slices = find_objects(labeled_borders)

    events = []
    for region_label in valid_region_labels:
        s = valid_slices[region_label - 1]
        start, end = s[0].start, s[0].stop

        duration_samples = end - start

        if min_samples <= duration_samples <= max_samples:
            events.append((start, end - 1))

    events.sort(key=lambda x: x[0])

    return events


def _internal_convert_2d_mask_to_1d(mask_2d: np.ndarray) -> np.ndarray:
    frequencies = np.linspace(CWT_PARAMS['freq_low'], CWT_PARAMS['freq_high'], CWT_PARAMS['freq_bins'])
    y_indices = np.where(
        (frequencies >= METRIC_PARAMS['spindle_freq_low']) &
        (frequencies <= METRIC_PARAMS['spindle_freq_high'])
    )[0]
    if len(y_indices) == 0:
        return np.zeros(mask_2d.shape[1], dtype=bool)
    spindle_band = mask_2d[y_indices, :]
    return np.any(spindle_band, axis=0)


def _internal_convert_2d_prob_to_1d(prob_2d: np.ndarray) -> np.ndarray:
    frequencies = np.linspace(CWT_PARAMS['freq_low'], CWT_PARAMS['freq_high'], CWT_PARAMS['freq_bins'])
    y_indices = np.where(
        (frequencies >= METRIC_PARAMS['spindle_freq_low']) &
        (frequencies <= METRIC_PARAMS['spindle_freq_high'])
    )[0]
    if len(y_indices) == 0:
        log.warning("No spindle frequency bins found for probability conversion.")
        return np.zeros(prob_2d.shape[1], dtype=prob_2d.dtype)

    spindle_band_probs = prob_2d[y_indices, :]
    return np.max(spindle_band_probs, axis=0)


def _calculate_iou(event1: Tuple[int, int], event2: Tuple[int, int]) -> float:
    start1, end1 = event1
    start2, end2 = event2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _stitch_predictions(all_preds: torch.Tensor, step_samples: int) -> torch.Tensor:
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


def compute_event_based_metrics(model, data_loader, threshold: float) -> Dict[str, float]:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model.to(device)
    model.eval()

    total_tp, total_fp, total_fn = 0, 0, 0
    iou_scores_of_tps = []
    iou_threshold = METRIC_PARAMS['iou_threshold']
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    peak_thresh = threshold
    border_thresh = FIXED_BORDER_THRESH
    log.info(
        f"Stitching predictions and calculating EVENT-BASED metrics (IoU: {iou_threshold}, Peak Thresh: {peak_thresh:.2f}, Border Thresh: {border_thresh:.2f})...")

    all_probs_list = []
    all_masks_list = []
    with torch.no_grad():
        for images_seq, masks_2d, _ in tqdm(data_loader, desc="1/3: Running Inference"):
            images_seq = images_seq.to(device)
            seg_logits = model(images_seq)
            outputs_2d_probs = torch.sigmoid(seg_logits)
            all_probs_list.append(outputs_2d_probs.cpu())
            all_masks_list.append(masks_2d.cpu())

    all_probs_tensor = torch.cat(all_probs_list, dim=0)
    all_masks_tensor = torch.cat(all_masks_list, dim=0)

    log.info("2/3: Stitching predictions into continuous signal...")
    h_out, w_out = all_probs_tensor.shape[2], all_probs_tensor.shape[3]
    all_masks_tensor_resized = F.interpolate(all_masks_tensor, size=(h_out, w_out), mode='nearest')

    stitched_prob_2d = _stitch_predictions(all_probs_tensor, step_samples)
    stitched_mask_2d = _stitch_predictions(all_masks_tensor_resized, step_samples)

    log.info("Converting 2D stitched images to 1D time series...")
    mask_1d_true_bool = _internal_convert_2d_mask_to_1d(stitched_mask_2d.squeeze().numpy())
    prob_1d_pred = _internal_convert_2d_prob_to_1d(stitched_prob_2d.squeeze().numpy())

    true_events = _find_events_dual_thresh(mask_1d_true_bool.astype(float), 0.5, 0.1, fs)
    pred_events = _find_events_dual_thresh(prob_1d_pred, peak_thresh, border_thresh, fs)

    log.info(f"3/3: Comparing events... Found {len(true_events)} true events and {len(pred_events)} predicted events.")

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
        "F1-score": dice_f1, "mIoU (TPs only)": miou, "Jaccard (Event IoU)": iou_jaccard,
        "Precision": precision, "Recall": recall, "TP (events)": total_tp,
        "FP (events)": total_fp, "FN (events)": total_fn
    }


def find_optimal_threshold(model, val_loader) -> float:
    log.info(f"Finding optimal PEAK threshold (using fixed border thresh: {FIXED_BORDER_THRESH})...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    all_probs_list = []
    all_masks_list = []
    with torch.no_grad():
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

    mask_1d_true_bool = _internal_convert_2d_mask_to_1d(stitched_mask_2d.squeeze().numpy())
    true_events = _find_events_dual_thresh(mask_1d_true_bool.astype(float), 0.5, 0.1, fs)  # Ground truth

    prob_1d_series = _internal_convert_2d_prob_to_1d(stitched_prob_2d.squeeze().numpy())  # Ennuste

    best_f1 = -1.0
    best_peak_threshold = 0.5
    border_thresh = FIXED_BORDER_THRESH

    search_space = np.arange(border_thresh + 0.05, 0.95, 0.05)
    log.info(f"Testing {len(search_space)} PEAK thresholds: {np.round(search_space, 2)}")

    for peak_thresh in search_space:

        pred_events = _find_events_dual_thresh(prob_1d_series, peak_thresh, border_thresh, fs)

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
        log.debug(f"Peak Thresh {peak_thresh:.2f} -> F1: {f1_score:.4f} (TP:{total_tp}, FP:{total_fp}, FN:{total_fn})")

        if f1_score > best_f1:
            best_f1 = f1_score
            best_peak_threshold = peak_thresh

    log.info(f"Optimal PEAK threshold found: {best_peak_threshold:.2f} (F1-score: {best_f1:.4f})")
    return best_peak_threshold
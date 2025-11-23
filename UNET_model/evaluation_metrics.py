# UNET_model/evaluation_metrics.py

import logging
import torch
import numpy as np
from scipy.ndimage import label, find_objects
from tqdm import tqdm
from typing import List, Dict, Tuple
from config import METRIC_PARAMS, DATA_PARAMS
import gc

log = logging.getLogger(__name__)

FIXED_BORDER_THRESH = 0.2


# --- POISTETTU: _verify_spindle_frequency (Aiheutti Recall-romahduksen) ---

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
    labels_with_peaks = np.unique(labeled_borders[peak_mask])
    labels_with_peaks = labels_with_peaks[labels_with_peaks > 0]

    valid_slices = find_objects(labeled_borders)
    events = []

    for label_idx in labels_with_peaks:
        s = valid_slices[label_idx - 1]
        start, end = s[0].start, s[0].stop
        duration = end - start

        if min_samples <= duration <= max_samples:
            events.append((start, end - 1))

    events.sort(key=lambda x: x[0])
    return events


def _calculate_iou(event1: Tuple[int, int], event2: Tuple[int, int]) -> float:
    start1, end1 = event1
    start2, end2 = event2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _stitch_predictions_1d(all_preds: torch.Tensor, step_samples: int) -> np.ndarray:
    num_windows, _, window_len = all_preds.shape
    final_len = (num_windows - 1) * step_samples + window_len
    stitched_sum = torch.zeros(final_len, dtype=torch.float32)
    stitched_weights = torch.zeros(final_len, dtype=torch.float32)
    window_weights = torch.hann_window(window_len, periodic=False)
    preds_flat = all_preds.squeeze(1).cpu()

    for i in range(num_windows):
        start = i * step_samples
        end = start + window_len
        stitched_sum[start:end] += preds_flat[i] * window_weights
        stitched_weights[start:end] += window_weights

    stitched_weights[stitched_weights == 0] = 1e-6
    return (stitched_sum / stitched_weights).numpy()


def compute_event_based_metrics(model, data_loader, threshold: float) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model.to(device)
    model.eval()

    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    all_probs_list = []
    all_masks_list = []

    with torch.no_grad():
        for inputs, masks in tqdm(data_loader, desc="Evaluating Events"):
            inputs = inputs.to(device)

            # --- NOVELTY: TEST TIME AUGMENTATION (TTA) ---
            # 1. Normaali ennuste
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # 2. Flipattu ennuste (Peilikuva)
            inputs_flipped = torch.flip(inputs, dims=[2])
            logits_flipped = model(inputs_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)
            probs_flipped = torch.flip(probs_flipped, dims=[2])  # Käännetään takaisin

            # Keskiarvo (Vähentää satunnaista kohinaa)
            probs_avg = (probs + probs_flipped) / 2.0

            all_probs_list.append(probs_avg.cpu().float())
            all_masks_list.append(masks.cpu().float())

    all_probs_tensor = torch.cat(all_probs_list, dim=0)
    all_masks_tensor = torch.cat(all_masks_list, dim=0).unsqueeze(1)

    del all_probs_list, all_masks_list
    gc.collect()

    log.info("Stitching predictions...")
    prob_1d_full = _stitch_predictions_1d(all_probs_tensor, step_samples)
    mask_1d_full = _stitch_predictions_1d(all_masks_tensor, step_samples)

    mask_1d_bool = mask_1d_full >= 0.4

    log.info(f"Finding events (Peak Thresh: {threshold:.2f})...")
    pred_events = _find_events_dual_thresh(prob_1d_full, threshold, FIXED_BORDER_THRESH, fs)
    true_events = _find_events_dual_thresh(mask_1d_bool.astype(float), 0.5, 0.1, fs)

    log.info(f"Found {len(true_events)} true events and {len(pred_events)} predicted events.")

    total_tp, total_fp = 0, 0
    matched_true_indices = set()
    iou_scores = []

    for pred in pred_events:
        best_iou = 0
        best_idx = -1

        for i, true_ev in enumerate(true_events):
            if i in matched_true_indices: continue

            iou = _calculate_iou(pred, true_ev)
            if iou > 0 and iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= METRIC_PARAMS['iou_threshold']:
            total_tp += 1
            matched_true_indices.add(best_idx)
            iou_scores.append(best_iou)
        else:
            total_fp += 1

    total_fn = len(true_events) - len(matched_true_indices)

    epsilon = 1e-6
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0

    return {
        "F1-score": f1,
        "Precision": precision,
        "Recall": recall,
        "TP (events)": total_tp,
        "FP (events)": total_fp,
        "FN (events)": total_fn,
        "mIoU (TPs)": mean_iou
    }


def find_optimal_threshold(model, val_loader) -> float:
    log.info("Finding optimal threshold with TTA...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model.to(device)
    model.eval()

    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    all_probs_list = []
    all_masks_list = []

    with torch.no_grad():
        for inputs, masks in tqdm(val_loader, desc="Optimizing"):
            inputs = inputs.to(device)

            # TTA Validointivaiheessa myös!
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            inputs_flipped = torch.flip(inputs, dims=[2])
            logits_flipped = model(inputs_flipped)
            probs_flipped = torch.flip(torch.sigmoid(logits_flipped), dims=[2])

            probs_avg = (probs + probs_flipped) / 2.0

            all_probs_list.append(probs_avg.cpu().float())
            all_masks_list.append(masks.cpu().float())

    all_probs_tensor = torch.cat(all_probs_list, dim=0)
    all_masks_tensor = torch.cat(all_masks_list, dim=0).unsqueeze(1)

    prob_1d_full = _stitch_predictions_1d(all_probs_tensor, step_samples)
    mask_1d_full = _stitch_predictions_1d(all_masks_tensor, step_samples)

    mask_1d_bool = mask_1d_full >= 0.4
    true_events = _find_events_dual_thresh(mask_1d_bool.astype(float), 0.5, 0.1, fs)

    best_f1 = 0.0
    best_thresh = 0.5
    thresholds = np.arange(0.2, 0.96, 0.05)

    for th in thresholds:
        pred_events = _find_events_dual_thresh(prob_1d_full, th, FIXED_BORDER_THRESH, fs)

        tp = 0
        matched = set()
        for p in pred_events:
            for i, t in enumerate(true_events):
                if i in matched: continue
                if _calculate_iou(p, t) >= METRIC_PARAMS['iou_threshold']:
                    tp += 1
                    matched.add(i)
                    break

        fp = len(pred_events) - tp
        fn = len(true_events) - tp
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    log.info(f"Optimal threshold found: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
    return best_thresh
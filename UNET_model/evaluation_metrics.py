# UNET_model/evaluation_metrics.py

import logging
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects
from scipy.signal import welch
from tqdm import tqdm
from typing import List, Dict, Tuple
from config import METRIC_PARAMS, DATA_PARAMS
import gc
import os

log = logging.getLogger(__name__)

FIXED_BORDER_THRESH = 0.5

def analyze_signal_properties(signal_segment: np.ndarray, fs: float):
    if len(signal_segment) < int(0.1 * fs): return 0.0, 0.0, 0.0
    nperseg = min(len(signal_segment), 256)
    try:
        freqs, psd = welch(signal_segment, fs, nperseg=nperseg)
    except Exception:
        return 0.0, 0.0, 0.0
    idx_sigma = np.where((freqs >= 9.0) & (freqs <= 16.0))[0]
    if len(idx_sigma) == 0: return 0.0, 0.0, 0.0
    peak_idx = idx_sigma[np.argmax(psd[idx_sigma])]
    peak_freq = freqs[peak_idx]
    mean_sigma_power = np.mean(psd[idx_sigma])
    total_power = np.sum(psd)
    relative_power = mean_sigma_power / total_power if total_power > 0 else 0.0
    return peak_freq, mean_sigma_power, relative_power


def generate_detailed_csv(true_events, pred_events, raw_signal_1d, probs_1d, fs, subject_id, output_dir):
    data_rows = []
    matched_true_indices = set()
    for pred_idx, pred in enumerate(pred_events):
        start, end = pred
        start, end = max(0, start), min(len(raw_signal_1d), end)
        pred_signal = raw_signal_1d[start:end]
        pred_probs = probs_1d[start:end]
        peak_freq, sigma_power, rel_power = analyze_signal_properties(pred_signal, fs)
        max_conf = np.max(pred_probs) if len(pred_probs) > 0 else 0.0
        best_iou = 0.0
        match_type = "FP"
        matched_true_idx = -1
        for t_idx, true_ev in enumerate(true_events):
            iou = _calculate_iou(pred, true_ev)
            if iou > best_iou:
                best_iou = iou
                matched_true_idx = t_idx
        if best_iou >= METRIC_PARAMS['iou_threshold']:
            match_type = "TP"
            if matched_true_idx != -1: matched_true_indices.add(matched_true_idx)
        data_rows.append(
            {'Subject': subject_id, 'Event_Type': match_type, 'Start_s': start / fs, 'Duration_s': (end - start) / fs,
             'IoU': best_iou, 'Model_Confidence': max_conf, 'Peak_Freq_Hz': peak_freq, 'Sigma_Power': sigma_power,
             'Relative_Power': rel_power, 'Notes': ''})

    for t_idx, true_ev in enumerate(true_events):
        if t_idx not in matched_true_indices:
            start, end = true_ev
            start, end = max(0, start), min(len(raw_signal_1d), end)
            true_signal = raw_signal_1d[start:end]
            peak_freq, sigma_power, rel_power = analyze_signal_properties(true_signal, fs)
            data_rows.append(
                {'Subject': subject_id, 'Event_Type': 'FN', 'Start_s': start / fs, 'Duration_s': (end - start) / fs,
                 'IoU': 0.0, 'Model_Confidence': 0.0, 'Peak_Freq_Hz': peak_freq, 'Sigma_Power': sigma_power,
                 'Relative_Power': rel_power, 'Notes': 'Missed'})

    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(os.path.join(output_dir, f"error_analysis_{subject_id}.csv"), index=False)


def _verify_spindle_power(signal_segment: np.ndarray, fs: float, threshold_ratio: float = 0.15) -> bool:
    n = len(signal_segment)
    if n < int(0.3 * fs): return False
    nperseg = min(n, 128)
    try:
        freqs, psd = welch(signal_segment, fs, nperseg=nperseg)
    except Exception:
        return False
    idx_sigma = np.where((freqs >= 11.0) & (freqs <= 16.0))[0]
    idx_total = np.where((freqs >= 0.5) & (freqs <= 30.0))[0]
    if len(idx_sigma) < 2 or len(idx_total) < 2: return False
    power_sigma = np.trapz(psd[idx_sigma], freqs[idx_sigma])
    power_total = np.trapz(psd[idx_total], freqs[idx_total])
    if power_total == 0: return False
    return (power_sigma / power_total) >= threshold_ratio


def _merge_close_events(events: List[Tuple[int, int]], fs: float, gap_thresh_sec: float = 0.3) -> List[Tuple[int, int]]:
    if not events: return []
    gap_samples = int(gap_thresh_sec * fs)
    merged = []
    curr_start, curr_end = events[0]
    for i in range(1, len(events)):
        next_start, next_end = events[i]
        if (next_start - curr_end) < gap_samples:
            curr_end = next_end
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged


def _find_events_dual_thresh(prob_1d: np.ndarray, peak_thresh: float, border_thresh: float, fs: float,
                             raw_signal: np.ndarray = None) -> List[Tuple[int, int]]:
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs
    border_mask = prob_1d > border_thresh
    labeled_borders, num_border_regions = label(border_mask)
    if num_border_regions == 0: return []
    peak_mask = prob_1d > peak_thresh
    labels_with_peaks = np.unique(labeled_borders[peak_mask])
    labels_with_peaks = labels_with_peaks[labels_with_peaks > 0]
    valid_slices = find_objects(labeled_borders)
    raw_events = []
    for label_idx in labels_with_peaks:
        s = valid_slices[label_idx - 1]
        raw_events.append((s[0].start, s[0].stop - 1))
    raw_events.sort(key=lambda x: x[0])
    merged_events = _merge_close_events(raw_events, fs, gap_thresh_sec=0.3)
    final_events = []
    for start, end in merged_events:
        duration = end - start
        if min_samples <= duration <= max_samples:
            if raw_signal is not None:
                segment = raw_signal[start:end]
                if _verify_spindle_power(segment, fs, threshold_ratio=0.15):
                    final_events.append((start, end))
            else:
                final_events.append((start, end))
    return final_events


def _calculate_iou(event1, event2):
    start1, end1 = event1
    start2, end2 = event2
    intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
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


def compute_event_based_metrics(model,
                                data_loader,
                                threshold: float,
                                subject_id: str = "unknown",
                                output_dir: str = ".",
                                use_power_check: bool = False) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')
    model.to(device);
    model.eval()
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)
    all_probs_list, all_masks_list, raw_signal_list = [], [], []

    with torch.no_grad():
        for inputs, masks, labels in tqdm(data_loader, desc=f"Evaluating ({subject_id})"):
            inputs = inputs.to(device)
            mask_logits, gate_logits = model(inputs)
            final_prob = torch.sigmoid(mask_logits) * torch.sigmoid(gate_logits).unsqueeze(2)

            # TTA
            inputs_flip = torch.flip(inputs, dims=[2])
            m_f, g_f = model(inputs_flip)
            final_prob_flip = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)

            avg_prob = (final_prob + final_prob_flip) / 2.0
            all_probs_list.append(avg_prob.cpu().float())
            all_masks_list.append(masks.cpu().float())
            raw_signal_list.append(inputs[:, 0, :].cpu().float())

    all_probs = torch.cat(all_probs_list, dim=0)
    all_masks = torch.cat(all_masks_list, dim=0).unsqueeze(1)
    all_raw = torch.cat(raw_signal_list, dim=0).unsqueeze(1)
    del all_probs_list, all_masks_list, raw_signal_list;
    gc.collect()

    prob_1d = _stitch_predictions_1d(all_probs, step_samples)
    mask_1d = _stitch_predictions_1d(all_masks, step_samples)
    raw_1d = _stitch_predictions_1d(all_raw, step_samples)

    log.info(f"Finding events (Thresh: {threshold:.2f}, PowerCheck: {use_power_check})...")

    signal_for_check = raw_1d if use_power_check else None

    pred_events = _find_events_dual_thresh(prob_1d, threshold, FIXED_BORDER_THRESH, fs, raw_signal=signal_for_check)
    true_events = _find_events_dual_thresh((mask_1d >= 0.4).astype(float), 0.5, 0.1, fs, raw_signal=None)

    log.info(f"Found {len(true_events)} true, {len(pred_events)} predicted.")

    try:
        generate_detailed_csv(true_events, pred_events, raw_1d, prob_1d, fs, subject_id, output_dir)
    except Exception as e:
        log.error(f"CSV gen error: {e}")

    # Metrics calculation
    tp = 0
    matched = set()
    iou_scores = []
    for p in pred_events:
        best_iou = 0
        best_idx = -1
        for i, t in enumerate(true_events):
            if i in matched: continue
            iou = _calculate_iou(p, t)
            if iou > best_iou: best_iou = iou; best_idx = i
        if best_iou >= METRIC_PARAMS['iou_threshold']:
            tp += 1;
            matched.add(best_idx);
            iou_scores.append(best_iou)

    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    eps = 1e-6
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * (prec * rec) / (prec + rec + eps)

    return {"F1-score": f1, "Precision": prec, "Recall": rec, "TP (events)": tp, "FP (events)": fp, "FN (events)": fn,
            "mIoU (TPs)": np.mean(iou_scores) if iou_scores else 0.0}


def find_optimal_threshold(model, val_loader) -> float:
    return 0.35  # Placeholder
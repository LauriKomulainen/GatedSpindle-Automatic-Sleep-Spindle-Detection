# core/reporting.py

import numpy as np
import pandas as pd
import os
from scipy.signal import welch
from configs.dreams_config import METRIC_PARAMS
from postprocessing.postprocessing import calculate_iou

def analyze_signal_properties(signal_segment: np.ndarray, fs: float):
    """Analyzes spectral properties of a signal segment."""
    if len(signal_segment) < int(0.1 * fs):
        return 0.0, 0.0, 0.0, 0.0

    nperseg = min(len(signal_segment), 256)
    try:
        freqs, psd = welch(signal_segment, fs, nperseg=nperseg)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

    # Sigma band (9-16Hz)
    idx_sigma = np.where((freqs >= 9.0) & (freqs <= 16.0))[0]

    if len(idx_sigma) == 0:
        return 0.0, 0.0, 0.0, 0.0

    peak_idx = idx_sigma[np.argmax(psd[idx_sigma])]
    peak_freq = freqs[peak_idx]
    mean_sigma_power = np.mean(psd[idx_sigma])

    total_power = np.sum(psd)
    relative_power = mean_sigma_power / total_power if total_power > 0 else 0.0

    return peak_freq, mean_sigma_power, relative_power, total_power

def generate_detailed_csv(true_events, pred_events, raw_signal_1d, probs_1d, fs, subject_id, output_dir):
    data_rows = []
    matched_true_indices = set()

    # 1. Analyze Predicted Events (TP & FP)
    for pred_idx, pred in enumerate(pred_events):
        start, end = pred
        start, end = max(0, start), min(len(raw_signal_1d), end)

        pred_signal = raw_signal_1d[start:end]
        pred_probs = probs_1d[start:end]

        # Signal Metrics
        peak_freq, sigma_power, rel_power, bg_power = analyze_signal_properties(pred_signal, fs)

        # Confidence Metrics
        max_conf = np.max(pred_probs) if len(pred_probs) > 0 else 0.0
        mean_conf = np.mean(pred_probs) if len(pred_probs) > 0 else 0.0

        # Border Confidence
        if len(pred_probs) >= 2:
            border_conf = (pred_probs[0] + pred_probs[-1]) / 2.0
        elif len(pred_probs) == 1:
            border_conf = pred_probs[0]
        else:
            border_conf = 0.0

        # Matching to Ground Truth
        best_iou = 0.0
        match_type = "FP"
        matched_true_idx = -1

        for t_idx, true_ev in enumerate(true_events):
            iou = calculate_iou(pred, true_ev)
            if iou > best_iou:
                best_iou = iou
                matched_true_idx = t_idx

        if best_iou >= METRIC_PARAMS['iou_threshold']:
            match_type = "TP"
            if matched_true_idx != -1:
                matched_true_indices.add(matched_true_idx)

        data_rows.append({
            'Subject': subject_id,
            'Event_Type': match_type,
            'Start_s': start / fs,
            'Duration_s': (end - start) / fs,
            'IoU': best_iou,
            'Model_Confidence_Max': max_conf,
            'Model_Confidence_Mean': mean_conf,
            'Border_Confidence': border_conf,
            'Peak_Freq_Hz': peak_freq,
            'Sigma_Power': sigma_power,
            'Relative_Power': rel_power,
            'Background_Power': bg_power,
            'Notes': ''
        })

    # 2. Analyze Missed Events (FN)
    for t_idx, true_ev in enumerate(true_events):
        if t_idx not in matched_true_indices:
            start, end = true_ev
            start, end = max(0, start), min(len(raw_signal_1d), end)

            true_signal = raw_signal_1d[start:end]
            peak_freq, sigma_power, rel_power, bg_power = analyze_signal_properties(true_signal, fs)

            missed_probs = probs_1d[start:end]
            max_conf = np.max(missed_probs) if len(missed_probs) > 0 else 0.0
            mean_conf = np.mean(missed_probs) if len(missed_probs) > 0 else 0.0

            data_rows.append({
                'Subject': subject_id,
                'Event_Type': 'FN',
                'Start_s': start / fs,
                'Duration_s': (end - start) / fs,
                'IoU': 0.0,
                'Model_Confidence_Max': max_conf,
                'Model_Confidence_Mean': mean_conf,
                'Border_Confidence': 0.0,
                'Peak_Freq_Hz': peak_freq,
                'Sigma_Power': sigma_power,
                'Relative_Power': rel_power,
                'Background_Power': bg_power,
                'Notes': 'Missed'
            })

    if data_rows:
        df = pd.DataFrame(data_rows)
        cols = ['Subject', 'Event_Type', 'Start_s', 'Duration_s', 'IoU',
                'Model_Confidence_Max', 'Model_Confidence_Mean', 'Border_Confidence',
                'Relative_Power', 'Background_Power', 'Sigma_Power', 'Peak_Freq_Hz', 'Notes']
        df = df[cols]
        df.to_csv(os.path.join(output_dir, f"error_analysis_{subject_id}.csv"), index=False)
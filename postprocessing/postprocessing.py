# post_processing/postprocessing.py

import torch
import numpy as np
from scipy.ndimage import label, find_objects
from typing import List, Tuple
from configs.dreams_config import METRIC_PARAMS, POST_PROCESSING_PARAMS

def merge_close_events(events: List[Tuple[int, int]], fs: float, gap_thresh_sec) -> List[Tuple[int, int]]:
    """Yhdistää tapahtumat, jotka ovat hyvin lähellä toisiaan."""
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

def find_events_dual_thresh(prob_1d: np.ndarray, peak_thresh: float, border_thresh: float, fs: float) -> List[Tuple[int, int]]:
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs

    # 1. Find regions > border_thresh
    border_mask = prob_1d > border_thresh
    labeled_borders, num_border_regions = label(border_mask)
    if num_border_regions == 0: return []

    # 2. Filter regions that don't contain a peak > peak_thresh
    peak_mask = prob_1d > peak_thresh
    labels_with_peaks = np.unique(labeled_borders[peak_mask])
    labels_with_peaks = labels_with_peaks[labels_with_peaks > 0]

    valid_slices = find_objects(labeled_borders)
    raw_events = []
    for label_idx in labels_with_peaks:
        s = valid_slices[label_idx - 1]
        raw_events.append((s[0].start, s[0].stop - 1))

    raw_events.sort(key=lambda x: x[0])
    gap_thresh_sec = POST_PROCESSING_PARAMS['gap_thresh_sec']
    
    # 3. Merge close events
    merged_events = merge_close_events(raw_events, fs, gap_thresh_sec)

    # 4. Filter by duration
    final_events = []
    for start, end in merged_events:
        duration = end - start
        if min_samples <= duration <= max_samples:
            final_events.append((start, end))

    return final_events

def stitch_predictions_1d(all_preds: torch.Tensor, step_samples: int) -> np.ndarray:
    num_windows, _, window_len = all_preds.shape
    final_len = (num_windows - 1) * step_samples + window_len
    stitched_sum = torch.zeros(final_len, dtype=torch.float32)
    stitched_weights = torch.zeros(final_len, dtype=torch.float32)

    # Center crop logic
    margin = (window_len - step_samples) // 2
    window_weights = torch.zeros(window_len, dtype=torch.float32)
    if margin < window_len // 2:
        window_weights[margin: window_len - margin] = 1.0
    else:
        window_weights[:] = 1.0

    preds_flat = all_preds.squeeze(1).cpu()
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_len
        stitched_sum[start:end] += preds_flat[i] * window_weights
        stitched_weights[start:end] += window_weights

    stitched_weights[stitched_weights == 0] = 1.0
    return (stitched_sum / stitched_weights).numpy()

def calculate_iou(event1, event2):
    start1, end1 = event1
    start2, end2 = event2
    intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0.0
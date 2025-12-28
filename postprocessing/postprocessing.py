# post_processing/postprocessing.py

import torch
import numpy as np
from typing import List, Tuple
from configs.dreams_config import METRIC_PARAMS, POST_PROCESSING_PARAMS

def merge_close_events(events: List[Tuple[int, int]], fs: float, gap_thresh_sec) -> List[Tuple[int, int]]:
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


def find_events_dual_thresh(prob_1d: np.ndarray, peak_thresh: float, border_thresh: float, fs: float) -> List[
    Tuple[int, int]]:
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs
    gap_thresh_sec = POST_PROCESSING_PARAMS['gap_thresh_sec']

    candidates = (prob_1d > border_thresh).astype(int)
    diff = np.diff(candidates, prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    raw_events = []
    for start, end in zip(starts, ends):
        segment_probs = prob_1d[start:end]
        if np.max(segment_probs) >= peak_thresh:
            raw_events.append((start, end))

    if not raw_events:
        return []

    merged_events = merge_close_events(raw_events, fs, gap_thresh_sec)

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
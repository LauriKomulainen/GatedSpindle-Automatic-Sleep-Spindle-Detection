# postprocessing/postprocessing.py

import torch
import numpy as np
from typing import List, Tuple
from core.config_loader import POST_PROCESSING_PARAMS

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


def find_events_dual_thresh(prob_1d: np.ndarray, peak_thresh: float,
                            border_thresh, fs: float,
                            gap_thresh_sec=None) -> List[Tuple[int, int]]:
    """Extract events from a 1D probability/mask series.

    Parameters
    ----------
    prob_1d : np.ndarray
        Per-sample probability or binary mask series.
    peak_thresh : float
        Minimum peak probability required for an event to be retained.
    border_thresh : float or False
        If False, dual-thresholding is disabled and `peak_thresh` is used as
        the single thresholding boundary. If a float (e.g. 0.2), candidates
        are first formed at `border_thresh` and then filtered by
        `peak_thresh`.
    fs : float
        Sampling rate (Hz).
    gap_thresh_sec : float or None
        If a float, events closer than this many seconds are merged and
        events shorter than `min_duration_sec` are discarded. If None, no
        post-processing is applied: events are returned as-is, preserving
        the raw event boundaries. Use None for ground-truth annotations,
        which represent the scorer's intended event boundaries and should
        not be modified.
    """
    min_samples = POST_PROCESSING_PARAMS['min_duration_sec'] * fs

    # Single-threshold mode: border disabled
    if border_thresh is False or border_thresh is None:
        candidates = (prob_1d > peak_thresh).astype(int)
    else:
        candidates = (prob_1d > border_thresh).astype(int)

    diff = np.diff(candidates, prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    raw_events = []
    for start, end in zip(starts, ends):
        if border_thresh is False or border_thresh is None:
            # Already thresholded at peak_thresh, no extra check needed
            raw_events.append((start, end))
        else:
            segment_probs = prob_1d[start:end]
            if np.max(segment_probs) >= peak_thresh:
                raw_events.append((start, end))

    if not raw_events:
        return []

    # Merging and minimum-duration filtering are opt-in.
    # Ground-truth annotations (called with gap_thresh_sec=None) bypass both
    # steps so that expert markings are preserved exactly as scored.
    if gap_thresh_sec is not None:
        events_to_filter = merge_close_events(raw_events, fs, gap_thresh_sec)
        final_events = []
        for start, end in events_to_filter:
            duration = end - start
            if min_samples <= duration:
                final_events.append((start, end))
    else:
        final_events = raw_events

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
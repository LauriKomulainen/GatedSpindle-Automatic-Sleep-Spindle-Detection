# UNET_model/evaluation_metrics.py

import logging
import torch
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
from typing import List, Dict
from training_parameters import CWT_PARAMS, METRIC_PARAMS, DATA_PARAMS

log = logging.getLogger(__name__)

def _internal_convert_2d_mask_to_1d(mask_2d: np.ndarray) -> np.ndarray:
    """
    Muuntaa 2D-ennustusmaskin 1D-aikasarjaksi.
    """
    frequencies = np.linspace(
        CWT_PARAMS['freq_low'],
        CWT_PARAMS['freq_high'],
        CWT_PARAMS['freq_bins']
    )
    y_indices = np.where(
        (frequencies >= METRIC_PARAMS['spindle_freq_low']) &
        (frequencies <= METRIC_PARAMS['spindle_freq_high'])
    )[0]

    if len(y_indices) == 0:
        return np.zeros(mask_2d.shape[1], dtype=bool)

    spindle_band = mask_2d[y_indices, :]
    mask_1d = np.any(spindle_band, axis=0)
    return mask_1d


def _get_events_from_mask(mask_1d: np.ndarray) -> List[tuple]:
    """
    Muuntaa 1D-binäärimaskin listaksi (alku, loppu) -tapahtumia.
    Suodattaa pois tapahtumat, jotka ovat liian lyhyitä tai pitkiä.
    """
    fs = DATA_PARAMS['fs']
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs

    labeled_array, num_features = label(mask_1d)
    events = []

    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        if len(indices) > 0:
            duration_samples = len(indices)

            if duration_samples >= min_samples and duration_samples <= max_samples:
                events.append((indices[0], indices[-1]))

    return events



def _calculate_iou(event1: tuple, event2: tuple) -> float:
    """
    Laskee Intersection over Union (IoU) kahdelle 1D-tapahtumalle.
    """
    start1, end1 = event1
    start2, end2 = event2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def compute_event_based_metrics(model, data_loader) -> Dict[str, float]:
    """
    Laskee segmentoinnin metriikat TAPAHTUMAPOHJAISESTI (Event-based).
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    iou_threshold = METRIC_PARAMS['iou_threshold']

    log.info(f"Calculating EVENT-BASED metrics (IoU threshold: {iou_threshold})...")

    with torch.no_grad():
        for images_2d, masks_2d, _ in tqdm(data_loader, desc="Testing (Event-based)"):
            images_2d = images_2d.to(device)
            outputs_2d = model(images_2d)
            outputs_2d = torch.sigmoid(outputs_2d)
            preds_2d = (outputs_2d > 0.5).float()

            for i in range(images_2d.shape[0]):
                mask_2d_true_np = masks_2d[i].cpu().numpy().squeeze()
                pred_2d_np = preds_2d[i].cpu().numpy().squeeze()

                mask_1d_true = _internal_convert_2d_mask_to_1d(mask_2d_true_np)
                mask_1d_pred = _internal_convert_2d_mask_to_1d(pred_2d_np)

                # Nyt molemmat tapahtumalistat suodatetaan keston mukaan
                true_events = _get_events_from_mask(mask_1d_true)
                pred_events = _get_events_from_mask(mask_1d_pred)

                tp_count = 0
                matched_true_events = []

                for pred in pred_events:
                    found_match = False
                    for j, true in enumerate(true_events):
                        if j in matched_true_events:
                            continue
                        iou = _calculate_iou(pred, true)
                        if iou > iou_threshold:
                            tp_count += 1
                            matched_true_events.append(j)
                            found_match = True
                            break
                    if not found_match:
                        total_fp += 1

                fn_count = len(true_events) - len(matched_true_events)
                total_tp += tp_count
                total_fn += fn_count

    epsilon = 1e-6
    precision = (total_tp + epsilon) / (total_tp + total_fp + epsilon)
    recall = (total_tp + epsilon) / (total_tp + total_fn + epsilon)
    dice_f1 = (2 * total_tp + epsilon) / (2 * total_tp + total_fp + total_fn + epsilon)
    iou_jaccard = (total_tp + epsilon) / (total_tp + total_fp + total_fn + epsilon)

    return {
        "F1-score": dice_f1,
        "IoU (Jaccard)": iou_jaccard,
        "Precision": precision,
        "Recall": recall,
        "TP (events)": total_tp,
        "FP (events)": total_fp,
        "FN (events)": total_fn
    }
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

# KORJAUS: Laskettu 0.2 -> 0.1 vastaamaan Ground Truth -logiikkaa
FIXED_BORDER_THRESH = 0.1


def _find_events_dual_thresh(prob_1d: np.ndarray,
                             peak_thresh: float,
                             border_thresh: float,
                             fs: float) -> List[Tuple[int, int]]:
    """
    Etsii tapahtumat 1D-todennäköisyyssarjasta käyttäen kaksoiskynnystä.
    TÄMÄ ON NOPEA, VEKTORISOITU TOTEUTUS.
    """
    min_samples = METRIC_PARAMS['min_duration_sec'] * fs
    max_samples = METRIC_PARAMS['max_duration_sec'] * fs

    # 1. Etsi KAIKKI alueet, jotka ylittävät MATALAN kynnyksen (border_thresh)
    border_mask = prob_1d > border_thresh
    labeled_borders, num_border_regions = label(border_mask)

    if num_border_regions == 0:
        return []

    # 2. Etsi KAIKKI pisteet, jotka ylittävät KORKEAN kynnyksen (peak_thresh)
    peak_mask = prob_1d > peak_thresh

    # 3. Selvitä, mitkä matalan kynnyksen alueet sisältävät vähintään yhden ytimen
    labels_of_peaks = labeled_borders[peak_mask]
    valid_region_labels = np.unique(labels_of_peaks[labels_of_peaks > 0])

    if len(valid_region_labels) == 0:
        return []

    # 4. Etsi validoitujen alueiden tarkat rajat
    valid_slices = find_objects(labeled_borders)

    events = []
    for region_label in valid_region_labels:
        s = valid_slices[region_label - 1]
        start, end = s[0].start, s[0].stop

        # 5. Suodata keston mukaan
        duration_samples = end - start

        if min_samples <= duration_samples <= max_samples:
            events.append((start, end - 1))

    events.sort(key=lambda x: x[0])
    return events


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


def _internal_convert_2d_prob_to_1d(prob_2d: np.ndarray) -> np.ndarray:
    """Muuntaa 2D CWT-todennäköisyyskartan 1D-todennäköisyyssarjaksi."""
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
    """Laskee Intersection over Union (IoU)."""
    start1, end1 = event1
    start2, end2 = event2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    return intersection / union if union > 0 else 0.0


def compute_event_based_metrics(model, data_loader, threshold: float) -> Dict[str, float]:
    """
    Laskee metriikat TAPAHTUMAPOHJAISESTI optimoidulla muistinhallinnalla (Streaming Stitching).
    """
    # Laitehallinta
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    fs = DATA_PARAMS['fs']
    window_len = int(DATA_PARAMS['window_sec'] * fs)
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)
    height = CWT_PARAMS['freq_bins']

    # Lasketaan lopullinen pituus
    num_windows = len(data_loader.dataset)
    final_len = (num_windows - 1) * step_samples + window_len

    log.info(f"Metrics: Streaming inference for {num_windows} windows. Final len: {final_len} samples.")

    # Varataan muisti CPU:lta lopulliselle kuvalle
    stitched_sum_probs = torch.zeros((height, final_len), dtype=torch.float32)
    stitched_sum_masks = torch.zeros((height, final_len), dtype=torch.float32)
    stitched_weights = torch.zeros((height, final_len), dtype=torch.float32)

    # Ikkunan painotukset
    window_weights = torch.hann_window(window_len, periodic=False).view(height, window_len)

    current_idx = 0

    with torch.no_grad():
        for images_seq, masks_2d, _ in tqdm(data_loader, desc="Inference & Stitching"):
            images_seq = images_seq.to(device)
            masks_2d = masks_2d.to(device)

            # 1. Ennustus
            logits = model(images_seq)
            probs = torch.sigmoid(logits)  # (B, 1, H, W)

            # 2. Interpoloi maskit vastaamaan ennusteen kokoa tarvittaessa
            if masks_2d.shape[2:] != probs.shape[2:]:
                masks_2d = F.interpolate(masks_2d, size=probs.shape[2:], mode='nearest')

            # 3. Siirrä CPU:lle käsittelyä varten
            probs = probs.cpu()
            masks_2d = masks_2d.cpu()

            batch_size = probs.shape[0]

            # 4. Streaming Stitching (lisätään suoraan isoon tensoriin)
            for b in range(batch_size):
                if current_idx >= num_windows:
                    break

                start = current_idx * step_samples
                end = start + window_len

                if end > final_len:
                    end = final_len
                    valid_len = end - start
                    # Leikataan painot ja data jos mennään yli (harvinainen edge case)
                    w_slice = window_weights[:, :valid_len]
                    p_slice = probs[b, 0, :, :valid_len]
                    m_slice = masks_2d[b, 0, :, :valid_len]
                else:
                    w_slice = window_weights
                    p_slice = probs[b, 0, :, :]
                    m_slice = masks_2d[b, 0, :, :]

                stitched_sum_probs[:, start:end] += p_slice * w_slice
                stitched_sum_masks[:, start:end] += m_slice * w_slice
                stitched_weights[:, start:end] += w_slice

                current_idx += 1

            # Vapautetaan GPU-muistia
            del images_seq, logits, masks_2d, probs

    # 5. Normalisointi
    stitched_weights[stitched_weights == 0] = 1e-6
    final_prob_2d = stitched_sum_probs / stitched_weights
    final_mask_2d = stitched_sum_masks / stitched_weights

    log.info("Converting to 1D time series...")
    # Muunnetaan numpyksi tässä vaiheessa
    prob_1d_pred = _internal_convert_2d_prob_to_1d(final_prob_2d.numpy())

    # Maski on 1, jos painotettu keskiarvo > 0.5 (soft voting)
    mask_1d_true_bool = _internal_convert_2d_mask_to_1d((final_mask_2d > 0.5).numpy())

    # 6. Tapahtumien etsintä
    true_events = _find_events_dual_thresh(mask_1d_true_bool.astype(float), 0.5, 0.1, fs)
    pred_events = _find_events_dual_thresh(prob_1d_pred, threshold, FIXED_BORDER_THRESH, fs)

    log.info(f"Comparing: {len(true_events)} true vs {len(pred_events)} pred events.")

    # 7. Metriikoiden laskenta
    total_tp, total_fp = 0, 0
    iou_scores_of_tps = []
    matched_true_events = []

    for pred in pred_events:
        found_match = False
        best_iou = 0
        best_true_idx = -1

        for j, true in enumerate(true_events):
            if j in matched_true_events:
                continue
            iou = _calculate_iou(pred, true)
            if iou > METRIC_PARAMS['iou_threshold'] and iou > best_iou:
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
    """
    Etsii parhaan 'peak_thresh' -kynnysarvon käyttäen Streaming Stitching -tekniikkaa.
    """
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
    window_len = int(DATA_PARAMS['window_sec'] * fs)
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)
    height = CWT_PARAMS['freq_bins']

    num_windows = len(val_loader.dataset)
    final_len = (num_windows - 1) * step_samples + window_len

    # Varataan muisti
    stitched_sum_probs = torch.zeros((height, final_len), dtype=torch.float32)
    stitched_sum_masks = torch.zeros((height, final_len), dtype=torch.float32)
    stitched_weights = torch.zeros((height, final_len), dtype=torch.float32)
    window_weights = torch.hann_window(window_len, periodic=False).view(height, window_len)

    current_idx = 0

    # --- Streaming Inference Loop ---
    with torch.no_grad():
        for images_seq, masks_2d, _ in tqdm(val_loader, desc="Optimizing Threshold (Inference)"):
            images_seq = images_seq.to(device)
            masks_2d = masks_2d.to(device)

            logits = model(images_seq)
            probs = torch.sigmoid(logits)

            if masks_2d.shape[2:] != probs.shape[2:]:
                masks_2d = F.interpolate(masks_2d, size=probs.shape[2:], mode='nearest')

            probs = probs.cpu()
            masks_2d = masks_2d.cpu()
            batch_size = probs.shape[0]

            for b in range(batch_size):
                if current_idx >= num_windows:
                    break
                start = current_idx * step_samples
                end = start + window_len

                if end > final_len:
                    # Edge case handling
                    end = final_len
                    valid_len = end - start
                    w_slice = window_weights[:, :valid_len]
                    p_slice = probs[b, 0, :, :valid_len]
                    m_slice = masks_2d[b, 0, :, :valid_len]
                else:
                    w_slice = window_weights
                    p_slice = probs[b, 0, :, :]
                    m_slice = masks_2d[b, 0, :, :]

                stitched_sum_probs[:, start:end] += p_slice * w_slice
                stitched_sum_masks[:, start:end] += m_slice * w_slice
                stitched_weights[:, start:end] += w_slice
                current_idx += 1

            del images_seq, logits, masks_2d, probs

    # Normalisointi ja 1D muunnos
    stitched_weights[stitched_weights == 0] = 1e-6
    final_prob_2d = stitched_sum_probs / stitched_weights
    final_mask_2d = stitched_sum_masks / stitched_weights

    mask_1d_true_bool = _internal_convert_2d_mask_to_1d((final_mask_2d > 0.5).numpy())
    true_events = _find_events_dual_thresh(mask_1d_true_bool.astype(float), 0.5, 0.1, fs)

    prob_1d_series = _internal_convert_2d_prob_to_1d(final_prob_2d.numpy())

    # --- Grid Search ---
    best_f1 = -1.0
    best_peak_threshold = 0.5
    search_space = np.arange(FIXED_BORDER_THRESH + 0.05, 0.95, 0.05)

    log.info(f"Testing {len(search_space)} PEAK thresholds...")

    for peak_thresh in search_space:
        pred_events = _find_events_dual_thresh(prob_1d_series, peak_thresh, FIXED_BORDER_THRESH, fs)

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

        if f1_score > best_f1:
            best_f1 = f1_score
            best_peak_threshold = peak_thresh

    log.info(f"Optimal PEAK threshold found: {best_peak_threshold:.2f} (F1-score: {best_f1:.4f})")
    return best_peak_threshold
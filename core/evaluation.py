# core/evaluation.py

import logging
import torch
import numpy as np
import gc
import os
import json
from tqdm import tqdm
from typing import Dict
from configs.dreams_config import METRIC_PARAMS, DATA_PARAMS, POST_PROCESSING_PARAMS
from postprocessing.postprocessing import stitch_predictions_1d, find_events_dual_thresh, calculate_iou
from utils.reporting import generate_detailed_csv

log = logging.getLogger(__name__)

def compute_event_based_metrics(model,
                                data_loader,
                                threshold: float,
                                subject_id: str = "unknown",
                                output_dir: str = ".") -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')
    model.to(device)
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
    del all_probs_list, all_masks_list, raw_signal_list
    gc.collect()

    prob_1d = stitch_predictions_1d(all_probs, step_samples)
    mask_1d = stitch_predictions_1d(all_masks, step_samples)
    raw_1d = stitch_predictions_1d(all_raw, step_samples)
    fixed_border_thresh = POST_PROCESSING_PARAMS['fixed_border_thresh']

    # 2. (DETECTION)
    pred_events = find_events_dual_thresh(
        prob_1d,
        threshold,
        fixed_border_thresh,
        fs
    )

    # Ground truth
    true_events = find_events_dual_thresh(
        (mask_1d >= 0.4).astype(float),
        0.5,
        0.1,
        fs
    )

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
            iou = calculate_iou(p, t)
            if iou > best_iou: best_iou = iou; best_idx = i
        if best_iou >= METRIC_PARAMS['iou_threshold']:
            tp += 1
            matched.add(best_idx)
            iou_scores.append(best_iou)

    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    eps = 1e-6
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * (prec * rec) / (prec + rec + eps)

    try:
        stats_payload = {
            "subject_id": subject_id,
            "threshold_used": float(threshold),
            "true_count": len(true_events),
            "pred_count": len(pred_events),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "mean_iou": float(np.mean(iou_scores))
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_filename = f"eval_stats_{subject_id}.json"
            json_path = os.path.join(output_dir, json_filename)

            with open(json_path, 'w') as f:
                json.dump(stats_payload, f, indent=4)

            log.info(f"Saved JSON stats to: {json_path}")

    except Exception as e:
        log.error(f"Failed to save JSON stats for {subject_id}: {e}")

    return {"F1-score": f1, "Precision": prec, "Recall": rec, "TP (events)": tp, "FP (events)": fp, "FN (events)": fn,
            "mIoU (TPs)": np.mean(iou_scores) if iou_scores else 0.0}


def find_optimal_threshold(model, val_loader) -> float:
    return 0.50
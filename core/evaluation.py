# core/evaluation.py

import logging
import torch
import numpy as np
import gc
import os
import json
from tqdm import tqdm
from typing import Dict
from postprocessing.postprocessing import stitch_predictions_1d, find_events_dual_thresh, calculate_iou
from core.config_loader import POST_PROCESSING_PARAMS, INFERENCE_PARAMS, TRAINING_PARAMS

log = logging.getLogger(__name__)


def aggregate_and_save_summary(repeat_metrics: Dict[str, list],
                               output_dir: str,
                               repeat_idx: int,
                               seed: int,
                               logger=None):
    """
    Calculates mean/std across folds for a repeat and saves to JSON.
    Logs only relative metrics (F1, Precision, Recall, IoU), hides raw counts (TP/FP/FN).
    """
    summary_stats = {
        "repeat_index": repeat_idx + 1,
        "seed": seed,
        "metrics": {}
    }

    if logger is None:
        logger = log

    logger.info(f"Summary for repeat: {repeat_idx + 1}")

    grand_results_update = {}

    metrics_to_hide_from_log = ["TP (events)", "FP (events)", "FN (events)"]

    if len(repeat_metrics) > 0:
        for key, values in repeat_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)

            if key not in metrics_to_hide_from_log:
                logger.info(f"{key:<15}: {mean_val:.3f} (± {std_val:.3f})")

            grand_results_update[key] = mean_val

            summary_stats["metrics"][key] = {
                "mean": float(mean_val),
                "std": float(std_val)
            }

        try:
            json_filename = f"summary_repeat_{repeat_idx + 1}.json"
            json_path = os.path.join(output_dir, json_filename)

            with open(json_path, 'w') as f:
                json.dump(summary_stats, f, indent=4)

            logger.info(f"Saved repeat summary JSON to: {json_path}")

        except Exception as e:
            logger.error(f"Failed to save summary JSON: {e}")

    else:
        logger.warning("No metrics collected for this repeat.")

    return grand_results_update


def save_final_experiment_summary(grand_results: Dict[str, list],
                                  output_dir: str,
                                  total_repeats: int,
                                  timestamp: str,
                                  logger=None):
    """
    Calculates final mean/std across all repeats, logs the results table,
    and saves the final experiment summary JSON.
    """
    if logger is None:
        logger = log

    logger.info(f"Final experiment results over {total_repeats} repeats.")

    final_summary_data = {
        "experiment_timestamp": timestamp,
        "total_repeats": total_repeats,
        "metrics": {}
    }

    metrics_to_hide_from_log = ["TP (events)", "FP (events)", "FN (events)"]

    if len(grand_results) > 0:
        logger.info(f"{'Metric':<15} {'Mean':<10} {'Std Dev':<10}")

        for key, values in grand_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)

            if key not in metrics_to_hide_from_log:
                logger.info(f"{key:<15} {mean_val:.3f} (± {std_val:.3f})")

            final_summary_data["metrics"][key] = {
                "mean": float(mean_val),
                "std": float(std_val)
            }

        try:
            final_json_path = os.path.join(output_dir, "final_experiment_summary.json")
            with open(final_json_path, 'w') as f:
                json.dump(final_summary_data, f, indent=4)

            logger.info(f"Saved final experiment summary to: {final_json_path}")

        except Exception as e:
            logger.error(f"Failed to save final summary JSON: {e}")
    else:
        logger.warning("No grand results collected.")


def compute_event_based_metrics(model,
                                data_loader,
                                threshold: float,
                                data_params: dict,
                                subject_id: str = "unknown",
                                output_dir: str = ".") -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gating = TRAINING_PARAMS.get('use_gating_branch')

    if torch.backends.mps.is_available(): device = torch.device('mps')
    model.to(device)
    model.eval()
    fs = data_params['fs']
    step_samples = int((data_params['window_sec'] - data_params['overlap_sec']) * fs)
    all_probs_list, all_masks_list = [], []

    with torch.no_grad():
        for inputs, masks, labels in tqdm(data_loader, desc=f"Evaluating ({subject_id})"):
            inputs = inputs.to(device)
            mask_logits, gate_logits = model(inputs)
            final_prob = torch.sigmoid(mask_logits)
            if use_gating:
                final_prob = final_prob * torch.sigmoid(gate_logits).unsqueeze(2)

            # TTA
            inputs_flip = torch.flip(inputs, dims=[2])
            m_f, g_f = model(inputs_flip)
            final_prob_flip = torch.flip(torch.sigmoid(m_f), dims=[2])
            if use_gating:
                final_prob_flip = final_prob_flip * torch.sigmoid(g_f).unsqueeze(2)

            avg_prob = (final_prob + final_prob_flip) / 2.0
            all_probs_list.append(avg_prob.cpu().float())
            all_masks_list.append(masks.cpu().float())

    all_probs = torch.cat(all_probs_list, dim=0)
    all_masks = torch.cat(all_masks_list, dim=0).unsqueeze(1)
    del all_probs_list, all_masks_list
    gc.collect()

    prob_1d = stitch_predictions_1d(all_probs, step_samples)
    mask_1d = stitch_predictions_1d(all_masks, step_samples)
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
        if best_iou >= INFERENCE_PARAMS['iou_threshold']:
            tp += 1
            matched.add(best_idx)
            iou_scores.append(best_iou)

    # Calculate false postives
    fp = len(pred_events) - tp

    # Calculate false negatives
    fn = len(true_events) - tp

    # Calculate precision
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # Calculate recall
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # Calculate F1-score
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

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
            "precision": float(precision),
            "recall": float(recall),
            "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0
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

    return {"F1-score": f1, "Precision": precision, "Recall": recall, "TP (events)": tp, "FP (events)": fp,
            "FN (events)": fn,
            "mIoU: ": np.mean(iou_scores) if iou_scores else 0.0}


def find_optimal_threshold(model, val_loader) -> float:
    return 0.50
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
from core.dataset import get_inference_data

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


def _event_majority_stage(start: int, end: int, stage_codes: np.ndarray) -> str:
    """Return the most common stage code in the [start, end] sample range."""
    n = len(stage_codes)
    s = max(0, int(start))
    e = min(n, int(end) + 1)
    if s >= e:
        return '?'
    segment = stage_codes[s:e]
    # np.unique returns sorted unique values + counts
    values, counts = np.unique(segment, return_counts=True)
    return str(values[counts.argmax()])


def _events_overlap_mask(events: list, stage_mask: np.ndarray,
                         stage_codes: np.ndarray = None) -> tuple:
    """Keep only events that have at least one sample inside stage_mask.

    Returns:
        (kept_events, n_discarded, discarded_per_stage)
        discarded_per_stage is a dict {stage_code: count} of where the discarded
        events fell (their majority stage). Empty dict if stage_codes is None.
    """
    if len(events) == 0:
        return [], 0, {}
    n = len(stage_mask)
    kept = []
    n_discarded = 0
    discarded_per_stage = {}
    for start, end in events:
        s = max(0, int(start))
        e = min(n, int(end) + 1)
        if s >= e:
            n_discarded += 1
            continue
        if stage_mask[s:e].max() > 0.5:
            kept.append((start, end))
        else:
            n_discarded += 1
            if stage_codes is not None:
                stage = _event_majority_stage(start, end, stage_codes)
                discarded_per_stage[stage] = discarded_per_stage.get(stage, 0) + 1
    return kept, n_discarded, discarded_per_stage


def _evaluate_single_subject(model, loader, stage_mask_full, stage_codes_full,
                             threshold, fs, step_samples, use_gating, use_tta,
                             device, subject_id):
    """Run inference on one subject's FULL recording, return (pred_events, true_events, info)."""
    all_probs_list, all_masks_list = [], []

    with torch.no_grad():
        for inputs, masks, _ in tqdm(loader, desc=f"Inferring ({subject_id})"):
            inputs = inputs.to(device)
            mask_logits, gate_logits = model(inputs)
            final_prob = torch.sigmoid(mask_logits)
            if use_gating:
                final_prob = final_prob * torch.sigmoid(gate_logits).unsqueeze(2)

            if use_tta:
                inputs_flip = torch.flip(inputs, dims=[2])
                m_f, g_f = model(inputs_flip)
                final_prob_flip = torch.flip(torch.sigmoid(m_f), dims=[2])
                if use_gating:
                    final_prob_flip = final_prob_flip * torch.sigmoid(g_f).unsqueeze(2)
                final_prob = (final_prob + final_prob_flip) / 2.0

            all_probs_list.append(final_prob.cpu().float())
            all_masks_list.append(masks.cpu().float())

    all_probs = torch.cat(all_probs_list, dim=0)
    all_masks = torch.cat(all_masks_list, dim=0).unsqueeze(1)
    del all_probs_list, all_masks_list

    # Stitch into a single 1D probability series — windows are in time order
    prob_1d = stitch_predictions_1d(all_probs, step_samples)
    mask_1d = stitch_predictions_1d(all_masks, step_samples)

    fixed_border_thresh = POST_PROCESSING_PARAMS['fixed_border_thresh']
    gap_thresh_sec = POST_PROCESSING_PARAMS['gap_thresh_sec']

    # Predicted events (across the entire recording) — model output gets merged
    pred_events = find_events_dual_thresh(
        prob_1d, threshold, fixed_border_thresh, fs,
        gap_thresh_sec=gap_thresh_sec
    )

    # Ground truth events — expert annotations are never merged
    true_events = find_events_dual_thresh(
        (mask_1d >= 0.4).astype(float), 0.5, False, fs,
        gap_thresh_sec=None
    )

    # Keep only events overlapping the N2 mask.
    # The stage_mask may differ in length from prob_1d by a few samples due to
    # the windowing — clip to the common length.
    n_common = min(len(stage_mask_full), len(prob_1d))
    stage_mask = stage_mask_full[:n_common]
    stage_codes = stage_codes_full[:n_common] if stage_codes_full is not None else None

    pred_total_before = len(pred_events)
    true_total_before = len(true_events)

    pred_events, n_pred_discarded, pred_discarded_by_stage = _events_overlap_mask(
        pred_events, stage_mask, stage_codes
    )
    true_events, n_true_discarded, _ = _events_overlap_mask(
        true_events, stage_mask, stage_codes
    )

    return pred_events, true_events, {
        "pred_total": pred_total_before,
        "pred_discarded_non_n2": n_pred_discarded,
        "pred_discarded_by_stage": pred_discarded_by_stage,
        "true_total": true_total_before,
        "true_discarded_non_n2": n_true_discarded,
    }


def _match_events(pred_events, true_events, iou_threshold):
    """Greedy IoU matching. Returns (tp, fp, fn, iou_scores)."""
    tp = 0
    matched = set()
    iou_scores = []
    for p in pred_events:
        best_iou = 0
        best_idx = -1
        for i, t in enumerate(true_events):
            if i in matched:
                continue
            iou = calculate_iou(p, t)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_idx)
            iou_scores.append(best_iou)

    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    return tp, fp, fn, iou_scores


def compute_event_based_metrics(model,
                                subject_ids: list,
                                processed_data_dir: str,
                                threshold: float,
                                data_params: dict,
                                identifier: str = "fold",
                                output_dir: str = ".") -> Dict[str, float]:
    """
    Each subject's full recording is predicted independently, events are
    extracted, filtered by N2 stage mask, then aggregated across subjects.
    """
    from core.dataset import get_inference_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    use_gating = TRAINING_PARAMS.get('use_gating_branch')
    use_tta = INFERENCE_PARAMS.get('use_tta', False)

    model.to(device)
    model.eval()
    fs = data_params['fs']
    step_samples = int((data_params['window_sec'] - data_params['overlap_sec']) * fs)
    batch_size = TRAINING_PARAMS.get('batch_size', 16)

    log.info(f"Inference: use_tta={use_tta}, use_gating={use_gating}, threshold={threshold:.3f}")

    all_pred_events_per_subj = []
    all_true_events_per_subj = []
    per_subject_stats = []
    total_pred_discarded = 0
    total_pred_before = 0
    total_true_discarded = 0
    total_true_before = 0
    aggregate_discarded_by_stage = {}

    for sid in subject_ids:
        try:
            loader, stage_mask_full, stage_codes_full = get_inference_data(
                processed_data_dir, sid, batch_size, data_params
            )
        except FileNotFoundError as e:
            log.warning(f"Skipping {sid}: {e}")
            continue

        pred_events, true_events, discard_info = _evaluate_single_subject(
            model, loader, stage_mask_full, stage_codes_full, threshold, fs,
            step_samples, use_gating, use_tta, device, sid
        )

        all_pred_events_per_subj.append(pred_events)
        all_true_events_per_subj.append(true_events)

        n_pred_kept = len(pred_events)
        n_pred_disc = discard_info["pred_discarded_non_n2"]
        n_pred_total = discard_info["pred_total"]
        pred_disc_pct = (n_pred_disc / n_pred_total * 100) if n_pred_total > 0 else 0.0
        per_stage = discard_info["pred_discarded_by_stage"]

        per_subject_stats.append({
            "subject": sid,
            "pred_total": n_pred_total,
            "pred_kept_in_n2": n_pred_kept,
            "pred_discarded_non_n2": n_pred_disc,
            "pred_discarded_by_stage": per_stage,
            "true_count": len(true_events),
        })
        total_pred_discarded += n_pred_disc
        total_pred_before += n_pred_total
        total_true_discarded += discard_info["true_discarded_non_n2"]
        total_true_before += discard_info["true_total"]
        for stage, n in per_stage.items():
            aggregate_discarded_by_stage[stage] = aggregate_discarded_by_stage.get(stage, 0) + n

        # Format per-stage breakdown for log
        if per_stage:
            stage_str = ", ".join(f"{s}={n}" for s, n in sorted(per_stage.items()))
            stage_breakdown_str = f" [{stage_str}]"
        else:
            stage_breakdown_str = ""

        log.info(
            f"  [{sid}] true={len(true_events)}, pred(N2)={n_pred_kept}, "
            f"pred(non-N2 discarded)={n_pred_disc}/{n_pred_total} "
            f"({pred_disc_pct:.1f}%){stage_breakdown_str}"
        )

    gc.collect()

    # Aggregate event-level metrics across subjects (micro-average)
    iou_threshold = INFERENCE_PARAMS['iou_threshold']
    tp_total, fp_total, fn_total = 0, 0, 0
    all_iou_scores = []

    for pred_events, true_events in zip(all_pred_events_per_subj, all_true_events_per_subj):
        tp, fp, fn, iou_scores = _match_events(pred_events, true_events, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        all_iou_scores.extend(iou_scores)

    # Aggregate non-N2 discard statistics
    pred_discard_pct = (total_pred_discarded / total_pred_before * 100) if total_pred_before > 0 else 0.0
    if aggregate_discarded_by_stage:
        agg_stage_str = ", ".join(
            f"{s}={n}" for s, n in sorted(aggregate_discarded_by_stage.items())
        )
        log.info(
            f"Non-N2 predictions discarded: {total_pred_discarded}/{total_pred_before} "
            f"({pred_discard_pct:.1f}%) — by stage: [{agg_stage_str}]"
        )
    else:
        log.info(
            f"Non-N2 predictions discarded: {total_pred_discarded}/{total_pred_before} "
            f"({pred_discard_pct:.1f}%)"
        )
    if total_true_discarded > 0:
        log.info(
            f"Non-N2 ground-truth events discarded: "
            f"{total_true_discarded}/{total_true_before} (sanity check)"
        )

    log.info(f"Aggregate: TP={tp_total}, FP={fp_total}, FN={fn_total}")

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_iou_scores)) if all_iou_scores else 0.0

    # Save per-subject + aggregate stats
    try:
        stats_payload = {
            "identifier": identifier,
            "threshold_used": float(threshold),
            "iou_threshold": float(iou_threshold),
            "tp": int(tp_total),
            "fp": int(fp_total),
            "fn": int(fn_total),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "mean_iou": mean_iou,
            "pred_total_before_n2_filter": int(total_pred_before),
            "pred_discarded_non_n2": int(total_pred_discarded),
            "pred_discard_pct": round(pred_discard_pct, 2),
            "pred_discarded_by_stage": aggregate_discarded_by_stage,
            "per_subject": per_subject_stats,
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"eval_stats_{identifier}.json")
            with open(json_path, 'w') as f:
                json.dump(stats_payload, f, indent=4)
            log.info(f"Saved JSON stats to: {json_path}")
    except Exception as e:
        log.error(f"Failed to save JSON stats: {e}")

    return {
        "F1-score": f1,
        "Precision": precision,
        "Recall": recall,
        "TP (events)": tp_total,
        "FP (events)": fp_total,
        "FN (events)": fn_total,
        "mIoU: ": mean_iou,
    }


def find_optimal_threshold(model, subject_ids, processed_data_dir, data_params,
                           threshold_grid=None, logger=None):
    """
    Find F1-optimal threshold by caching probability series per subject and
    sweeping thresholds over the cached series. Inference runs once per subject.
    """


    if logger is None:
        logger = log
    if threshold_grid is None:
        threshold_grid = np.arange(0.40, 0.90, 0.05)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    use_gating = TRAINING_PARAMS.get('use_gating_branch')
    use_tta = INFERENCE_PARAMS.get('use_tta', False)
    fs = data_params['fs']
    step_samples = int((data_params['window_sec'] - data_params['overlap_sec']) * fs)
    batch_size = TRAINING_PARAMS.get('batch_size', 16)
    fixed_border_thresh = POST_PROCESSING_PARAMS['fixed_border_thresh']
    gap_thresh_sec = POST_PROCESSING_PARAMS['gap_thresh_sec']
    iou_threshold = INFERENCE_PARAMS['iou_threshold']

    model.to(device).eval()

    # 1) Cache stitched probabilities + ground truth per subject (inference ONCE)
    cached = []  # list of dicts: prob_1d, true_events, stage_mask, stage_codes
    for sid in subject_ids:
        try:
            loader, stage_mask_full, stage_codes_full = get_inference_data(
                processed_data_dir, sid, batch_size, data_params
            )
        except FileNotFoundError as e:
            logger.warning(f"Skipping {sid}: {e}")
            continue

        all_probs_list, all_masks_list = [], []
        with torch.no_grad():
            for inputs, masks, _ in tqdm(loader, desc=f"Caching ({sid})"):
                inputs = inputs.to(device)
                mask_logits, gate_logits = model(inputs)
                final_prob = torch.sigmoid(mask_logits)
                if use_gating:
                    final_prob = final_prob * torch.sigmoid(gate_logits).unsqueeze(2)
                if use_tta:
                    inputs_flip = torch.flip(inputs, dims=[2])
                    m_f, g_f = model(inputs_flip)
                    final_prob_flip = torch.flip(torch.sigmoid(m_f), dims=[2])
                    if use_gating:
                        final_prob_flip = final_prob_flip * torch.sigmoid(g_f).unsqueeze(2)
                    final_prob = (final_prob + final_prob_flip) / 2.0
                all_probs_list.append(final_prob.cpu().float())
                all_masks_list.append(masks.cpu().float())

        all_probs = torch.cat(all_probs_list, dim=0)
        all_masks = torch.cat(all_masks_list, dim=0).unsqueeze(1)
        prob_1d = stitch_predictions_1d(all_probs, step_samples)
        mask_1d = stitch_predictions_1d(all_masks, step_samples)

        # Ground truth events are threshold-independent — compute once.
        # Expert annotations are never merged.
        true_events = find_events_dual_thresh(
            (mask_1d >= 0.4).astype(float), 0.5, False, fs,
            gap_thresh_sec=None
        )
        n_common = min(len(stage_mask_full), len(prob_1d))
        stage_mask = stage_mask_full[:n_common]
        stage_codes = stage_codes_full[:n_common] if stage_codes_full is not None else None

        # Filter true events by N2 once
        true_events, _, _ = _events_overlap_mask(true_events, stage_mask, stage_codes)

        cached.append({
            "sid": sid,
            "prob_1d": prob_1d,
            "true_events": true_events,
            "stage_mask": stage_mask,
            "stage_codes": stage_codes,
        })
        del all_probs_list, all_masks_list, all_probs, all_masks, mask_1d
        gc.collect()

    # 2) Sweep thresholds over cached probabilities (no GPU work)
    best_f1, best_t = -1.0, float(threshold_grid[0])
    for t in threshold_grid:
        tp_total, fp_total, fn_total = 0, 0, 0
        for c in cached:
            pred_events = find_events_dual_thresh(
                c["prob_1d"], float(t), fixed_border_thresh, fs,
                gap_thresh_sec=gap_thresh_sec
            )
            pred_events, _, _ = _events_overlap_mask(
                pred_events, c["stage_mask"], c["stage_codes"]
            )
            tp, fp, fn, _ = _match_events(pred_events, c["true_events"], iou_threshold)
            tp_total += tp; fp_total += fp; fn_total += fn

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    logger.info(f"Optimal threshold: {best_t:.3f}  (F1={best_f1:.3f})")
    return best_t
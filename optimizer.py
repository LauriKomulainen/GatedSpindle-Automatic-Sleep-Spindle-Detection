# optimizer.py

import torch
import numpy as np
import logging
import itertools
from pathlib import Path
from scipy.ndimage import label, find_objects
from scipy.signal import welch
from tqdm import tqdm

from utils.logger import setup_logging
from data_preprocess.dataset import get_dataloaders
from UNET_model.model import GatedUNet
from UNET_model.evaluation_metrics import _stitch_predictions_1d, _calculate_iou
from config import DATA_PARAMS

RUN_TIMESTAMP = "2025-11-25_23-43-58"
ALL_SUBJECTS = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']

FIXED_MERGE_GAP = 0.3

PARAM_GRID = {
    'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'min_duration': [0.3, 0.5],
    'use_power_check': [False, True],
    'power_ratio': [0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
}

setup_logging("master_optimization.log")
log = logging.getLogger(__name__)

def verify_spindle_power(signal_segment, fs, threshold_ratio):
    n = len(signal_segment)
    if n < int(0.3 * fs): return False
    nperseg = min(n, 128)
    try:
        freqs, psd = welch(signal_segment, fs, nperseg=nperseg)
    except:
        return False

    idx_sigma = np.where((freqs >= 11.0) & (freqs <= 16.0))[0]
    idx_total = np.where((freqs >= 0.5) & (freqs <= 30.0))[0]

    if len(idx_sigma) < 2 or len(idx_total) < 2: return False

    power_sigma = np.trapz(psd[idx_sigma], freqs[idx_sigma])
    power_total = np.trapz(psd[idx_total], freqs[idx_total])

    if power_total == 0: return False
    return (power_sigma / power_total) >= threshold_ratio


def find_events_fast(prob_1d, raw_1d, thresh, min_samples, merge_samples, use_power, power_ratio, fs):
    mask = prob_1d > thresh
    labeled, num_features = label(mask)
    if num_features == 0: return []

    slices = find_objects(labeled)
    raw_events = [(s[0].start, s[0].stop) for s in slices]

    if not raw_events: return []
    merged = []
    curr_start, curr_end = raw_events[0]
    for i in range(1, len(raw_events)):
        next_start, next_end = raw_events[i]
        if (next_start - curr_end) < merge_samples:
            curr_end = next_end
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))

    final_events = []
    for start, end in merged:
        if (end - start) >= min_samples:
            if use_power:
                if verify_spindle_power(raw_1d[start:end], fs, power_ratio):
                    final_events.append((start, end))
            else:
                final_events.append((start, end))

    return final_events


def calculate_f1_fast(pred_events, true_events):
    tp = 0
    matched = set()
    for p in pred_events:
        best_iou = 0
        best_idx = -1
        for j, t in enumerate(true_events):
            if j in matched: continue
            s1, e1 = p
            s2, e2 = t
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou: best_iou = iou; best_idx = j

        if best_iou >= 0.2:
            tp += 1
            matched.add(best_idx)

    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return f1, prec, rec, tp, fp


def run_master_optimization():
    log.info("--- STARTING MASTER OPTIMIZATION & PERSONAL BEST TRACKING ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent
    processed_data_path = project_root / "diagnostics_plots" / "processed_data"
    reports_dir = project_root / "model_reports" / f"LOSO_run_{RUN_TIMESTAMP}"

    subject_cache = {}
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    log.info("Phase 1: Caching Model Predictions...")

    for i, subject_id in enumerate(ALL_SUBJECTS):
        fold_num = i + 1
        try:
            _, _, test_loader = get_dataloaders(str(processed_data_path), 16, [], [], [subject_id], 1.0)
        except:
            continue

        fold_dir = reports_dir / f"Fold_{fold_num}_(Test={subject_id})"
        model_best = GatedUNet(dropout_rate=0.0).to(device)
        model_swa = GatedUNet(dropout_rate=0.0).to(device)

        p_best = fold_dir / 'unet_model_best.pth'
        p_swa = fold_dir / 'unet_model_swa.pth'

        has_best = False;
        has_swa = False
        if p_best.exists():
            model_best.load_state_dict(torch.load(p_best, map_location=device))
            model_best.eval();
            has_best = True
        if p_swa.exists():
            model_swa.load_state_dict(torch.load(p_swa, map_location=device))
            model_swa.eval();
            has_swa = True

        if not has_best and not has_swa: continue

        all_probs, all_masks, all_raw = [], [], []
        with torch.no_grad():
            for inputs, masks, _ in test_loader:
                inputs = inputs.to(device)
                preds_list = []
                if has_best:
                    m, g = model_best(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)
                    inputs_flip = torch.flip(inputs, dims=[2])
                    m_f, g_f = model_best(inputs_flip)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)
                    preds_list.append((p + p_f) / 2.0)
                if has_swa:
                    m, g = model_swa(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)
                    inputs_flip = torch.flip(inputs, dims=[2])
                    m_f, g_f = model_swa(inputs_flip)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)
                    preds_list.append((p + p_f) / 2.0)

                final_prob = torch.mean(torch.stack(preds_list), dim=0)
                all_probs.append(final_prob.cpu().float())
                all_masks.append(masks.cpu().float())
                all_raw.append(inputs[:, 0, :].cpu().float())

        prob_tensor = torch.cat(all_probs, dim=0)
        mask_tensor = torch.cat(all_masks, dim=0).unsqueeze(1)
        raw_tensor = torch.cat(all_raw, dim=0).unsqueeze(1)

        prob_1d = _stitch_predictions_1d(prob_tensor, step_samples)
        mask_1d = _stitch_predictions_1d(mask_tensor, step_samples)
        raw_1d = _stitch_predictions_1d(raw_tensor, step_samples)

        true_mask = mask_1d > 0.5
        lbl, _ = label(true_mask)
        slc = find_objects(lbl)
        true_events = [(s[0].start, s[0].stop) for s in slc]

        subject_cache[subject_id] = {
            'prob': prob_1d, 'raw': raw_1d, 'true': true_events
        }

    # 2. GRID SEARCH & TRACKING
    log.info("Phase 2: Grid Search & Optimization...")

    keys = PARAM_GRID.keys()
    values = (PARAM_GRID[key] for key in keys)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_global_f1 = 0.0
    best_global_cfg = {}

    best_personal_results = {sub: {'f1': 0.0, 'res_str': '', 'cfg': {}} for sub in ALL_SUBJECTS}

    fixed_merge_samp = int(FIXED_MERGE_GAP * fs)

    for cfg in tqdm(combinations, desc="Testing Configs"):
        if not cfg['use_power_check'] and cfg['power_ratio'] != PARAM_GRID['power_ratio'][0]:
            continue

        subject_f1s = []
        min_samp = int(cfg['min_duration'] * fs)

        for subject_id in ALL_SUBJECTS:
            if subject_id not in subject_cache: continue
            data = subject_cache[subject_id]

            preds = find_events_fast(
                data['prob'], data['raw'],
                cfg['threshold'], min_samp, fixed_merge_samp,
                cfg['use_power_check'], cfg['power_ratio'], fs
            )

            f1, prec, rec, tp, fp = calculate_f1_fast(preds, data['true'])
            subject_f1s.append(f1)

            if f1 > best_personal_results[subject_id]['f1']:
                res_str = f"Thresh {cfg['threshold']}: F1 {f1:.4f} (P: {prec:.2f}, R: {rec:.2f}) - TP {tp} / FP {fp}"
                best_personal_results[subject_id] = {
                    'f1': f1,
                    'res_str': res_str,
                    'cfg': cfg.copy()
                }

        avg_f1 = np.mean(subject_f1s)

        if avg_f1 > best_global_f1:
            best_global_f1 = avg_f1
            best_global_cfg = cfg.copy()


    # 1. GLOBAL BEST
    log.info("=" * 60)
    log.info(f"WINNING GLOBAL CONFIGURATION (Highest Average F1: {best_global_f1:.4f})")
    log.info(f"Settings: {best_global_cfg}")
    log.info("-" * 60)
    log.info("Detailed Results with Global Config:")

    global_stats = {'f1': [], 'prec': [], 'rec': []}

    for subject_id in ALL_SUBJECTS:
        if subject_id not in subject_cache: continue
        data = subject_cache[subject_id]
        cfg = best_global_cfg

        preds = find_events_fast(
            data['prob'], data['raw'],
            cfg['threshold'], int(cfg['min_duration'] * fs), fixed_merge_samp,
            cfg['use_power_check'], cfg['power_ratio'], fs
        )
        f1, prec, rec, tp, fp = calculate_f1_fast(preds, data['true'])

        global_stats['f1'].append(f1)
        global_stats['prec'].append(prec)
        global_stats['rec'].append(rec)

        log.info(f"{subject_id}: F1 {f1:.4f} (P: {prec:.2f}, R: {rec:.2f}) - TP {tp} / FP {fp}")

    log.info("-" * 40)
    log.info(f"GLOBAL AVERAGE F1: {np.mean(global_stats['f1']):.4f}")
    log.info(f"GLOBAL AVERAGE PREC: {np.mean(global_stats['prec']):.4f}")
    log.info(f"GLOBAL AVERAGE REC:  {np.mean(global_stats['rec']):.4f}")
    log.info("=" * 60)

    # 2. PERSONAL BESTS (ORACLE)
    log.info("BEST POSSIBLE RESULT PER SUBJECT (Oracle / Personal Best)")
    log.info("This shows the potential if params were tuned per subject:")
    log.info("-" * 60)

    oracle_f1s = []

    for subject_id in ALL_SUBJECTS:
        res = best_personal_results[subject_id]
        oracle_f1s.append(res['f1'])
        log.info(f"BEST RESULT for {subject_id}: {res['res_str']}")
        log.info(f"   Config: {res['cfg']}")

    log.info("-" * 40)
    log.info(f"ORACLE AVERAGE F1: {np.mean(oracle_f1s):.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_master_optimization()
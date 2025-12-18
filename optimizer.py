# optimizer.py

import torch
import numpy as np
import pandas as pd
import logging
import itertools
import argparse
import sys
from pathlib import Path
from scipy.ndimage import label, find_objects
from scipy.signal import welch
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.logger import setup_logging
from core.dataset import get_dataloaders
from core.model import GatedUNet
from core.metrics import _stitch_predictions_1d_spindleunet
from configs.dreams_config import DATA_PARAMS, METRIC_PARAMS

# --- CONFIGURATION ---
ALL_SUBJECTS = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']
MANUAL_RUN_NAME = 'X1-2'

PARAM_GRID = {
    'model_mode': ['Best', 'SWA', 'Ensemble'],
    'threshold': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    'min_duration': [0.5],
    'merge_gap': [0.15],
    'use_power_check': [False],
    'power_ratio': [0.12, 0.15, 0.18],
    'freq_low': [11.0],
    'freq_high': [16.0]
}

setup_logging("optimizer.log")
log = logging.getLogger(__name__)


def verify_spindle_power(signal_segment, fs, threshold_ratio, f_low, f_high):
    n = len(signal_segment)
    if n < int(0.1 * fs): return False
    try:
        freqs, psd = welch(signal_segment, fs, nperseg=min(n, 256))
    except:
        return False
    idx_sigma = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    idx_total = np.where((freqs >= 0.5) & (freqs <= 30.0))[0]
    if len(idx_sigma) < 1 or len(idx_total) < 1: return False
    power_sigma = np.trapz(psd[idx_sigma], freqs[idx_sigma])
    power_total = np.trapz(psd[idx_total], freqs[idx_total])
    return (power_sigma / (power_total + 1e-9)) >= threshold_ratio


def process_single_combination(cfg, subject_cache, fs):
    total_tp, total_fp, total_fn = 0, 0, 0
    subject_results = {}
    min_samp = int(cfg['min_duration'] * fs)
    max_samp = int(METRIC_PARAMS['max_duration_sec'] * fs)
    merge_samp = int(cfg['merge_gap'] * fs)
    iou_thresh = METRIC_PARAMS['iou_threshold']

    for subject_id, data in subject_cache.items():
        mode = cfg['model_mode']
        if mode == 'Ensemble':
            if data['best'] is not None and data['swa'] is not None:
                prob_map = (data['best'] + data['swa']) / 2.0
            else:
                prob_map = data['best'] if data['best'] is not None else data['swa']
        elif mode == 'Best':
            prob_map = data['best']
        else:  # SWA
            prob_map = data['swa']

        if prob_map is None:
            subject_results[subject_id] = 0.0
            continue

        mask = prob_map > cfg['threshold']
        labeled, _ = label(mask)
        raw_slices = find_objects(labeled)
        raw_events = [(s[0].start, s[0].stop) for s in raw_slices] if raw_slices else []

        merged = []
        if raw_events:
            curr_start, curr_end = raw_events[0]
            for i in range(1, len(raw_events)):
                next_start, next_end = raw_events[i]
                if (next_start - curr_end) < merge_samp:
                    curr_end = next_end
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        final_preds = []
        for start, end in merged:
            dur = end - start
            if min_samp <= dur <= max_samp:
                if cfg['use_power_check']:
                    segment = data['raw'][start:end]
                    if verify_spindle_power(segment, fs, cfg['power_ratio'], cfg['freq_low'], cfg['freq_high']):
                        final_preds.append((start, end))
                else:
                    final_preds.append((start, end))

        tp, matched = 0, set()
        true_events = data['true']
        for p in final_preds:
            best_iou, best_idx = 0, -1
            for j, t in enumerate(true_events):
                if j in matched: continue
                s1, e1 = p;
                s2, e2 = t
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou: best_iou = iou; best_idx = j
            if best_iou >= iou_thresh:
                tp += 1;
                matched.add(best_idx)

        fp, fn = len(final_preds) - tp, len(true_events) - tp
        total_tp += tp;
        total_fp += fp;
        total_fn += fn
        prec, rec = tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6)
        subject_results[subject_id] = 2 * prec * rec / (prec + rec + 1e-6)

    avg_f1 = np.mean(list(subject_results.values()))
    return {**cfg, 'Avg_F1': avg_f1, 'TP': total_tp, 'FP': total_fp, 'FN': total_fn, **subject_results}


def run_optimizer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=MANUAL_RUN_NAME)
    args = parser.parse_args()

    log.info("--- STARTING OPTIMIZATION WITH MODEL COMPARISON ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    project_root = Path(__file__).resolve().parent
    reports_root = project_root / "model_reports"
    latest_run_dir = reports_root / args.run_name

    subject_cache = {}
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    # Phase 1: Inference
    log.info("PHASE 1: Generating model predictions...")
    for i, subject_id in enumerate(ALL_SUBJECTS):
        log.info(f"Processing {subject_id}...")
        fold_dir = list(latest_run_dir.glob(f"*Fold_{i + 1}*"))
        if not fold_dir: continue
        fold_dir = fold_dir[0]

        _, _, test_loader = get_dataloaders(str(project_root / "data" / "processed"), 16, [], [], [subject_id], 1.0)

        m_best = GatedUNet(dropout_rate=0.0).to(device)
        m_swa = GatedUNet(dropout_rate=0.0).to(device)

        has_b, has_s = False, False
        if (fold_dir / 'unet_model_best.pth').exists():
            m_best.load_state_dict(torch.load(fold_dir / 'unet_model_best.pth', map_location=device))
            m_best.eval();
            has_b = True
        if (fold_dir / 'unet_model_swa.pth').exists():
            m_swa.load_state_dict(torch.load(fold_dir / 'unet_model_swa.pth', map_location=device))
            m_swa.eval();
            has_s = True

        pb_list, ps_list, m_list, r_list = [], [], [], []
        with torch.no_grad():
            for inputs, masks, _ in tqdm(test_loader, desc=f"  Inference {subject_id}", leave=False):
                inputs = inputs.to(device)
                inv = torch.flip(inputs, dims=[2])

                if has_b:
                    m, g = m_best(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)
                    m_f, g_f = m_best(inv)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)
                    pb_list.append(((p + p_f) / 2).cpu())

                if has_s:
                    m, g = m_swa(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)
                    m_f, g_f = m_swa(inv)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)
                    ps_list.append(((p + p_f) / 2).cpu())

                m_list.append(masks.cpu());
                r_list.append(inputs[:, 0, :].cpu())

        mask_1d = _stitch_predictions_1d_spindleunet(torch.cat(m_list, 0).unsqueeze(1), step_samples)
        lbl, _ = label(mask_1d > 0.5)
        true_ev = [(s[0].start, s[0].stop) for s in find_objects(lbl)] if lbl.any() else []

        subject_cache[subject_id] = {
            'raw': _stitch_predictions_1d_spindleunet(torch.cat(r_list, 0).unsqueeze(1), step_samples),
            'true': true_ev,
            'best': _stitch_predictions_1d_spindleunet(torch.cat(pb_list, 0), step_samples) if pb_list else None,
            'swa': _stitch_predictions_1d_spindleunet(torch.cat(ps_list, 0), step_samples) if ps_list else None
        }

    # Phase 2: Grid Search
    log.info("PHASE 2: Running Grid Search...")
    keys = PARAM_GRID.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*(PARAM_GRID[k] for k in keys))]

    results = Parallel(n_jobs=-1)(
        delayed(process_single_combination)(cfg, subject_cache, fs)
        for cfg in tqdm(combinations, desc="Grid Search")
    )

    df = pd.DataFrame(results).sort_values('Avg_F1', ascending=False)
    df.to_csv("optimization_results_full.csv", index=False)

    # Vertailu eri mallityyppien välillä
    log.info("\n" + "-" * 30 + "\nMODEL MODE COMPARISON:\n" + "-" * 30)
    for mode in ['Best', 'SWA', 'Ensemble']:
        mode_best = df[df['model_mode'] == mode].iloc[0]
        log.info(f"{mode:8} -> F1: {mode_best['Avg_F1']:.4f} (Thresh: {mode_best['threshold']})")

    # Voittaja-asetukset
    best = df.iloc[0]
    log.info("\n" + "=" * 50 + "\nWINNER CONFIGURATION DETAILS\n" + "=" * 50)
    log.info(f"Model Mode:   {best['model_mode']}")
    log.info(f"Threshold:    {best['threshold']}")
    log.info(f"Avg F1:       {best['Avg_F1']:.4f}")
    log.info(f"Total TP/FP/FN: {int(best['TP'])} / {int(best['FP'])} / {int(best['FN'])}")
    log.info("-" * 30)
    for s in ALL_SUBJECTS:
        if s in best: log.info(f"{s}: {best[s]:.4f}")
    log.info("=" * 50)


if __name__ == "__main__":
    run_optimizer()
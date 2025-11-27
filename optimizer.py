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
from UNET_model.evaluation_metrics import _stitch_predictions_1d
from config import DATA_PARAMS

RUN_TIMESTAMP = "2025-11-27_21-25-13"
ALL_SUBJECTS = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']

# GRID
PARAM_GRID = {
    'threshold': [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99],
    'min_duration': [0.5],
    'merge_gap': [0.1, 0.2, 0.3],
    'use_power_check': [False, True],
    'power_ratio': [0.10, 0.15, 0.20, 0.25, 0.30]
}

# Mitä moodia testataan?
TEST_MODES = ['Best', 'SWA', 'Ensemble']

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
    log.info("--- STARTING MASTER OPTIMIZATION (SEPARATE MODES) ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent
    processed_data_path = project_root / "diagnostics_plots" / "processed_data"

    # Etsitään uusin LOSO-ajo
    reports_root = project_root / "model_reports"
    all_runs = sorted([d for d in reports_root.iterdir() if d.is_dir() and "LOSO_run" in d.name])
    if not all_runs:
        log.error("No LOSO runs found in model_reports!")
        return

    latest_run_dir = all_runs[-1]
    log.info(f"Using models from LATEST run: {latest_run_dir.name}")

    subject_cache = {}
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    log.info("Phase 1: Caching Model Predictions (Best & SWA separated)...")

    for i, subject_id in enumerate(ALL_SUBJECTS):
        fold_num = i + 1
        try:
            _, _, test_loader = get_dataloaders(str(processed_data_path), 16, [], [], [subject_id], 1.0)
        except:
            continue

        fold_dirs = [d for d in latest_run_dir.iterdir() if f"Fold_{fold_num}" in d.name]
        if not fold_dirs: continue
        fold_dir = fold_dirs[0]

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

        # Listat erikseen
        probs_best, probs_swa = [], []
        all_masks, all_raw = [], []

        with torch.no_grad():
            for inputs, masks, _ in test_loader:
                inputs = inputs.to(device)
                inputs_flip = torch.flip(inputs, dims=[2])

                # 1. Best Model Prediction
                if has_best:
                    m, g = model_best(inputs)
                    # Gating on disabloitu inferenssissä (palauttaa 1.0), joten pelkkä m
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)

                    m_f, g_f = model_best(inputs_flip)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)

                    avg_p = (p + p_f) / 2.0
                    probs_best.append(avg_p.cpu().float())

                # 2. SWA Model Prediction
                if has_swa:
                    m, g = model_swa(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)

                    m_f, g_f = model_swa(inputs_flip)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)

                    avg_p = (p + p_f) / 2.0
                    probs_swa.append(avg_p.cpu().float())

                all_masks.append(masks.cpu().float())
                all_raw.append(inputs[:, 0, :].cpu().float())

        # Stitching
        cache_entry = {'raw': None, 'true': None, 'best': None, 'swa': None}

        # Raw & True
        mask_tensor = torch.cat(all_masks, dim=0).unsqueeze(1)
        raw_tensor = torch.cat(all_raw, dim=0).unsqueeze(1)

        mask_1d = _stitch_predictions_1d(mask_tensor, step_samples)
        raw_1d = _stitch_predictions_1d(raw_tensor, step_samples)

        true_mask = mask_1d > 0.5
        lbl, _ = label(true_mask)
        slc = find_objects(lbl)
        true_events = [(s[0].start, s[0].stop) for s in slc]

        cache_entry['raw'] = raw_1d
        cache_entry['true'] = true_events

        # Stitch Best
        if probs_best:
            p_tensor = torch.cat(probs_best, dim=0)
            cache_entry['best'] = _stitch_predictions_1d(p_tensor, step_samples)

        # Stitch SWA
        if probs_swa:
            p_tensor = torch.cat(probs_swa, dim=0)
            cache_entry['swa'] = _stitch_predictions_1d(p_tensor, step_samples)

        subject_cache[subject_id] = cache_entry

    # 2. GRID SEARCH
    log.info("Phase 2: Grid Search & Optimization...")

    keys = PARAM_GRID.keys()
    values = (PARAM_GRID[key] for key in keys)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Tracking Global Bests
    best_global_results = {
        'Best': {'f1': 0.0, 'cfg': {}},
        'SWA': {'f1': 0.0, 'cfg': {}},
        'Ensemble': {'f1': 0.0, 'cfg': {}}
    }

    # Tracking Personal Bests (per Subject per Mode)
    # { 'excerpt1': {'Best': {f1, cfg}, 'SWA': ...} }
    personal_bests = {sub: {m: {'f1': 0.0, 'res_str': ''} for m in TEST_MODES} for sub in ALL_SUBJECTS}

    for cfg in tqdm(combinations, desc="Testing Configs"):
        if not cfg['use_power_check'] and cfg['power_ratio'] != PARAM_GRID['power_ratio'][0]:
            continue

        min_samp = int(cfg['min_duration'] * fs)
        merge_samp = int(cfg['merge_gap'] * fs)

        # Temp storage for averages
        mode_f1s = {'Best': [], 'SWA': [], 'Ensemble': []}

        for subject_id in ALL_SUBJECTS:
            if subject_id not in subject_cache: continue
            data = subject_cache[subject_id]

            # Loopataan moodit
            for mode in TEST_MODES:
                prob_map = None

                if mode == 'Best':
                    prob_map = data['best']
                elif mode == 'SWA':
                    prob_map = data['swa']
                elif mode == 'Ensemble':
                    if data['best'] is not None and data['swa'] is not None:
                        prob_map = (data['best'] + data['swa']) / 2.0
                    elif data['best'] is not None:
                        prob_map = data['best']
                    elif data['swa'] is not None:
                        prob_map = data['swa']

                if prob_map is None: continue

                # Find Events
                preds = find_events_fast(
                    prob_map, data['raw'],
                    cfg['threshold'], min_samp, merge_samp,
                    cfg['use_power_check'], cfg['power_ratio'], fs
                )

                f1, prec, rec, tp, fp = calculate_f1_fast(preds, data['true'])
                mode_f1s[mode].append(f1)

                # Update Personal Best for this mode
                if f1 > personal_bests[subject_id][mode]['f1']:
                    res_str = f"[{mode}] F1 {f1:.4f} (P:{prec:.2f} R:{rec:.2f}) | Th:{cfg['threshold']} G:{cfg['merge_gap']} Pow:{cfg['power_ratio']}"
                    personal_bests[subject_id][mode] = {
                        'f1': f1,
                        'res_str': res_str,
                        'cfg': cfg.copy()
                    }

        # Update Global Averages
        for mode in TEST_MODES:
            if not mode_f1s[mode]: continue
            avg_f1 = np.mean(mode_f1s[mode])

            if avg_f1 > best_global_results[mode]['f1']:
                best_global_results[mode]['f1'] = avg_f1
                best_global_results[mode]['cfg'] = cfg.copy()

    # --- REPORTING ---
    log.info("=" * 80)
    log.info("FINAL RESULTS BY MODE")
    log.info("=" * 80)

    for mode in TEST_MODES:
        res = best_global_results[mode]
        log.info(f"MODE: {mode.upper()}")
        log.info(f"  Best Average F1: {res['f1']:.4f}")
        log.info(f"  Best Config:     {res['cfg']}")
        log.info("-" * 40)

    log.info("\nORACLE RESULTS PER SUBJECT (Best found configuration for each)")
    log.info("-" * 80)

    for sub in ALL_SUBJECTS:
        best_mode = max(TEST_MODES, key=lambda m: personal_bests[sub][m]['f1'])
        best_entry = personal_bests[sub][best_mode]

        log.info(f"{sub} WINNER -> {best_entry['res_str']}")


if __name__ == "__main__":
    run_master_optimization()
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
from data_preprocess.dataset import get_dataloaders
from UNET_model.model import GatedUNet
from UNET_model.evaluation_metrics import _stitch_predictions_1d
from config import DATA_PARAMS

# --- CONFIGURATION ---
ALL_SUBJECTS = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']

# Jos et käytä komentoriviargumenttia, voit asettaa kansion nimen tähän manuaalisesti.
# Esim: "LOSO_run_2025-12-02_21-58-15"
MANUAL_RUN_NAME = '1_1'

# LAAJENNETTU GRID (Perustuu virheanalyysiin)
PARAM_GRID = {
    # Kokeillaan korkeampia kynnyksiä FP:n vähentämiseksi
    'threshold': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],

    # Sallitaan lyhyemmät pätkät Recallin parantamiseksi
    'min_duration': [0.5],

    # Yhdistetäänkö lähekkäiset?
    'merge_gap': [0.1],

    # Power Check -parametrit (Tärkeä Excerpt 4:lle)
    'use_power_check': [True, False],
    'power_ratio': [0.10, 0.12, 0.15, 0.18, 0.20],  # Matalampi ratio heikoille sukkuloille

    # Taajuusrajat power checkille
    'freq_low': [10.0, 11.0],
    'freq_high': [16.0, 18.0]
}

# Keskitytään vain Ensembleen optimoinnin nopeuttamiseksi
TEST_MODES = ['Ensemble']

setup_logging("optimizer.log")
log = logging.getLogger(__name__)


def verify_spindle_power(signal_segment, fs, threshold_ratio, f_low, f_high):
    """
    Tarkistaa onko segmentissä tarpeeksi sukkulatehoa suhteessa kokonaistehoon.
    Käyttää dynaamisia taajuusrajoja.
    """
    n = len(signal_segment)
    # Sallitaan hieman lyhyemmät segmentit tarkistuksessa (0.1s)
    if n < int(0.1 * fs): return False

    try:
        # Käytetään nperseg joka on max segmentin pituus tai 256
        freqs, psd = welch(signal_segment, fs, nperseg=min(n, 256))
    except:
        return False

    # Dynaamiset rajat gridistä
    idx_sigma = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    idx_total = np.where((freqs >= 0.5) & (freqs <= 30.0))[0]

    if len(idx_sigma) < 1 or len(idx_total) < 1: return False

    power_sigma = np.trapz(psd[idx_sigma], freqs[idx_sigma])
    power_total = np.trapz(psd[idx_total], freqs[idx_total])

    if power_total == 0: return False
    return (power_sigma / power_total) >= threshold_ratio


def process_single_combination(cfg, subject_cache, fs):
    """
    Laskee metriikat yhdelle parametriyhdistelmälle (cfg) kaikille potilaille.
    Tätä funktiota ajetaan rinnakkain (Parallel).
    """

    total_tp, total_fp, total_fn = 0, 0, 0
    subject_results = {}

    min_samp = int(cfg['min_duration'] * fs)
    merge_samp = int(cfg['merge_gap'] * fs)

    f_low = cfg.get('freq_low', 11.0)
    f_high = cfg.get('freq_high', 16.0)

    for subject_id, data in subject_cache.items():
        prob_map = None

        # Ensemble Logic: (Best + SWA) / 2
        if data['best'] is not None and data['swa'] is not None:
            prob_map = (data['best'] + data['swa']) / 2.0
        elif data['best'] is not None:
            prob_map = data['best']
        elif data['swa'] is not None:
            prob_map = data['swa']

        if prob_map is None:
            subject_results[subject_id] = 0.0
            continue

        # --- Event Detection Logic ---
        mask = prob_map > cfg['threshold']
        labeled, num_features = label(mask)
        raw_slices = find_objects(labeled)
        raw_events = [(s[0].start, s[0].stop) for s in raw_slices]

        # 1. Merging close events
        merged = []
        if raw_events:
            curr_start, curr_end = raw_events[0]
            for i in range(1, len(raw_events)):
                next_start, next_end = raw_events[i]
                if (next_start - curr_end) < merge_samp:
                    curr_end = next_end  # Yhdistä
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        # 2. Filtering (Duration & Power)
        final_preds = []
        for start, end in merged:
            if (end - start) >= min_samp:
                if cfg['use_power_check']:
                    # Hae vastaava pätkä raakasignaalista
                    segment = data['raw'][start:end]
                    if verify_spindle_power(segment, fs, cfg['power_ratio'], f_low, f_high):
                        final_preds.append((start, end))
                else:
                    final_preds.append((start, end))

        # 3. Scoring (IoU matching)
        tp = 0
        matched = set()
        true_events = data['true']

        # Yksinkertaistettu IoU matching optimointia varten
        for p in final_preds:
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

            if best_iou >= 0.2:  # IoU threshold fixed at 0.2 usually
                tp += 1
                matched.add(best_idx)

        fp = len(final_preds) - tp
        fn = len(true_events) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Subject specific F1
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        subject_results[subject_id] = f1

    # Global Average F1 (Macro Average over subjects)
    avg_f1 = np.mean(list(subject_results.values()))

    # Palautetaan tulos dictionaryna
    return {
        'threshold': cfg['threshold'],
        'min_duration': cfg['min_duration'],
        'merge_gap': cfg['merge_gap'],
        'use_power_check': cfg['use_power_check'],
        'power_ratio': cfg['power_ratio'],
        'freq_low': f_low,
        'freq_high': f_high,
        'Avg_F1': avg_f1,
        'Total_TP': total_tp,
        'Total_FP': total_fp,
        'Total_FN': total_fn,
        **subject_results  # Purkaa potilaskohtaiset tulokset sarakkeiksi
    }


def run_optimizer():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Run Grid Search Optimization on Trained Models")
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of the run folder in model_reports (e.g., LOSO_run_...)')
    args = parser.parse_args()

    # Määritä ajettava kansio
    run_folder_name = args.run_name if args.run_name else MANUAL_RUN_NAME

    log.info("--- STARTING OPTIMIZATION ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    # Path setup
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent
    processed_data_path = project_root / "diagnostics_plots" / "processed_data"
    reports_root = project_root / "model_reports"

    if not reports_root.exists():
        log.error(f"Report directory {reports_root} does not exist.")
        return

    # Folder selection logic
    if run_folder_name:
        latest_run_dir = reports_root / run_folder_name
        if not latest_run_dir.exists():
            log.error(f"Manual run directory not found: {latest_run_dir}")
            return
        log.info(f"Using SPECIFIED run: {latest_run_dir.name}")
    else:
        all_runs = sorted([d for d in reports_root.iterdir() if d.is_dir() and "LOSO_run" in d.name])
        if not all_runs:
            log.error("No LOSO runs found!")
            return
        latest_run_dir = all_runs[-1]
        log.info(f"Using LATEST run: {latest_run_dir.name}")

    # --- PHASE 1: CACHE PREDICTIONS (LOAD ONCE) ---
    log.info("Phase 1: Loading Data & Caching Model Predictions...")

    subject_cache = {}
    fs = DATA_PARAMS['fs']
    step_samples = int((DATA_PARAMS['window_sec'] - DATA_PARAMS['overlap_sec']) * fs)

    for i, subject_id in enumerate(ALL_SUBJECTS):
        fold_num = i + 1
        try:
            # Lataa vain Test-data (käytetään tyhjiä listoja train/val)
            _, _, test_loader = get_dataloaders(str(processed_data_path), 16, [], [], [subject_id], 1.0)
        except Exception as e:
            log.warning(f"Could not load data for {subject_id}: {e}")
            continue

        if len(test_loader) == 0:
            continue

        # Etsi fold-kansio
        fold_dirs = [d for d in latest_run_dir.iterdir() if f"Fold_{fold_num}" in d.name]
        if not fold_dirs:
            log.warning(f"Model folder for Fold {fold_num} not found. Skipping {subject_id}.")
            continue
        fold_dir = fold_dirs[0]

        # Alusta mallit
        model_best = GatedUNet(dropout_rate=0.0).to(device)
        model_swa = GatedUNet(dropout_rate=0.0).to(device)

        p_best = fold_dir / 'unet_model_best.pth'
        p_swa = fold_dir / 'unet_model_swa.pth'

        has_best = False
        has_swa = False

        if p_best.exists():
            model_best.load_state_dict(torch.load(p_best, map_location=device))
            model_best.eval()
            has_best = True

        if p_swa.exists():
            model_swa.load_state_dict(torch.load(p_swa, map_location=device))
            model_swa.eval()
            has_swa = True

        if not has_best and not has_swa:
            log.warning(f"No model weights found for {subject_id}. Skipping.")
            continue

        # Inference loop
        probs_best, probs_swa = [], []
        all_masks, all_raw = [], []

        with torch.no_grad():
            for inputs, masks, _ in test_loader:
                inputs = inputs.to(device)
                inputs_flip = torch.flip(inputs, dims=[2])

                # 1. Best Model
                if has_best:
                    m, g = model_best(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)
                    # TTA
                    m_f, g_f = model_best(inputs_flip)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)
                    avg_p = (p + p_f) / 2.0
                    probs_best.append(avg_p.cpu().float())

                # 2. SWA Model
                if has_swa:
                    m, g = model_swa(inputs)
                    p = torch.sigmoid(m) * torch.sigmoid(g).unsqueeze(2)
                    # TTA
                    m_f, g_f = model_swa(inputs_flip)
                    p_f = torch.flip(torch.sigmoid(m_f), dims=[2]) * torch.sigmoid(g_f).unsqueeze(2)
                    avg_p = (p + p_f) / 2.0
                    probs_swa.append(avg_p.cpu().float())

                all_masks.append(masks.cpu().float())
                all_raw.append(inputs[:, 0, :].cpu().float())  # Tallenna Channel 1 (Raw EEG)

        # Stitching (yhdistä ikkunat jatkuvaksi signaaliksi)
        cache_entry = {'raw': None, 'true': None, 'best': None, 'swa': None}

        mask_tensor = torch.cat(all_masks, dim=0).unsqueeze(1)
        raw_tensor = torch.cat(all_raw, dim=0).unsqueeze(1)

        mask_1d = _stitch_predictions_1d(mask_tensor, step_samples)
        raw_1d = _stitch_predictions_1d(raw_tensor, step_samples)

        # Etsi Ground Truth tapahtumat kerran
        true_mask = mask_1d > 0.5
        lbl, _ = label(true_mask)
        slc = find_objects(lbl)
        true_events = [(s[0].start, s[0].stop) for s in slc]

        cache_entry['raw'] = raw_1d
        cache_entry['true'] = true_events

        if probs_best:
            p_tensor = torch.cat(probs_best, dim=0)
            cache_entry['best'] = _stitch_predictions_1d(p_tensor, step_samples)

        if probs_swa:
            p_tensor = torch.cat(probs_swa, dim=0)
            cache_entry['swa'] = _stitch_predictions_1d(p_tensor, step_samples)

        subject_cache[subject_id] = cache_entry
        log.info(f"Cached predictions for {subject_id}")

    if not subject_cache:
        log.error("No valid data cached. Aborting.")
        return

    # --- PHASE 2: PARALLEL GRID SEARCH ---
    log.info("Phase 2: Running Parallel Grid Search...")

    keys = PARAM_GRID.keys()
    values = (PARAM_GRID[key] for key in keys)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Suodata turhat kombinaatiot pois (jos power_check=False, ratio/freq ei vaikuta)
    filtered_combinations = []
    seen_configs = set()

    for cfg in combinations:
        # Luo yksilöivä avain
        if not cfg['use_power_check']:
            # Jos check on pois, pakota nämä arvot vakioiksi duplikaattien välttämiseksi
            cfg['power_ratio'] = 0.0
            cfg['freq_low'] = 0.0
            cfg['freq_high'] = 0.0

        # Muuta dictionary tupleksi jotta voidaan käyttää setissä
        cfg_tuple = tuple(sorted(cfg.items()))
        if cfg_tuple not in seen_configs:
            seen_configs.add(cfg_tuple)
            filtered_combinations.append(cfg)

    log.info(f"Testing {len(filtered_combinations)} unique combinations using all CPU cores...")

    # Aja rinnakkain (n_jobs=-1 käyttää kaikkia ytimiä)
    results = Parallel(n_jobs=-1)(
        delayed(process_single_combination)(cfg, subject_cache, fs)
        for cfg in tqdm(filtered_combinations, desc="Optimizing")
    )

    # --- SAVE RESULTS ---
    df = pd.DataFrame(results)
    df = df.sort_values(by='Avg_F1', ascending=False)

    output_csv = "optimization_results_full.csv"
    df.to_csv(output_csv, index=False)

    log.info("=" * 80)
    log.info(f"OPTIMIZATION COMPLETE. Saved to {output_csv}")
    log.info("=" * 80)

    best_row = df.iloc[0]
    log.info(f"WINNER CONFIGURATION (F1: {best_row['Avg_F1']:.4f}):")
    log.info(f"  Threshold:    {best_row['threshold']}")
    log.info(f"  Min Duration: {best_row['min_duration']} s")
    log.info(f"  Merge Gap:    {best_row['merge_gap']} s")
    log.info(f"  Power Check:  {best_row['use_power_check']}")
    if best_row['use_power_check']:
        log.info(f"  Power Ratio:  {best_row['power_ratio']}")
        log.info(f"  Freq Band:    {best_row['freq_low']} - {best_row['freq_high']} Hz")
    log.info("-" * 40)

    # Näytä Excerpt 4:n tulos erikseen
    if 'excerpt4' in best_row:
        log.info(f"Excerpt 4 F1 with this config: {best_row['excerpt4']:.4f}")


if __name__ == "__main__":
    run_optimizer()
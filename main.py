# main.py

import gc
import os
import torch
import torch.nn as nn
import logging
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import paths

from utils.logger import setup_logging
from core.dataset import get_dataloaders
from core.model import GatedUNet, train_model
from core.evaluation import compute_event_based_metrics, find_optimal_threshold
from configs.dreams_config import (
    TRAINING_PARAMS,
    DATA_PARAMS,
    CV_CONFIG,
    INFERENCE_PARAMS,
    METRIC_PARAMS
)

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")


# --- Helper Class for Ensemble ---
class EnsembleWrapper(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        m1, g1 = self.model_a(x)
        m2, g2 = self.model_b(x)
        return (m1 + m2) / 2.0, (g1 + g2) / 2.0

    def eval(self):
        self.model_a.eval()
        self.model_b.eval()


def log_metrics(logger, label, m):
    """Helper function for clean logging of metrics."""
    logger.info(
        f"[{label:<4}] "
        f"F1: {m['F1-score']:.4f} | "
        f"Prec: {m['Precision']:.4f} | "
        f"Rec: {m['Recall']:.4f} | "
        f"TP: {int(m['TP (events)']):<3} | "
        f"FP: {int(m['FP (events)']):<3} | "
        f"FN: {int(m['FN (events)']):<3}"
    )


def log_param_dict(logger, name, d):
    """Helper function to pretty-print configuration dictionaries."""
    logger.info(f"--- {name} ---")
    for k, v in d.items():
        logger.info(f"  {k:<25}: {v}")


if __name__ == "__main__":
    set_seed(1)

    # Ensure output directory exists
    os.makedirs(paths.REPORTS_DIR, exist_ok=True)

    setup_logging("training.log")
    log = logging.getLogger(__name__)

    # --- 1. LOG CONFIGURATION ---
    log.info("EXPERIMENT CONFIGURATION")
    log_param_dict(log, "TRAINING_PARAMS", TRAINING_PARAMS)
    log_param_dict(log, "DATA_PARAMS", DATA_PARAMS)
    log_param_dict(log, "INFERENCE_PARAMS", INFERENCE_PARAMS)
    log_param_dict(log, "METRIC_PARAMS", METRIC_PARAMS)
    log_param_dict(log, "CV_CONFIG", CV_CONFIG)

    params = TRAINING_PARAMS

    # Check SWA config
    USE_SWA = params.get('use_swa', False)
    log.info(f"Starting Training. SWA={USE_SWA}, Inference Mode={INFERENCE_PARAMS['inference_mode']}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(paths.REPORTS_DIR, f"LOSO_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    all_subjects = DATA_PARAMS['subjects_list']
    all_metrics = defaultdict(list)

    selected_folds = CV_CONFIG['folds_to_run']
    folds_to_iterate = selected_folds if selected_folds else range(len(all_subjects))

    for k in folds_to_iterate:
        test_subject_id = [all_subjects[k]]
        val_subject_id = [all_subjects[(k + 1) % len(all_subjects)]]
        train_subject_ids = [s for s in all_subjects if s != test_subject_id[0] and s != val_subject_id[0]]

        fold_name = f"Fold_{k + 1}_(Test={test_subject_id[0]})"
        log.info(f"\n{'=' * 20} STARTING FOLD: {fold_name} {'=' * 20}")

        fold_output_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_output_dir, exist_ok=True)

        # 1. Load Data
        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                processed_data_dir=paths.PROCESSED_DATA_DIR,
                batch_size=params['batch_size'],
                train_subject_ids=train_subject_ids,
                val_subject_ids=val_subject_id,
                test_subject_ids=test_subject_id,
            )
        except Exception as e:
            log.error(f"Data loading failed: {e}")
            continue

        model = GatedUNet(dropout_rate=params['dropout_rate'])

        # 2. Train
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_type=params['optimizer_type'],
            learning_rate=params['learning_rate'],
            num_epochs=params['num_epochs'],
            early_stopping_patience=params['early_stopping_patience'],
            output_dir=fold_output_dir,
            fs=DATA_PARAMS['fs'],
            use_swa=USE_SWA
        )

        # 3. Evaluation - COMPARE METHODS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Best Model
        model_best = GatedUNet(dropout_rate=0.0).to(device)
        best_path = os.path.join(fold_output_dir, 'unet_model_best.pth')
        if os.path.exists(best_path):
            model_best.load_state_dict(torch.load(best_path, map_location=device))
        else:
            log.error("Best model not found!")
            continue

        # Load SWA Model
        model_swa = None
        if USE_SWA:
            swa_path = os.path.join(fold_output_dir, 'unet_model_swa.pth')
            if os.path.exists(swa_path):
                model_swa = GatedUNet(dropout_rate=0.0).to(device)
                model_swa.load_state_dict(torch.load(swa_path, map_location=device))
            else:
                log.warning("SWA model wanted but not found.")

        # Determine Threshold (using Best model usually)
        if INFERENCE_PARAMS['fixed_threshold'] is not None:
            optimal_thresh = INFERENCE_PARAMS['fixed_threshold']
            log.info(f"Using FIXED threshold: {optimal_thresh}")
        else:
            log.info("Finding optimal threshold...")
            optimal_thresh = find_optimal_threshold(model_best, val_loader)
            log.info(f"Optimal threshold determined: {optimal_thresh:.2f}")

        log.info(f"\n--- COMPARISON FOR {test_subject_id[0]} (Thresh: {optimal_thresh}) ---")
        log.info(f"{'Type':<6} {'F1':<8} {'Prec':<8} {'Rec':<8} {'TP':<5} {'FP':<5} {'FN':<5}")
        log.info("-" * 60)

        # A. Evaluate BEST
        metrics_best = compute_event_based_metrics(model_best, test_loader, optimal_thresh, test_subject_id[0],
                                                   fold_output_dir)
        log_metrics(log, "BEST", metrics_best)

        metrics_swa = None
        metrics_ens = None

        # B. Evaluate SWA
        if model_swa:
            metrics_swa = compute_event_based_metrics(model_swa, test_loader, optimal_thresh, test_subject_id[0],
                                                      fold_output_dir)
            log_metrics(log, "SWA", metrics_swa)

            # C. Evaluate ENSEMBLE
            ensemble_model = EnsembleWrapper(model_best, model_swa).to(device)
            metrics_ens = compute_event_based_metrics(ensemble_model, test_loader, optimal_thresh, test_subject_id[0],
                                                      fold_output_dir)
            log_metrics(log, "ENS", metrics_ens)

        log.info("-" * 60)

        # 4. Select Final Metrics based on Config
        selected_mode = INFERENCE_PARAMS['inference_mode']
        final_metrics = metrics_best  # default

        if selected_mode == 'swa' and metrics_swa:
            final_metrics = metrics_swa
        elif selected_mode == 'ensemble' and metrics_ens:
            final_metrics = metrics_ens

        log.info(f"--> SELECTED FINAL METRICS ({selected_mode.upper()}): F1 {final_metrics['F1-score']:.4f}")

        for key, value in final_metrics.items():
            all_metrics[key].append(value)

        del model, model_best, model_swa
        torch.cuda.empty_cache()
        gc.collect()

    log.info("\n" + "=" * 80)
    log.info("FULL LOSO CROSS-VALIDATION COMPLETE")
    if len(all_metrics) > 0:
        log.info(f"{'Metric':<15} {'Mean':<10} {'Std':<10}")
        log.info("-" * 40)
        for key, values in all_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            log.info(f"{key:<15} {mean:.4f}     (± {std:.4f})")

    log.info(f"All results saved to directory: {output_dir}")
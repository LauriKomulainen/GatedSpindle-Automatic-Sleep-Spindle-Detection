# utils/utils.py

"""
Shared experiment utilities used by run_mass.py and run_dreams.py.

Contains seed setup, logging helpers, CV split logic, output-directory
handling, ensemble model wrapping, and single-fold evaluation.
"""

import gc
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import paths
from core.model import GatedUNet
from core.evaluation import compute_event_based_metrics, find_optimal_threshold
from core.config_loader import INFERENCE_PARAMS

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def log_metrics(logger, label: str, metrics: dict):
    """Log evaluation metrics in a formatted single line."""
    if not metrics:
        return
    logger.info(
        f"{label:<4} F1: {metrics['F1-score']:.3f} | "
        f"Prec: {metrics['Precision']:.3f} | Rec: {metrics['Recall']:.3f} | "
        f"TP: {int(metrics['TP (events)']):<3} | "
        f"FP: {int(metrics['FP (events)']):<3} | "
        f"FN: {int(metrics['FN (events)']):<3}"
    )


def log_params(logger, name: str, params: dict):
    """Log a parameter dictionary."""
    logger.info(f"{name} :")
    for key, value in params.items():
        logger.info(f"  {key:<25}: {value}")


class EnsembleWrapper(nn.Module):
    """Wrapper for averaging predictions from two models."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module):
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
        return self


# Cross-Validation Splits

def get_kfold_splits(subjects: list, fold_idx: int, num_folds: int = 5):
    """K-Fold cross-validation split.

    Returns train/val/test subject IDs for a single fold. The val set is the
    "next" fold (wrapping around).
    """
    n = len(subjects)
    fold_sizes = np.full(num_folds, n // num_folds, dtype=int)
    fold_sizes[: n % num_folds] += 1

    folds, current = [], 0
    for size in fold_sizes:
        folds.append(subjects[current : current + size])
        current += size

    test_ids = folds[fold_idx]
    val_ids = folds[(fold_idx + 1) % num_folds]
    train_ids = [
        s for i, fold in enumerate(folds) for s in fold
        if i not in (fold_idx, (fold_idx + 1) % num_folds)
    ]
    fold_name = f"Fold_{fold_idx + 1}"
    return train_ids, val_ids, test_ids, fold_name, fold_name


def get_loso_splits(subjects: list, fold_idx: int):
    """Leave-One-Subject-Out split."""
    n = len(subjects)
    test_ids = [subjects[fold_idx]]
    val_ids = [subjects[(fold_idx + 1) % n]]
    train_ids = [s for s in subjects if s not in test_ids + val_ids]
    return train_ids, val_ids, test_ids, f"Fold_{fold_idx + 1}_(Test={test_ids[0]})", test_ids[0]


def get_cv_splits(cv_strategy: str, subjects: list, fold_idx: int, num_folds: int):
    """Dispatch to the correct CV split strategy."""
    if cv_strategy == 'loso':
        return get_loso_splits(subjects, fold_idx)
    elif cv_strategy == 'kfold':
        return get_kfold_splits(subjects, fold_idx, num_folds)
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}. Expected 'kfold' or 'loso'.")


def get_cv_parameters(cv_strategy: str, subjects: list, n_folds: int, logger):
    """Determine the effective number of folds and a human-readable name."""
    if cv_strategy == 'loso':
        num_folds = len(subjects)
        strategy_name = f"LOSO ({num_folds} folds, {len(subjects)} subjects)"
    elif cv_strategy == 'kfold':
        num_folds = min(n_folds, len(subjects))
        strategy_name = f"{num_folds}-Fold CV ({len(subjects)} subjects)"
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}. Expected 'kfold' or 'loso'.")
    return num_folds, strategy_name


# Output directory handling

def setup_output_dir(timestamp: str) -> str:
    """Create and return the master output directory."""
    os.makedirs(paths.REPORTS_DIR, exist_ok=True)
    master_dir = os.path.join(paths.REPORTS_DIR, f"LOSO_Experiment_{timestamp}")
    os.makedirs(master_dir, exist_ok=True)
    return master_dir


def parse_eval_directories(run_dir: str):
    """Parse evaluation directories and return master dir and repeat indices."""
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    dir_name = os.path.basename(run_dir.rstrip(os.sep))

    if dir_name.startswith("Repeat_"):
        repeat_idx = int(dir_name.split("_")[-1]) - 1
        print(f"Evaluation mode: Single repeat detected ({dir_name})")
        return os.path.dirname(run_dir.rstrip(os.sep)), [repeat_idx]

    master_dir = run_dir
    repeats = sorted(
        int(d.split("_")[-1]) - 1
        for d in os.listdir(master_dir)
        if d.startswith("Repeat_") and os.path.isdir(os.path.join(master_dir, d))
    )
    if not repeats:
        raise ValueError(f"No 'Repeat_X' folders found in {master_dir}")

    print(f"Evaluation mode: Found {len(repeats)} repeats in {dir_name}")
    return master_dir, repeats


# Fold evaluation

def evaluate_fold(fold_dir, test_subject_ids, processed_data_dir, val_loader,
                  num_channels, identifier, use_swa, logger, data_params):
    """Evaluate a single fold using best, SWA, and ensemble models.

    Per-subject inference on FULL recordings; events filtered by
    N2 stage_mask after prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = os.path.join(fold_dir, "unet_model_best.pth")

    if not os.path.exists(best_path):
        logger.warning(f"Best model not found: {best_path}")
        return None

    # Load best model
    model_best = GatedUNet(num_channels, dropout_rate=0.0).to(device)
    model_best.load_state_dict(torch.load(best_path, map_location=device))
    logger.info(f"Loaded model from {best_path}")

    # Determine threshold
    threshold = INFERENCE_PARAMS["fixed_threshold"]
    if threshold is None:
        threshold = find_optimal_threshold(model_best, val_loader)

    mode = INFERENCE_PARAMS["inference_mode"]

    # Evaluate best model
    save_dir = fold_dir if mode == "none" else None
    metrics_best = compute_event_based_metrics(
        model_best, test_subject_ids, processed_data_dir, threshold,
        data_params, f"{identifier}_best", save_dir,
    )
    log_metrics(logger, "BEST:", metrics_best)

    metrics_swa, metrics_ens, model_swa = None, None, None

    # Evaluate SWA and ensemble if enabled
    if use_swa:
        swa_path = os.path.join(fold_dir, "unet_model_swa.pth")
        if os.path.exists(swa_path):
            model_swa = GatedUNet(num_channels, dropout_rate=0.0).to(device)
            model_swa.load_state_dict(torch.load(swa_path, map_location=device))

            save_dir = fold_dir if mode == "swa" else None
            metrics_swa = compute_event_based_metrics(
                model_swa, test_subject_ids, processed_data_dir, threshold,
                data_params, f"{identifier}_swa", save_dir,
            )
            log_metrics(logger, "SWA:", metrics_swa)

            ensemble = EnsembleWrapper(model_best, model_swa).to(device)
            save_dir = fold_dir if mode == "ensemble" else None
            metrics_ens = compute_event_based_metrics(
                ensemble, test_subject_ids, processed_data_dir, threshold,
                data_params, f"{identifier}_ens", save_dir,
            )
            log_metrics(logger, "ENS:", metrics_ens)
        else:
            logger.info("SWA model not found, skipping SWA/Ensemble evaluation")

    # Select final metrics based on mode
    final_metrics = metrics_best
    if mode == "swa" and metrics_swa:
        final_metrics = metrics_swa
    elif mode == "ensemble" and metrics_ens:
        final_metrics = metrics_ens

    # Cleanup
    del model_best
    if model_swa:
        del model_swa
    torch.cuda.empty_cache()
    gc.collect()

    return final_metrics
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
def evaluate_fold(fold_dir, test_subject_ids, val_subject_ids, processed_data_dir,
                  num_channels, identifier, use_swa, logger, data_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    best_path = os.path.join(fold_dir, "unet_model_best.pth")

    if not os.path.exists(best_path):
        logger.warning(f"Best model not found: {best_path}")
        return None

    # Normalize mode: accept None, "none", "best" all as "best-only"
    mode = INFERENCE_PARAMS["inference_mode"]
    if mode in (None, "none", "best"):
        mode = "best"
    elif mode not in ("swa", "ensemble"):
        logger.warning(f"Unknown inference_mode '{mode}', defaulting to 'best'")
        mode = "best"

    threshold = INFERENCE_PARAMS["fixed_threshold"]
    if threshold is None:
        # Optimize threshold on validation subjects using full event-based pipeline.
        # Inference runs once per subject; thresholds are swept over cached probs.
        model_for_thr = GatedUNet(num_channels, dropout_rate=0.0).to(device)
        model_for_thr.load_state_dict(torch.load(best_path, map_location=device))
        threshold = find_optimal_threshold(
            model_for_thr,
            val_subject_ids,
            processed_data_dir,
            data_params,
            logger=logger,
        )
        del model_for_thr
        torch.cuda.empty_cache()
        gc.collect()

    # Always load BEST (needed for "best" and "ensemble" modes)
    model_best = GatedUNet(num_channels, dropout_rate=0.0).to(device)
    model_best.load_state_dict(torch.load(best_path, map_location=device))
    logger.info(f"Loaded model from {best_path}")
    logger.info(f"Using threshold: {threshold:.3f}")

    final_metrics = None

    if mode == "best":
        final_metrics = compute_event_based_metrics(
            model_best, test_subject_ids, processed_data_dir, threshold,
            data_params, f"{identifier}_best", fold_dir,
        )
        log_metrics(logger, "BEST:", final_metrics)

    elif mode == "swa":
        swa_path = os.path.join(fold_dir, "unet_model_swa.pth")
        if not os.path.exists(swa_path):
            logger.error(f"SWA mode requested but {swa_path} not found")
            del model_best
            return None
        model_swa = GatedUNet(num_channels, dropout_rate=0.0).to(device)
        model_swa.load_state_dict(torch.load(swa_path, map_location=device))
        final_metrics = compute_event_based_metrics(
            model_swa, test_subject_ids, processed_data_dir, threshold,
            data_params, f"{identifier}_swa", fold_dir,
        )
        log_metrics(logger, "SWA:", final_metrics)
        del model_swa

    elif mode == "ensemble":
        swa_path = os.path.join(fold_dir, "unet_model_swa.pth")
        if not os.path.exists(swa_path):
            logger.error(f"Ensemble mode requested but {swa_path} not found")
            del model_best
            return None
        model_swa = GatedUNet(num_channels, dropout_rate=0.0).to(device)
        model_swa.load_state_dict(torch.load(swa_path, map_location=device))
        ensemble = EnsembleWrapper(model_best, model_swa).to(device)
        final_metrics = compute_event_based_metrics(
            ensemble, test_subject_ids, processed_data_dir, threshold,
            data_params, f"{identifier}_ens", fold_dir,
        )
        log_metrics(logger, "ENS:", final_metrics)
        del model_swa

    del model_best
    torch.cuda.empty_cache()
    gc.collect()
    return final_metrics
# main.py

import gc
import os
import shutil
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import paths
from utils.logger import setup_logging
from core.dataset import get_dataloaders
from core.train_model import train_model
from core.model import GatedUNet
from core.evaluation import (
    compute_event_based_metrics,
    find_optimal_threshold,
    aggregate_and_save_summary,
    save_final_experiment_summary,
)
from configs.model_config import INFERENCE_PARAMS, TRAINING_PARAMS, POST_PROCESSING_PARAMS
from configs.config_loader import DATA_PARAMS, CV_CONFIG, SELECTED_DATASET


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def filter_valid_subjects(subjects: list, data_dir: str, logger) -> list:
    """Filter subjects that have valid data files on disk."""
    valid, missing = [], []
    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_X_1D.npy")
        y_path = os.path.join(data_dir, f"{subj}_Y_1D.npy")
        (valid if os.path.exists(x_path) and os.path.exists(y_path) else missing).append(subj)

    if missing:
        logger.warning(f"Filtered out {len(missing)} subjects with missing data: {missing}")
    logger.info(f"Valid subjects available: {len(valid)}")
    return valid


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


def get_loso_splits(subjects: list, fold_idx: int):
    """Leave-One-Subject-Out split."""
    n = len(subjects)
    test_ids = [subjects[fold_idx]]
    val_ids = [subjects[(fold_idx + 1) % n]]
    train_ids = [s for s in subjects if s not in test_ids + val_ids]
    return train_ids, val_ids, test_ids, f"Fold_{fold_idx + 1}_(Test={test_ids[0]})", test_ids[0]


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


def evaluate_fold(fold_dir, test_loader, val_loader, num_channels, identifier, use_swa, logger):
    """Evaluate a single fold using best, SWA, and ensemble models."""
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
    metrics_best = compute_event_based_metrics(model_best, test_loader, threshold, f"{identifier}_best", save_dir)
    log_metrics(logger, "BEST:", metrics_best)

    metrics_swa, metrics_ens, model_swa = None, None, None

    # Evaluate SWA and ensemble if enabled
    if use_swa:
        swa_path = os.path.join(fold_dir, "unet_model_swa.pth")
        if os.path.exists(swa_path):
            model_swa = GatedUNet(num_channels, dropout_rate=0.0).to(device)
            model_swa.load_state_dict(torch.load(swa_path, map_location=device))

            save_dir = fold_dir if mode == "swa" else None
            metrics_swa = compute_event_based_metrics(model_swa, test_loader, threshold, f"{identifier}_swa", save_dir)
            log_metrics(logger, "SWA:", metrics_swa)

            ensemble = EnsembleWrapper(model_best, model_swa).to(device)
            save_dir = fold_dir if mode == "ensemble" else None
            metrics_ens = compute_event_based_metrics(ensemble, test_loader, threshold, f"{identifier}_ens", save_dir)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1, nargs="?", const=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--shuffle_folds", action="store_true")
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(1, 99999)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Setup directories and logging
    if args.mode == "train":
        master_dir = setup_output_dir(timestamp)
        repeats = list(range(args.repeats))
        log_file = "training.log"
    else:
        if not args.run_dir:
            raise ValueError("--run_dir is required for evaluate mode")
        master_dir, repeats = parse_eval_directories(args.run_dir)
        log_file = "evaluation_rerun.log"

    setup_logging(log_file)
    log = logging.getLogger(__name__)

    log.info(f"Dataset: {SELECTED_DATASET}")
    log_params(log, "Training params", TRAINING_PARAMS)
    log_params(log, "Data params", DATA_PARAMS)
    log_params(log, "Data params", INFERENCE_PARAMS)
    log_params(log, "Data params", POST_PROCESSING_PARAMS)

    # Filter valid subjects
    all_subjects = filter_valid_subjects(list(DATA_PARAMS["subjects_list"]), paths.PROCESSED_DATA_DIR, log)
    if not all_subjects:
        log.error("No valid subjects found! Check the processed_data directory.")
        return

    # DREAMS uses LOSO cross-validation
    use_swa = TRAINING_PARAMS.get("use_swa", False)
    num_folds = len(all_subjects)
    cv_strategy = "LOSO"

    log.info(f"Cross validation strategy: {cv_strategy} ({num_folds} folds)")
    folds_to_run = CV_CONFIG.get("folds_to_run") or range(num_folds)
    grand_results = defaultdict(list)

    # Main training/evaluation loop
    for repeat_idx in repeats:
        current_seed = base_seed + repeat_idx
        set_seed(current_seed)
        log.info(f"Repeat {repeat_idx + 1} / {len(repeats)} (Seed: {current_seed})")

        repeat_dir = os.path.join(master_dir, f"Repeat_{repeat_idx + 1}")
        if args.mode == "train":
            os.makedirs(repeat_dir, exist_ok=True)
        elif not os.path.exists(repeat_dir):
            continue

        subjects = all_subjects.copy()
        if args.shuffle_folds:
            random.shuffle(subjects)
            log.info("Subjects shuffled.")

        repeat_metrics = defaultdict(list)

        for fold_idx in folds_to_run:
            # Get train/val/test split (LOSO)
            train_ids, val_ids, test_ids, fold_name, identifier = get_loso_splits(subjects, fold_idx)

            log.info(f"{fold_name}")
            log.info(f"Train ({len(train_ids)}): {train_ids}")
            log.info(f"Val ({len(val_ids)}): {val_ids}")
            log.info(f"Test ({len(test_ids)}): {test_ids}")

            fold_dir = os.path.join(repeat_dir, fold_name)
            if args.mode == "train":
                os.makedirs(fold_dir, exist_ok=True)
            elif not os.path.exists(fold_dir):
                continue

            # Load data
            try:
                train_loader, val_loader, test_loader = get_dataloaders(
                    processed_data_dir=paths.PROCESSED_DATA_DIR,
                    batch_size=TRAINING_PARAMS["batch_size"],
                    train_subject_ids=train_ids,
                    val_subject_ids=val_ids,
                    test_subject_ids=test_ids,
                )
                if len(train_loader) == 0:
                    log.error("Train loader is empty! Skipping.")
                    continue

                sample = val_loader.dataset[0][0] if len(val_loader) > 0 else train_loader.dataset[0][0]
                num_channels = sample.shape[0]
            except Exception as e:
                log.error(f"Data loading failed: {e}")
                continue

            # Train model
            if args.mode == "train":
                model = GatedUNet(num_channels, dropout_rate=TRAINING_PARAMS["dropout_rate"])
                train_model(
                    model,
                    train_loader,
                    val_loader,
                    TRAINING_PARAMS["learning_rate"],
                    TRAINING_PARAMS["num_epochs"],
                    TRAINING_PARAMS["early_stopping_patience"],
                    fold_dir,
                    use_swa,
                )
                del model
                torch.cuda.empty_cache()

            # Evaluate
            try:
                metrics = evaluate_fold(fold_dir, test_loader, val_loader, num_channels, identifier, use_swa, log)
                if metrics:
                    for key, value in metrics.items():
                        repeat_metrics[key].append(value)
            except Exception as e:
                log.error(f"Evaluation failed: {e}")

        # Aggregate results
        summary = aggregate_and_save_summary(repeat_metrics, repeat_dir, repeat_idx, current_seed, log)
        for key, val in summary.items():
            grand_results[key].append(val)

    # Save final summary
    save_final_experiment_summary(
        grand_results, master_dir, len(repeats), timestamp if args.mode == "train" else "eval", log
    )

    # Copy log file to output directory
    logging.shutdown()
    try:
        shutil.copy2(os.path.join("logs", log_file), os.path.join(master_dir, log_file))
    except Exception:
        pass


if __name__ == "__main__":
    main()
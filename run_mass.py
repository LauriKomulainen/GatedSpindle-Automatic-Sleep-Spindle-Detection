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
from configs.mass_model_config import INFERENCE_PARAMS, TRAINING_PARAMS, POST_PROCESSING_PARAMS
from configs.mass_config import DATA_PARAMS, CV_CONFIG


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


# Check which subjects have the required data files on disk.
def filter_valid_subjects(subjects: list, data_dir: str, logger, scorer_mode: str = None) -> list:
    valid, missing = [], []
    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_X_1D.npy")

        # Check scorer-specific Y file first, fall back to default
        y_found = False
        if scorer_mode and scorer_mode in ('E1', 'E2', 'UNION'):
            y_scorer_path = os.path.join(data_dir, f"{subj}_Y_{scorer_mode}.npy")
            if os.path.exists(y_scorer_path):
                # For E2 mode, also verify the file is non-trivial (has actual spindle annotations)
                if scorer_mode == 'E2':
                    try:
                        y_data = np.load(y_scorer_path, mmap_mode='r')
                        if y_data.max() < 0.5:
                            logger.info(f"  {subj}: Y_E2.npy exists but has no spindle annotations, skipping")
                            missing.append(subj)
                            continue
                    except Exception:
                        pass
                y_found = True

        if not y_found:
            y_default = os.path.join(data_dir, f"{subj}_Y_1D.npy")
            y_found = os.path.exists(y_default)

        if os.path.exists(x_path) and y_found:
            valid.append(subj)
        else:
            missing.append(subj)

    if missing:
        logger.warning(f"Filtered out {len(missing)} subjects (scorer={scorer_mode or 'default'}): {missing}")
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
        logger.info(f"{key:<25}: {value}")


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


# Cross-Validation Split Strategies
def generate_seed_folds(subject_ids: list, n_folds: int = 5, seed: int = 0):
    """
    Generate random CV folds matching SEED's cv_split logic.

    Uses a seeded random permutation to split subjects into n_folds groups,
    then assigns test = fold[i], val = fold[(i+1) % n_folds], train = rest.
    Includes SEED's duplicate-avoidance retry mechanism for edge cases.

    Args:
        subject_ids: list of subject ID strings
        n_folds: number of folds (default 5)
        seed: random seed (varies per repeat to produce different partitions)

    Returns:
        list of dicts with 'test' and 'val' keys for each fold
    """
    subjects = np.array(subject_ids)
    n_subjects = len(subjects)
    n_per_fold = int(np.ceil(n_subjects / n_folds))

    attempts = 1
    while True:
        perm_1 = np.random.RandomState(seed=seed).permutation(n_subjects)
        perm_2 = np.random.RandomState(seed=seed + attempts).permutation(n_subjects)
        extended = np.concatenate([perm_1, perm_2])

        test_folds = []
        for i in range(n_folds):
            start = i * n_per_fold
            end = (i + 1) * n_per_fold
            indices = np.unique(extended[start:end])
            test_folds.append(subjects[indices].tolist())

        # Validate: no duplicates across adjacent folds (test + val)
        total_elements = sum(len(f) for f in test_folds)
        cond1 = total_elements == n_per_fold * n_folds

        val_folds = [test_folds[(i + 1) % n_folds] for i in range(n_folds)]
        cond2 = all(
            len(set(t) & set(v)) == 0
            for t, v in zip(test_folds, val_folds)
        )

        if cond1 and cond2:
            break
        attempts += 1000

    folds = []
    for i in range(n_folds):
        folds.append({
            'test': test_folds[i],
            'val': test_folds[(i + 1) % n_folds],
        })
    return folds


def get_kfold_splits(subjects: list, fold_idx: int, num_folds: int = 5):
    """K-Fold cross-validation split."""
    n = len(subjects)
    fold_sizes = np.full(num_folds, n // num_folds, dtype=int)
    fold_sizes[: n % num_folds] += 1

    folds, current = [], 0
    for size in fold_sizes:
        folds.append(subjects[current : current + size])
        current += size

    test_ids = folds[fold_idx]
    val_ids = folds[(fold_idx + 1) % num_folds]
    train_ids = [s for i, fold in enumerate(folds) for s in fold if i not in (fold_idx, (fold_idx + 1) % num_folds)]

    return train_ids, val_ids, test_ids, f"Fold_{fold_idx + 1}", f"Fold_{fold_idx + 1}"


def get_explicit_fold_splits(fold_config: dict, all_subjects: list, fold_idx: int):
    """
    Get train/val/test splits from explicitly defined fold configurations.
    Used for SEED and Spindle-UMamba comparison experiments.
    """
    fold = fold_config['folds'][fold_idx]
    test_ids = fold['test']
    val_ids = fold['val']
    train_ids = [s for s in all_subjects if s not in test_ids and s not in val_ids]

    test_str = ",".join([s.split("-")[-1].lstrip("0") or "0" for s in test_ids])
    fold_name = f"Fold_{fold_idx + 1}_(Test={test_str})"
    identifier = f"Fold_{fold_idx + 1}"

    return train_ids, val_ids, test_ids, fold_name, identifier


def get_cv_splits(cv_strategy: str, subjects: list, fold_idx: int, num_folds: int, seed: int = 0):
    """Router function that dispatches to the correct CV split strategy.

    Note: 'subjects' is the scorer-filtered subject list from get_cv_parameters.
    For explicit fold strategies, subjects within each fold that are not in
    the eligible list are automatically excluded.
    """
    if cv_strategy == 'seed_15':
        from configs.mass_config import SEED_CONFIG
        eligible = [s for s in SEED_CONFIG['subjects_15'] if s in subjects]
        folds = generate_seed_folds(eligible, n_folds=SEED_CONFIG['n_folds'], seed=seed)
        fold_config = {'folds': folds}
        return get_explicit_fold_splits(fold_config, eligible, fold_idx)

    elif cv_strategy == 'umamba_19':
        from configs.mass_config import UMAMBA_CONFIG
        # Filter each fold's test/val to only include eligible subjects
        filtered_folds = []
        for fold in UMAMBA_CONFIG['folds']:
            filtered_folds.append({
                'test': [s for s in fold['test'] if s in subjects],
                'val': [s for s in fold['val'] if s in subjects],
            })
        fold_config = {'folds': filtered_folds}
        return get_explicit_fold_splits(fold_config, subjects, fold_idx)

    elif cv_strategy == 'loso':
        return get_loso_splits(subjects, fold_idx)

    else:  # 'kfold' or default
        return get_kfold_splits(subjects, fold_idx, num_folds)


def get_loso_splits(subjects: list, fold_idx: int):
    """Leave-One-Subject-Out split."""
    n = len(subjects)
    test_ids = [subjects[fold_idx]]
    val_ids = [subjects[(fold_idx + 1) % n]]
    train_ids = [s for s in subjects if s not in test_ids + val_ids]
    return train_ids, val_ids, test_ids, f"Fold_{fold_idx + 1}_(Test={test_ids[0]})", test_ids[0]


def get_cv_parameters(cv_strategy: str, all_subjects: list, logger, scorer_mode: str = None):
    """
    Determine the subject list and number of folds for a given CV strategy.

    The subject list is first filtered by scorer_mode (E1=19, E2=15, UNION=15)
    to ensure only subjects with valid annotations are included in CV.
    """
    # Filter subjects by scorer mode
    if scorer_mode and scorer_mode in ('E1', 'E2', 'UNION'):
        from configs.mass_config import SUBJECTS_BY_SCORER
        scorer_subjects = SUBJECTS_BY_SCORER[scorer_mode]
        eligible_subjects = [s for s in all_subjects if s in scorer_subjects]
        logger.info(f"Scorer mode '{scorer_mode}': {len(eligible_subjects)}/{len(all_subjects)} subjects eligible")
    else:
        eligible_subjects = all_subjects

    if cv_strategy == 'seed_15':
        from configs.mass_config import SEED_CONFIG
        subjects = [s for s in SEED_CONFIG['subjects_15'] if s in eligible_subjects]
        num_folds = SEED_CONFIG['n_folds']
        strategy_name = f"SEED-style 5-Fold CV ({len(subjects)}/15 subjects, random splits per repeat)"

    elif cv_strategy == 'umamba_19':
        from configs.mass_config import UMAMBA_CONFIG
        subjects = [s for s in UMAMBA_CONFIG['subjects_19'] if s in eligible_subjects]
        num_folds = len(UMAMBA_CONFIG['folds'])
        strategy_name = f"UMamba-style 5-Fold CV ({len(subjects)}/{len(UMAMBA_CONFIG['subjects_19'])} subjects)"

    elif cv_strategy == 'loso':
        subjects = eligible_subjects
        num_folds = len(subjects)
        strategy_name = f"LOSO ({num_folds} folds)"

    else:  # 'kfold'
        subjects = eligible_subjects
        num_folds = min(5, len(subjects))
        strategy_name = f"5-Fold CV ({num_folds} folds, {len(subjects)} subjects)"

    return subjects, num_folds, strategy_name


# =============================================================================
# Evaluation
# =============================================================================

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
    metrics_best = compute_event_based_metrics(model_best, test_loader, threshold, DATA_PARAMS, f"{identifier}_best", save_dir)
    log_metrics(logger, "BEST:", metrics_best)

    metrics_swa, metrics_ens, model_swa = None, None, None

    # Evaluate SWA and ensemble if enabled
    if use_swa:
        swa_path = os.path.join(fold_dir, "unet_model_swa.pth")
        if os.path.exists(swa_path):
            model_swa = GatedUNet(num_channels, dropout_rate=0.0).to(device)
            model_swa.load_state_dict(torch.load(swa_path, map_location=device))

            save_dir = fold_dir if mode == "swa" else None
            metrics_swa = compute_event_based_metrics(model_swa, test_loader, threshold, DATA_PARAMS, f"{identifier}_swa", save_dir)
            log_metrics(logger, "SWA:", metrics_swa)

            ensemble = EnsembleWrapper(model_best, model_swa).to(device)
            save_dir = fold_dir if mode == "ensemble" else None
            metrics_ens = compute_event_based_metrics(ensemble, test_loader, threshold, DATA_PARAMS, f"{identifier}_ens", save_dir)
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


# =============================================================================
# Main
# =============================================================================

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

    log.info(f"Dataset: MASS")
    log_params(log, "Training params", TRAINING_PARAMS)
    log_params(log, "Data params", DATA_PARAMS)
    log_params(log, "Data params", INFERENCE_PARAMS)
    log_params(log, "Data params", POST_PROCESSING_PARAMS)


    # Filter valid subjects
    scorer_mode = DATA_PARAMS.get("scorer_mode", None)
    all_subjects = filter_valid_subjects(list(DATA_PARAMS["subjects_list"]), paths.PROCESSED_DATA_DIR, log, scorer_mode)
    if not all_subjects:
        log.error("No valid subjects found! Check the processed_data directory.")
        return

    # Determine CV strategy
    use_swa = TRAINING_PARAMS.get("use_swa", False)
    cv_strategy = CV_CONFIG.get("cv_strategy", "kfold")

    # Get subjects and fold count for chosen strategy
    subjects, num_folds, strategy_name = get_cv_parameters(cv_strategy, all_subjects, log, scorer_mode)

    log.info(f"Cross validation strategy: {strategy_name}")
    log.info(f"Scorer mode: {DATA_PARAMS.get('scorer_mode', 'N/A')}")

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

        # For explicit fold strategies, shuffling is disabled
        fold_subjects = subjects.copy()
        if args.shuffle_folds and cv_strategy not in ('seed_15', 'umamba_19'):
            random.shuffle(fold_subjects)
            log.info("Subjects shuffled.")
        elif args.shuffle_folds and cv_strategy in ('seed_15', 'umamba_19'):
            log.warning("--shuffle_folds ignored for explicit fold strategies")

        repeat_metrics = defaultdict(list)

        for fold_idx in folds_to_run:
            # Get train/val/test split
            train_ids, val_ids, test_ids, fold_name, identifier = get_cv_splits(
                cv_strategy, fold_subjects, fold_idx, num_folds, seed=current_seed
            )

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

        # Aggregate results for this repeat
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
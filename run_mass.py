# run_mass.py

import os
import shutil
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import paths
from utils.logger import setup_logging
from core.dataset import get_dataloaders
from core.train_model import train_model
from core.model import ResidualUNet1D
from utils.run_utils import (
    set_seed,
    log_params,
    setup_output_dir,
    parse_eval_directories,
    evaluate_fold,
    get_cv_splits,
    get_cv_parameters,
)
from core.evaluation import (
    aggregate_and_save_summary,
    save_final_experiment_summary,
)
from configs.mass_model_config import (
    INFERENCE_PARAMS,
    TRAINING_PARAMS,
    POST_PROCESSING_PARAMS,
)
from configs.mass_config import (
    DATA_PARAMS, CV_CONFIG,
    get_subjects_for_scorer,
    get_explicit_splits,
)


def filter_valid_subjects(subjects: list, data_dir: str, logger, scorer_mode: str) -> list:
    """Check which subjects have the required data files on disk for the given scorer_mode.

    Strict check: the scorer-specific Y file must exist. No fallback to Y_1D.npy.
    """
    valid, missing = [], []
    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_X_1D.npy")
        y_scorer_path = os.path.join(data_dir, f"{subj}_Y_{scorer_mode}.npy")

        if not os.path.exists(x_path):
            missing.append(subj)
            logger.warning(f"  {subj}: missing X file {x_path}")
            continue

        if not os.path.exists(y_scorer_path):
            missing.append(subj)
            logger.warning(f"  {subj}: missing Y_{scorer_mode}.npy")
            continue

        # For E2 mode, verify the file is non-trivial (has actual spindle annotations)
        if scorer_mode == 'E2':
            try:
                y_data = np.load(y_scorer_path, mmap_mode='r')
                if y_data.max() < 0.5:
                    logger.info(f"  {subj}: Y_E2.npy exists but has no spindle annotations, skipping")
                    missing.append(subj)
                    continue
            except Exception as e:
                logger.warning(f"  {subj}: failed to read Y_E2.npy: {e}")
                missing.append(subj)
                continue

        valid.append(subj)

    if missing:
        logger.warning(f"Filtered out {len(missing)} subjects (scorer={scorer_mode}): {missing}")
    logger.info(f"Valid subjects available: {len(valid)}")
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, nargs="?", const=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--shuffle_folds", action="store_true")
    parser.add_argument(
        "--fold_split",
        type=str,
        default="shuffled",
        choices=["shuffled", "sequential"],
        help="How to assign subjects to CV folds. "
             "'shuffled' (default): seeded random permutation per repeat, as in SEED. "
             "'sequential': subjects assigned to folds in their listed order, "
             "producing fixed contiguous test sets (e.g. fold 1 = subjects 1-4, "
             "fold 2 = subjects 5-8, ...). Same across repeats. "
             "Both options are ignored when CV_CONFIG['use_explicit_splits'] is True.",
    )
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
    log_params(log, "Inference params", INFERENCE_PARAMS)
    log_params(log, "Post-processing params", POST_PROCESSING_PARAMS)
    log_params(log, "CV config", CV_CONFIG)

    # Read CV configuration
    scorer_mode = CV_CONFIG["scorer_mode"]
    cv_strategy = CV_CONFIG["cv_strategy"]
    n_folds_cfg = CV_CONFIG.get("n_folds")
    use_swa = TRAINING_PARAMS.get("use_swa")
    use_explicit = CV_CONFIG.get("use_explicit_splits", False)

    if use_explicit:
        log.info(f"Using explicit splits from mass_config.MASS_SS2_5FOLD_SPLITS "
                 f"(scorer={scorer_mode}). Seed-based subject shuffling is bypassed.")

    # Determine eligible subjects from scorer_mode, then filter by what's on disk
    scorer_subjects = get_subjects_for_scorer(scorer_mode)
    log.info(f"Scorer mode '{scorer_mode}': {len(scorer_subjects)} subjects eligible")

    subjects = filter_valid_subjects(scorer_subjects, paths.PROCESSED_DATA_DIR, log, scorer_mode)
    if not subjects:
        log.error("No valid subjects found! Check the processed_data directory.")
        return

    # Determine fold count for the chosen strategy
    num_folds, strategy_name = get_cv_parameters(cv_strategy, subjects, n_folds_cfg, log)

    log.info(f"Cross validation strategy: {strategy_name}")
    log.info(f"Scorer mode: {scorer_mode}")
    if not use_explicit:
        log.info(f"Fold split mode: {args.fold_split}")

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

        # Determine fold specifications for this repeat
        explicit_folds = None
        fold_subjects = None

        if use_explicit:
            try:
                explicit_folds = get_explicit_splits(scorer_mode, repeat_idx)
            except ValueError as e:
                log.error(f"Explicit splits validation failed for "
                          f"scorer={scorer_mode}, Repeat_{repeat_idx + 1}: {e}")
                continue
            log.info(f"Loaded {len(explicit_folds)} explicit folds for "
                     f"scorer={scorer_mode}, Repeat_{repeat_idx + 1}")
            iter_folds = list(range(len(explicit_folds)))
        else:
            # Determine subject order for fold assignment
            fold_subjects = subjects.copy()
            if args.fold_split == "shuffled":
                # SEED-style: seeded random permutation, differs per repeat
                rng = random.Random(current_seed)
                rng.shuffle(fold_subjects)
                log.info(f"Subjects shuffled for repeat (seed={current_seed}).")
            else:
                # Sequential: subjects kept in their original listed order.
                # Folds become fixed contiguous blocks, identical across repeats.
                log.info(f"Subjects kept in listed order (sequential fold assignment).")
            iter_folds = folds_to_run

        repeat_metrics = defaultdict(list)

        for fold_idx in iter_folds:
            if use_explicit:
                spec = explicit_folds[fold_idx]
                train_ids = list(spec['train'])
                val_ids = list(spec['val'])
                test_ids = list(spec['test'])
                fold_name = spec['fold_name']
                identifier = test_ids[0] if len(test_ids) == 1 else fold_name
            else:
                train_ids, val_ids, test_ids, fold_name, identifier = get_cv_splits(
                    cv_strategy, fold_subjects, fold_idx, num_folds
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

            try:
                train_loader, val_loader, test_loader = get_dataloaders(
                    processed_data_dir=paths.PROCESSED_DATA_DIR,
                    batch_size=TRAINING_PARAMS["batch_size"],
                    train_subject_ids=train_ids,
                    val_subject_ids=val_ids,
                    test_subject_ids=test_ids,
                    data_params={**DATA_PARAMS, "scorer_mode": scorer_mode},
                )
                if len(train_loader) == 0:
                    log.error("Train loader is empty! Skipping.")
                    continue

                sample = val_loader.dataset[0][0] if len(val_loader) > 0 else train_loader.dataset[0][0]
                num_channels = sample.shape[0]
            except Exception as e:
                log.error(f"Data loading failed: {e}")
                continue

            if args.mode == "train":
                model = ResidualUNet1D(num_channels, dropout_rate=TRAINING_PARAMS["dropout_rate"])
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

            try:
                metrics = evaluate_fold(
                    fold_dir,
                    test_ids,
                    val_ids,
                    paths.PROCESSED_DATA_DIR,
                    num_channels,
                    identifier,
                    use_swa,
                    log,
                    {**DATA_PARAMS, "scorer_mode": scorer_mode},
                )
                if metrics:
                    for key, value in metrics.items():
                        repeat_metrics[key].append(value)
            except Exception as e:
                log.error(f"Evaluation failed: {e}")

        summary = aggregate_and_save_summary(repeat_metrics, repeat_dir, repeat_idx, current_seed, log)
        for key, val in summary.items():
            grand_results[key].append(val)

    save_final_experiment_summary(
        grand_results, master_dir, len(repeats), timestamp if args.mode == "train" else "eval", log
    )

    logging.shutdown()
    try:
        shutil.copy2(os.path.join("logs", log_file), os.path.join(master_dir, log_file))
    except Exception:
        pass


if __name__ == "__main__":
    main()
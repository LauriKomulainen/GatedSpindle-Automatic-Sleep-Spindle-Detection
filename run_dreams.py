# run_dreams.py

import os
import shutil
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict
import torch
import paths
from utils.logger import setup_logging
from utils.run_utils import (
    set_seed,
    log_params,
    setup_output_dir,
    parse_eval_directories,
    evaluate_fold,
    get_loso_splits,
)
from core.dataset import get_dataloaders
from core.train_model import train_model
from core.model import ResidualUNet1D
from core.evaluation import (
    aggregate_and_save_summary,
    save_final_experiment_summary,
)
from configs.dreams_model_config import INFERENCE_PARAMS, TRAINING_PARAMS, POST_PROCESSING_PARAMS
from configs.dreams_config import DATA_PARAMS, CV_CONFIG, get_explicit_splits


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, nargs="?", const=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--shuffle_folds", action="store_true",
                        help="Shuffle the subject order before fold assignment. "
                             "Ignored when CV_CONFIG['use_explicit_splits'] is True.")
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

    log.info(f"Dataset: DREAMS")
    log_params(log, "Training params", TRAINING_PARAMS)
    log_params(log, "Data params", DATA_PARAMS)
    log_params(log, "Inference params", INFERENCE_PARAMS)
    log_params(log, "Post-processing params", POST_PROCESSING_PARAMS)

    use_explicit = CV_CONFIG.get("use_explicit_splits", False)
    use_swa = TRAINING_PARAMS.get("use_swa", False)
    cv_strategy = "LOSO"

    # Filter valid subjects
    all_subjects = filter_valid_subjects(list(DATA_PARAMS["subjects_list"]), paths.PROCESSED_DATA_DIR, log)
    if not all_subjects:
        log.error("No valid subjects found! Check the processed_data directory.")
        return

    # Load and validate explicit splits once if enabled. DREAMS LOSO splits are
    # deterministic and identical across repeats, so a single load is sufficient.
    explicit_folds = None
    if use_explicit:
        log.info("Using explicit DREAMS LOSO splits from "
                 "dreams_config.DREAMS_LOSO_SPLITS. Seed-based shuffling is bypassed.")
        try:
            explicit_folds = get_explicit_splits()
        except ValueError as e:
            log.error(f"Explicit splits validation failed: {e}")
            return
        num_folds = len(explicit_folds)
    else:
        num_folds = len(all_subjects)

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
        if not use_explicit and args.shuffle_folds:
            random.shuffle(subjects)
            log.info("Subjects shuffled.")

        repeat_metrics = defaultdict(list)

        for fold_idx in folds_to_run:
            if use_explicit:
                spec = explicit_folds[fold_idx]
                train_ids = list(spec['train'])
                val_ids = list(spec['val'])
                test_ids = list(spec['test'])
                fold_name = spec['fold_name']
                identifier = test_ids[0] if len(test_ids) == 1 else fold_name
            else:
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

            try:
                train_loader, val_loader, test_loader = get_dataloaders(
                    processed_data_dir=paths.PROCESSED_DATA_DIR,
                    batch_size=TRAINING_PARAMS["batch_size"],
                    train_subject_ids=train_ids,
                    val_subject_ids=val_ids,
                    test_subject_ids=test_ids,
                    data_params=DATA_PARAMS,
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
                    DATA_PARAMS,
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
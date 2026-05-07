# run_cross_dataset.py
"""
Cross-dataset evaluation: train master model on MASS-SS2, evaluate on DREAMS.

This script trains a single 'master' model on MASS-SS2 (union of the two
expert annotations) and evaluates it on DREAMS (n=6) subject without any
retraining. It is intentionally one-directional: MASS-SS2 is large enough
to use as a source, while DREAMS is too small to serve as a useful source
for transfer to MASS-SS2.

Workflow & usage:
  1. Split MASS-SS2 subjects into train and validation sets using a
     seeded shuffle. The validation subjects are held out for early
     stopping during training.
        --> python build_mass_dataset.py
  2. Train the master model
        --> python run_cross_dataset.py --mode train
  3. Build DREAMS dataset
        --> python build_dreams_dataset.py
  4. Run evaluation
        --> python run_cross_dataset.py --mode evaluate --run_dir model_reports/MASS-SS2/Repeat_1
"""

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
    evaluate_fold,
)
from core.dataset import get_dataloaders
from core.train_model import train_model
from core.model import GatedUNet
from core.evaluation import (
    aggregate_and_save_summary,
    save_final_experiment_summary,
)

# Source (MASS-SS2) training and data parameters. Used only in train phase.
from configs.mass_model_config import TRAINING_PARAMS as MASS_TRAINING_PARAMS
from configs.mass_config import (
    DATA_PARAMS as MASS_DATA_PARAMS,
    get_subjects_for_scorer,
)

# Target (DREAMS) data parameters. Used only in evaluate phase.
from configs.dreams_config import DATA_PARAMS as DREAMS_DATA_PARAMS

CROSS_DATASET_CONFIG = {
    'source_scorer_mode': 'UNION',  # MASS-SS2 ground truth: union of E1 and E2
    'val_fraction': 0.2,  # Fraction of MASS subjects held out for early stopping
    'min_val_subjects': 2,  # Floor on the number of validation subjects
}


def filter_valid_subjects(subjects: list, data_dir: str, scorer_mode: str,
                          dataset_name: str, logger) -> list:
    """Return only the subjects whose required .npy files are on disk."""
    valid, missing = [], []
    for subj in subjects:
        x_path = os.path.join(data_dir, f"{subj}_X_1D.npy")
        if scorer_mode in ('E1', 'E2', 'UNION'):
            y_path = os.path.join(data_dir, f"{subj}_Y_{scorer_mode}.npy")
        else:
            y_path = os.path.join(data_dir, f"{subj}_Y_1D.npy")

        if os.path.exists(x_path) and os.path.exists(y_path):
            valid.append(subj)
        else:
            missing.append(subj)

    if missing:
        logger.warning(f"Filtered out {len(missing)} {dataset_name} subjects: {missing}")
    logger.info(f"Valid {dataset_name} subjects: {len(valid)}")
    return valid


def split_train_val(subjects: list, val_fraction: float, min_val: int,
                    seed: int, logger) -> tuple:
    """Split a subject list into (train_ids, val_ids) using a seeded shuffle."""
    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_val = max(min_val, int(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)  # Ensure at least 1 train subject

    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]

    logger.info(f"Train ({len(train_ids)}): {train_ids}")
    logger.info(f"Val ({len(val_ids)}): {val_ids}")

    return train_ids, val_ids


def train_master_model(train_ids: list, val_ids: list, output_dir: str,
                       source_scorer_mode: str, log) -> int:
    """Train the master model on MASS-SS2."""
    log.info("=" * 60)
    log.info("PHASE 1: Training master model on MASS-SS2")
    log.info("=" * 60)

    use_swa = MASS_TRAINING_PARAMS.get('use_swa', False)

    try:
        train_loader, val_loader, _ = get_dataloaders(
            processed_data_dir=paths.PROCESSED_DATA_DIR,
            batch_size=MASS_TRAINING_PARAMS["batch_size"],
            train_subject_ids=train_ids,
            val_subject_ids=val_ids,
            test_subject_ids=[],
            data_params={**MASS_DATA_PARAMS, "scorer_mode": source_scorer_mode},
        )
        if len(train_loader) == 0:
            log.error("Train loader is empty! Aborting.")
            return None
        if len(val_loader) == 0:
            log.error("Validation loader is empty! Aborting.")
            return None

        sample = val_loader.dataset[0][0]
        num_channels = sample.shape[0]
        log.info(f"Input channels: {num_channels}")
    except Exception as e:
        log.error(f"Data loading failed: {e}")
        return None

    model = GatedUNet(num_channels, dropout_rate=MASS_TRAINING_PARAMS["dropout_rate"])
    train_model(
        model,
        train_loader,
        val_loader,
        MASS_TRAINING_PARAMS["learning_rate"],
        MASS_TRAINING_PARAMS["num_epochs"],
        MASS_TRAINING_PARAMS["early_stopping_patience"],
        output_dir,
        use_swa,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Master model saved to: {output_dir}")
    return num_channels


def evaluate_on_dreams(master_dir: str, dreams_subjects: list,
                       num_channels: int, log) -> dict:
    """Evaluate the master model on DREAMS, one subject at a time."""
    try:
        from core.config_loader import INFERENCE_PARAMS
        INFERENCE_PARAMS['threshold'] = 0.5
        log.info("Explicitly forced zero-shot threshold to 0.500 in config_loader")
    except Exception as e:
        log.warning(f"Could not force threshold: {e}")

    use_swa = MASS_TRAINING_PARAMS.get('use_swa', False)

    eval_data_params = {**DREAMS_DATA_PARAMS, "scorer_mode": None}
    log_params(log, "Evaluation data params (DREAMS-side)", eval_data_params)

    repeat_metrics = defaultdict(list)

    for subject_id in dreams_subjects:
        test_ids = [subject_id]
        val_ids = []

        log.info(f"Evaluating DREAMS subject: {subject_id}")
        log.info(f"Test:  {test_ids}")
        log.info(f"Val (threshold tuning): {len(val_ids)} subjects")

        try:
            metrics = evaluate_fold(
                master_dir,
                test_ids,
                val_ids,
                paths.PROCESSED_DATA_DIR,
                num_channels,
                f"DREAMS_{subject_id}",
                use_swa,
                log,
                eval_data_params,
            )
            if metrics:
                for key, value in metrics.items():
                    repeat_metrics[key].append(value)
        except Exception as e:
            log.error(f"Evaluation failed for {subject_id}: {e}")

    return repeat_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation: MASS-SS2 -> DREAMS"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["train", "evaluate"],
        help="Pipeline mode (default: full)"
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Required for --mode evaluate: path to existing master model directory or a specific repeat directory"
    )
    parser.add_argument(
        "--seed", type=int, default=0, nargs="?", const=None,
        help="Base random seed (default: 0). Repeat r uses seed = base + r."
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of training repeats with different seeds"
    )
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(1, 99999)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    source_name = "MASS-SS2"
    target_name = "DREAMS"
    source_scorer_mode = CROSS_DATASET_CONFIG['source_scorer_mode']

    if args.mode in "train":
        base_output = setup_output_dir(timestamp)
        master_dir = os.path.join(
            os.path.dirname(base_output),
            f"cross_{source_name}_to_{target_name}_{timestamp}",
        )
        if base_output != master_dir:
            os.rename(base_output, master_dir)
        log_file = "cross_dataset_training.log"
    else:
        if not args.run_dir:
            raise ValueError("--run_dir is required for --mode evaluate")
        master_dir = args.run_dir
        log_file = "cross_dataset_evaluation.log"

    setup_logging(log_file)
    log = logging.getLogger(__name__)

    source_subjects = []
    if args.mode in "train":
        raw_source_ids = get_subjects_for_scorer(source_scorer_mode)
        source_subjects = filter_valid_subjects(
            raw_source_ids, paths.PROCESSED_DATA_DIR, source_scorer_mode,
            source_name, log,
        )
        if not source_subjects:
            log.error(f"No valid {source_name} subjects found! Build the dataset first.")
            return

    target_subjects = []
    if args.mode in "evaluate":
        raw_target_ids = list(DREAMS_DATA_PARAMS['subjects_list'])
        target_subjects = filter_valid_subjects(
            raw_target_ids, paths.PROCESSED_DATA_DIR, None,
            target_name, log,
        )
        if not target_subjects:
            log.error(f"No valid {target_name} subjects found! Build the dataset first.")
            return
        if len(target_subjects) < 2:
            log.error(
                f"LOSO threshold tuning requires at least 2 DREAMS subjects. Found {len(target_subjects)}."
            )
            return

    grand_results = defaultdict(list)
    repeat_tasks = []

    if args.mode in "train":
        for r in range(args.repeats):
            repeat_tasks.append({
                "idx": r,
                "path": os.path.join(master_dir, f"Repeat_{r + 1}")
            })
    else:
        # Evaluate-only mode
        norm_dir = os.path.normpath(master_dir)
        base_name = os.path.basename(norm_dir)

        if base_name.startswith("Repeat_"):
            try:
                r_idx = int(base_name.split("_")[1]) - 1
            except (IndexError, ValueError):
                r_idx = 0

            repeat_tasks.append({
                "idx": r_idx,
                "path": norm_dir
            })

            master_dir = os.path.dirname(norm_dir)
            log.info(f"Evaluate-only mode: Single repeat target detected ({base_name}).")
        else:
            repeat_dirs = sorted(
                d for d in os.listdir(master_dir)
                if d.startswith("Repeat_") and os.path.isdir(os.path.join(master_dir, d))
            )
            if not repeat_dirs:
                log.error(f"No Repeat_* directories found under {master_dir}")
                return
            for d in repeat_dirs:
                try:
                    r_idx = int(d.split("_")[1]) - 1
                except (IndexError, ValueError):
                    r_idx = 0
                repeat_tasks.append({
                    "idx": r_idx,
                    "path": os.path.join(master_dir, d)
                })
            log.info(f"Evaluate-only mode: found {len(repeat_tasks)} existing repeats.")

    log.info(f"Output / Master directory: {master_dir}")

    for task in repeat_tasks:
        repeat_idx = task["idx"]
        repeat_dir = task["path"]
        current_seed = base_seed + repeat_idx
        set_seed(current_seed)

        if args.mode in "train":
            os.makedirs(repeat_dir, exist_ok=True)
            train_ids, val_ids = split_train_val(
                source_subjects,
                CROSS_DATASET_CONFIG['val_fraction'],
                CROSS_DATASET_CONFIG['min_val_subjects'],
                current_seed, log,
            )

            num_channels = train_master_model(
                train_ids, val_ids, repeat_dir, source_scorer_mode, log,
            )
            if num_channels is None:
                log.error(f"Training failed for {os.path.basename(repeat_dir)}, skipping evaluation")
                continue
        else:
            if not os.path.exists(repeat_dir):
                log.warning(f"Repeat directory not found: {repeat_dir}, skipping")
                continue

            try:
                _, val_loader_tmp, _ = get_dataloaders(
                    processed_data_dir=paths.PROCESSED_DATA_DIR,
                    batch_size=1,
                    train_subject_ids=[],
                    val_subject_ids=[target_subjects[0]],
                    test_subject_ids=[],
                    data_params={**DREAMS_DATA_PARAMS, "scorer_mode": None},
                )
                sample = val_loader_tmp.dataset[0][0]
                num_channels = sample.shape[0]
                log.info(f"Inferred input channels from {target_name} sample: {num_channels}")
            except Exception as e:
                log.error(f"Failed to infer num_channels: {e}")
                continue

        if args.mode in "evaluate":
            fold_dirs = sorted(
                d for d in os.listdir(repeat_dir)
                if d.startswith("Fold_") and os.path.isdir(os.path.join(repeat_dir, d))
            )

            if not fold_dirs:
                fold_dirs = [""]

            for fold_name in fold_dirs:
                eval_dir = os.path.join(repeat_dir, fold_name) if fold_name else repeat_dir

                if fold_name:
                    log.info(f" Evaluating {fold_name} in {os.path.basename(repeat_dir)}")

                repeat_metrics = evaluate_on_dreams(
                    eval_dir, target_subjects, num_channels, log,
                )

                summary = aggregate_and_save_summary(
                    repeat_metrics, eval_dir, repeat_idx, current_seed, log,
                )

                for key, val in summary.items():
                    grand_results[key].append(val)

    if grand_results:
        num_evaluations = len(next(iter(grand_results.values())))

        save_final_experiment_summary(
            grand_results,
            master_dir,
            num_evaluations,
            timestamp if args.mode in ("full", "train") else "eval",
            log,
        )

    log.info(f"Results saved to: {master_dir}")

    logging.shutdown()
    try:
        shutil.copy2(os.path.join("logs", log_file), os.path.join(master_dir, log_file))
    except Exception:
        pass


if __name__ == "__main__":
    main()
# main.py

import gc
import os
import shutil
import torch
import torch.nn as nn
import logging
import random
import argparse
import numpy as np
import paths

from datetime import datetime
from collections import defaultdict
from utils.logger import setup_logging
from core.dataset import get_dataloaders
from core.model import GatedUNet, train_model
from core.evaluation import (
    compute_event_based_metrics,
    find_optimal_threshold,
    aggregate_and_save_summary,
    save_final_experiment_summary
)
from configs.dreams_config import (
    TRAINING_PARAMS,
    DATA_PARAMS,
    CV_CONFIG,
    INFERENCE_PARAMS
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")


# Helper Class for Ensemble
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
    if m is None:
        return
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
    logger.info(f"--- {name} ---")
    for k, v in d.items():
        logger.info(f"  {k:<25}: {v}")


if __name__ == "__main__":
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description="Sleep Spindle Detection Pipeline")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help="Mode: 'train' starts new training, 'evaluate' tests existing models.")
    parser.add_argument('--run_dir', type=str, default=None,
                        help="Path to the existing run directory (required if mode='evaluate').")
    parser.add_argument('--seed', type=int, default=1, nargs='?', const=None,
                        help="Base random seed. Subsequent repeats will increment this.")
    parser.add_argument('--repeats', type=int, default=1,
                        help="How many times to repeat the full Cross-Validation experiment.")
    parser.add_argument('--shuffle_folds', action='store_true',
                        help="If set, shuffles the subject order for each repeat to vary validation sets.")

    args = parser.parse_args()

    # Base seed handling
    base_seed = args.seed
    if base_seed is None:
        base_seed = random.randint(1, 99999)
        print(f"NOTE: Random execution requested. Base generated seed: {base_seed}")

    # Ensure output directory exists base
    os.makedirs(paths.REPORTS_DIR, exist_ok=True)

    # SETUP MASTER OUTPUT DIRECTORY
    if args.mode == 'train':
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        master_output_dir = os.path.join(paths.REPORTS_DIR, f"LOSO_Experiment_{timestamp}")
        os.makedirs(master_output_dir, exist_ok=True)
        log_file_name = "training.log"
    else:
        # Evaluation mode
        if args.run_dir is None:
            raise ValueError("In 'evaluate' mode, you MUST provide --run_dir.")
        if not os.path.exists(args.run_dir):
            raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
        master_output_dir = args.run_dir
        log_file_name = "evaluation_rerun.log"

    setup_logging(log_file_name)
    log = logging.getLogger(__name__)

    # 1. LOG CONFIGURATION
    log.info(f"EXPERIMENT CONFIGURATION (Mode: {args.mode.upper()})")
    log.info(f"Repeats requested: {args.repeats}")

    log_param_dict(log, "TRAINING_PARAMS", TRAINING_PARAMS)
    log_param_dict(log, "DATA_PARAMS", DATA_PARAMS)
    log_param_dict(log, "INFERENCE_PARAMS", INFERENCE_PARAMS)
    log_param_dict(log, "CV_CONFIG", CV_CONFIG)

    grand_results = defaultdict(list)

    # REPEATS LOOP
    for repeat_idx in range(args.repeats):
        current_seed = base_seed + repeat_idx
        set_seed(current_seed)
        log.info(f"STARTING REPEAT {repeat_idx + 1} / {args.repeats} (Seed: {current_seed})")

        # Define run directory for this repeat
        if args.mode == 'train':
            run_output_dir = os.path.join(master_output_dir, f"Repeat_{repeat_idx + 1}")
            os.makedirs(run_output_dir, exist_ok=True)
        else:
            # In eval mode, we assume the structure exists.
            possible_subdir = os.path.join(master_output_dir, f"Repeat_{repeat_idx + 1}")
            if os.path.exists(possible_subdir):
                run_output_dir = possible_subdir
            else:
                run_output_dir = master_output_dir

        params = TRAINING_PARAMS
        USE_SWA = params.get('use_swa', False)

        all_subjects = list(DATA_PARAMS['subjects_list'])

        if args.shuffle_folds:
            random.shuffle(all_subjects)
            log.info(f"Subject order shuffled for this repeat: {all_subjects}")

        repeat_metrics = defaultdict(list)

        selected_folds = CV_CONFIG['folds_to_run']
        folds_to_iterate = selected_folds if selected_folds else range(len(all_subjects))

        # CROSS-VALIDATION LOOP
        for k in folds_to_iterate:
            test_subject_id = [all_subjects[k]]
            val_subject_id = [all_subjects[(k + 1) % len(all_subjects)]]
            train_subject_ids = [s for s in all_subjects if s != test_subject_id[0] and s != val_subject_id[0]]

            fold_name = f"Fold_{k + 1}_(Test={test_subject_id[0]})"
            log.info(f"Repeat {repeat_idx + 1}: {fold_name}")

            fold_output_dir = os.path.join(run_output_dir, fold_name)

            if args.mode == 'train':
                os.makedirs(fold_output_dir, exist_ok=True)
            elif not os.path.exists(fold_output_dir):
                log.warning(f"Directory {fold_output_dir} not found! Skipping.")
                continue

            # 1. Load Data
            try:
                train_loader, val_loader, test_loader = get_dataloaders(
                    processed_data_dir=paths.PROCESSED_DATA_DIR,
                    batch_size=params['batch_size'],
                    train_subject_ids=train_subject_ids,
                    val_subject_ids=val_subject_id,
                    test_subject_ids=test_subject_id,
                )

                if len(val_loader) > 0:
                    first_sample_data = val_loader.dataset[0][0]
                else:
                    first_sample_data = train_loader.dataset[0][0]

                num_channels = first_sample_data.shape[0]
                log.info(f"Detected input channels: {num_channels}")

            except Exception as e:
                log.error(f"Data loading failed: {e}")
                continue

            # 2. Train (ONLY IF MODE IS TRAIN)
            if args.mode == 'train':
                model = GatedUNet(num_channels, dropout_rate=params['dropout_rate'])
                train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer_type=params['optimizer_type'],
                    learning_rate=params['learning_rate'],
                    num_epochs=params['num_epochs'],
                    early_stopping_patience=params['early_stopping_patience'],
                    output_dir=fold_output_dir,
                    use_swa=USE_SWA
                )
                del model
                torch.cuda.empty_cache()

                # 3. Evaluation (Common logic for both Train & Evaluate)
                metrics_best = None
                metrics_swa = None
                metrics_ens = None
                final_metrics = None

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Load Best Model
                best_path = os.path.join(fold_output_dir, 'unet_model_best.pth')
                if not os.path.exists(best_path):
                    log.warning(f"Best model not found at {best_path}. Skipping evaluation for this fold.")
                    continue

                try:
                    model_best = GatedUNet(num_channels, dropout_rate=0.0).to(device)
                    model_best.load_state_dict(torch.load(best_path, map_location=device))
                    log.info(f"Loaded model from {best_path}")

                    # Threshold
                    if INFERENCE_PARAMS['fixed_threshold'] is not None:
                        optimal_thresh = INFERENCE_PARAMS['fixed_threshold']
                    else:
                        optimal_thresh = find_optimal_threshold(model_best, val_loader)

                    target_mode = INFERENCE_PARAMS['inference_mode']

                    # A. Evaluate BEST
                    dir_for_best = fold_output_dir if target_mode == 'none' else None

                    metrics_best = compute_event_based_metrics(
                        model_best, test_loader, optimal_thresh,
                        f"{test_subject_id[0]}_best", dir_for_best
                    )
                    log_metrics(log, "BEST", metrics_best)

                    # B. Evaluate SWA
                    model_swa = None
                    if USE_SWA:
                        swa_path = os.path.join(fold_output_dir, 'unet_model_swa.pth')
                        if os.path.exists(swa_path):
                            model_swa = GatedUNet(num_channels, dropout_rate=0.0).to(device)
                            model_swa.load_state_dict(torch.load(swa_path, map_location=device))

                            dir_for_swa = fold_output_dir if target_mode == 'swa' else None

                            metrics_swa = compute_event_based_metrics(
                                model_swa, test_loader, optimal_thresh,
                                f"{test_subject_id[0]}_swa", dir_for_swa
                            )
                            log_metrics(log, "SWA", metrics_swa)

                            # C. Evaluate ENSEMBLE
                            ensemble_model = EnsembleWrapper(model_best, model_swa).to(device)

                            dir_for_ens = fold_output_dir if target_mode == 'ensemble' else None

                            metrics_ens = compute_event_based_metrics(
                                ensemble_model, test_loader, optimal_thresh,
                                f"{test_subject_id[0]}_ens", dir_for_ens
                            )
                            log_metrics(log, "ENS", metrics_ens)
                        else:
                            if args.mode == 'evaluate':
                                log.info("SWA model not found, skipping SWA/Ensemble.")

                        # 4. Select Final Metrics
                        selected_mode = INFERENCE_PARAMS['inference_mode']
                        final_metrics = metrics_best  # Default fallback

                        if selected_mode == 'swa' and metrics_swa:
                            final_metrics = metrics_swa
                        elif selected_mode == 'ensemble' and metrics_ens:
                            final_metrics = metrics_ens

                        log.info(
                            f"Repeat {repeat_idx + 1} | Fold {k + 1} -> Final ({selected_mode.upper()}) F1: {final_metrics['F1-score']:.4f}")

                        # Save metrics for summary
                        for key, value in final_metrics.items():
                            repeat_metrics[key].append(value)

                        # Cleanup
                        del model_best
                        if model_swa: del model_swa
                        torch.cuda.empty_cache()
                        gc.collect()

                except Exception as e:
                    log.error(f"Error during evaluation of {test_subject_id[0]}: {e}")
                    import traceback
                    traceback.print_exc()

        # REPEAT SUMMARY
        summary_updates = aggregate_and_save_summary(
            repeat_metrics=repeat_metrics,
            output_dir=run_output_dir,
            repeat_idx=repeat_idx,
            seed=current_seed,
            logger=log
        )

        for key, val in summary_updates.items():
            grand_results[key].append(val)

    save_final_experiment_summary(
        grand_results=grand_results,
        output_dir=master_output_dir,
        total_repeats=args.repeats,
        timestamp=timestamp if args.mode == 'train' else "evaluation_run",
        logger=log
    )

    logging.shutdown()
    log_dir = "logs"
    source_log_path = os.path.join(log_dir, log_file_name)
    target_log_path = os.path.join(master_output_dir, log_file_name)

    if os.path.exists(source_log_path):
        try:
            shutil.copy2(source_log_path, target_log_path)
            print(f"Training log file copied from {source_log_path} to: {target_log_path}")
        except Exception as e:
            print(f"Could not move training log file. It remains at {source_log_path}. Error: {e}")
# main.py

import gc
import os
import torch
import logging
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

from utils.logger import setup_logging
from data_preprocess.dataset import get_dataloaders
from UNET_model.model import GatedUNet, train_model
from UNET_model.evaluation_metrics import compute_event_based_metrics, find_optimal_threshold
from config import TRAINING_PARAMS, DATA_PARAMS, TEST_FAST_FRACTION, CV_CONFIG, INFERENCE_PARAMS

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

if __name__ == "__main__":
    set_seed(1)
    setup_logging("training.log")
    log = logging.getLogger(__name__)

    FAST_TEST_FRACTION = TEST_FAST_FRACTION['FAST_TEST_FRACTION']

    params = TRAINING_PARAMS
    if FAST_TEST_FRACTION < 1.0:
        log.warning(f"RUNNING IN FAST TEST MODE (DATA FRACTION: {FAST_TEST_FRACTION})")
        params['num_epochs'] = 1
        params['early_stopping_patience'] = 5

    log.info("Starting GATED U-Net Training: Leave-One-Subject-Out (LOSO)")

    result_dir = "model_reports"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(result_dir, f"LOSO_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    all_subjects = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']
    all_metrics = defaultdict(list)

    selected_folds = CV_CONFIG['folds_to_run']
    folds_to_iterate = selected_folds if selected_folds else range(len(all_subjects))

    for k in folds_to_iterate:
        test_subject_id = [all_subjects[k]]
        val_subject_id = [all_subjects[(k + 1) % len(all_subjects)]]
        train_subject_ids = [s for s in all_subjects if s != test_subject_id[0] and s != val_subject_id[0]]

        fold_name = f"Fold_{k + 1}_(Test={test_subject_id[0]})"
        log.info(f"STARTING FOLD: {fold_name}")

        fold_output_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_output_dir, exist_ok=True)

        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                processed_data_dir="./diagnostics_plots/processed_data",
                batch_size=params['batch_size'],
                train_subject_ids=train_subject_ids,
                val_subject_ids=val_subject_id,
                test_subject_ids=test_subject_id,
                use_fraction=FAST_TEST_FRACTION
            )
        except Exception as e:
            log.error(f"Data loading failed: {e}")
            continue

        model = GatedUNet(dropout_rate=params['dropout_rate'])

        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_type=params['optimizer_type'],
            learning_rate=params['learning_rate'],
            num_epochs=params['num_epochs'],
            early_stopping_patience=params['early_stopping_patience'],
            output_dir=fold_output_dir,
            fs=DATA_PARAMS['fs']
        )

        best_model_path = os.path.join(fold_output_dir, 'unet_model_best.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))

        fixed_thresh = INFERENCE_PARAMS['fixed_threshold']

        if fixed_thresh is not None:
            log.info(f"Using FIXED threshold from config: {fixed_thresh}")
            optimal_thresh = fixed_thresh
        else:
            log.info("Finding optimal decision threshold using validation data...")
            optimal_thresh = find_optimal_threshold(model, val_loader)
            log.info(f"Optimal threshold determined: {optimal_thresh:.2f}")

        log.info(f"Computing metrics (Threshold: {optimal_thresh}, Power Check: {INFERENCE_PARAMS['use_power_check']})")

        metrics = compute_event_based_metrics(
            model,
            test_loader,
            threshold=optimal_thresh,
            subject_id=test_subject_id[0],
            output_dir=fold_output_dir,
            use_power_check=INFERENCE_PARAMS['use_power_check']
        )

        log.info(f"Fold Complete.")
        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        log.info(f"Results for {test_subject_id[0]}:\n")
        for key, value in metrics.items():
            log.info(f"  {key}: {value:.4f}")
            all_metrics[key].append(value)

    log.info("=" * 80)
    log.info("FULL LOSO CROSS-VALIDATION COMPLETE")
    if len(all_metrics) > 0:
        for key, values in all_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            log.info(f"Average {key}: {mean:.4f} (± {std:.4f})")

    log.info(f"All results saved to directory: {output_dir}")
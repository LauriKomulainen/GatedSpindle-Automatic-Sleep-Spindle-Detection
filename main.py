# main.py

import gc
import os
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

from utils.logger import setup_logging
from data_preprocess.dataset import get_dataloaders
from UNET_model.model import UNet, train_model
from UNET_model.evaluation_metrics import compute_event_based_metrics, find_optimal_threshold
from training_parameters import TRAINING_PARAMS, DATA_PARAMS, TEST_FAST_FRACTION, CV_CONFIG

if __name__ == "__main__":

    setup_logging("training.log")
    log = logging.getLogger(__name__)

    FAST_TEST_FRACTION = TEST_FAST_FRACTION['FAST_TEST_FRACTION']

    params = TRAINING_PARAMS
    if FAST_TEST_FRACTION < 1.0:
        log.warning(f"RUNNING IN FAST TEST MODE (DATA FRACTION: {FAST_TEST_FRACTION})")
        params['num_epochs'] = 1
        params['early_stopping_patience'] = 5

    log.info("Starting U-Net Training: Leave-One-Subject-Out (LOSO)")
    log.info(f"Using hyperparameters: {params}")

    result_dir = "model_reports"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(result_dir, f"LOSO_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    all_subjects = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']
    all_metrics = defaultdict(list)

    selected_folds = CV_CONFIG['folds_to_run']

    if selected_folds is None:
        folds_to_iterate = range(len(all_subjects))
        log.info(f"Starting ALL {len(all_subjects)}-Fold LOSO Cross-Validation")
    else:
        folds_to_iterate = selected_folds
        log.info(f"Starting SPECIFIC folds only: {folds_to_iterate}")

    for k in folds_to_iterate:
        test_subject_id = [all_subjects[k]]
        val_subject_id = [all_subjects[(k + 1) % len(all_subjects)]]
        train_subject_ids = [s for s in all_subjects if s != test_subject_id[0] and s != val_subject_id[0]]

        fold_name = f"Fold_{k + 1}_(Test={test_subject_id[0]})"
        log.info(f"STARTING FOLD: {fold_name}")
        log.info("=" * 80)

        fold_output_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_output_dir, exist_ok=True)

        log.info("Loading and splitting data for this fold...")
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
            log.error("Make sure 'data_handler.py' has been run successfully and 'processed_data' is not empty.")
            continue

        log.info("Initializing U-Net model (from scratch)...")
        model = UNet(dropout_rate=params['dropout_rate'])

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
            log.info(f"Loading best model ('{best_model_path}') for evaluation.")
            model.load_state_dict(torch.load(best_model_path))
        else:
            log.warning("No best model found. Using the last model state for evaluation.")

        if train_losses and val_losses:
            plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title(f'Loss Progression - {fold_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Tversky Loss')
            plt.legend()
            loss_plot_path = os.path.join(fold_output_dir, 'loss_progression.png')
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()
            log.info(f"Loss curve saved: {loss_plot_path}")

        log.info("Finding optimal decision threshold using validation data...")
        optimal_thresh = find_optimal_threshold(model, val_loader)

        log.info(f"Computing final test metrics using optimal threshold: {optimal_thresh:.2f}")
        metrics = compute_event_based_metrics(model, test_loader, threshold=optimal_thresh)

        log.info(f"---FOLD {fold_name} COMPLETE ---")

        log.info("Cleaning up memory...")
        del model
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        log.info(f"Test results for patient {test_subject_id[0]}:\n")
        for key, value in metrics.items():
            log.info(f"  {key}: {value:.4f}")
            all_metrics[key].append(value)

        log.info("Saving final prediction images from TEST data...")

    log.info("FULL LOSO CROSS-VALIDATION COMPLETE")
    log.info(f"Final average results across all {len(all_subjects)} folds:")

    for key, values in all_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        log.info(f"Average {key}: {mean:.4f} (± {std:.4f})")

    log.info(f"All results saved to directory: {output_dir}")
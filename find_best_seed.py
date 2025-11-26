# find_best_seed.py

import torch
import logging
import numpy as np
import random
import gc
import os
from utils.logger import setup_logging
from data_preprocess.dataset import get_dataloaders
from UNET_model.model import GatedUNet, train_model
from UNET_model.evaluation_metrics import compute_event_based_metrics, find_optimal_threshold
from config import TRAINING_PARAMS, DATA_PARAMS, TEST_FAST_FRACTION, INFERENCE_PARAMS

SEEDS_TO_TRY = [0, 1, 123, 2024, 3407]
TARGET_FOLD = "excerpt4"

setup_logging("seed_search.log")
log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_seed_search():
    log.info(f"--- Starting Seed Search for {TARGET_FOLD} ---")

    all_subjects = ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4', 'excerpt5', 'excerpt6']
    test_id = [TARGET_FOLD]
    val_id = ['excerpt5']
    train_ids = [s for s in all_subjects if s not in test_id + val_id]

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            processed_data_dir="./diagnostics_plots/processed_data",
            batch_size=TRAINING_PARAMS['batch_size'],
            train_subject_ids=train_ids,
            val_subject_ids=val_id,
            test_subject_ids=test_id,
            use_fraction=TEST_FAST_FRACTION['FAST_TEST_FRACTION']
        )
    except Exception as e:
        log.error(f"Data load failed: {e}")
        return

    results = {}

    for seed in SEEDS_TO_TRY:
        log.info("=" * 40)
        log.info(f"TESTING SEED: {seed}")
        set_seed(seed)

        fold_dir = f"seed_search_results/seed_{seed}"
        os.makedirs(fold_dir, exist_ok=True)

        model = GatedUNet(dropout_rate=TRAINING_PARAMS['dropout_rate'])

        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_type=TRAINING_PARAMS['optimizer_type'],
            learning_rate=TRAINING_PARAMS['learning_rate'],
            num_epochs=TRAINING_PARAMS['num_epochs'],  # Tai esim. 50 jos haluat nopeammin testata
            early_stopping_patience=TRAINING_PARAMS['early_stopping_patience'],
            output_dir=fold_dir,
            fs=DATA_PARAMS['fs']
        )

        best_path = os.path.join(fold_dir, 'unet_model_best.pth')
        model.load_state_dict(torch.load(best_path))

        optimal_thresh = find_optimal_threshold(model, val_loader)

        metrics = compute_event_based_metrics(
            model, test_loader, optimal_thresh,
            subject_id=test_id[0], output_dir=fold_dir,
            use_power_check=False
        )

        log.info(f"SEED {seed} RESULT: F1: {metrics['F1-score']:.4f}")
        results[seed] = metrics['F1-score']

        del model
        torch.cuda.empty_cache()
        gc.collect()

    log.info("=" * 40)
    log.info("FINAL RESULTS:")
    best_seed = max(results, key=results.get)
    for s, f1 in results.items():
        log.info(f"Seed {s}: F1 {f1:.4f}")

    log.info(f"WINNER: Seed {best_seed} (F1 {results[best_seed]:.4f})")
    log.info(f"Update your main.py with set_seed({best_seed})!")


if __name__ == "__main__":
    run_seed_search()
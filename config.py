# config.py

import os

# --- File Paths ---
PATHS = {
    "raw_data_dir": "Path",
    "processed_data_dir": "./data/processed",
    "output_dir": "./new_model_reports_v3"
}

os.makedirs(PATHS["processed_data_dir"], exist_ok=True)
os.makedirs(PATHS["output_dir"], exist_ok=True)

# --- Data & Preprocessing Parameters ---
DATA_PARAMS = {
    'fs': 200.0,
    'window_sec': 5.0,
    'overlap_sec': 2.5,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True,
    'included_stages': [2, 1, 0],
    'hypnogram_resolution_sec': 5.0,

    # --- SUBJECT SELECTION ---
    'subjects_list': [
        'excerpt1', 'excerpt2', 'excerpt3', 'excerpt4',
        'excerpt5', 'excerpt6'
    ],
}

# --- Training Hyperparameters ---
TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.2,
    'optimizer_type': 'Adam',
    'weight_decay': 1e-4,
    'ademamix_alpha': 5.0,
    'ademamix_betas': (0.9, 0.999, 0.9999),
    'num_epochs': 200,
    'early_stopping_patience': 25,
    'use_swa': True
}

CV_CONFIG = {
    'folds_to_run': None  # [0, 1]
    #'folds_to_run': [0]
}

METRIC_PARAMS = {
    'spindle_freq_low': 11.0,
    'spindle_freq_high': 16.0,
    'iou_threshold': 0.2,
    'min_duration_sec': 0.5,
    'max_duration_sec': 3.0
}

TEST_FAST_FRACTION = {
    'FAST_TEST_FRACTION': 1.0
}

INFERENCE_PARAMS = {
    'fixed_threshold': 0.55,
    'use_power_check': False,
    'inference_mode': 'ensemble',
    'use_hybrid_filter': False,
    'hybrid_high_conf': 0.85,
    'hybrid_min_power': 0.03
}
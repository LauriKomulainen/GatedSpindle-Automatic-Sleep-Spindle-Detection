# training_parameters.py

"""
Central configuration file for all training hyperparameters and data settings.
"""

# --- Training Hyperparameters ---
TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.2,
    'optimizer_type': 'Adam',
    'num_epochs': 100,
    'early_stopping_patience': 15
}

CV_CONFIG = {
    # 'folds_to_run': None,
    # 'folds_to_run': [3],
    'folds_to_run': None
}

# --- Data & Preprocessing Parameters ---
DATA_PARAMS = {
    'fs': 100.0,
    'window_sec': 5.0,
    'overlap_sec': 2.5,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4
}

# --- Event Metric Parameters ---
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
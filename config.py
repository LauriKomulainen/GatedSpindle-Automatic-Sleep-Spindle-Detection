# config.py

"""
Central configuration file for all training hyperparameters and data settings.
"""

# --- Training Hyperparameters ---
TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.2,
    'optimizer_type': 'Adam',
    'num_epochs': 200,
    'early_stopping_patience': 25
}

CV_CONFIG = {
    'folds_to_run': None
    #'folds_to_run': [3,4,5] # Excerpt 4
}

# --- Data & Preprocessing Parameters ---
DATA_PARAMS = {
    'fs': 200.0,
    'window_sec': 5.0,
    'overlap_sec': 2.5,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,

    # --- NEW: Instance Normalization & Hypnogram Filtering ---
    # True: Normalisoi jokainen 5s ikkuna erikseen (suositeltu).
    # False: Normalisoi koko signaali kerralla ennen pilkkomista (vanha tapa).
    'use_instance_norm': True,

    # Mitkä univaiheet otetaan mukaan koulutusdataan?
    # DREAMS koodaus:
    # 5=Wake, 4=REM, 3=S1, 2=S2, 1=S3, 0=S4
    'included_stages': [2, 1, 0],

    # Hypnogrammin resoluutio sekunneissa (DREAMS readme/kuva mainitsi 5s)
    'hypnogram_resolution_sec': 5.0
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

INFERENCE_PARAMS = {
    'fixed_threshold': 0.55,
    'use_power_check': False
}
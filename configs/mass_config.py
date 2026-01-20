# configs/mass_config.py

# 1. Data & Preprocessing Parameters
DATA_PARAMS = {
    'fs': 256.0,  # Resample target frequency
    'window_sec': 20.0,
    'overlap_sec': 0,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True,
    'included_stages': ['N2', 'N3'],
    'hypnogram_resolution_sec': 20.0,
    'page_duration': 20,

    # KAIKKI SUBJECTIT
    'subjects_list': [
        '01-02-0001', '01-02-0002', '01-02-0003',
        '01-02-0005', '01-02-0006', '01-02-0007',
        '01-02-0009', '01-02-0010', '01-02-0011', '01-02-0012',
        '01-02-0013', '01-02-0014',
        '01-02-0017', '01-02-0018', '01-02-0019'
    ],

    'channels': ['EEG C3-CLE'], # Tai pelkkä 'EEG C3-CLE'
}

# Training Hyperparameters
TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.3,
    'optimizer_type': 'Adam',
    'weight_decay': 1e-4,
    'num_epochs': 20,
    'early_stopping_patience': 25,
    'use_swa': True
}

CV_CONFIG = {
    'folds_to_run': None
    #'folds_to_run': [2,3]
}

INFERENCE_PARAMS = {
    'iou_threshold': 0.2,
    'fixed_threshold': 0.75,
    'inference_mode': 'ensemble', # Options: none (best), swa, ensemble
    'save_error_analysis': False,
}

POST_PROCESSING_PARAMS = {
    'min_duration_sec': 0.5,
    'gap_thresh_sec': 0.3,
    'fixed_border_thresh': 0.1,
}

SIGNAL_VISUALIZATION_PARAMS = {
    'channel_names': ["EEG (0.3-30Hz)", "Sigma (11-16Hz)", "Hilbert Envelope"]
}
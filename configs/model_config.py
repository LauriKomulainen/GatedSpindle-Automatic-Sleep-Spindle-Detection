TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.2,
    'optimizer_type': 'Adam',
    'weight_decay': 1e-4,
    'num_epochs': 200,
    'early_stopping_patience': 25,
    'use_swa': True
}

INFERENCE_PARAMS = {
    'iou_threshold': 0.2,
    'fixed_threshold': 0.65,
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
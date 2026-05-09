# configs/mass_model_config.py

TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.2,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'use_swa': False,               # Options: False, True
    'use_gating_branch': False,     # Options: False, True
    'seg_loss_weight': 0.85,
    'kernel_size': 11,
    'lr_scheduler': 'plateau',      # Options: 'cosine', 'plateau'
}

TRAINING_PARAMS['padding'] = (TRAINING_PARAMS['kernel_size'] - 1) // 2
TRAINING_PARAMS['scheduler_patience'] = (TRAINING_PARAMS['early_stopping_patience'] - 1) // 2

INFERENCE_PARAMS = {
    'iou_threshold': 0.2,
    'fixed_threshold': None,    # None, thresholds in the range 0.3–0.9 (steps of 0.1), done during of validation.
    'inference_mode': 'best',   # Options: 'best', 'swa', 'ensemble'
    'use_tta': True,            # Options: False, True
}

POST_PROCESSING_PARAMS = {
    'min_duration_sec': 0.5,
    'gap_thresh_sec': 0.3,
    'fixed_border_thresh': False,   # False = disabled (single-threshold mode).
                                    # Set to a float (e.g. 0.2) to enable
                                    # dual-thresholding with that border.
}

SIGNAL_VISUALIZATION_PARAMS = {
    'channel_names': ["EEG (0.3-30Hz)", "Sigma (11-16Hz)", "Hilbert Envelope"],
    'input_examples': 2
}
# config.py

# Data & Preprocessing Parameters
DATA_PARAMS = {
    'fs': 200.0,
    'window_sec': 5.0,
    'overlap_sec': 2.5,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True, # Windows are processed on-the-fly during the training when True
    'included_stages': [2, 1, 0], # In DREAMS dataset: Stages 2, 1 and 0 equals for N2 & N3 sleep stages
    'hypnogram_resolution_sec': 5.0,

    # SUBJECT SELECTION
    'subjects_list': [
        'excerpt1', 'excerpt2', 'excerpt3', 'excerpt4',
        'excerpt5', 'excerpt6'
    ],
}

CV_CONFIG = {
    'folds_to_run': None
    #'folds_to_run': [2,3]
}
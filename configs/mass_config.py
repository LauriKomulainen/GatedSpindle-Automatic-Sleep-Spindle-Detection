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
    'included_stages': ['2'],
    'hypnogram_resolution_sec': 20.0,
    'page_duration': 20,

    # Scorer mode: 'E1' = Expert 1 only, 'E2' = Expert 2 only, 'UNION' = merge both
    'scorer_mode': 'UNION',

    # Subject list
    'subjects_list': [
        '01-02-0001', '01-02-0002', '01-02-0003',
        '01-02-0004', '01-02-0005', '01-02-0006',
        '01-02-0007', '01-02-0008', '01-02-0009',
        '01-02-0010', '01-02-0011', '01-02-0012',
        '01-02-0013', '01-02-0014', '01-02-0015',
        '01-02-0015', '01-02-0017', '01-02-0018',
        '01-02-0019'
    ],
    'channels': ['EEG C3-CLE'],
}

CV_CONFIG = {
    'folds_to_run': None
}

'''
Same ID's as in SEED: http://github.com/nicolasigor/Sleep-EEG-Event-Detector/blob/main/sleeprnn/data/mass_ss.py

To evaluate againts SEED model, use this testing config
'''
TESTING_CONFIG = {
    'IDS_TEST': ['2','6','12','13'],
    'TRAINING_SET': 75.0
}


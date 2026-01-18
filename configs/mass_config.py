# configs/mass_config.py

# 1. Data & Preprocessing Parameters
DATA_PARAMS = {
    'fs': 256.0,  # Resample target frequency
    'window_sec': 5.0,
    'overlap_sec': 2.5,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True,
    'included_stages': ['2', '3', 'N2', 'N3'], # TODO: check how stages are named
    'hypnogram_resolution_sec': 30.0, # MASS epochs are typically 30s

    # SUBJECT SELECTION TODO: INCLUDE ONLY UNION OF TWO EXPERTS
    'subjects_list': [
        '01-02-0001', '01-02-0002', '01-02-0003', '01-02-0004',
        '01-02-0005', '01-02-0006', '01-02-0007', '01-02-0008',
        '01-02-0009', '01-02-0010', '01-02-0011', '01-02-0012',
        '01-02-0013', '01-02-0014', '01-02-0015', '01-02-0016',
        '01-02-0017', '01-02-0018', '01-02-0019'
    ],
}

SIGNAL_VISUALIZATION_PARAMS = {
    'channel_names': ["EEG (0.3-30Hz)", "Sigma (11-16Hz)", "Hilbert Envelope"]
}
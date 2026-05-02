# configs/mass_config.py

"""
MASS Dataset Configuration
"""

# 1. Data & Preprocessing Parameters
DATA_PARAMS = {
    'fs': 200.0,
    'window_sec': 10.0,
    'overlap_sec': 5.0,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True,
    'included_stages': ['2'],
    'hypnogram_resolution_sec': 20.0,
    'page_duration': 20,

    # All 19 MASS-SS2 subjects (build_dataset processes all of them)
    'subjects_list': [
        '01-02-0001', '01-02-0002', '01-02-0003',
        '01-02-0004', '01-02-0005', '01-02-0006',
        '01-02-0007', '01-02-0008', '01-02-0009',
        '01-02-0010', '01-02-0011', '01-02-0012',
        '01-02-0013', '01-02-0014', '01-02-0015',
        '01-02-0016', '01-02-0017', '01-02-0018',
        '01-02-0019'
    ],
    'channels': ['EEG C3-CLE'],
}

# 2. Cross-Validation Configuration
CV_CONFIG = {
    'scorer_mode': 'E2',    # 'E1' | 'E2' | 'UNION'
    'cv_strategy': 'kfold', # 'kfold' | 'loso'
    'n_folds': 5,
    'folds_to_run': None,   # None = run all folds; else list of fold indices
}

# Subjects WITHOUT E2 annotations (no Spindles_E2.edf)
SUBJECTS_NO_E2 = ['01-02-0004', '01-02-0008', '01-02-0015', '01-02-0016']

def get_subjects_for_scorer(scorer_mode: str) -> list:
    """Return the list of subjects eligible for the given scorer mode.

    E1    -> all 19 subjects (Expert 1 annotated everyone)
    E2    -> 15 subjects     (Expert 2 annotated 15 of them)
    UNION -> 15 subjects     (requires both annotations to exist)
    """
    if scorer_mode == 'E1':
        return list(DATA_PARAMS['subjects_list'])
    elif scorer_mode in ('E2', 'UNION'):
        return [s for s in DATA_PARAMS['subjects_list'] if s not in SUBJECTS_NO_E2]
    else:
        raise ValueError(f"Unknown scorer_mode: {scorer_mode}. Expected 'E1', 'E2', or 'UNION'.")
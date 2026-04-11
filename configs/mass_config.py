# configs/mass_config.py

"""
MASS Dataset Configuration
===========================
Supports multiple cross-validation strategies and scorer modes
for fair comparison against published methods (SEED, Spindle-UMamba).

Usage:
    - Set 'scorer_mode' to 'E1', 'E2', or 'UNION'
    - Set CV_CONFIG['cv_strategy'] to 'seed_15', 'umamba_19', 'kfold', or 'loso'
    - build_dataset.py processes all 19 patients once (no rebuild needed for scorer change)
"""

# =============================================================================
# 1. Data & Preprocessing Parameters
# =============================================================================
DATA_PARAMS = {
    'fs': 200.0,
    'window_sec': 20.0,
    'overlap_sec': 10.0,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True,
    'included_stages': ['2', '3'],
    'hypnogram_resolution_sec': 20.0,
    'page_duration': 20,

    # Scorer mode: 'E1' = Expert 1 only, 'E2' = Expert 2 only, 'UNION' = merge both
    # No rebuild needed - dataset.py selects the correct Y file at runtime
    'scorer_mode': 'E1',

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


# =============================================================================
# 1b. Scorer-Dependent Subject Lists
# =============================================================================
# E1 annotated all 19 subjects. E2 annotated only 15 (missing 4, 8, 15, 16).
# UNION requires both E1 and E2, so also limited to 15 subjects.
#
# This mapping determines which subjects are eligible for training AND evaluation
# based on the chosen scorer_mode, matching published protocols:
#   - E1:    use all 19 subjects  (as in Spindle-UMamba, SpindleU-Net)
#   - E2:    use 15 subjects      (as in SpindleU-Net, SEED)
#   - UNION: use 15 subjects      (needs both E1 & E2 annotations)

# Subjects WITHOUT E2 annotations (no Spindles_E2.edf)
SUBJECTS_NO_E2 = ['01-02-0004', '01-02-0008', '01-02-0015', '01-02-0016']

SUBJECTS_BY_SCORER = {
    'E1': DATA_PARAMS['subjects_list'],                                          # all 19
    'E2': [s for s in DATA_PARAMS['subjects_list'] if s not in SUBJECTS_NO_E2],  # 15
    'UNION': [s for s in DATA_PARAMS['subjects_list'] if s not in SUBJECTS_NO_E2],  # 15
}


# =============================================================================
# 2. Cross-Validation Configuration
# =============================================================================
CV_CONFIG = {
    # Options: 'seed_15', 'umamba_19', 'kfold', 'loso'
    #
    # 'seed_15'    -> Reproduces SEED paper Table 2 "15 subjects" results:
    #                 5-fold CV on the 15 subjects that have BOTH E1 and E2.
    #                 Patients 4, 8, 15, 16 are excluded (no E2 annotations).
    #                 This matches SEED's "biased evaluation" which is the
    #                 standard comparison point used by all subsequent papers.
    #                 Compare to: SEED E1 F1=80.8%, SEED E2 F1=86.1%
    #
    # 'umamba_19'  -> Reproduces Spindle-UMamba paper: 5-fold CV on all 19 subjects
    #                 Compare to: UMamba E1 F1=80.0%, UMamba E2 F1=79.9%
    #
    # 'kfold'      -> Standard 5-fold CV on all available subjects
    # 'loso'       -> Leave-One-Subject-Out
    'cv_strategy': 'umamba_19',

    # Only run specific folds (None = run all)
    'folds_to_run': None,
}


# =============================================================================
# 3. SEED Paper Configuration (15 subjects, 5-fold CV)
# =============================================================================
# From SEED source code (mass_ss.py):
#   IDS_INVALID = [4, 8, 15, 16]  -> excluded entirely (no E2 annotations)
#   IDS_TEST = [2, 6, 12, 13]     -> held out during SEED's own development
#
# SEED Table 2 "15 subjects" = 5-fold CV on ALL 15 valid subjects.
# The held-out split (IDS_TEST) was only for SEED's internal design validation
# and is NOT needed for external comparison.
#
# SEED ran 3 repeats × 5 folds = 15 partitions.
# Use --repeats 3 for exact protocol match.

SEED_CONFIG = {
    # 15 subjects with both E1 and E2 annotations
    # = all 19 minus IDS_INVALID [4, 8, 15, 16]
    'subjects_15': [
        '01-02-0001', '01-02-0002', '01-02-0003',
        '01-02-0005', '01-02-0006', '01-02-0007',
        '01-02-0009', '01-02-0010', '01-02-0011',
        '01-02-0012', '01-02-0013', '01-02-0014',
        '01-02-0017', '01-02-0018', '01-02-0019'
    ],

    # Folds are generated dynamically by generate_seed_folds() using a
    # random permutation seeded per repeat, matching SEED's cv_split logic.
    # 5-fold CV: 3 test, 3 val, 9 train per fold.
    'n_folds': 5,
}


# =============================================================================
# 4. Spindle-UMamba Paper Configuration (19 subjects, 5-fold CV)
# =============================================================================
# UMamba uses all 19 subjects. Exact splits not published.
# We follow their protocol: 5-fold CV, ~3-4 test subjects per fold.

UMAMBA_CONFIG = {
    'subjects_19': [
        '01-02-0001', '01-02-0002', '01-02-0003', '01-02-0004',
        '01-02-0005', '01-02-0006', '01-02-0007', '01-02-0008',
        '01-02-0009', '01-02-0010', '01-02-0011', '01-02-0012',
        '01-02-0013', '01-02-0014', '01-02-0015', '01-02-0016',
        '01-02-0017', '01-02-0018', '01-02-0019'
    ],

    # 5-fold CV: folds of size [4, 4, 4, 4, 3]
    'folds': [
        {
            'test':  ['01-02-0001', '01-02-0002', '01-02-0003', '01-02-0004'],
            'val':   ['01-02-0005', '01-02-0006', '01-02-0007', '01-02-0008'],
        },
        {
            'test':  ['01-02-0005', '01-02-0006', '01-02-0007', '01-02-0008'],
            'val':   ['01-02-0009', '01-02-0010', '01-02-0011', '01-02-0012'],
        },
        {
            'test':  ['01-02-0009', '01-02-0010', '01-02-0011', '01-02-0012'],
            'val':   ['01-02-0013', '01-02-0014', '01-02-0015', '01-02-0016'],
        },
        {
            'test':  ['01-02-0013', '01-02-0014', '01-02-0015', '01-02-0016'],
            'val':   ['01-02-0017', '01-02-0018', '01-02-0019'],
        },
        {
            'test':  ['01-02-0017', '01-02-0018', '01-02-0019'],
            'val':   ['01-02-0001', '01-02-0002', '01-02-0003', '01-02-0004'],
        },
    ],
}
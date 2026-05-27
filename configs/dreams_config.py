# configs/dreams_config.py

DATA_PARAMS = {
    'fs': 200.0,
    'window_sec': 5.0,
    'overlap_sec': 2.5,
    'lowcut': 0.3,
    'highcut': 30.0,
    'filter_order': 4,
    'use_instance_norm': True,
    'included_stages': [2, 1, 0],
    'hypnogram_resolution_sec': 5.0,

    'subjects_list': [
        'excerpt1', 'excerpt2', 'excerpt3', 'excerpt4',
        'excerpt5', 'excerpt6'
    ],
    'count_spindle_stats': False,
}

CV_CONFIG = {
    'folds_to_run': None,
    'use_explicit_splits': False,  # If True, DREAMS_LOSO_SPLITS below is used
                                   # for every repeat (LOSO is deterministic,
                                   # so all repeats see the same folds).
}

# DREAMS LOSO splits — 6 folds, one subject held out at a time.
# Used to produce the results in Section 9.2.
DREAMS_LOSO_SPLITS = [
    {
        'fold_name': 'Fold_1',
        'test':  ['excerpt1'],
        'val':   ['excerpt2'],
        'train': ['excerpt3', 'excerpt4', 'excerpt5', 'excerpt6'],
    },
    {
        'fold_name': 'Fold_2',
        'test':  ['excerpt2'],
        'val':   ['excerpt3'],
        'train': ['excerpt1', 'excerpt4', 'excerpt5', 'excerpt6'],
    },
    {
        'fold_name': 'Fold_3',
        'test':  ['excerpt3'],
        'val':   ['excerpt4'],
        'train': ['excerpt1', 'excerpt2', 'excerpt5', 'excerpt6'],
    },
    {
        'fold_name': 'Fold_4',
        'test':  ['excerpt4'],
        'val':   ['excerpt5'],
        'train': ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt6'],
    },
    {
        'fold_name': 'Fold_5',
        'test':  ['excerpt5'],
        'val':   ['excerpt6'],
        'train': ['excerpt1', 'excerpt2', 'excerpt3', 'excerpt4'],
    },
    {
        'fold_name': 'Fold_6',
        'test':  ['excerpt6'],
        'val':   ['excerpt1'],
        'train': ['excerpt2', 'excerpt3', 'excerpt4', 'excerpt5'],
    },
]


def get_explicit_splits() -> list:
    """Return the canonical DREAMS LOSO splits with validation."""
    eligible = set(DATA_PARAMS['subjects_list'])
    seen_in_test = []

    for i, fold in enumerate(DREAMS_LOSO_SPLITS):
        for split_name in ('train', 'val', 'test'):
            if split_name not in fold:
                raise ValueError(f"DREAMS Fold {i + 1} missing '{split_name}' key.")

        all_in_fold = set(fold['train']) | set(fold['val']) | set(fold['test'])
        not_eligible = all_in_fold - eligible
        if not_eligible:
            raise ValueError(
                f"DREAMS Fold {i + 1} contains unknown subjects: {sorted(not_eligible)}"
            )

        train_set, val_set, test_set = set(fold['train']), set(fold['val']), set(fold['test'])
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError(f"DREAMS Fold {i + 1} has overlapping splits.")

        seen_in_test.extend(fold['test'])

    test_counts = {s: seen_in_test.count(s) for s in eligible}
    not_tested = [s for s, c in test_counts.items() if c == 0]
    multi_tested = [s for s, c in test_counts.items() if c > 1]
    if not_tested or multi_tested:
        raise ValueError(
            f"DREAMS LOSO coverage error: never tested={not_tested}, "
            f"tested multiple times={multi_tested}"
        )

    return DREAMS_LOSO_SPLITS
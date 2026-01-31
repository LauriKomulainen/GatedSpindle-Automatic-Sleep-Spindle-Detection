# configs/config_loader.py

"""
Centralized configuration loader.

Usage:
    from configs.config_loader import DATA_PARAMS, CV_CONFIG, TESTING_CONFIG
"""

from paths import SELECTED_DATASET

_SUPPORTED_DATASETS = {"DREAMS", "MASS"}

if SELECTED_DATASET not in _SUPPORTED_DATASETS:
    raise ValueError(f"Unknown dataset: {SELECTED_DATASET}. Supported: {_SUPPORTED_DATASETS}")

if SELECTED_DATASET == "DREAMS":
    from configs.dreams_config import DATA_PARAMS, CV_CONFIG
    TESTING_CONFIG = None

elif SELECTED_DATASET == "MASS":
    from configs.mass_config import DATA_PARAMS, CV_CONFIG, TESTING_CONFIG


__all__ = ["DATA_PARAMS", "CV_CONFIG", "TESTING_CONFIG", "SELECTED_DATASET"]
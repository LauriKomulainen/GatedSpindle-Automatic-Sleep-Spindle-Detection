# core/config_loader.py

"""
Reads the dataset name from subject_stats.json in PROCESSED_DATA_DIR
and exposes the matching model config params. Import the params from
this module anywhere they are needed.

Example:
    from config_loader import TRAINING_PARAMS, INFERENCE_PARAMS, POST_PROCESSING_PARAMS
"""

import json
import logging
import importlib
import paths

log = logging.getLogger(__name__)


def _load_dataset_config():
    """
    Read dataset name from subject_stats.json in PROCESSED_DATA_DIR,
    then dynamically import the matching model config module.
    """
    stats_path = paths.PROCESSED_DATA_DIR / "subject_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Could not find {stats_path}. Run the dataset build script first."
        )

    with open(stats_path, "r") as f:
        stats = json.load(f)

    dataset_name = stats.get("dataset")
    if dataset_name is None:
        raise KeyError(
            f"'dataset' key missing from {stats_path}. "
            "Re-run the dataset build script to regenerate it."
        )

    config_module_name = {
        "DREAMS": "configs.dreams_model_config",
        "MASS": "configs.mass_model_config",
    }.get(dataset_name)

    if config_module_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    log.info(f"Loading config for dataset: {dataset_name} ({config_module_name})")
    return dataset_name, importlib.import_module(config_module_name)


DATASET_NAME, _config = _load_dataset_config()
TRAINING_PARAMS = _config.TRAINING_PARAMS
INFERENCE_PARAMS = _config.INFERENCE_PARAMS
POST_PROCESSING_PARAMS = _config.POST_PROCESSING_PARAMS
SIGNAL_VISUALIZATION_PARAMS = _config.SIGNAL_VISUALIZATION_PARAMS
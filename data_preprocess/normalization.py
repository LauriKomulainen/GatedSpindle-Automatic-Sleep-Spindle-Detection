# data_preprocess/normalization.py

import logging
import numpy as np
log = logging.getLogger(__name__)

def normalize_data(data):
    """
    Normalizes data using Z-score (StandardScaler).
    """
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            log.warning("Signal standard deviation is 0. Returning zeroed signal.")
            return data - mean
        normalized_data = (data - mean) / std
        return normalized_data
    except Exception as e:
        log.error(f"Normalization failed: {e}")
        return data
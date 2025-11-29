# data_preprocess/normalization.py

import logging
import numpy as np

log = logging.getLogger(__name__)

def normalize_data(data):
    """
    Robust Z-score normalization using Median and IQR.
    Handling small windows (Instance Norm) robustness included.
    """
    # Clip extreme outliers to prevent scaling issues
    p05 = np.percentile(data, 0.5)
    p995 = np.percentile(data, 99.5)
    data = np.clip(data, p05, p995)

    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    # Estä nollalla jako, jos signaali on täysin tasainen (esim. sensorivirhe)
    if iqr == 0:
        return np.zeros_like(data)

    # 1.349 on kerroin, jolla IQR vastaa keskihajontaa (std) normaalijakaumassa
    return (data - median) / (iqr / 1.349)
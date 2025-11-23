# data_preprocess/normalization.py

import logging
import numpy as np

log = logging.getLogger(__name__)


def normalize_data(data):
    p05 = np.percentile(data, 0.5)
    p995 = np.percentile(data, 99.5)
    data = np.clip(data, p05, p995)

    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    if iqr == 0:
        return data

    return (data - median) / (iqr / 1.349)
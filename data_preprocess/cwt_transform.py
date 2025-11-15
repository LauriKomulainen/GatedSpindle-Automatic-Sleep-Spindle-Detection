# data_preprocess/cwt_transform.py

import numpy as np
import pywt
from tqdm import tqdm
import logging
from typing import Tuple

from training_parameters import CWT_PARAMS, METRIC_PARAMS

log = logging.getLogger(__name__)

CWT_FREQ_BINS = CWT_PARAMS['freq_bins']
CWT_FREQ_LOW = CWT_PARAMS['freq_low']
CWT_FREQ_HIGH = CWT_PARAMS['freq_high']
WAVELET_NAME = CWT_PARAMS['wavelet_name']
SPINDLE_FREQ_LOW = METRIC_PARAMS['spindle_freq_low']
SPINDLE_FREQ_HIGH = METRIC_PARAMS['spindle_freq_high']


def _create_cwt_image(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    """
    Converts one 1D signal window into a 2D CWT image.
    """
    frequencies = np.linspace(CWT_FREQ_LOW, CWT_FREQ_HIGH, CWT_FREQ_BINS)
    try:
        center_freq = pywt.central_frequency(WAVELET_NAME)
    except Exception:
        center_freq = 0.8125  # Fallback for 'morl'

    scales = (center_freq * fs) / frequencies
    scales = scales[::-1]

    try:
        cwt_complex, _ = pywt.cwt(signal_1d, scales=scales, wavelet=WAVELET_NAME)
    except Exception as e:
        log.error(f"PyWavelets CWT failed: {e}. Returning empty image.")
        return np.zeros((CWT_FREQ_BINS, len(signal_1d)), dtype=np.float32)

    cwt_complex = cwt_complex[::-1, :]
    cwt_mag = np.abs(cwt_complex)
    cwt_min, cwt_max = cwt_mag.min(), cwt_mag.max()

    if cwt_max - cwt_min == 0:
        return np.zeros_like(cwt_mag, dtype=np.float32)

    cwt_norm = (cwt_mag - cwt_min) / (cwt_max - cwt_min)
    return cwt_norm.astype(np.float32)


def _create_cwt_mask(mask_1d: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Converts a 1D annotation mask into a 2D CWT mask.
    """
    (height, width) = image_shape
    mask_2d = np.zeros(image_shape, dtype=np.float32)
    frequencies = np.linspace(CWT_FREQ_LOW, CWT_FREQ_HIGH, CWT_FREQ_BINS)
    y_indices = np.where(
        (frequencies >= SPINDLE_FREQ_LOW) &
        (frequencies <= SPINDLE_FREQ_HIGH)
    )[0]
    x_indices = np.where(mask_1d == 1)[0]

    if len(y_indices) > 0 and len(x_indices) > 0:
        mask_2d[np.ix_(y_indices, x_indices)] = 1.0
    return mask_2d


def transform_windows_to_images(x_windows_1d: np.ndarray,
                                y_masks_1d: np.ndarray,
                                fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main conversion function. Converts 1D window arrays to 2D image arrays.
    """
    num_windows, window_length = x_windows_1d.shape
    image_shape = (CWT_FREQ_BINS, window_length)
    all_x_images = np.zeros((num_windows, *image_shape), dtype=np.float32)
    all_y_images = np.zeros((num_windows, *image_shape), dtype=np.float32)

    log.info(f"Converting {num_windows} 1D windows to 2D CWT images...")
    for i in tqdm(range(num_windows), desc="CWT Conversion"):
        all_x_images[i] = _create_cwt_image(x_windows_1d[i], fs)
        all_y_images[i] = _create_cwt_mask(y_masks_1d[i], image_shape)

    x_images_final = np.expand_dims(all_x_images, axis=1)
    y_images_final = np.expand_dims(all_y_images, axis=1)

    return x_images_final, y_images_final
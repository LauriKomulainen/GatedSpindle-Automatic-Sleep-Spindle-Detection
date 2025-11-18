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

# KORJAUS: Määritellään taajuuskaistat uusiksi
DELTA_FREQ_LOW = 0.5
DELTA_FREQ_HIGH = 4.0

# VAIHDETTU: Muscle (20-30Hz) -> Alpha (8-12Hz) vähentämään vääriä positiivisia
CONTEXT_2_FREQ_LOW = 8.0
CONTEXT_2_FREQ_HIGH = 12.0


def _normalize_channel(channel_data: np.ndarray) -> np.ndarray:
    """
    KORJATTU: Ei enää pakota dataa 0-1 välille ikkunakohtaisesti.
    Tämä säilyttää hiljaiset jaksot hiljaisina.
    """
    SCALING_FACTOR = 10.0
    return np.clip(channel_data / SCALING_FACTOR, 0, 5.0).astype(np.float32)


def _create_cwt_image_3channel(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    image_shape = (CWT_FREQ_BINS, len(signal_1d))
    frequencies = np.linspace(CWT_FREQ_LOW, CWT_FREQ_HIGH, CWT_FREQ_BINS)

    try:
        center_freq = pywt.central_frequency(WAVELET_NAME)
    except Exception:
        center_freq = 0.8125

    scales = (center_freq * fs) / frequencies
    scales = scales[::-1]

    try:
        cwt_complex, _ = pywt.cwt(signal_1d, scales=scales, wavelet=WAVELET_NAME)
    except Exception as e:
        log.error(f"PyWavelets CWT failed: {e}. Returning empty image.")
        return np.zeros((3, *image_shape), dtype=np.float32)

    cwt_complex = cwt_complex[::-1, :]
    cwt_mag = np.abs(cwt_complex)

    # Channel 0: Koko taajuuskaista (normalisoitu)
    channel_0 = _normalize_channel(cwt_mag)

    # Channel 1: Delta (Context)
    delta_indices = np.where(
        (frequencies >= DELTA_FREQ_LOW) & (frequencies <= DELTA_FREQ_HIGH)
    )[0]
    channel_1 = np.zeros(image_shape, dtype=np.float32)
    if len(delta_indices) > 0:
        delta_band_raw = cwt_mag[delta_indices, :]
        channel_1[delta_indices, :] = _normalize_channel(delta_band_raw)

    # Channel 2: NYT ALPHA (Context) - Aiemmin Muscle
    # Tämä auttaa mallia näkemään, onko kyseessä 12Hz sukkula vai 10Hz alpha
    alpha_indices = np.where(
        (frequencies >= CONTEXT_2_FREQ_LOW) & (frequencies <= CONTEXT_2_FREQ_HIGH)
    )[0]
    channel_2 = np.zeros(image_shape, dtype=np.float32)
    if len(alpha_indices) > 0:
        alpha_band_raw = cwt_mag[alpha_indices, :]
        channel_2[alpha_indices, :] = _normalize_channel(alpha_band_raw)

    cwt_3_channel = np.stack([channel_0, channel_1, channel_2], axis=0)
    return cwt_3_channel.astype(np.float32)


def _create_cwt_mask(mask_1d: np.ndarray, image_shape: tuple) -> np.ndarray:
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
    num_windows, window_length = x_windows_1d.shape

    image_shape_2d = (CWT_FREQ_BINS, window_length)
    image_shape_3ch = (3, CWT_FREQ_BINS, window_length)

    all_x_images = np.zeros((num_windows, *image_shape_3ch), dtype=np.float32)
    all_y_images = np.zeros((num_windows, *image_shape_2d), dtype=np.float32)

    log.info(f"Converting {num_windows} 1D windows to 3-Channel 2D CWT images (Alpha Context)...")
    for i in tqdm(range(num_windows), desc="3-Channel CWT Conversion"):
        all_x_images[i] = _create_cwt_image_3channel(x_windows_1d[i], fs)
        all_y_images[i] = _create_cwt_mask(y_masks_1d[i], image_shape_2d)

    y_images_final = np.expand_dims(all_y_images, axis=1)

    return all_x_images, y_images_final
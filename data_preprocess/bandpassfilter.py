# data_preprocess/bandpassfilter.py

from scipy.signal import butter, filtfilt
import logging
log = logging.getLogger(__name__)

def apply_bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Applies a zero-phase Butterworth bandpass filter.
    """
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    except Exception as e:
        log.error(f"Bandpass filtering failed: {e}")
        return data
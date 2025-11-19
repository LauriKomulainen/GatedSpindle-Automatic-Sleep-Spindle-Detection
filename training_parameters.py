# training_parameters.py

"""
Central configuration file for all training hyperparameters and data settings.
"""

# --- Training Hyperparameters ---
TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'dropout_rate': 0.4,
    'optimizer_type': 'Adam',
    'num_epochs': 30,  # Max epochs to run
    'early_stopping_patience': 6  # How long to wait for improvement
}

# --- Data & Preprocessing Parameters ---
DATA_PARAMS = {
    'fs': 100.0,  # Target sample rate (all data will be resampled to this)
    'window_sec': 5.0,  # Window length in seconds
    'overlap_sec': 2.5,  # Overlap between windows
    'lowcut': 0.3,  # Bandpass filter low cut (Hz)
    'highcut': 35.0,  # Bandpass filter high cut (Hz)
    'filter_order': 4  # Order of the Butterworth filter
}

# --- CWT Parameters ---
CWT_PARAMS = {
    'freq_bins': 64,  # Image height in pixels
    'freq_low': 1.0,  # Lowest frequency in the image
    'freq_high': 35.0,  # Highest frequency in the image
    'wavelet_name': 'morl'  # Wavelet to use (Morlet)
}

# --- Event Metric Parameters ---
METRIC_PARAMS = {
    'spindle_freq_low': 11.0,  # Start frequency for defining a spindle
    'spindle_freq_high': 16.0,  # End frequency for defining a spindle
    'iou_threshold': 0.2,  # 20% overlap = True Positive
    'min_duration_sec': 0.4,  # Min duration for a prediction to be valid
    'max_duration_sec': 3.0  # Max duration for a prediction to be valid
}

TEST_FAST_FRACTION = {
    'FAST_TEST_FRACTION': 1.0 # E.g., 0.2 takes only 20% of data. Use 1.0 for full training
}
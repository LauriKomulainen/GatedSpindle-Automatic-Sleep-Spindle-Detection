# GatedSpindle: Automatic Sleep Spindle Detection with Gated U-Net and Ensemble framework

This repository contains a deep learning framework designed for the robust detection of sleep spindles in electroencephalography (EEG) signals. The system utilizes a 1D Gated U-Net architecture combined with Stochastic Weight Averaging (SWA) and an ensemble inference strategy to address the challenges of low signal-to-noise ratios and high inter-subject variability inherent in sleep EEG data.

The method has been developed and validated using the DREAMS Sleep Spindles Database.

## Methodology

### 1. Data Preprocessing
The pipeline processes raw EEG signals as follows:
* **Bandpass Filtering:** 4th-order Butterworth filter (0.3–30 Hz).
* **Sleep stage stratification:** Analysis restricted to NREM stages (N2 and N3); Wake and REM stages are excluded.

### 2. Model Inputs
The model operates on fixed-length 1D EEG segments extracted from preprocessed recordings.
Each input sample corresponds to a **5-second window** and consists of a three-channel
time-series representation:

* Channel 1: Bandpass-filtered raw EEG signal (0.3–30 Hz). 
* Channel 2: Sigma-band EEG signal (11–16 Hz), extracted using a 4th-order Butterworth filter.
* Channel 3: Hilbert Envelope from Sigma-band

Independent robust Z-score normalization is applied per 5-second window using
the median and interquartile range (IQR). To reduce the influence of high-amplitude signal excursions,
signals are clipped to the 0.5–99.5 percentile range prior to normalization.

### 3. Model Architecture
**1D Gated U-Net** designed for time-series segmentation:
* **Encoder-Decoder:** Symmetric structure with skip connections to preserve high-resolution temporal features.
* **Gating Mechanism:** Sigmoid-based Attention Gates in skip connections filter irrelevant features (noise) before merging with decoder layers.

### 4. Optimization Strategy
Techniques for improved stability and generalization:
* **Stochastic Weight Averaging (SWA):** Weights averaged over final epochs to approximate a broader, more robust local minimum.
* **Ensemble Inference:** The final detection utilizes a dual-model ensemble strategy that combines:
    1.  **Best Model:** The model checkpoint with the lowest validation loss (peak performance).
    2.  **SWA Model:** The model with averaged weights from the end of training (better generalization).
    
The ensemble is implemented by averaging the raw **logits** (pre-activation outputs) of both models before applying the sigmoid function. This "logit averaging" approach creates a smoother decision boundary and prevents a single overconfident model from dominating the prediction, resulting in more reliable event detection than simple probability averaging.

## Project Structure (NOTE! Outdated)

```text
├── configs/
│   ├── dreams_config.py        # Hyperparameters, recording constants, and path configurations
│   └── mass_config.py          # TBD
│
├── core/
│   ├── model.py                # 1D Gated U-Net architecture
│   ├── dataset.py              # Dataset, DataLoader logic, and Augmentation
│   └── evaluation.py           # Event-based metrics
│
├── data_loaders/
│   ├── dreams_loader.py        # Parsers for DREAMS .edf signals and .txt annotations
│   └── mass_loader.py          # TBD
│
├── postprocessing/         
│   └── postprocessing.py       # Dual-threshold event detection, merging logic, and window stitching
│
├── signal_processing/          
│   ├── bandpassfilter.py       # Butterworth bandpass filter implementation
│   ├── normalization.py        # Robust Z-score normalization (IQR-based)
│   └── transforms.py           # Model input channels
│
├── utils/                
│   ├── logger.py               # Centralized logging configuration
│   ├── reporting.py            # Generates detailed CSV error analysis and signal stats
│   ├── plot_results.py         # Visualization tool for performance charts
│   └── signal_visualization.py # Plots RAW signal & input signals for model
│
├── main.py                     # Orchestrator for LOSO cross-validation, training, and inference
├── build_dataset.py            # Offline preprocessing: converts raw EDFs to optimized .npy tensors
└── paths.py                    # Global path definitions & which dataset is used (DREAMS or MASS)
```

## Usage Instructions

### Environment Notes

The model and training pipeline were developed and tested on macOS (Apple Silicon, MacBook Pro 2024, M4).
The code is expected to be platform-agnostic, but has not been
explicitly tested on Windows systems.

### 1. Installation

To ensure reproducibility, please follow these setup steps.

1. Install Python `3.13.0`

**Note**: the project has been tested with this specific version. Correct functionality cannot be guaranteed with other Python versions.
2. Install the required Python packages using pip:
```bash
Pip install -r requirements.txt
```

### 2. Other instructions TBD...

## License & Citation
This project is open-source and available under the MIT License (see the LICENSE file for details). You are free to use, modify, and distribute this software for research and development purposes.

Citation request: If you use this model or code in your research or develop it further, please credit this repository.

Contact: If you encounter issues with the model or have questions regarding the implementation, please contact: laurikom(at)student.uef.fi
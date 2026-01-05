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
Each input sample corresponds to a **5-second window** and consists of a two-channel
time-series representation:

* Channel 1: Bandpass-filtered raw EEG signal (0.3–30 Hz). 
* Channel 2: Sigma-band EEG signal (11–16 Hz), extracted using a 4th-order Butterworth filter.

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

## Project Structure

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
│   ├── base_loader.py          # TBD
│   ├── dreams_loader.py        # Parsers for DREAMS .edf signals and .txt annotations
│   ├── mass_loader.py          # TBD        
│   └── user_data_loader.py     # TBD
│
├── postprocessing/         
│   └── postprocessing.py       # Dual-threshold event detection, merging logic, and window stitching
│
├── preprocessing/          
│   ├── bandpassfilter.py       # Butterworth bandpass filter implementation     
│   └── normalization.py        # Robust Z-score normalization (IQR-based)
│
├── utils/                
│   ├── logger.py               # Centralized logging configuration
│   ├── reporting.py            # Generates detailed CSV error analysis and signal stats
│   └── signal_visualization.py # Plots RAW signal & input signals for model
│
├── main.py                     # Orchestrator for LOSO cross-validation, training, and inference
├── data_handler.py             # Offline preprocessing: converts raw EDFs to optimized .npy tensors
├── plot_results.py             # Visualization tool for performance charts
└── paths.py                    # Global path definitions
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

### 2. Data Setup

1. Download the DREAMS Sleep Spindle Database. Other datasets are not currently supported without code changes.
2. Create a folder at data/DREAMS in the project root directory.
3. Move all downloaded files (both .edf recordings and .txt annotations) directly into this folder.
4. Open `paths.py` and ensure the RAW_DREAMS_DATA_DIR variable matches your data location (default is data/DREAMS).

### 3. Data Preprocessing

Before training, the raw EEG data must be converted into processed tensors (.npy format).
```bash
python data_handler.py
```
* This script performs bandpass filtering (0.3-30Hz), segmentation, and Z-score normalization.
* Processed files are saved to the data/processed directory (defined in paths.py).
* If you modify filtering parameters in `dreams_config.py`, you must re-run this script to regenerate the data.

### 4. Model Training

Run the main training loop to start the Leave-One-Subject-Out (LOSO) cross-validation. For detailed benchmarking, reproducibility, and seed configuration, see the **Performance Evaluation** section below.

## Performance Evaluation

To ensure the reliability of the results and fair comparison with existing literature, the model is evaluated using a Leave-One-Subject-Out (LOSO) cross-validation protocol.

### 1. Standard benchmark (No shuffle)
This experiment evaluates the stability of the model across different random initializations (random seeds) while keeping the subject folds fixed. This setup enables direct and fair comparison.

The initial random seed is selected uniformly at random from the range **1–99,999**. After each complete LOSO iteration, the seed is incremented by one. This process is repeated until a total of three (--repeats 3) LOSO runs are completed. To reproduce the reported results, execute the following command:
```bash
python main.py --mode train --repeats 3 --seed 51143
```
Average performance of 3 runs (No shuffle):
* F1-score: 0.7920 ± 0.0140
* Precision: 0.7902 ± 0.0133
* Recall: 0.7983 ± 0.0212
* mIoU: 0.7933 ± 0.0074

For non-deterministic verification runs, the `--seed` argument may be omitted. When unspecified, the initial seed is sampled uniformly at random from **1–99,999**, and subsequently incremented by one after each complete LOSO cycle.
```bash
python main.py --mode train --repeats 3 --seed
```

### 2. Robustness test (With training fold shuffling)
This experiment tests the model's ability to generalize across different subject combinations. By shuffling the validation subjects for each repeat (ensure the performance is not biased toward a specific subject pairing).
```bash
python main.py --mode train --repeats 3 --seed --shuffle_folds
```
Average performance of 3 runs (With training fold shuffling):
* F1-score: TBD
* Precision: TBD
* Recall: TBD
* mIoU: TBD

### 3. Reference Single-Run Performance (Seed = 1)

This experiment reports the performance of a single deterministic Leave-One-Subject-Out (LOSO) run using a fixed random initialization.

The model is trained and evaluated using a fixed random seed (`seed = 1`) and a single LOSO
cycle (`--repeats 1`). Unlike the standard benchmark, this experiment does not assess
stability across random initializations. To reproduce the reported results, execute the following command:
```bash
python main.py
```
Average performance of 1 run (fixed seed):
* F1-score: 0.8199 
* Precision: 0.7852 
* Recall: 0.8615 
* mIoU: 0.7800

## License & Citation
This project is open-source and available under the MIT License (see the LICENSE file for details). You are free to use, modify, and distribute this software for research and development purposes.

Citation request: If you use this model or code in your research or develop it further, please credit this repository.

Contact: If you encounter issues with the model or have questions regarding the implementation, please contact: laurikom(at)student.uef.fi
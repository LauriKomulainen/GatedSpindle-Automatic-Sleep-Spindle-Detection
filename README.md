# GatedSpindle: Automatic Sleep Spindle Detection with Gated U-Net and Ensemble framework

This repository contains a deep learning framework designed for the robust detection of sleep spindles in electroencephalography (EEG) signals. The system utilizes a 1D Gated U-Net architecture combined with Stochastic Weight Averaging (SWA) and an ensemble inference strategy to address the challenges of low signal-to-noise ratios and high inter-subject variability inherent in sleep EEG data.

The method has been developed and validated using the DREAMS Sleep Spindles Database.

## Methodology

### 1. Data Preprocessing
The pipeline processes raw EEG signals as follows:
* **Bandpass Filtering:** 4th-order Butterworth filter (0.3–30 Hz).
* **Sleep Stage Stratification:** Analysis restricted to NREM stages (N2 and N3); Wake and REM epochs excluded to minimize false positives.
* **Instance Normalization:** Independent Z-score normalization for each 5-second window to mitigate inter-subject amplitude variability.

### 2. Model Architecture
**1D Gated U-Net** designed for time-series segmentation:
* **Encoder-Decoder:** Symmetric structure with skip connections to preserve high-resolution temporal features.
* **Gating Mechanism:** Sigmoid-based Attention Gates in skip connections filter irrelevant features (noise) before merging with decoder layers.

### 3. Optimization Strategy
Techniques for improved stability and generalization:
* **Stochastic Weight Averaging (SWA):** Weights averaged over final epochs to approximate a broader, more robust local minimum.
* **Ensemble Inference:** Final detection averages predictions from the "Best Model" (lowest validation loss) and the "SWA Model".

## Project Structure

```text
├── configs/
│   └── dreams_config.py    # Main configuration for DREAMS dataset
├── core/
│   ├── model.py            # Gated U-Net architecture implementation
│   ├── dataset.py          # PyTorch Dataset and augmentation logic
│   └── metrics.py          # Evaluation metrics and event detection logic
├── data_loaders/           # Loaders for DREAMS/MASS datasets
│   └── dreams_config.py 
├── data_preprocess/        # Signal filtering and normalization tools
│   ├── bandpassfilter.py            
│   └── normalization.py          
├── utils/                  # Logging and helper utilities
│   ├── find_best_seed.py            
│   └── logger.py  
├── main.py                 # Main training and evaluation script
├── data_handler.py         # Script for raw data preprocessing
└── paths.py                # System path definitions
```

## Usage Instructions

### 1. Data Setup
The project uses `paths.py` to locate data.
1.  Open `paths.py` and verify the `RAW_DREAMS_DATA_DIR` variable (default is `data/DREAMS`).
2.  Place all DREAMS database files (both `.edf` and `.txt` annotations) directly into this folder.

### 2. Data Preprocessing
Before training, convert the raw EEG data into processed tensors (`.npy` format).
```bash
python data_handler.py
```
* This script performs bandpass filtering (0.3-30Hz), segmentation, and Z-score normalization.
* Output: Processed files are saved to the data/processed directory (defined in paths.py).
* Important: If you modify filtering parameters in dreams_config.py, you must re-run this script.

### 3. Model Training

Run the main training loop. This script performs the Leave-One-Subject-Out (LOSO) cross-validation.
```bash
python main.py
```
* The script automatically loads the preprocessed data.
* It trains a model for each fold (holding one subject out for testing) using the settings in configs/dreams_config.py.
* Logs, checkpoints, and detailed CSV error analysis reports are saved to model_reports/.

## Performance Evaluation

The model's performance was evaluated using Leave-One-Subject-Out (LOSO) cross-validation on the DREAMS database. The final predictions were generated using an ensemble of the best validation model and the SWA model.

### Model Performance by Subject (LOSO Cross-Validation)

| Subject       | F1-score | Precision | Recall | TP (Count) | FP (Count) | FN (Count) | mIoU (TPs) |
|:--------------| :--- | :--- | :--- |:-----------|:-----------|:-----------| :--- |
| **Excerpt 1** | 0.8073 | 0.7817 | 0.8346 | 111 | 31 | 22 | - |
| **Excerpt 2** | 0.7947 | 0.7792 | 0.8108 | 60 | 17 | 14 | - |
| **Excerpt 3** | 0.8434 | 0.8140 | 0.8750 | 35 | 8 | 5 | - |
| **Excerpt 4** | 0.7600 | 0.7308 | 0.7917 | 19 | 7 | 5 | - |
| **Excerpt 5** | 0.8000 | 0.7706 | 0.8317 | 84 | 25 | 17 | - |
| **Excerpt 6** | 0.8195 | 0.8936 | 0.7568 | 84 | 10 | 27 | - |
| **Average**   | **0.8041** | **0.7950** | **0.8168** | **-** | **-** | **-** | **0.7873** |

## License & Citation
This project is open-source and available under the MIT License (see the LICENSE file for details). You are free to use, modify, and distribute this software for research and development purposes.

Citation Request: If you use this model or code in your research or develop it further, please credit this repository.

Contact: If you encounter issues with the model or have questions regarding the implementation, please contact: laurikom(at)student.uef.fi
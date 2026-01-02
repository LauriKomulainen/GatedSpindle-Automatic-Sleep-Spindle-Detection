# GatedSpindle: Automatic Sleep Spindle Detection with Gated U-Net and Ensemble framework

This repository contains a deep learning framework designed for the robust detection of sleep spindles in electroencephalography (EEG) signals. The system utilizes a 1D Gated U-Net architecture combined with Stochastic Weight Averaging (SWA) and an ensemble inference strategy to address the challenges of low signal-to-noise ratios and high inter-subject variability inherent in sleep EEG data.

The method has been developed and validated using the DREAMS Sleep Spindles Database.

## Methodology

### 1. Data Preprocessing
The pipeline processes raw EEG signals as follows:
* **Bandpass Filtering:** 4th-order Butterworth filter (0.3–30 Hz).
* **Sleep Stage Stratification:** Analysis restricted to NREM stages (N2 and N3); Wake and REM stages are excluded.
* **Instance Normalization:** Independent Z-score normalization for each 5-second window to mitigate inter-subject amplitude variability.

### 2. Model Architecture
**1D Gated U-Net** designed for time-series segmentation:
* **Encoder-Decoder:** Symmetric structure with skip connections to preserve high-resolution temporal features.
* **Gating Mechanism:** Sigmoid-based Attention Gates in skip connections filter irrelevant features (noise) before merging with decoder layers.

### 3. Optimization Strategy
Techniques for improved stability and generalization:
* **Stochastic Weight Averaging (SWA):** Weights averaged over final epochs to approximate a broader, more robust local minimum.
* **Ensemble Inference:** The final detection utilizes a dual-model ensemble strategy that combines:
    1.  **Best Model:** The model checkpoint with the lowest validation loss (peak performance).
    2.  **SWA Model:** The model with averaged weights from the end of training (better generalization).
    
The ensemble is implemented by averaging the raw **logits** (pre-activation outputs) of both models before applying the sigmoid function. This "logit averaging" approach creates a smoother decision boundary and prevents a single overconfident model from dominating the prediction, resulting in more reliable event detection than simple probability averaging.

## Project Structure

```text
├── configs/
│   └── dreams_config.py        # Hyperparameters, recording constants, and path configurations
│
├── core/
│   ├── model.py                # 1D Gated U-Net architecture
│   ├── dataset.py              # Dataset, DataLoader logic, and Augmentation
│   └── evaluation.py           # Event-based metrics
│
├── data_loaders/           
│   └── dreams_loader.py        # Parsers for DREAMS .edf signals and .txt annotations
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

### 1. Installation

To ensure reproducibility, please follow these setup steps.

1. Install Python `3.13.0`
* Note: The project has been tested with this specific version. Correct functionality cannot be guaranteed with other Python versions.
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

Run the main training loop to start the Leave-One-Subject-Out (LOSO) cross-validation.
```bash
python main.py
```
* The script automatically loads the preprocessed data. 
* It trains a model for each fold (holding one subject out for testing) using the hyperparameters defined in configs/dreams_config.py. 
* Logs, model checkpoints, and detailed CSV error analysis reports are saved to the model_reports/ directory.

NOTE: If you have already trained models and want to re-calculate metrics (e.g., to test a different threshold strategy) without retraining, use the evaluate mode:
```bash
python main.py --mode evaluate --run_dir model_reports/LOSO_run_YYYY-MM-DD_HH-MM-SS
```
* `--mode evaluate`: Skips the training process and loads saved models (.pth files). 
* `--run_dir`: The path to the specific directory created during a previous training run.


## Performance Evaluation

The model's performance was evaluated using Leave-One-Subject-Out (LOSO) cross-validation on the DREAMS database. The final predictions were generated using an ensemble of the best validation model and the SWA model.

### Model Performance by Subject (LOSO Cross-Validation)

| Subject       | F1-score | Precision | Recall | TP (Count) | FP (Count) | FN (Count) | mIoU (TPs) |
|:--------------| :--- | :--- | :--- |:-----------|:-----------|:-----------| :--- |
| **Excerpt 1** | 0.8223 | 0.7712 | 0.8806 | 118 | 35 | 16 | - |
| **Excerpt 2** | 0.8153 | 0.7711 | 0.8649 | 64 | 19 | 10 | - |
| **Excerpt 3** | 0.8471 | 0.8000 | 0.9000 | 36 | 9 | 4 | - |
| **Excerpt 4** | 0.8000 | 0.7333 | 0.8800 | 22 | 8 | 3 | - |
| **Excerpt 5** | 0.8257 | 0.7826 | 0.8738 | 90 | 25 | 13 | - |
| **Excerpt 6** | 0.8093 | 0.8529 | 0.7699 | 87 | 15 | 26 | - |
| **Average** | **0.8199** | **0.7852** | **0.8615** | **-** | **-** | **-** | **0.7800** |
## License & Citation
This project is open-source and available under the MIT License (see the LICENSE file for details). You are free to use, modify, and distribute this software for research and development purposes.

Citation Request: If you use this model or code in your research or develop it further, please credit this repository.

Contact: If you encounter issues with the model or have questions regarding the implementation, please contact: laurikom(at)student.uef.fi
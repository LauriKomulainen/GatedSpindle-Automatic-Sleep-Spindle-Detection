# GatedSpindle: Automatic Sleep Spindle Detection with Gated U-Net and Ensemble framework

This repository contains a deep learning framework designed for the robust detection of sleep spindles in electroencephalography (EEG) signals. The system utilizes a 1D Gated U-Net architecture combined with Stochastic Weight Averaging (SWA) and an ensemble inference strategy to address the challenges of low signal-to-noise ratios and high inter-subject variability inherent in sleep EEG data.

The method has been developed and validated using the DREAMS Sleep Spindles Database.

## System Environment

The code was developed and tested in the following environment:
* **Hardware:** MacBook Pro (M4 Chip, Apple Silicon)
* **Software:** Python 3.13.0, PyTorch
* **Dependencies:** MNE, PyTorch, SciPy, NumPy, Pandas (see requirements).

## Performance Evaluation

The model's performance was evaluated using Leave-One-Subject-Out (LOSO) cross-validation on the DREAMS database. The final predictions were generated using an ensemble of the best validation model and the SWA model.

**Training Configuration:**
* **Optimizer:** Adam (`weight_decay=1e-4`)
* **Regularization:** Dropout (0.2) + SWA
* **Preprocessing:** 200 Hz sampling rate, 0.3–30 Hz bandpass filter

**LOSO Cross-Validation Results:**

| Subject | F1-score | Precision | Recall | TP (Count) | FP (Count) | FN (Count) | mIoU (TPs) |
| :--- | :--- | :--- | :--- |:-----------|:-----------|:-----------| :--- |
| **Excerpt 1** | 0.8218 | 0.7958 | 0.8496 | 113        | 29         | 20         | 0.7958 |
| **Excerpt 2** | 0.7947 | 0.7895 | 0.8000 | 60         | 16         | 15         | 0.8095 |
| **Excerpt 3** | 0.8434 | 0.8140 | 0.8750 | 35         | 8          | 5          | 0.7735 |
| **Excerpt 4** | 0.7308 | 0.7037 | 0.7600 | 19         | 8          | 6          | 0.7911 |
| **Excerpt 5** | 0.7981 | 0.7757 | 0.8218 | 83         | 24         | 18         | 0.8241 |
| **Excerpt 6** | 0.8213 | 0.8947 | 0.7589 | 85         | 10         | 27         | 0.8248 |
| **MEAN** | **0.8017** | **0.7956** | **0.8109** | **-**      | **-**      | **-**      | **0.8031** |

## Methodology

### 1. Data Preprocessing Pipeline
To ensure high-quality input for the neural network, the raw EEG signals undergo a strict preprocessing pipeline:
* **Bandpass Filtering:** Signals are filtered between 0.3 Hz and 30 Hz using a 4th-order Butterworth filter to remove DC drift and high-frequency muscle artifacts.
* **Sleep Stage Stratification:** Training and inference are explicitly restricted to NREM sleep stages (N2 and N3). Epochs classified as Wake or REM are excluded to reduce false positives.
* **Instance Normalization:** Each 5-second input window is normalized independently (Z-score normalization). This approach mitigates the issue of amplitude variability between different subjects.

### 2. Model Architecture
The core of the system is a 1D Gated U-Net architecture adapted for time-series segmentation.
* **Encoder-Decoder Structure:** The network uses a symmetric U-Net design with skip connections to preserve spatial information.
* **Gating Mechanism:** Sigmoid-based gating units are applied within the skip connections. These gates learn to filter out irrelevant features (noise) from the encoder before they are merged with the decoder features.

### 3. Optimization Strategy
The training process incorporates advanced techniques to ensure stability:
* **Stochastic Weight Averaging (SWA):** Weights are averaged over the final training epochs to approximate a broader local minimum in the loss landscape.
* **Ensemble Inference:** The final detection is an average of the predictions from the "Best Model" (lowest validation loss) and the "SWA Model".

## Configuration and Data Setup

This framework is configured via `config.py`. You must set up your data paths before running the code.

### 1. Path Configuration
Open `config.py` and locate the `PATHS` dictionary. Update `raw_data_dir` to point to the folder where you downloaded the DREAMS database (.edf and .txt files).

```
PATHS = {
    "raw_data_dir": "/absolute/path/to/your/data/raw_DREAMS",
    "processed_data_dir": "./data/processed",
    "output_dir": "./model_reports"
}
```

### 2. Annotation Merging Strategy

The DREAMS database provides annotations from two independent experts. You can configure how these are combined using `DATA_PARAMS` in `config.py`:
* UNION: A spindle is accepted if marked by Expert 1 OR Expert 2. This maximizes the number of training events but may include more uncertain spindles. (Default)
* INTERSECTION: A spindle is accepted only if marked by BOTH scorers.

## Usage Instructions

### 1. Data Preprocessing

Before training, the raw EDF and text files must be processed into tensor format. This step performs filtering, segmentation, and annotation merging based on your configuration.
```
python data_handler.py
```
* Output: Saves .npy files to processed_data_dir. 
* Important: If you change filtering parameters (e.g. frequency bands) or the merge mode in `config.py`, you must re-run this script to generate new data

### 2. Model Training

Run the main training loop. This script performs Leave-One-Subject-Out (LOSO) cross-validation.
```
python main.py
```
* The script loads the preprocessed data. 
* It trains a model for each fold (holding one subject out for testing). 
* Logs and detailed CSV error analysis reports are saved to model_reports/.

## Project Structure
* `config.py`: Central configuration file (paths, hyperparameters). 
* `main.py`: Primary script for training and evaluation. 
* `data_handler.py`: Script for loading and preprocessing raw data. 
* `UNET_model/`: Contains `model.py` (Gated U-Net) and `evaluation_metrics.py`. 
* `data_preprocess/`: Helper modules (`handler.py`, `bandpassfilter.py`, `normalization.py`). 
* `utils/`: Logging utilities.

## License & Citation
This project is open-source and available under the MIT License (see the LICENSE file for details). You are free to use, modify, and distribute this software for research and development purposes.

Citation Request: If you use this model or code in your research or develop it further, please credit this repository.

Contact: If you encounter issues with the model or have questions regarding the implementation, please contact: laurikom(at)student.uef.fi
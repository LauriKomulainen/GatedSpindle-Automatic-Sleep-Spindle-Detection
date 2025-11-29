# Sleep Spindle Detection: Gated U-Net & Dual-Model Ensemble

This repository implements a deep learning framework for robust sleep spindle detection using a **Gated U-Net** architecture. The system is specifically engineered to handle high inter-subject variability and suppress false positives in noisy EEG recordings by enforcing physiological validity through stage-stratified learning.

## Key Architectural Features

### 1. Advanced Data Pipeline (Physiologically Aware)
To ensure the model learns valid physiological features rather than artifacts, the pipeline employs strict quality control:
* **Stage-Stratified Filtering:** Training and inference are explicitly restricted to NREM stages 2 and 3 (N2+N3). This eliminates "Wake" and "REM" epochs where spindle-like artifacts (e.g., alpha waves) often cause false positives.
* **Instance Normalization:** Instead of global normalization, each 5-second window is normalized independently (Z-score). This allows the model to detect spindles based on morphology and relative frequency content, regardless of amplitude fluctuations across the recording.
* **Dual-Channel Input:** The model receives a 2-channel time-series (200 Hz):
    1.  **Raw EEG:** Provides broad spectral context.
    2.  **Sigma-filtered (11-16 Hz):** Explicitly highlights the spindle frequency band.

### 2. Gated U-Net (Architecture) The core model is a specialized 1D U-Net designed for segmentation in low-SNR environments:
* **Residual Blocks:** The encoder and decoder utilize residual connections (Conv1d + InstanceNorm + ReLU) to facilitate deep feature learning and prevent gradient degradation.
* **Global Gating Mechanism:** A parallel classification branch analyzes the entire window to output a global probability score (0-1). This "gate" suppresses the segmentation output if the window is unlikely to contain a spindle, significantly reducing False Positives.

### 3. Robust Inference Strategy
The final prediction is not the result of a single pass but a stabilized ensemble:
* **Dual-Model Ensemble:** Predictions are averaged from two distinct model checkpoints:
    1.  **Best Val Loss:** The model snapshot with the lowest validation loss.
    2.  **SWA (Stochastic Weight Averaging):** A model utilizing weights averaged over the final training epochs (e.g., 90-150), ensuring a flatter and more generalizable local minimum.
* **Test-Time Augmentation (TTA):** Signals are processed in both original and time-flipped orientations, with results averaged to reduce uncertainty.

---

## Performance Results (LOSO Cross-Validation)

The model was evaluated using Leave-One-Subject-Out (LOSO) cross-validation on the DREAMS database. By filtering out non-NREM artifacts (Stage Filtering):

| Subject | F1-score | Precision | Recall | TP (Events) | FP (Events) | mIoU (TPs) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Excerpt 1** | **0.8061** | 0.82 | 0.80 | 106 | 24 | 0.7744 |
| **Excerpt 2** | **0.8000** | 0.80 | 0.80 | 60 | 15 | 0.8013 |
| **Excerpt 3** | **0.8205** | 0.84 | 0.80 | 32 | 6 | 0.8063 |
| **Excerpt 4** | **0.7755** | 0.79 | 0.76 | 19 | 5 | 0.7596 |
| **Excerpt 5** | **0.7467** | 0.68 | 0.83 | 84 | 40 | 0.8016 |
| **Excerpt 6** | **0.7668** | 0.91 | 0.66 | 74 | 7 | 0.8050 |
| **AVERAGE** | **0.7859** <br> (± 0.0252) | **0.8067** <br> (± 0.0704) | **0.7749** <br> (± 0.0551) | **62.5** | **16.2** | **0.7914** <br> (± 0.0179) |

*Note: Evaluation is restricted to N2 and N3 sleep stages to ensure biological validity and comparability with ground truth, removing artifacts present in Wake/REM stages.*
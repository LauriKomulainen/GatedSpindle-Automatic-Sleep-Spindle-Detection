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
| **Excerpt 1** | **0.8061** | 0.8154 | 0.7970 | 106 | 24 | 0.7744 |
| **Excerpt 2** | **0.7761** | 0.8814 | 0.6933 | 52 | 7 | 0.7921 |
| **Excerpt 3** | **0.8831** | 0.9189 | 0.8500 | 34 | 3 | 0.7834 |
| **Excerpt 4** | **0.6977** | 0.8333 | 0.6000 | 15 | 3 | 0.7449 |
| **Excerpt 5** | **0.7959** | 0.8211 | 0.7723 | 78 | 17 | 0.8124 |
| **Excerpt 6** | **0.7923** | 0.8632 | 0.7321 | 82 | 13 | 0.8190 |
| **AVERAGE** | **0.7919** <br> (± 0.0543) | **0.8555** <br> (± 0.0366) | **0.7408** <br> (± 0.0798) | **61.2** | **11.2** | **0.7877** <br> (± 0.0246) |

*Note: Evaluation is restricted to N2 and N3 sleep stages to ensure biological validity and comparability with ground truth, removing artifacts present in Wake/REM stages.*
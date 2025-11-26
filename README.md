# Sleep Spindle Detection: Gated U-Net & Dual-Model Ensemble

This repository implements a deep learning framework for robust sleep spindle detection using a **Gated U-Net** architecture. The system is specifically engineered to handle high inter-subject variability and suppress false positives in noisy EEG recordings without sacrificing recall on clean data.

## Key Architectural Features

### 1. Gated U-Net (Architecture)
Unlike standard segmentation models that treat every window equally, our **Gated U-Net** incorporates a **Global Classification Branch (Gating Mechanism)** alongside the standard segmentation path.
* **Mechanism:** The gating branch analyzes the entire 5-second window to determine the global probability of a spindle's presence.

### 2. Dual-Model Ensemble (Inference)
To maximize stability and generalization, the inference pipeline utilizes a **Dual-Model Ensemble** strategy:
* **Model A:** The "Best Single Epoch" model (lowest validation loss).
* **Model B:** The **Stochastic Weight Averaging (SWA)** model, which aggregates weights across the final trajectory of training to find a flatter, more robust minimum.
* **Result:** The final prediction is the average of these two models, smoothed further by **Test-Time Augmentation (TTA)** (signal flipping).

### 3. Optimized Data Pipeline
* **Sampling Rate:** Standardized to **200 Hz** to capture high-frequency sigma characteristics accurately.
* **Inputs:** 2-Channel Time-Series:
    * **Raw EEG** (Context)
    * **Sigma-band filtered signal** (11-16 Hz) (Focus)

---

## Performance Results (LOSO Cross-Validation)

The model was evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation on the DREAMS database (n=6, union of two experts). The results below reflect the **Global Optimized Configuration** (Threshold: 0.6, Min Duration: 0.5s, Merge Gap: 0.3s), demonstrating valid, automated performance without subject-specific tuning.

| Subject | F1-score | Precision | Recall | TP (Events) | FP (Events) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Excerpt 1** | **0.8248** | 0.81 | 0.84 | 113 | 27 |
| **Excerpt 2** | **0.7481** | 0.91 | 0.64 | 49 | 5 |
| **Excerpt 3** | **0.8222** | 0.80 | 0.84 | 37 | 9 |
| **Excerpt 4** | **0.6316** | 0.60 | 0.67 | 42 | 28 |
| **Excerpt 5** | **0.7739** | 0.80 | 0.75 | 77 | 19 |
| **Excerpt 6** | **0.7647** | 0.90 | 0.67 | 78 | 9 |
| **AVERAGE** | **0.7609** | **0.8029** | **0.7336** | **66.0** | **16.1** |

## Configuration & Reproducibility
TBD
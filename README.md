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

The model was evaluated using Leave-One-Subject-Out (LOSO) cross-validation on the DREAMS database.

| Subject | F1-score | Precision | Recall | TP (Events) | FP (Events) | mIoU (TPs) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Excerpt 1** | **0.8248** | 0.81 | 0.84 | 113 | 27 | 0.7743 |
| **Excerpt 2** | **0.7481** | 0.91 | 0.64 | 49 | 5 | 0.7922 |
| **Excerpt 3** | **0.8222** | 0.80 | 0.84 | 37 | 9 | 0.7739 |
| **Excerpt 4** | **0.5838** | 0.44 | 0.89 | 54 | 70 | 0.6966 |
| **Excerpt 5** | **0.7739** | 0.80 | 0.75 | 77 | 19 | 0.8262 |
| **Excerpt 6** | **0.7826** | 0.85 | 0.72 | 81 | 14 | 0.8213 |
| **AVERAGE** | **0.7559** <br> (± 0.0815) | **0.7680** <br> (± 0.1537) | **0.7800** <br> (± 0.0837) | **68.5** | **24.0** | **0.7807** <br> (± 0.0428) |

## Configuration & Reproducibility
TBD
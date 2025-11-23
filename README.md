# Model Performance Report (Latest Run)
This repository implements a robust deep learning framework for sleep spindle detection using a customized **Res-U-Net** architecture.

## Key Architectural Features
1. **Backbone:** U-Net with Residual Blocks (Res-U-Net).
2. **Global Context:** Transformer Encoder Bottleneck to capture long-range temporal dependencies.
3. **Domain Generalization:** **Instance Normalization** is used instead of Batch Normalization to make the model invariant to signal amplitude differences between subjects (solving the "quiet vs. loud" subject problem).
4. **Attention Mechanism:** **CBAM (Convolutional Block Attention Module)** is integrated into the encoder to help the model focus on relevant morphological features (spindle shape) and ignore noise.
5. **Inference:** **Test-Time Augmentation (TTA)** is used during prediction (averaging predictions of the original and flipped signal) to improve robustness.

## Input Data
Multi-view 3-Channel Time-Series:
- **Raw EEG** (Context)
- **Sigma-band filtered signal** (11-16 Hz) (Focus)
- **Teager Energy Operator (TEO)** signal (Energy)

## Performance Results (LOSO Cross-Validation)
The model was evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation on the DREAMS database.

| Subject (Excerpt) | F1-score | Precision | Recall | TP (Events) | FP (Events) | FN (Events) | mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Excerpt 1** | 0.7727 | 0.7846 | 0.7612 | 102 | 28 | 32 | 0.8056 |
| **Excerpt 2** | 0.6772 | 0.8600 | 0.5584 | 43 | 7 | 34 | 0.7686 |
| **Excerpt 3** | 0.7475 | 0.6727 | 0.8409 | 37 | 18 | 7 | 0.7424 |
| **Excerpt 4** | 0.6165 | 0.5775 | 0.6613 | 41 | 30 | 21 | 0.7277 |
| **Excerpt 5** | 0.7721 | 0.7411 | 0.8058 | 83 | 29 | 20 | 0.8245 |
| **Excerpt 6** | 0.7822 | 0.8148 | 0.7521 | 88 | 20 | 29 | 0.8142 |
| **AVERAGE** | **0.7280 (± 0.0609)** | **0.7418 (± 0.0939)** | **0.7300 (± 0.0946)** | **65.7** | **22.0** | **23.8** | **0.7805 (± 0.0367)** |
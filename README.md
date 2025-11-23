# Model Performance Report (Latest Run)
This repository implements a robust deep learning framework for sleep spindle detection using a customized **Res-U-Net** architecture. The model is designed for high inter-subject generalization using advanced optimization and attention techniques.

## Key Architectural Features
1. **Backbone:** U-Net with Residual Blocks (Res-U-Net).
2. **Global Context:** Transformer Encoder Bottleneck to capture long-range temporal dependencies.
3. **Domain Generalization:** **Instance Normalization** is used instead of Batch Normalization to make the model invariant to signal amplitude differences between subjects (solving the "quiet vs. loud" subject problem).
4. **Attention Mechanism:** **CBAM (Convolutional Block Attention Module)** is integrated into the encoder to help the model focus on relevant morphological features (spindle shape) and ignore noise.
5. **Training Optimization:** **Stochastic Weight Averaging (SWA)** combined with **Cosine Annealing Warm Restarts** is used to find a robust minimum in the loss landscape, ensuring superior generalization.
6. **Post-Processing:** **Test-Time Augmentation (TTA)** and **Event Merging** (combining closely spaced predictions) are used to maximize event recall and minimize fragmentation.

## Input Data
Multi-view 3-Channel Time-Series:
- **Raw EEG** (Context)
- **Sigma-band filtered signal** (11-16 Hz) (Focus)
- **Teager Energy Operator (TEO)** signal (Energy)

## Performance Results (LOSO Cross-Validation)
The model was evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation on the DREAMS database.

| Subject (Excerpt) | F1-score | Precision | Recall | TP (Events) | FP (Events) | FN (Events) | mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Excerpt 1** | 0.7661 | 0.6975 | 0.8496 | 113 | 49 | 20 | 0.7959 |
| **Excerpt 2** | 0.7576 | 0.9091 | 0.6494 | 50 | 5 | 27 | 0.8135 |
| **Excerpt 3** | 0.7500 | 0.6923 | 0.8182 | 36 | 16 | 8 | 0.7914 |
| **Excerpt 4** | 0.6165 | 0.5694 | 0.6721 | 41 | 31 | 20 | 0.7321 |
| **Excerpt 5** | 0.7477 | 0.7080 | 0.7921 | 80 | 33 | 21 | 0.8205 |
| **Excerpt 6** | 0.7892 | 0.7928 | 0.7857 | 88 | 23 | 24 | 0.8146 |
| **AVERAGE** | **0.7379 (± 0.0560)** | **0.7282 (± 0.1039)** | **0.7612 (± 0.0742)** | **68.0** | **26.2** | **20.0** | **0.7947 (± 0.0298)** |
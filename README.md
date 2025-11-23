# Model Performance Report (Work in progress, latest run)
This repository implements a deep learning framework for sleep spindle detection using a U-Net architecture with a Transformer Bottleneck. The model leverages a multi-view input approach (Raw EEG, Sigma-filtered, and Teager Energy Operator) to robustly identify spindle events across diverse subjects.

## Architecture Overview
1. Backbone: U-Net (1D Convolutional Neural Network).

2. Bottleneck: Transformer Encoder (replacing the traditional LSTM/BiLSTM). This allows the model to capture global temporal dependencies and long-range context within the signal window.

3. Input: 3-Channel Time-Series:
- Raw EEG 
- Sigma-band filtered signal (11-16 Hz)
- Teager Energy Operator (TEO) signal

4. Loss Function: DiceBCE Loss (Combination of Dice Loss and Binary Cross-Entropy).

## Performance Results (LOSO Cross-Validation)
The model was evaluated using Leave-One-Subject-Out (LOSO) cross-validation on the DREAMS database.

| Subject (Excerpt) | F1-score | Precision | Recall | TP (Events) | FP (Events) | FN (Events) | mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Excerpt 1** | 0.7722 | 0.8000 | 0.7463 | 100 | 25 | 34 | 0.8050 |
| **Excerpt 2** | 0.6207 | 0.9231 | 0.4675 | 36 | 3 | 41 | 0.8088 |
| **Excerpt 3** | 0.7674 | 0.7857 | 0.7500 | 33 | 9 | 11 | 0.7578 |
| **Excerpt 4** | 0.5217 | 0.3724 | 0.8710 | 54 | 91 | 8 | 0.7051 |
| **Excerpt 5** | 0.7736 | 0.7523 | 0.7961 | 82 | 27 | 21 | 0.8188 |
| **Excerpt 6** | 0.7925 | 0.8842 | 0.7179 | 84 | 11 | 33 | 0.7953 |
| **AVERAGE** | **0.7080 (± 0.1012)** | **0.7530 (± 0.1799)** | **0.7248 (± 0.1250)** | **64.8** | **27.7** | **24.7** | **0.7818 (± 0.0393)** |

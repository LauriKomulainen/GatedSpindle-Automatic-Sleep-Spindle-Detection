# Automatic Sleep Spindle Detection

A deep learning framework for detecting sleep spindles in EEG signals using a 1D U-Net. Developed and validated on the DREAMS and MASS-SS2 sleep databases.

## Method

The model is a 1D U-Net with residual encoder/decoder blocks and instance normalization. Inputs are multi-channel time series derived from a single EEG channel (raw bandpass-filtered signal plus task-relevant frequency bands). Training uses a composite Dice + BCE loss and supports Stochastic Weight Averaging together with ensemble inference, where the best checkpoint and the SWA model are combined by averaging logits before the final sigmoid.

Exact hyperparameters, channel definitions, filter bands, window lengths, and post-processing thresholds live in the per-dataset config files under `configs/`.

## Datasets

Place raw recordings (**DREAMS** or **MASS-SS2**) under the directories defined in `paths.py`. Preprocessing (`build_dreams_dataset.py` and `build_mass_dataset.py`) converts EDFs into per-subject `.npy` tensors used by the training pipeline.

## Usage

```bash
# Preprocess raw recordings into .npy tensors
python build_mass_dataset.py

# Train with cross-validation
python run_mass.py --mode train --repeats 3

# Re-evaluate an existing run (e.g. with different post-processing)
python run_mass.py --mode evaluate --run_dir results/<run_timestamp>
```

DREAMS uses the same interface via `run_dreams.py`. See `--help` for full options.

## Environment

Developed on macOS (Apple Silicon, MPS backend). Linux with CUDA should work but is untested. Python and package versions are pinned in `requirements.txt`.

## Project Layout

- `core/` — model, dataset, training, and evaluation logic
- `configs/` — per-dataset hyperparameters and cross-validation configuration
- `data_loaders/` — dataset-specific I/O
- `signal_processing/` — filtering, normalization, and input channel construction
- `postprocessing/` — event extraction and window stitching
- `utils/` — logging, build & run helpers, and plotting

## License & Citation

This project is open-source and available under the MIT License (see the LICENSE file for details). You are free to use, modify, and distribute this software for research and development purposes.

Citation request: If you use this model or code in your research or develop it further, please credit this repository.

Contact: If you encounter issues with the model or have questions regarding the implementation, please contact: laurikom(at)student.uef.fi
# data_preprocess/dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import logging
import random
from scipy.signal import butter, filtfilt

log = logging.getLogger(__name__)


# --- APUFUNKTIOT ---

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def teager_energy_operator(signal):
    """
    Novelty: Korostaa sukkuloiden energiapiikkejä epälineaarisesti.
    """
    teo = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
    teo_padded = np.pad(teo, (1, 1), mode='constant', constant_values=0)
    return np.clip(teo_padded, 0, None)


class RandomAugment1D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        if random.random() < self.p:
            gain = random.uniform(0.8, 1.2)
            signal = signal * gain
            noise_level = random.uniform(0.0, 0.05)
            noise = torch.randn_like(signal) * noise_level
            signal = signal + noise
        return signal


# --- DATASET ---

class SpindleDataset(Dataset):
    def __init__(self, x_1d_path, y_1d_path, seq_len=1, augment=False):
        self.x_1d_path = x_1d_path
        self.y_path = y_1d_path

        # Ladataan kevyet 1D-tiedostot
        self.x_mmap = np.load(x_1d_path, mmap_mode='r')
        self.y_mmap = np.load(y_1d_path, mmap_mode='r')

        self.length = self.x_mmap.shape[0]
        self.fs = 100.0

        self.augment = augment
        self.augmentor = RandomAugment1D(p=0.5)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Raakasignaali
        raw_signal = np.array(self.x_mmap[idx], dtype=np.float32)

        # --- TRI-MODAL SENSORY FUSION (KORJATTU) ---

        # CH1: Context (Raw Broadband)
        ch1 = torch.tensor(raw_signal, dtype=torch.float32).unsqueeze(0)

        # CH2: Focus (Sigma Band 11-16Hz)
        sigma_signal = butter_bandpass_filter(raw_signal, 11.0, 16.0, self.fs, order=4)

        # --- KORJAUS 1: Normalisointi poistettu ---
        # Raakadatan globaali normalisointi on jo tehty (data_handler.py).
        # Ikkunakohtainen Z-score tuhosi hiljaisten alueiden informaation.
        # Poistettu rivit: if np.std > 1e-6: sigma = (sigma - mean) / std

        ch2 = torch.tensor(sigma_signal.copy(), dtype=torch.float32).unsqueeze(0)

        # CH3: Energy (TEO)
        teo_signal = teager_energy_operator(sigma_signal)

        # --- PARANNUS: Skaalaus ennen Tanh-funktiota ---
        # Teager energy -arvot ovat usein hyvin pieniä (esim. 0.01), jolloin tanh on lineaarinen.
        # Kerrotaan 10:llä, jotta voimakkaat piikit "saturoituvat" kohti 1.0:aa
        # ja taustakohina pysyy lähellä nollaa.
        teo_signal = np.tanh(teo_signal * 10.0)

        ch3 = torch.tensor(teo_signal.copy(), dtype=torch.float32).unsqueeze(0)

        # Yhdistetään
        signal_tensor = torch.cat([ch1, ch2, ch3], dim=0)

        # Augmentaatio
        if self.augment:
            signal_tensor = self.augmentor(signal_tensor)

        # 2. Maski (Target)
        mask_1d = np.array(self.y_mmap[idx], dtype=np.float32)
        mask_tensor = torch.tensor(mask_1d, dtype=torch.float32)

        return signal_tensor, mask_tensor


def get_dataloaders(processed_data_dir: str,
                    batch_size: int,
                    train_subject_ids: list,
                    val_subject_ids: list,
                    test_subject_ids: list,
                    use_fraction: float = 1.0):
    log.info(f"Training data: {train_subject_ids}")
    log.info(f"Validation data: {val_subject_ids}")
    log.info(f"Test data: {test_subject_ids}")

    datasets = {}
    all_subjects = list(set(train_subject_ids + val_subject_ids + test_subject_ids))

    for subject_id in all_subjects:
        x_1d_path = os.path.join(processed_data_dir, f"{subject_id}_X_1D.npy")
        y_1d_path = os.path.join(processed_data_dir, f"{subject_id}_Y_1D.npy")

        if not (os.path.exists(x_1d_path) and os.path.exists(y_1d_path)):
            log.warning(f"Skipping {subject_id}, required files not found.")
            continue

        # Ladataan perusdata
        datasets[subject_id] = SpindleDataset(x_1d_path, y_1d_path, augment=False)

    # Rakennetaan Loaderit
    train_list = []
    for sid in train_subject_ids:
        if sid in datasets:
            # Augmentaatio päälle koulutusdatalle
            ds = SpindleDataset(datasets[sid].x_1d_path, datasets[sid].y_path, augment=True)
            train_list.append(ds)

    train_ds = ConcatDataset(train_list) if train_list else []
    val_ds = ConcatDataset([datasets[sid] for sid in val_subject_ids if sid in datasets])
    test_ds = ConcatDataset([datasets[sid] for sid in test_subject_ids if sid in datasets])

    if use_fraction < 1.0:
        def get_subset(ds):
            if len(ds) == 0: return ds
            indices = random.sample(range(len(ds)), int(len(ds) * use_fraction))
            return Subset(ds, indices)

        train_ds = get_subset(train_ds)
        val_ds = get_subset(val_ds)
        test_ds = get_subset(test_ds)

    common = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True}
    return DataLoader(train_ds, shuffle=True, **common), \
        DataLoader(val_ds, shuffle=False, **common), \
        DataLoader(test_ds, shuffle=False, **common)
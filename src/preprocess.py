import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple


def bandpass_filter(epoch: np.ndarray, low: float = 1.0, high: float = 40.0,
                    sfreq: float = 128.0, order: int = 5) -> np.ndarray:
    b, a = butter(order, [low / (sfreq / 2), high / (sfreq / 2)], btype='band')
    return filtfilt(b, a, epoch, axis=-1)


def notch_filter(epoch: np.ndarray, freq: float = 50.0,
                 sfreq: float = 128.0, Q: float = 30.0) -> np.ndarray:
    w0 = freq / (sfreq / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, epoch, axis=-1)


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    mean = np.mean(epoch, axis=1, keepdims=True)
    std = np.std(epoch, axis=1, keepdims=True) + 1e-8
    return (epoch - mean) / std


def preprocess_epochs(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float = 128.0,
    band_low: float = 1.0,
    band_high: float = 40.0,
    notch_freq: float = 50.0,
    notch_Q: float = 30.0,
    artifact_threshold: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    clean_epochs = []
    clean_labels = []
    for epoch, label in zip(X, y):
        # 1) Notch & bandpass
        epoch_notch = notch_filter(epoch, freq=notch_freq, sfreq=sfreq, Q=notch_Q)
        epoch_bp    = bandpass_filter(epoch_notch, low=band_low, high=band_high, sfreq=sfreq)
        # 2) Artifact check
        if np.max(np.abs(epoch_bp)) > artifact_threshold:
            continue  # drop both epoch and its label
        # 3) Normalize & store
        clean_epochs.append(normalize_epoch(epoch_bp))
        clean_labels.append(label)
    return np.array(clean_epochs), np.array(clean_labels)

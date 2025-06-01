# src/eeg_knn_bhho/data_loading.py
"""
Data loading utilities for EEG KNN+BHHO pipeline.
Includes functions to load MAT and STEW datasets and epoch raw data with overlap.
"""
import os
import glob
import numpy as np
import mne
from typing import Tuple, List


def epoch_data_overlap(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 1.0,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Split continuous EEG data into overlapping epochs.

    Parameters
    ----------
    data : np.ndarray
        Raw data array of shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency (Hz).
    epoch_length : float
        Epoch duration in seconds.
    overlap : float
        Fractional overlap between consecutive epochs (0 <= overlap < 1).

    Returns
    -------
    epochs : np.ndarray, shape (n_epochs, n_channels, n_samples_epoch)
    """
    samples_per_epoch = int(round(epoch_length * sfreq))
    if samples_per_epoch <= 0:
        raise ValueError("epoch_length * sfreq must be positive")

    step = int(round(samples_per_epoch * (1 - overlap)))
    if step < 1:
        raise ValueError("overlap too high: step size < 1 sample")

    n_channels, n_samples = data.shape
    starts = np.arange(0, n_samples - samples_per_epoch + 1, step)
    if starts.size == 0:
        return np.empty((0, n_channels, samples_per_epoch), dtype=data.dtype)

    epochs = np.stack([data[:, start:start + samples_per_epoch] for start in starts], axis=0)
    return epochs


def load_mat_epochs(
    base_dir: str,
    epoch_length: float = 1.0,
    resample_sfreq: float = 128.0,
    overlap: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MAT dataset EDF files and split into overlapping epochs.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing `Subject??_1.edf` and `_2.edf` files.
    epoch_length : float
        Epoch length in seconds.
    resample_sfreq : float
        Sampling frequency to resample the raw data.
    overlap : float
        Fractional overlap between epochs.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray, shape (n_epochs,)
        Labels: 0 for rest, 1 for task.
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"MAT directory not found: {base_dir}")

    rest_files = sorted(glob.glob(os.path.join(base_dir, "Subject??_1.edf")))
    if not rest_files:
        raise FileNotFoundError(f"No MAT rest files in {base_dir}")

    rest_epochs: List[np.ndarray] = []
    task_epochs: List[np.ndarray] = []

    for rest_path in rest_files:
        task_path = rest_path.replace("_1.edf", "_2.edf")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Missing task file for {rest_path}")

        raw_rest = mne.io.read_raw_edf(rest_path, preload=True, verbose=False)
        raw_task = mne.io.read_raw_edf(task_path, preload=True, verbose=False)

        raw_rest.resample(resample_sfreq)
        raw_task.resample(resample_sfreq)

        data_r = raw_rest.get_data()
        data_t = raw_task.get_data()

        epochs_r = epoch_data_overlap(data_r, resample_sfreq, epoch_length, overlap)
        epochs_t = epoch_data_overlap(data_t, resample_sfreq, epoch_length, overlap)

        rest_epochs.append(epochs_r)
        task_epochs.append(epochs_t)

    rest_all = np.vstack(rest_epochs)
    task_all = np.vstack(task_epochs)
    X = np.concatenate([rest_all, task_all], axis=0)
    y = np.concatenate([
        np.zeros(len(rest_all), dtype=int),
        np.ones(len(task_all), dtype=int)
    ], axis=0)

    return X, y


def load_stew_epochs(
    base_dir: str,
    epoch_length: float = 1.0,
    sfreq: float = 256.0,
    overlap: float = 0.5,
    file_pattern: str = "sub??_*.txt"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load STEW dataset text files and split into overlapping epochs.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing STEW .txt files.
    epoch_length : float
        Epoch length in seconds.
    sfreq : float
        Sampling frequency of the raw text data.
    overlap : float
        Fractional overlap between epochs.
    file_pattern : str
        Glob pattern to match STEW files (rest vs. hi).

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray, shape (n_epochs,)
        Labels: 0 for rest, 1 for high workload.
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"STEW directory not found: {base_dir}")

    files = sorted(glob.glob(os.path.join(base_dir, file_pattern)))
    if not files:
        raise FileNotFoundError(f"No STEW files in {base_dir}")

    rest_epochs: List[np.ndarray] = []
    task_epochs: List[np.ndarray] = []

    for path in files:
        label = 1 if "_hi" in os.path.basename(path) else 0
        raw = np.loadtxt(path).T  # shape (n_channels, n_samples)
        epochs = epoch_data_overlap(raw, sfreq, epoch_length, overlap)

        if label == 0:
            rest_epochs.append(epochs)
        else:
            task_epochs.append(epochs)

    rest_all = np.vstack(rest_epochs)
    task_all = np.vstack(task_epochs)
    X = np.concatenate([rest_all, task_all], axis=0)
    y = np.concatenate([
        np.zeros(len(rest_all), dtype=int),
        np.ones(len(task_all), dtype=int)
    ], axis=0)

    return X, y

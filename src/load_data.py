import os
import glob
import numpy as np
import mne
from src.epoching import epoch_data_overlap  # we’ll define this in a sec


def epoch_data_overlap(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 1.0,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Split continuous data into overlapping epochs.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz.
    epoch_length : float
        Length of each epoch (seconds).
    overlap : float
        Fraction overlap between consecutive epochs [0,1).

    Returns
    -------
    epochs : np.ndarray, shape (n_epochs, n_channels, n_samples_per_epoch)
    """
    samples_per_epoch = int(round(epoch_length * sfreq))
    if samples_per_epoch <= 0:
        raise ValueError("epoch_length * sfreq must be > 0")
    step = int(round(samples_per_epoch * (1 - overlap)))
    if step < 1:
        raise ValueError("overlap too high, step becomes < 1 sample")

    n_channels, n_samples = data.shape
    # compute start indices
    starts = np.arange(0, n_samples - samples_per_epoch + 1, step)
    if len(starts) == 0:
        return np.empty((0, n_channels, samples_per_epoch), dtype=data.dtype)

    # build epochs
    epochs = np.stack([
        data[:, i : i + samples_per_epoch]
        for i in starts
    ], axis=0)  # shape (n_epochs, n_channels, samples_per_epoch)

    return epochs


def load_mat_epochs(
    base_dir: str,
    epoch_length: float = 1.0,
    resample_sfreq: float = 128.0,
    overlap: float = 0.5
) -> (np.ndarray, np.ndarray):
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"MAT directory not found: {base_dir}")

    # should match Subject01_1.edf, Subject02_1.edf, etc.
    rest_files = sorted(glob.glob(os.path.join(base_dir, "Subject??_1.edf")))
    if not rest_files:
        raise FileNotFoundError(f"No MAT rest files found in {base_dir}")

    rest_epochs, task_epochs = [], []
    for rest_path in rest_files:
        task_path = rest_path.replace("_1.edf", "_2.edf")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Missing task file for {rest_path}")

        # load & resample
        raw_rest = mne.io.read_raw_edf(rest_path, preload=True, verbose=False)
        raw_task = mne.io.read_raw_edf(task_path, preload=True, verbose=False)
        raw_rest.resample(resample_sfreq)
        raw_task.resample(resample_sfreq)

        data_r = raw_rest.get_data()  # (n_channels, samples)
        data_t = raw_task.get_data()

        # epoch with overlap
        epochs_r = epoch_data_overlap(data_r, raw_rest.info['sfreq'], epoch_length, overlap)
        epochs_t = epoch_data_overlap(data_t, raw_task.info['sfreq'], epoch_length, overlap)

        rest_epochs.append(epochs_r)
        task_epochs.append(epochs_t)

    # stack, label, return
    rest_all = np.vstack(rest_epochs)   # now non-empty
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
) -> (np.ndarray, np.ndarray):
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"STEW directory not found: {base_dir}")

    files = sorted(glob.glob(os.path.join(base_dir, file_pattern)))
    if not files:
        raise FileNotFoundError(f"No STEW files found in {base_dir}")

    rest_epochs, task_epochs = [], []
    for path in files:
        label = 1 if "_hi" in os.path.basename(path) else 0
        raw = np.loadtxt(path).T  # (n_channels, samples)
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

import numpy as np


def epoch_data_overlap(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 1.0,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Split a continuous multichannel EEG signal into overlapping epochs.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Raw EEG time-series for one recording session.
    sfreq : float
        Sampling frequency in Hz.
    epoch_length : float
        Length of each epoch in seconds.
    overlap : float
        Fractional overlap between consecutive epochs (0 <= overlap < 1).

    Returns
    -------
    epochs : np.ndarray, shape (n_epochs, n_channels, samples_per_epoch)
        Array of overlapping epochs extracted from `data`.

    Notes
    -----
    - If overlap=0.5, then each epoch starts half an epoch-length after the previous.
    - The last segment is only included if it exactly fits; leftover samples are discarded.
    """
    samples_per_epoch = int(epoch_length * sfreq)
    step = int(samples_per_epoch * (1 - overlap))
    n_channels, n_samples = data.shape

    epochs = []
    # Slide window by `step` until we can no longer get a full epoch
    for start in range(0, n_samples - samples_per_epoch + 1, step):
        end = start + samples_per_epoch
        epochs.append(data[:, start:end])

    return np.array(epochs)

# src/eeg_knn_bhho/preprocessing.py
"""
Preprocessing utilities for EEG KNN+BHHO pipeline.
Includes Transformers for bandpass/notch filtering and epoch normalization.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, TransformerMixin


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sfreq: float,
    order: int = 5
) -> np.ndarray:
    """
    Apply a zero-phase bandpass Butterworth filter to multichannel data.

    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_channels, n_samples).
    low_freq : float
        Low cutoff frequency in Hz.
    high_freq : float
        High cutoff frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered : np.ndarray
        Bandpass-filtered data of same shape as input.
    """
    nyq = 0.5 * sfreq
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, data, axis=-1)


def notch_filter(
    data: np.ndarray,
    freq: float,
    sfreq: float,
    quality: float = 30.0
) -> np.ndarray:
    """
    Apply a zero-phase notch filter at a specific frequency.

    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_channels, n_samples).
    freq : float
        Notch (center) frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    quality : float
        Quality factor for the notch filter.

    Returns
    -------
    filtered : np.ndarray
        Notch-filtered data of same shape as input.
    """
    nyq = 0.5 * sfreq
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, data, axis=-1)


def normalize_epoch(
    epoch: np.ndarray
) -> np.ndarray:
    """
    Z-score normalize each channel of an epoch.

    Parameters
    ----------
    epoch : np.ndarray
        Single-epoch data of shape (n_channels, n_samples).

    Returns
    -------
    normalized : np.ndarray
    """
    mean = np.mean(epoch, axis=1, keepdims=True)
    std = np.std(epoch, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (epoch - mean) / std


class BandpassNotchFilter(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a notch filter followed by a bandpass filter to each epoch.

    This wrapper can be used inside a scikit-learn Pipeline.
    """
    def __init__(
        self,
        band_low: float = 1.0,
        band_high: float = 40.0,
        notch_freq: float = 50.0,
        notch_Q: float = 30.0,
        sfreq: float = 128.0,
        order: int = 5
    ):
        self.band_low = band_low
        self.band_high = band_high
        self.notch_freq = notch_freq
        self.notch_Q = notch_Q
        self.sfreq = sfreq
        self.order = order

    def fit(self, X: np.ndarray, y=None):
        """
        No fitting necessary; this transformer is stateless.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply notch and bandpass filtering to each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)

        Returns
        -------
        X_filtered : np.ndarray
            Filtered epochs of same shape.
        """
        n_epochs, n_ch, n_s = X.shape
        X_filtered = np.zeros_like(X, dtype=np.float64)

        for i in range(n_epochs):
            epoch = X[i]  # shape: (n_channels, n_samples)
            # Notch filter
            epoch_notch = notch_filter(epoch, self.notch_freq, self.sfreq, self.notch_Q)
            # Bandpass filter
            epoch_bp = bandpass_filter(epoch_notch, self.band_low, self.band_high, self.sfreq, self.order)
            X_filtered[i] = epoch_bp

        return X_filtered


class EpochNormalizer(BaseEstimator, TransformerMixin):
    """
    Transformer that z-score normalizes each channel of each epoch.

    This wrapper can be used inside a scikit-learn Pipeline.
    """
    def fit(self, X: np.ndarray, y=None):
        """
        No fitting necessary; this transformer is stateless.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Z-score normalize each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)

        Returns
        -------
        X_norm : np.ndarray, same shape as X
        """
        n_epochs, n_ch, n_s = X.shape
        X_norm = np.zeros_like(X, dtype=np.float64)

        for i in range(n_epochs):
            X_norm[i] = normalize_epoch(X[i])

        return X_norm

# src/eeg_knn_bhho/decomposition.py
"""
CiSSA decomposition module for EEG KNN+BHHO pipeline.
Provides a transformer to extract narrow-band IMFs per channel via Circulant SSA.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


def cissa_1d(signal: np.ndarray, n_imfs: int = 3) -> np.ndarray:
    """
    Circulant SSA decomposition to extract top n_imfs narrow-band components.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        1-D time series.
    n_imfs : int
        Number of components to extract (excluding DC).

    Returns
    -------
    comps : np.ndarray, shape (n_imfs, n_samples)
        Reconstructed narrow-band components.
    """
    N = signal.shape[0]
    Xf = np.fft.fft(signal)
    mag = np.abs(Xf)
    half = N // 2 + 1

    # Exclude DC (index 0) from ranking if desired: here we rank mag[1:half]
    idx_sorted = np.argsort(mag[1:half])[::-1] + 1  # shift by +1 to compare against original FFT indices
    selected = idx_sorted[:n_imfs]

    comps = np.zeros((n_imfs, N), dtype=float)
    for i, k in enumerate(selected):
        mask = np.zeros(N, dtype=bool)
        mask[k] = True
        if 0 < k < half:
            mask[-k] = True
        Xc = np.zeros_like(Xf)
        Xc[mask] = Xf[mask]
        comps[i] = np.real(np.fft.ifft(Xc))
    return comps


def decompose_epoch_cissa(epoch: np.ndarray, n_imfs: int = 3) -> np.ndarray:
    """
    Apply cissa_1d to each channel in an epoch and stack.

    Parameters
    ----------
    epoch : np.ndarray, shape (n_channels, n_samples)
    n_imfs : int
        Number of IMFs per channel.

    Returns
    -------
    stacked : np.ndarray, shape (n_channels * n_imfs, n_samples)
    """
    n_ch, _ = epoch.shape
    channel_comps = [cissa_1d(epoch[ch], n_imfs) for ch in range(n_ch)]
    # Each entry in channel_comps has shape (n_imfs, n_samples)
    # Stack them so total shape = (n_ch * n_imfs, n_samples)
    return np.vstack(channel_comps)


class CISSADecomposer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies CiSSA decomposition to each epoch.

    Parameters
    ----------
    n_imfs : int
        Number of IMFs to extract per channel.
    n_jobs : int
        Number of parallel jobs for decomposition. -1 uses all cores.
    """
    def __init__(self, n_imfs: int = 3, n_jobs: int = 1):
        self.n_imfs = n_imfs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        No fitting necessary; CiSSA is applied transform-time.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CiSSA decomposition to each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)

        Returns
        -------
        X_cissa : np.ndarray, shape (n_epochs, n_channels * n_imfs, n_samples)
        """
        # Parallel processing over epochs
        decomposed = Parallel(n_jobs=self.n_jobs)(
            delayed(decompose_epoch_cissa)(epoch, self.n_imfs)
            for epoch in X
        )
        return np.array(decomposed)

# src/eeg_knn_bhho/connectivity.py
"""
Compute functional connectivity features for EEG epochs:
 - Pairwise spectral coherence between all channel pairs via Welch’s method.

The output for each epoch is a 1D vector of length C = n_channels * (n_channels-1) / 2.
"""

import numpy as np
from scipy.signal import coherence
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


class ConnectivityExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer to compute pairwise spectral coherence for each epoch.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    nperseg : int
        Segment length for Welch-based coherence.
    n_jobs : int
        Parallel jobs for computing coherence across epochs.
    """
    def __init__(self, sfreq: float, nperseg: int = 64, n_jobs: int = 1):
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        No fitting needed; stateless transformer.
        """
        return self

    def _epoch_coherence(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute pairwise coherence for one epoch.

        Parameters
        ----------
        epoch : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        coh_vector : np.ndarray, shape (n_pairs,)
            Flattened upper-triangular coherence values (averaged over all frequencies).
        """
        n_ch, n_s = epoch.shape
        # Prepare container for pairwise coherence
        pairs = []
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                f, Cxy = coherence(epoch[i], epoch[j], fs=self.sfreq, nperseg=self.nperseg)
                # Average coherence across frequencies
                mean_coh = np.mean(Cxy)
                pairs.append(mean_coh)
        return np.array(pairs, dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute connectivity for all epochs.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)

        Returns
        -------
        conn_feats : np.ndarray, shape (n_epochs, n_pairs)
        """
        n_epochs = X.shape[0]
        # Parallelize across epochs
        coh_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._epoch_coherence)(X[i]) for i in range(n_epochs)
        )
        return np.vstack(coh_list)  # shape: (n_epochs, n_pairs)

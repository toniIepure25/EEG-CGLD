# src/eeg_knn_bhho/ica_preprocessing.py

"""
ICA-based artifact removal for EEG epochs.
Uses MNE’s ICA on continuous data to identify and remove artifact components,
then reconstructs cleaned epochs.

Steps:
 1. Concatenate epochs into a continuous MNE Raw object (with dummy info).
 2. Fit ICA on that continuous data.
 3. Identify components to exclude based on statistical heuristics (e.g., kurtosis).
 4. Apply inverse transform to get cleaned data.
 5. Re-slice into epochs.

Note: In a real study, you might identify EOG/ECG artifacts via correlation, but
here we demonstrate a simple statistical approach. You can refine component
selection rules (e.g., manually inspect top‐k components, use automated methods).
"""
import numpy as np
import mne
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class ICACleaner(BaseEstimator, TransformerMixin):
    """
    Transformer that applies ICA to a set of EEG epochs to remove artifact components.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the epochs (Hz).
    n_components : Optional[int]
        Number of ICA components to retain. If None, uses all channels.
    kurtosis_thresh : float
        Threshold on absolute kurtosis of IC activations to mark as artifact. 
        Any component whose absolute kurtosis exceeds this is excluded.
    random_state : Optional[int]
        Seed for reproducibility of ICA.
    """
    def __init__(
        self,
        sfreq: float,
        n_components: Optional[int] = None,
        kurtosis_thresh: float = 5.0,
        random_state: Optional[int] = 42
    ):
        self.sfreq = sfreq
        self.n_components = n_components
        self.kurtosis_thresh = kurtosis_thresh
        self.random_state = random_state
        self._ica = None           # will hold the fitted mne.preprocessing.ICA instance
        self._info = None          # will hold channel info for Raw creation
        self._orig_ch_names = None # original channel names

    def fit(self, X: np.ndarray, y=None):
        """
        Fit ICA on the concatenated epochs.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)
            The raw EEG epochs.

        Returns
        -------
        self
        """
        n_epochs, n_ch, n_samp = X.shape
        self._orig_ch_names = [f"EEG{i}" for i in range(n_ch)]

        # 1. Create MNE Info (no actual montage, dummy channel names)
        self._info = mne.create_info(
            ch_names=self._orig_ch_names,
            sfreq=self.sfreq,
            ch_types=["eeg"] * n_ch
        )

        # 2. Concatenate epochs into one big array (shape: n_ch, n_epochs * n_samp)
        data_cat = X.transpose(1, 0, 2).reshape(n_ch, n_epochs * n_samp)

        # 3. Create RawArray
        raw = mne.io.RawArray(data_cat, self._info, verbose=False)

        # 4. Fit ICA
        ica = mne.preprocessing.ICA(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter="auto",
            verbose=False
        )
        ica.fit(raw)

        # 5. Identify components to exclude by kurtosis heuristic
        sources = ica.get_sources(raw).get_data()  # shape: (n_components, total_samples)
        # Compute kurtosis along time‐axis for each component
        comp_kurtosis = np.apply_along_axis(
            lambda u: (((u - u.mean()) / (u.std() + 1e-12))**4).mean() - 3,
            axis=1,
            arr=sources
        )
        # Mark any component whose |kurtosis| > threshold as artifact
        exclude = list(np.where(np.abs(comp_kurtosis) > self.kurtosis_thresh)[0])
        ica.exclude = exclude

        # Store fitted ICA
        self._ica = ica
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted ICA to remove artifact components and reconstruct cleaned epochs.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_samples)
            The raw EEG epochs.

        Returns
        -------
        X_clean : np.ndarray, shape (n_epochs, n_channels, n_samples)
            The ICA‐cleaned EEG epochs.
        """
        n_epochs, n_ch, n_samp = X.shape
        if self._ica is None:
            raise RuntimeError("ICACleaner not fitted. Call `.fit(X)` first.")

        # 1. Concatenate input epochs the same way as in fit
        data_cat = X.transpose(1, 0, 2).reshape(n_ch, n_epochs * n_samp)
        raw = mne.io.RawArray(data_cat, self._info, verbose=False)

        # 2. Apply ICA removal
        self._ica.apply(raw)

        # 3. Retrieve cleaned data and reshape back to epochs
        cleaned_data = raw.get_data()  # shape: (n_ch, n_epochs * n_samp)
        X_clean = cleaned_data.reshape(n_ch, n_epochs, n_samp).transpose(1, 0, 2)
        return X_clean

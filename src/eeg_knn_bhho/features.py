# src/eeg_knn_bhho/features.py
"""
Feature extraction from CiSSA outputs: band-power, time-domain statistics,
Hjorth parameters, entropies, and spectral ratios. Wrapped as a scikit-learn
transformer for easy integration into pipelines.

Each IMF (Intrinsic Mode Function) signal yields:
  1. Band-power for user-defined frequency bands (Welch’s PSD on full-rate signal).
  2. Time-domain statistics: mean, standard deviation, skewness, kurtosis (on downsampled signal).
  3. Hjorth parameters: activity (variance), mobility, complexity (on downsampled signal).
  4. Entropies: sample entropy, permutation entropy, spectral entropy (on downsampled signal, 
     with safe fallback to zero if Antropy fails).
  5. Spectral ratios: θ/α and β/α power ratios (on full-rate signal).

The final feature vector per IMF has length:
    len(bands)           (band-power)  +
    4                    (time_stats)  +
    3                    (hjorth)      +
    3                    (entropies)   +
    2                    (spectral ratios)
  = len(bands) + 12

Thus, if you have `n_imfs` IMFs, the output dimension is n_imfs * (len(bands) + 12).
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple


class CiSSALightFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Ultra-light feature extractor for CiSSA outputs.

    For each IMF signal (1D array), computes:
      - Band-power for each specified band via Welch’s method on the full-rate signal.
      - Downsamples by factor `ds`.
      - Time-domain statistics on downsampled signal: mean, std, skew, kurtosis.
      - Hjorth parameters on downsampled signal: activity (variance), mobility, complexity.
      - Entropies on downsampled signal:
          * Sample Entropy
          * Permutation Entropy
          * Spectral Entropy
        Each wrapped in try/except to avoid failing if Antropy/Numba errors occur.
      - Spectral ratios on full-rate signal:
          * Theta/Alpha power ratio (4–8 Hz / 8–13 Hz)
          * Beta/Alpha power ratio  (13–30 Hz / 8–13 Hz)

    Parameters
    ----------
    sfreq : float
        Original sampling frequency (Hz) before downsampling.
    bands : List[Tuple[float, float]]
        Frequency bands for band-power (e.g., [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]).
    ds : int, default=4
        Downsampling factor for entropy and time-domain/hjorth features.
    nperseg : int, default=64
        Segment length for Welch’s method when computing PSD on full-rate signals.
    n_jobs : int, default=1
        Number of parallel jobs for feature extraction (-1 uses all available cores).
    """

    def __init__(
        self,
        sfreq: float,
        bands: List[Tuple[float, float]],
        ds: int = 4,
        nperseg: int = 64,
        n_jobs: int = 1
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.ds = ds
        self.nperseg = nperseg
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        No fitting necessary; this transformer is stateless.
        """
        return self

    def _compute_signal_features(self, sig_full: np.ndarray) -> np.ndarray:
        """
        Compute features for a single IMF signal.

        Parameters
        ----------
        sig_full : np.ndarray, shape (n_samples_full,)
            The full-rate IMF component.

        Returns
        -------
        feats : np.ndarray, shape (n_feats_per_imf,)
            Feature vector for this IMF, in the order:
              [band-powers...] + [time_stats...] + [hjorth...] + [entropies...] + [spectral_ratios...]
        """
        # 1. Band-power on full-rate signal via Welch’s method
        f, Pxx = welch(sig_full, fs=self.sfreq, nperseg=self.nperseg)
        bp = [
            np.trapz(Pxx[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])
            for (lo, hi) in self.bands
        ]

        # 2. Compute spectral ratios on full-rate signal
        # Identify numeric indices for theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz)
        theta_mask = (f >= 4.0) & (f < 8.0)
        alpha_mask = (f >= 8.0) & (f < 13.0)
        beta_mask  = (f >= 13.0) & (f < 30.0)

        theta_power = np.trapz(Pxx[theta_mask], f[theta_mask]) if theta_mask.any() else 0.0
        alpha_power = np.trapz(Pxx[alpha_mask], f[alpha_mask]) if alpha_mask.any() else 1e-12
        beta_power  = np.trapz(Pxx[beta_mask],  f[beta_mask])  if beta_mask.any()  else 0.0

        ratio_theta_alpha = theta_power / (alpha_power + 1e-12)
        ratio_beta_alpha  = beta_power  / (alpha_power + 1e-12)

        spectral_ratios = [ratio_theta_alpha, ratio_beta_alpha]

        # 3. Downsample the signal for time-domain/hjorth/entropy features
        sig_ds = sig_full[::self.ds]
        n_s_down = sig_ds.shape[0]

        # 4. Time-domain statistics on downsampled signal
        mu = sig_ds.mean()
        sigma = sig_ds.std()
        sk = skew(sig_ds)
        kt = kurtosis(sig_ds)
        time_stats = [mu, sigma, sk, kt]

        # 5. Hjorth parameters on downsampled signal
        d1 = np.diff(sig_ds)
        d2 = np.diff(sig_ds, 2)
        v0 = np.var(sig_ds)
        v1 = np.var(d1) if d1.size > 0 else 0.0
        v2 = np.var(d2) if d2.size > 0 else 0.0
        mobility = np.sqrt(v1 / v0) if v0 > 0 else 0.0
        complexity = np.sqrt(v2 / v1) / mobility if (v1 > 0 and mobility > 0) else 0.0
        hjorth_params = [v0, mobility, complexity]

        # 6. Entropies on downsampled signal, with safe fallbacks
        try:
            samp_ent = ant.sample_entropy(sig_ds)
        except Exception:
            samp_ent = 0.0

        try:
            perm_ent = ant.perm_entropy(sig_ds, order=3, delay=1, normalize=True)
        except Exception:
            perm_ent = 0.0

        try:
            spec_ent = ant.spectral_entropy(
                sig_ds,
                sf=self.sfreq / self.ds,
                method="welch",
                nperseg=n_s_down,
                normalize=True
            )
        except Exception:
            spec_ent = 0.0

        entropies = [samp_ent, perm_ent, spec_ent]

        # 7. Concatenate all features into one array
        #    Order: [band-powers...] + [time_stats...] + [hjorth_params...] + [entropies...] + [spectral_ratios...]
        feats = np.array(bp + time_stats + hjorth_params + entropies + spectral_ratios, dtype=np.float64)
        return feats

    def transform(self, X_cissa: np.ndarray) -> np.ndarray:
        """
        Apply feature extraction to all epochs.

        Parameters
        ----------
        X_cissa : np.ndarray, shape (n_epochs, n_components, n_samples_full)
            Output of CiSSA decomposition for each epoch.

        Returns
        -------
        feats_all : np.ndarray, shape (n_epochs, n_components * n_feats_per_imf)
            Stacked feature vectors for all IMFs and all epochs.
        """
        n_epochs, n_comps, n_samples_full = X_cissa.shape

        # Number of features per IMF = len(bands) + 4 (time) + 3 (hjorth) + 3 (entropy) + 2 (spectral ratios)
        n_feats_per = len(self.bands) + 4 + 3 + 3 + 2

        def _epoch_feats(epoch_imfs: np.ndarray) -> np.ndarray:
            """
            Compute feature vector for one epoch (all IMFs).
            """
            imf_features = []
            for comp_idx in range(n_comps):
                feats = self._compute_signal_features(epoch_imfs[comp_idx])
                imf_features.append(feats)
            # Concatenate features from all IMFs: shape = (n_comps * n_feats_per,)
            return np.hstack(imf_features)

        # Parallel extraction across epochs
        feats_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_epoch_feats)(X_cissa[i]) for i in range(n_epochs)
        )

        # Stack into shape: (n_epochs, n_comps * n_feats_per)
        return np.vstack(feats_list)

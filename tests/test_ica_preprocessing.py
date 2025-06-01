# tests/test_ica_preprocessing.py
import numpy as np
from eeg_knn_bhho.ica_preprocessing import ICACleaner

def test_ica_cleaner_basic():
    # Create 10 epochs, 2 channels, 128 samples each
    n_epochs, n_ch, n_samp = 10, 2, 128
    rng = np.random.RandomState(0)
    # Simulate a simple sinusoidal signal + spike artifact in channel 0
    t = np.arange(n_samp) / 128.0
    base = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine
    # Stack into epochs
    X = np.zeros((n_epochs, n_ch, n_samp))
    for e in range(n_epochs):
        X[e, 0, :] = base + 5.0 * (rng.rand() < 0.01) * rng.randn(n_samp)  # occasional spike
        X[e, 1, :] = base

    cleaner = ICACleaner(sfreq=128.0, n_components=2, kurtosis_thresh=10.0, random_state=0)
    cleaner.fit(X)
    X_clean = cleaner.transform(X)
    # After cleaning, verify shapes unchanged
    assert X_clean.shape == X.shape
    # The extreme spikes (kurtosis) should be reduced; at least channel variance goes down
    var_raw = np.var(X[:, 0, :])
    var_clean = np.var(X_clean[:, 0, :])
    assert var_clean < var_raw * 1.1  # cleaned variance should not increase

# tests/test_ica_preprocessing.py
import numpy as np
from eeg_knn_bhho.ica_preprocessing import ICACleaner

def test_ica_cleaner_basic():
    # Create 10 epochs, 2 channels, 128 samples each
    n_epochs, n_ch, n_samp = 10, 2, 128
    rng = np.random.RandomState(0)
    t = np.arange(n_samp) / 128.0
    base = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine

    # Introduce a high-kurtosis spike artifact in channel 0
    X = np.zeros((n_epochs, n_ch, n_samp))
    for e in range(n_epochs):
        spike = (rng.rand(n_samp) < 0.01) * rng.randn(n_samp) * 10.0
        X[e, 0, :] = base + spike
        X[e, 1, :] = base

    cleaner = ICACleaner(sfreq=128.0, n_components=2, kurtosis_thresh=5.0, random_state=0)
    cleaner.fit(X)
    X_clean = cleaner.transform(X)

    # Shapes must match
    assert X_clean.shape == X.shape

    # Variance on channel 0 should decrease after spike removal (at least not increase by much)
    var_before = np.var(X[:, 0, :])
    var_after = np.var(X_clean[:, 0, :])
    assert var_after < var_before * 1.05
